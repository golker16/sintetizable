import sys
import numpy as np
import soundfile as sf

# librosa imports mínimos (sin importar librosa “entero”)
from librosa.core.audio import load as lr_load
from librosa.core.pitch import pyin as lr_pyin
from librosa.core.convert import note_to_hz as lr_note_to_hz, hz_to_midi as lr_hz_to_midi

from PySide6.QtCore import QObject, QThread, Signal, Qt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QProgressBar,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QMessageBox,
    QSpacerItem,
    QSizePolicy,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ----------------- DEFAULTS -----------------

FMIN_NOTE = "C2"
FMAX_NOTE = "C7"
DEFAULT_FRAME_LENGTH = 2048
DEFAULT_HOP_LENGTH = 256


# ----------------- DSP -----------------

def load_mono(path: str):
    y, sr = lr_load(path, sr=None, mono=True)
    return y.astype(np.float32, copy=False), int(sr)


def frame_rms(y: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < frame_length:
        y = np.pad(y, (0, frame_length - n))
        n = len(y)

    n_frames = 1 + (n - frame_length) // hop_length
    rms = np.empty(n_frames, dtype=float)

    for i in range(n_frames):
        start = i * hop_length
        frame = y[start:start + frame_length]
        rms[i] = np.sqrt(np.mean(frame * frame) + 1e-12)

    return rms


def one_pole_smooth(x: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 1e-6, 1.0))
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
    return y


def extract_pitch_voicing_and_env(
    y: np.ndarray,
    sr: int,
    frame_length: int,
    hop_length: int,
    env_smooth_alpha: float,
    f0_smooth_alpha: float,
    gate_unvoiced: bool,
):
    # Pitch tracking
    f0, voiced_flag, voiced_prob = lr_pyin(
        y,
        fmin=lr_note_to_hz(FMIN_NOTE),
        fmax=lr_note_to_hz(FMAX_NOTE),
        frame_length=frame_length,
        hop_length=hop_length,
    )

    f0 = np.asarray(f0, dtype=float)
    voiced = np.asarray(voiced_flag, dtype=bool)

    valid = np.where(~np.isnan(f0))[0]
    if len(valid) == 0:
        raise RuntimeError("No se pudo detectar pitch (F0) en el audio fuente.")

    # Rellenar NaNs para continuidad (el gate luego decide si suena)
    f0_clean = f0.copy()
    last = f0_clean[valid[0]]
    for i in range(len(f0_clean)):
        if np.isnan(f0_clean[i]):
            f0_clean[i] = last
        else:
            last = f0_clean[i]

    # Suavizar F0 para evitar “temblor”/vibrato raro
    f0_clean = one_pole_smooth(f0_clean, alpha=f0_smooth_alpha)

    # Envelope por RMS
    env = frame_rms(y, frame_length=frame_length, hop_length=hop_length)
    env = env / (np.max(env) + 1e-12)

    if gate_unvoiced:
        env = env * voiced.astype(float)

    env = one_pole_smooth(env, alpha=env_smooth_alpha)

    return f0_clean, voiced, env


def synth_from_f0(
    f0_frames_hz: np.ndarray,
    env_frames: np.ndarray,
    sr: int,
    n_samples: int,
    frame_length: int,
    hop_length: int,
    waveform: str,
    harmonics: int,
):
    f0_frames_hz = np.asarray(f0_frames_hz, float)
    env_frames = np.asarray(env_frames, float)

    # centros de frame en samples
    centers = (np.arange(len(f0_frames_hz)) * hop_length + frame_length / 2.0)
    centers = np.clip(centers, 0, max(0, n_samples - 1))

    t = np.arange(n_samples, dtype=float)

    # Interpolar a nivel sample
    f0 = np.interp(t, centers, f0_frames_hz)
    env = np.interp(t, centers, env_frames)

    f0 = np.clip(f0, 1.0, sr / 2.0)
    phase = np.cumsum(2.0 * np.pi * f0 / sr)

    if waveform == "sine":
        sig = np.sin(phase)

    elif waveform == "saw":
        H = max(1, int(harmonics))
        sig = np.zeros_like(phase)
        for k in range(1, H + 1):
            sig += (1.0 / k) * np.sin(k * phase)
        sig /= (np.max(np.abs(sig)) + 1e-12)

    elif waveform == "square":
        H = max(1, int(harmonics))
        sig = np.zeros_like(phase)
        # impares
        for k in range(1, 2 * H, 2):
            sig += (1.0 / k) * np.sin(k * phase)
        sig /= (np.max(np.abs(sig)) + 1e-12)

    else:
        raise ValueError("waveform debe ser 'sine', 'saw' o 'square'")

    out = sig * env
    out = np.clip(out, -1.0, 1.0)
    return out.astype(np.float32, copy=False)


# ----------------- PLOT -----------------

class PitchPlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(5, 2))
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        self.setMinimumHeight(170)

    def plot_midi(self, midi_curve: np.ndarray):
        self.ax.clear()
        self.ax.plot(midi_curve, linewidth=1)
        self.ax.set_title("Curva de afinación (MIDI)")
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Nota MIDI")
        self.ax.grid(True, alpha=0.3)
        self.draw()

    def clear_plot(self):
        self.ax.clear()
        self.draw()


# ----------------- WORKER -----------------

class AudioWorker(QObject):
    progress = Signal(int)
    log = Signal(str)
    finished = Signal()
    error = Signal(str)
    pitch_midi = Signal(object)

    def __init__(
        self,
        src_path: str,
        out_path: str,
        waveform: str,
        harmonics: int,
        hop_length: int,
        frame_length: int,
        env_alpha: float,
        f0_alpha: float,
        gate_unvoiced: bool,
        output_gain: float,
    ):
        super().__init__()
        self.src_path = src_path
        self.out_path = out_path
        self.waveform = waveform
        self.harmonics = harmonics
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.env_alpha = env_alpha
        self.f0_alpha = f0_alpha
        self.gate_unvoiced = gate_unvoiced
        self.output_gain = output_gain

    def run(self):
        try:
            self.log.emit("Procesando: Pitch map + Envelope → Síntesis")

            self.progress.emit(5)
            self.log.emit("Cargando audio fuente...")
            y, sr = load_mono(self.src_path)
            self.log.emit(f"  Fuente: {len(y)} samples, sr={sr}")

            self.progress.emit(25)
            self.log.emit("Extrayendo F0/voicing + envelope (RMS)...")
            f0_frames, voiced, env_frames = extract_pitch_voicing_and_env(
                y=y,
                sr=sr,
                frame_length=self.frame_length,
                hop_length=self.hop_length,
                env_smooth_alpha=self.env_alpha,
                f0_smooth_alpha=self.f0_alpha,
                gate_unvoiced=self.gate_unvoiced,
            )

            midi_frames = lr_hz_to_midi(f0_frames)
            self.pitch_midi.emit(midi_frames)

            self.progress.emit(60)
            self.log.emit("Sintetizando señal...")
            out = synth_from_f0(
                f0_frames_hz=f0_frames,
                env_frames=env_frames,
                sr=sr,
                n_samples=len(y),
                frame_length=self.frame_length,
                hop_length=self.hop_length,
                waveform=self.waveform,
                harmonics=self.harmonics,
            )

            self.progress.emit(80)
            if self.output_gain != 1.0:
                out = np.clip(out * float(self.output_gain), -1.0, 1.0)

            self.log.emit(f"Guardando WAV: {self.out_path}")
            self.progress.emit(95)
            sf.write(self.out_path, out, sr)

            self.progress.emit(100)
            self.log.emit("Listo.")
            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))


# ----------------- UI -----------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pitch+Envelope Synth (sin sklearn)")
        self.resize(980, 680)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        # Paths
        self.src_edit = QLineEdit()
        self.out_edit = QLineEdit()

        btn_src = QPushButton("Examinar…")
        btn_out = QPushButton("Guardar como…")
        btn_src.clicked.connect(self.browse_src)
        btn_out.clicked.connect(self.browse_out)

        row_src = QHBoxLayout()
        row_src.addWidget(QLabel("Audio fuente:"))
        row_src.addWidget(self.src_edit)
        row_src.addWidget(btn_src)

        row_out = QHBoxLayout()
        row_out.addWidget(QLabel("Salida WAV:"))
        row_out.addWidget(self.out_edit)
        row_out.addWidget(btn_out)

        layout.addLayout(row_src)
        layout.addLayout(row_out)

        # Controls
        controls = QHBoxLayout()

        self.wave_combo = QComboBox()
        self.wave_combo.addItems(["sine", "saw", "square"])

        self.harm_spin = QSpinBox()
        self.harm_spin.setRange(1, 64)
        self.harm_spin.setValue(12)

        self.hop_spin = QSpinBox()
        self.hop_spin.setRange(64, 4096)
        self.hop_spin.setSingleStep(64)
        self.hop_spin.setValue(DEFAULT_HOP_LENGTH)

        self.env_alpha = QDoubleSpinBox()
        self.env_alpha.setRange(0.01, 1.0)
        self.env_alpha.setSingleStep(0.05)
        self.env_alpha.setValue(0.25)

        self.f0_alpha = QDoubleSpinBox()
        self.f0_alpha.setRange(0.01, 1.0)
        self.f0_alpha.setSingleStep(0.05)
        self.f0_alpha.setValue(0.20)

        self.gate_check = QCheckBox("Mutear unvoiced")
        self.gate_check.setChecked(True)

        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0.1, 3.0)
        self.gain_spin.setSingleStep(0.1)
        self.gain_spin.setValue(1.0)

        controls.addWidget(QLabel("Wave:"))
        controls.addWidget(self.wave_combo)
        controls.addSpacing(10)

        controls.addWidget(QLabel("Armónicos:"))
        controls.addWidget(self.harm_spin)
        controls.addSpacing(10)

        controls.addWidget(QLabel("Hop:"))
        controls.addWidget(self.hop_spin)
        controls.addSpacing(10)

        controls.addWidget(QLabel("Env α:"))
        controls.addWidget(self.env_alpha)
        controls.addSpacing(10)

        controls.addWidget(QLabel("F0 α:"))
        controls.addWidget(self.f0_alpha)
        controls.addSpacing(10)

        controls.addWidget(self.gate_check)
        controls.addSpacing(10)

        controls.addWidget(QLabel("Gain:"))
        controls.addWidget(self.gain_spin)

        controls.addStretch()
        layout.addLayout(controls)

        # Process button
        self.btn_process = QPushButton("Procesar (sintetizar)")
        self.btn_process.clicked.connect(self.start_processing)
        layout.addWidget(self.btn_process)

        # Progress + Plot + Logs
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        self.pitch_canvas = PitchPlotCanvas(self)
        layout.addWidget(self.pitch_canvas)

        self.logs = QTextEdit()
        self.logs.setReadOnly(True)
        layout.addWidget(self.logs, stretch=1)

        layout.addItem(QSpacerItem(0, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))
        footer = QLabel("© 2025 Gabriel Golker")
        footer.setAlignment(Qt.AlignCenter)
        layout.addWidget(footer)

        self.thread = None
        self.worker = None

    def browse_src(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar audio fuente", "", "Audio files (*.wav *.flac *.ogg *.mp3);;Todos (*.*)"
        )
        if path:
            self.src_edit.setText(path)

    def browse_out(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar salida", "resultado.wav", "WAV (*.wav);;Todos (*.*)"
        )
        if path:
            self.out_edit.setText(path)

    def log(self, msg: str):
        self.logs.append(msg)

    def start_processing(self):
        src = self.src_edit.text().strip()
        outp = self.out_edit.text().strip()

        if not src or not outp:
            QMessageBox.warning(self, "Falta info", "Selecciona audio fuente y ruta de salida.")
            return

        wave = self.wave_combo.currentText()
        harm = int(self.harm_spin.value())
        hop = int(self.hop_spin.value())
        frame_length = DEFAULT_FRAME_LENGTH
        env_a = float(self.env_alpha.value())
        f0_a = float(self.f0_alpha.value())
        gate = bool(self.gate_check.isChecked())
        gain = float(self.gain_spin.value())

        self.logs.clear()
        self.pitch_canvas.clear_plot()
        self.progress.setValue(0)
        self.btn_process.setEnabled(False)

        self.thread = QThread()
        self.worker = AudioWorker(
            src_path=src,
            out_path=outp,
            waveform=wave,
            harmonics=harm,
            hop_length=hop,
            frame_length=frame_length,
            env_alpha=env_a,
            f0_alpha=f0_a,
            gate_unvoiced=gate,
            output_gain=gain,
        )
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress.setValue)
        self.worker.log.connect(self.log)
        self.worker.pitch_midi.connect(lambda arr: self.pitch_canvas.plot_midi(np.array(arr)))
        self.worker.finished.connect(self.on_done)
        self.worker.error.connect(self.on_err)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.error.connect(self.thread.quit)
        self.worker.error.connect(self.worker.deleteLater)

        self.thread.start()

    def on_done(self):
        self.btn_process.setEnabled(True)
        QMessageBox.information(self, "Listo", "Audio sintetizado correctamente.")

    def on_err(self, msg: str):
        self.btn_process.setEnabled(True)
        self.log(f"ERROR: {msg}")
        QMessageBox.critical(self, "Error", msg)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    import qdarkstyle
    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyside6"))

    w = MainWindow()
    w.show()
    sys.exit(app.exec())

