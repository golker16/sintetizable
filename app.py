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

WT_FRAME_SIZE = 2048
WT_MIP_LEVELS = 8


# ----------------- WAVETABLE (drop-in) -----------------

def _to_mono_float(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    x = x.astype(np.float32, copy=False)
    # normalizar si viene en int
    if np.issubdtype(x.dtype, np.integer):
        mx = np.iinfo(x.dtype).max
        x = x.astype(np.float32) / float(mx)
    return x


def _fade_edges(frame: np.ndarray, fade: int = 8) -> np.ndarray:
    """Suaviza los bordes del ciclo para reducir clicks si el ciclo no cierra perfecto."""
    if fade <= 0 or 2 * fade >= len(frame):
        return frame
    w = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    out = frame.copy()
    out[:fade] *= w
    out[-fade:] *= w[::-1]
    return out


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    m = np.max(np.abs(frame)) + 1e-12
    return (frame / m).astype(np.float32, copy=False)


def _linear_resample(x: np.ndarray, new_len: int) -> np.ndarray:
    """Resample lineal 1D: suficiente para precomputar mipmaps sin dependencias extras."""
    n = len(x)
    if new_len == n:
        return x.astype(np.float32, copy=False)
    src = np.linspace(0.0, 1.0, n, endpoint=False, dtype=np.float32)
    dst = np.linspace(0.0, 1.0, new_len, endpoint=False, dtype=np.float32)
    return np.interp(dst, src, x).astype(np.float32, copy=False)


def load_wavetable_wav(
    path: str,
    frame_size: int = WT_FRAME_SIZE,
    normalize_each_frame: bool = True,
    edge_fade: int = 8,
):
    """
    Carga WAV mono.
    - Si el WAV tiene longitud == frame_size -> 1 frame
    - Si el WAV tiene longitud múltiplo de frame_size -> multi-frame concatenado
    Retorna: frames shape = (n_frames, frame_size) float32
    """
    audio, sr = sf.read(path, always_2d=False)
    audio = _to_mono_float(audio)
    n = len(audio)

    if n < frame_size:
        audio = np.pad(audio, (0, frame_size - n))
        n = len(audio)

    n_frames = n // frame_size
    if n_frames < 1:
        n_frames = 1

    use_len = n_frames * frame_size
    audio = audio[:use_len]

    frames = audio.reshape(n_frames, frame_size).copy()

    for i in range(n_frames):
        f = frames[i]
        f = f - np.mean(f)            # quitar DC
        f = _fade_edges(f, edge_fade) # opcional
        if normalize_each_frame:
            f = _normalize_frame(f)
        frames[i] = f

    return frames.astype(np.float32, copy=False)


def build_wavetable_mipmaps(frames: np.ndarray, levels: int = WT_MIP_LEVELS):
    """
    Crea mipmaps por downsampling de la tabla (reduce armónicos).
    Retorna lista mipmaps[level] con shape=(n_frames, frame_size_level)
    """
    frames = np.asarray(frames, dtype=np.float32)
    n_frames, frame_size = frames.shape

    mipmaps = []
    cur = frames
    cur_size = frame_size

    for lvl in range(levels):
        mipmaps.append(cur)

        next_size = max(32, cur_size // 2)
        if next_size == cur_size:
            break

        nxt = np.zeros((n_frames, next_size), dtype=np.float32)
        for fi in range(n_frames):
            nxt[fi] = _linear_resample(cur[fi], next_size)

        cur = nxt
        cur_size = next_size

    return mipmaps


def _lerp(a, b, t):
    return a + (b - a) * t


def _table_read_linear(table_1d: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Lee una tabla 1D con fase [0,1), interpolación lineal."""
    n = len(table_1d)
    idx = phase * n
    i0 = np.floor(idx).astype(np.int32)
    frac = idx - i0
    i1 = (i0 + 1) % n
    return (1.0 - frac) * table_1d[i0] + frac * table_1d[i1]


def render_wavetable_osc(
    f0_hz: np.ndarray,          # por-sample
    sr: int,
    mipmaps: list,              # salida de build_wavetable_mipmaps
    position: float = 0.0,      # 0..1
    phase0: float = 0.0,
    mip_strength: float = 1.0,  # 0..1
):
    """
    Oscilador wavetable con morph entre frames + mipmaps simple.
    Devuelve (audio, phase_final).
    """
    f0_hz = np.asarray(f0_hz, dtype=np.float32)
    n_samples = len(f0_hz)
    n_levels = len(mipmaps)

    base_frames = mipmaps[0]
    n_frames = base_frames.shape[0]

    pos = float(np.clip(position, 0.0, 1.0))
    fidx = pos * (n_frames - 1)
    f0i = int(np.floor(fidx))
    ft = float(fidx - f0i)
    f1i = min(f0i + 1, n_frames - 1)

    phase = np.empty(n_samples, dtype=np.float32)
    ph = float(phase0 % 1.0)
    for i in range(n_samples):
        phase[i] = ph
        ph += float(f0_hz[i]) / float(sr)
        ph -= np.floor(ph)

    f_ref = 55.0
    ratio = np.maximum(f0_hz / f_ref, 1e-6)
    lvl_float = np.log2(ratio) * float(np.clip(mip_strength, 0.0, 1.0))
    lvl = np.clip(np.floor(lvl_float).astype(np.int32), 0, n_levels - 1)

    out = np.zeros(n_samples, dtype=np.float32)

    for L in range(n_levels):
        mask = (lvl == L)
        if not np.any(mask):
            continue

        tables_L = mipmaps[L]
        t0 = tables_L[f0i]
        t1 = tables_L[f1i]
        table = _lerp(t0, t1, ft)

        out[mask] = _table_read_linear(table, phase[mask])

    return out, float(ph)


def synth_wavetable_from_f0_env(
    f0_frames_hz: np.ndarray,
    env_frames: np.ndarray,
    sr: int,
    n_samples: int,
    frame_length: int,
    hop_length: int,
    mipmaps: list,
    position: float,
    mip_strength: float,
):
    f0_frames_hz = np.asarray(f0_frames_hz, dtype=np.float32)
    env_frames = np.asarray(env_frames, dtype=np.float32)

    centers = (np.arange(len(f0_frames_hz)) * hop_length + frame_length / 2.0)
    centers = np.clip(centers, 0, max(0, n_samples - 1))

    t = np.arange(n_samples, dtype=np.float32)
    f0 = np.interp(t, centers.astype(np.float32), f0_frames_hz).astype(np.float32)
    env = np.interp(t, centers.astype(np.float32), env_frames).astype(np.float32)

    f0 = np.clip(f0, 1.0, sr / 2.0)

    osc, _ = render_wavetable_osc(
        f0_hz=f0,
        sr=sr,
        mipmaps=mipmaps,
        position=position,
        phase0=0.0,
        mip_strength=mip_strength,
    )

    out = osc * env
    out = np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)
    return out


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

    f0_clean = f0.copy()
    last = f0_clean[valid[0]]
    for i in range(len(f0_clean)):
        if np.isnan(f0_clean[i]):
            f0_clean[i] = last
        else:
            last = f0_clean[i]

    f0_clean = one_pole_smooth(f0_clean, alpha=f0_smooth_alpha)

    env = frame_rms(y, frame_length=frame_length, hop_length=hop_length)
    env = env / (np.max(env) + 1e-12)

    n = min(len(f0_clean), len(voiced), len(env))
    f0_clean = f0_clean[:n]
    voiced = voiced[:n]
    env = env[:n]

    if gate_unvoiced:
        env = env * voiced.astype(float)

    env = one_pole_smooth(env, alpha=env_smooth_alpha)

    return f0_clean.astype(np.float32), voiced, env.astype(np.float32)


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
        hop_length: int,
        frame_length: int,
        env_alpha: float,
        f0_alpha: float,
        gate_unvoiced: bool,
        output_gain: float,
        wavetable_path: str,
        wt_position: float,
        wt_mip_strength: float,
    ):
        super().__init__()
        self.src_path = src_path
        self.out_path = out_path
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.env_alpha = env_alpha
        self.f0_alpha = f0_alpha
        self.gate_unvoiced = gate_unvoiced
        self.output_gain = output_gain
        self.wavetable_path = wavetable_path
        self.wt_position = wt_position
        self.wt_mip_strength = wt_mip_strength

    def run(self):
        try:
            self.log.emit("Procesando: Pitch map + Envelope → Wavetable Synth")

            if not self.wavetable_path:
                raise RuntimeError("Selecciona un archivo WAV de wavetable antes de procesar.")

            self.progress.emit(5)
            self.log.emit("Cargando audio fuente...")
            y, sr = load_mono(self.src_path)
            self.log.emit(f"  Fuente: {len(y)} samples, sr={sr}")

            self.progress.emit(15)
            self.log.emit("Cargando wavetable...")
            frames = load_wavetable_wav(self.wavetable_path, frame_size=WT_FRAME_SIZE)
            mipmaps = build_wavetable_mipmaps(frames, levels=WT_MIP_LEVELS)
            self.log.emit(f"  Wavetable: {frames.shape[0]} frame(s), size={frames.shape[1]}, mips={len(mipmaps)}")

            self.progress.emit(35)
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

            self.progress.emit(65)
            self.log.emit("Sintetizando wavetable...")
            out = synth_wavetable_from_f0_env(
                f0_frames_hz=f0_frames,
                env_frames=env_frames,
                sr=sr,
                n_samples=len(y),
                frame_length=self.frame_length,
                hop_length=self.hop_length,
                mipmaps=mipmaps,
                position=self.wt_position,
                mip_strength=self.wt_mip_strength,
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
        self.setWindowTitle("Pitch+Envelope Wavetable Synth (sin sklearn)")
        self.resize(1020, 720)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        # Paths
        self.src_edit = QLineEdit()
        self.out_edit = QLineEdit()
        self.wt_edit = QLineEdit()

        btn_src = QPushButton("Examinar…")
        btn_out = QPushButton("Guardar como…")
        btn_wt = QPushButton("Wavetable…")

        btn_src.clicked.connect(self.browse_src)
        btn_out.clicked.connect(self.browse_out)
        btn_wt.clicked.connect(self.browse_wt)

        row_src = QHBoxLayout()
        row_src.addWidget(QLabel("Audio fuente:"))
        row_src.addWidget(self.src_edit)
        row_src.addWidget(btn_src)

        row_wt = QHBoxLayout()
        row_wt.addWidget(QLabel("Wavetable WAV:"))
        row_wt.addWidget(self.wt_edit)
        row_wt.addWidget(btn_wt)

        row_out = QHBoxLayout()
        row_out.addWidget(QLabel("Salida WAV:"))
        row_out.addWidget(self.out_edit)
        row_out.addWidget(btn_out)

        layout.addLayout(row_src)
        layout.addLayout(row_wt)
        layout.addLayout(row_out)

        # Controls
        controls = QHBoxLayout()

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

        self.wt_pos = QDoubleSpinBox()
        self.wt_pos.setRange(0.0, 1.0)
        self.wt_pos.setSingleStep(0.01)
        self.wt_pos.setValue(0.0)

        self.wt_mip = QDoubleSpinBox()
        self.wt_mip.setRange(0.0, 1.0)
        self.wt_mip.setSingleStep(0.05)
        self.wt_mip.setValue(1.0)

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
        controls.addSpacing(18)

        controls.addWidget(QLabel("WT Position:"))
        controls.addWidget(self.wt_pos)
        controls.addSpacing(10)

        controls.addWidget(QLabel("WT Mip:"))
        controls.addWidget(self.wt_mip)

        controls.addStretch()
        layout.addLayout(controls)

        # Process button
        self.btn_process = QPushButton("Procesar (wavetable)")
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

    def browse_wt(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar wavetable WAV", "", "WAV (*.wav);;Todos (*.*)"
        )
        if path:
            self.wt_edit.setText(path)

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
        wt = self.wt_edit.text().strip()

        if not src or not outp:
            QMessageBox.warning(self, "Falta info", "Selecciona audio fuente y ruta de salida.")
            return

        hop = int(self.hop_spin.value())
        frame_length = DEFAULT_FRAME_LENGTH
        env_a = float(self.env_alpha.value())
        f0_a = float(self.f0_alpha.value())
        gate = bool(self.gate_check.isChecked())
        gain = float(self.gain_spin.value())
        wt_pos = float(self.wt_pos.value())
        wt_mip = float(self.wt_mip.value())

        self.logs.clear()
        self.pitch_canvas.clear_plot()
        self.progress.setValue(0)
        self.btn_process.setEnabled(False)

        self.thread = QThread()
        self.worker = AudioWorker(
            src_path=src,
            out_path=outp,
            hop_length=hop,
            frame_length=frame_length,
            env_alpha=env_a,
            f0_alpha=f0_a,
            gate_unvoiced=gate,
            output_gain=gain,
            wavetable_path=wt,
            wt_position=wt_pos,
            wt_mip_strength=wt_mip,
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

