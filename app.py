import sys
import numpy as np
import soundfile as sf
from scipy.signal import hilbert

# ✅ librosa imports mínimos (evita importar librosa completo y su dependencia sklearn)
from librosa.core.audio import load as lr_load, resample as lr_resample
from librosa.core.pitch import pyin as lr_pyin
from librosa.core.convert import note_to_hz as lr_note_to_hz, hz_to_midi as lr_hz_to_midi
from librosa.effects import pitch_shift as lr_pitch_shift

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
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# ----------------- CONSTANTES GLOBALES -----------------

HOP_LENGTH = 256        # hop para análisis de pitch
BASE_MIDI = 60.0        # C4 como referencia del molde
FMIN_NOTE = "C2"
FMAX_NOTE = "C7"


# ----------------- UTILIDADES DE AUDIO -----------------


def load_mono(path: str):
    """Carga un audio en mono, sin cambiar el samplerate."""
    y, sr = lr_load(path, sr=None, mono=True)
    return y, sr


def compute_envelope(signal: np.ndarray) -> np.ndarray:
    """
    Calcula la envolvente por transformada de Hilbert.
    Devuelve un array positivo del mismo tamaño.
    """
    analytic = hilbert(signal)
    env = np.abs(analytic)
    return env


def apply_envelope(dst: np.ndarray, env: np.ndarray, safety_gain: float = 0.5) -> np.ndarray:
    """
    Aplica la envolvente al audio destino YA AFINADO.

    - Si el destino es más largo que la envolvente,
      SOLO se usan los primeros len(env) samples.
    - Si la envolvente es más larga, se recorta al tamaño del destino.
    """
    min_len = min(len(dst), len(env))

    dst_seg = dst[:min_len]
    env_seg = env[:min_len]

    # Normalizar envolvente a [0,1]
    max_env = np.max(env_seg)
    if max_env > 0:
        env_norm = env_seg / max_env
    else:
        env_norm = env_seg

    # Bajar un poco el destino para evitar clipping
    dst_scaled = dst_seg * safety_gain

    # Aplicar envolvente
    out = dst_scaled * env_norm

    # Limitar a [-1, 1] por seguridad
    out = np.clip(out, -1.0, 1.0)
    return out


def extract_midi_curve(y: np.ndarray, sr: int, hop_length: int = HOP_LENGTH) -> np.ndarray:
    """
    Extrae curva de pitch en notas MIDI usando librosa.pyin.
    Rellena huecos (NaN) con la última nota válida.
    Si no se detecta ningún pitch, lanza una excepción.
    """
    f0, voiced_flag, voiced_prob = lr_pyin(
        y,
        fmin=lr_note_to_hz(FMIN_NOTE),
        fmax=lr_note_to_hz(FMAX_NOTE),
        frame_length=2048,
        hop_length=hop_length,
    )

    midi = lr_hz_to_midi(f0)  # puede contener NaN
    midi_clean = np.array(midi, dtype=float)

    valid_idx = np.where(~np.isnan(midi_clean))[0]
    if len(valid_idx) == 0:
        raise RuntimeError(
            "No se pudo detectar afinación en el audio fuente "
            "(demasiado ruidoso, polifónico o volumen muy bajo)."
        )

    # Rellenar NaNs con la última nota válida
    last = midi_clean[valid_idx[0]]
    for i in range(len(midi_clean)):
        if np.isnan(midi_clean[i]):
            midi_clean[i] = last
        else:
            last = midi_clean[i]

    return midi_clean


def time_varying_pitch_shift(
    y: np.ndarray,
    sr: int,
    midi_curve: np.ndarray,
    base_midi: float = BASE_MIDI,
    hop_length: int = HOP_LENGTH,
) -> np.ndarray:
    """
    Pitch-shift tiempo-variable usando una curva de notas MIDI.

    - y: audio molde (en C originalmente)
    - midi_curve: curva de notas MIDI del audio fuente
    - base_midi: nota base del molde (C4 = 60)
    """
    offsets = midi_curve - base_midi  # semitonos
    frame_len = hop_length * 4       # ventana para overlap-add
    win = np.hanning(frame_len)

    out = np.zeros_like(y, dtype=float)

    for i, semitones in enumerate(offsets):
        start = i * hop_length
        end = start + frame_len
        if end > len(y):
            break

        frame = y[start:end]

        # pitch-shift de la ventanita
        shifted = lr_pitch_shift(frame, sr=sr, n_steps=float(semitones))

        # aseguramos longitud frame_len
        if len(shifted) > frame_len:
            shifted = shifted[:frame_len]
        elif len(shifted) < frame_len:
            shifted = np.pad(shifted, (0, frame_len - len(shifted)))

        shifted *= win
        out[start:end] += shifted

    max_abs = np.max(np.abs(out))
    if max_abs > 0:
        out = out / max_abs * 0.95

    return out


# ----------------- WIDGET PARA PLOT DE PITCH -----------------


class PitchPlotCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig = Figure(figsize=(5, 2))
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)
        self.setMinimumHeight(150)
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Nota MIDI")
        self.ax.grid(True, alpha=0.3)

    def plot_curve(self, midi_curve: np.ndarray):
        self.ax.clear()
        self.ax.plot(midi_curve, linewidth=1)
        self.ax.set_xlabel("Frame")
        self.ax.set_ylabel("Nota MIDI")
        self.ax.grid(True, alpha=0.3)
        self.draw()

    def clear_plot(self):
        self.ax.clear()
        self.draw()


# ----------------- WORKER EN HILO SEPARADO -----------------


class AudioWorker(QObject):
    progress = Signal(int)      # 0-100
    log = Signal(str)
    finished = Signal()
    error = Signal(str)
    pitch_curve = Signal(object)  # np.ndarray para dibujar curva

    def __init__(self, src_path: str, dst_path: str, out_path: str):
        super().__init__()
        self.src_path = src_path
        self.dst_path = dst_path
        self.out_path = out_path

    def run(self):
        try:
            self.log.emit("Iniciando proceso: afinación + envolvente")

            # 1) Cargar audios
            self.log.emit("Cargando audio fuente...")
            self.progress.emit(5)
            src, sr_src = load_mono(self.src_path)
            self.log.emit(f"  Fuente: {len(src)} muestras, sr = {sr_src} Hz")

            self.log.emit("Cargando audio molde (en C)...")
            self.progress.emit(15)
            dst, sr_dst = load_mono(self.dst_path)
            self.log.emit(f"  Molde: {len(dst)} muestras, sr = {sr_dst} Hz")

            # 2) Igualar samplerate
            if sr_src != sr_dst:
                self.log.emit(
                    f"Samplerates distintos (fuente={sr_src}, molde={sr_dst}). "
                    "Remuestreando molde al de la fuente..."
                )
                dst = lr_resample(dst, orig_sr=sr_dst, target_sr=sr_src)
                sr_dst = sr_src
                self.log.emit(f"  Nuevo molde: {len(dst)} muestras, sr = {sr_dst} Hz")

            # 3) Hacer que la duración la marque la fuente
            min_len = min(len(src), len(dst))
            if min_len < len(dst):
                self.log.emit(
                    "Recortando molde a la duración de la fuente "
                    "(ej: molde 55s → se usan primeros 20s)."
                )
            src = src[:min_len]
            dst = dst[:min_len]

            # 4) Envolvente de la fuente
            self.log.emit("Calculando envolvente de la fuente (Hilbert)...")
            self.progress.emit(30)
            env = compute_envelope(src)

            # 5) Curva de pitch del fuente
            self.log.emit("Extrayendo curva de pitch (pyin)...")
            self.progress.emit(50)
            midi_curve = extract_midi_curve(src, sr_src, hop_length=HOP_LENGTH)
            self.pitch_curve.emit(midi_curve)
            mean_midi = float(np.mean(midi_curve))
            std_midi = float(np.std(midi_curve))
            self.log.emit(
                f"Curva de pitch: media = {mean_midi:.2f} MIDI, desviación = {std_midi:.2f}"
            )

            # 6) Aplicar curva de pitch al molde
            self.log.emit("Aplicando curva de pitch al molde...")
            self.progress.emit(70)
            pitched = time_varying_pitch_shift(
                dst, sr_src, midi_curve, base_midi=BASE_MIDI, hop_length=HOP_LENGTH
            )

            # 7) Aplicar envolvente de la fuente al audio ya afinado
            self.log.emit("Aplicando envolvente de la fuente al molde afinado...")
            self.progress.emit(85)
            out = apply_envelope(pitched, env, safety_gain=0.5)

            # 8) Guardar resultado
            self.log.emit(f"Guardando resultado en: {self.out_path}")
            self.progress.emit(95)
            sf.write(self.out_path, out, sr_src)

            self.progress.emit(100)
            self.log.emit("Proceso completado (afinación + envolvente).")
            self.finished.emit()

        except Exception as e:
            self.error.emit(str(e))


# ----------------- VENTANA PRINCIPAL -----------------


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Copiador de Envolvente + Afinación")
        self.resize(900, 600)

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        # --- Texto de modo ---
        mode_label = QLabel("Modo: Copiar afinación (portamento) + envolvente en un solo paso")
        main_layout.addWidget(mode_label)

        # --- Rutas de archivos ---
        self.src_edit = QLineEdit()
        self.dst_edit = QLineEdit()
        self.out_edit = QLineEdit()

        btn_src = QPushButton("Examinar…")
        btn_dst = QPushButton("Examinar…")
        btn_out = QPushButton("Guardar como…")

        btn_src.clicked.connect(self.browse_src)
        btn_dst.clicked.connect(self.browse_dst)
        btn_out.clicked.connect(self.browse_out)

        # Fila fuente
        row_src = QHBoxLayout()
        row_src.addWidget(QLabel("Audio fuente:"))
        row_src.addWidget(self.src_edit)
        row_src.addWidget(btn_src)

        # Fila molde
        row_dst = QHBoxLayout()
        row_dst.addWidget(QLabel("Audio molde (en C):"))
        row_dst.addWidget(self.dst_edit)
        row_dst.addWidget(btn_dst)

        # Fila salida
        row_out = QHBoxLayout()
        row_out.addWidget(QLabel("Audio de salida:"))
        row_out.addWidget(self.out_edit)
        row_out.addWidget(btn_out)

        main_layout.addLayout(row_src)
        main_layout.addLayout(row_dst)
        main_layout.addLayout(row_out)

        # --- Botón procesar ---
        self.process_button = QPushButton("Procesar (afinación + envolvente)")
        self.process_button.clicked.connect(self.start_processing)
        main_layout.addWidget(self.process_button)

        # --- Barra de progreso ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        # --- Plot de pitch ---
        self.pitch_canvas = PitchPlotCanvas(self)
        main_layout.addWidget(self.pitch_canvas)

        # --- Logs ---
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        main_layout.addWidget(self.log_text, stretch=1)

        # Spacer + copyright
        main_layout.addItem(QSpacerItem(0, 10, QSizePolicy.Minimum, QSizePolicy.Expanding))

        self.footer_label = QLabel("© 2025 Gabriel Golker")
        self.footer_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.footer_label)

        # Hilo y worker
        self.thread = None
        self.worker = None

    # ---------- Métodos de browse ----------

    def browse_src(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar audio fuente",
            "",
            "Audio files (*.wav *.flac *.ogg *.mp3);;Todos los archivos (*.*)",
        )
        if path:
            self.src_edit.setText(path)

    def browse_dst(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Seleccionar audio molde (en C)",
            "",
            "Audio files (*.wav *.flac *.ogg *.mp3);;Todos los archivos (*.*)",
        )
        if path:
            self.dst_edit.setText(path)

    def browse_out(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Seleccionar archivo de salida",
            "resultado.wav",
            "Archivos WAV (*.wav);;Todos los archivos (*.*)",
        )
        if path:
            self.out_edit.setText(path)

    # ---------- Procesamiento ----------

    def start_processing(self):
        src_path = self.src_edit.text().strip()
        dst_path = self.dst_edit.text().strip()
        out_path = self.out_edit.text().strip()

        if not src_path or not dst_path or not out_path:
            QMessageBox.warning(self, "Campos incompletos", "Completa todas las rutas de archivo.")
            return

        self.log_text.clear()
        self.progress_bar.setValue(0)
        self.pitch_canvas.clear_plot()

        self.process_button.setEnabled(False)

        self.thread = QThread()
        self.worker = AudioWorker(src_path, dst_path, out_path)
        self.worker.moveToThread(self.thread)

        # Conexiones
        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.log.connect(self.append_log)
        self.worker.finished.connect(self.on_finished)
        self.worker.error.connect(self.on_error)
        self.worker.pitch_curve.connect(self.on_pitch_curve)

        # Limpieza de hilo
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.error.connect(self.thread.quit)
        self.worker.error.connect(self.worker.deleteLater)

        self.thread.start()

    def append_log(self, text: str):
        self.log_text.append(text)

    def on_finished(self):
        self.process_button.setEnabled(True)
        QMessageBox.information(self, "Listo", "Proceso completado correctamente.")

    def on_error(self, message: str):
        self.process_button.setEnabled(True)
        QMessageBox.critical(self, "Error", f"Ocurrió un error:\n{message}")
        self.append_log(f"ERROR: {message}")

    def on_pitch_curve(self, midi_curve):
        self.pitch_canvas.plot_curve(np.array(midi_curve))


# ----------------- ENTRADA PRINCIPAL -----------------


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # QDarkStyle
    import qdarkstyle

    app.setStyleSheet(qdarkstyle.load_stylesheet(qt_api="pyside6"))

    window = MainWindow()
    window.show()
    sys.exit(app.exec())

