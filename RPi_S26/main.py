import sys
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QFileDialog,
    QComboBox, QVBoxLayout, QHBoxLayout, QLabel,
    QStackedWidget, QDesktopWidget, QSlider, QCheckBox
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QFontMetrics, QPainter, QPainterPath, QPen, QColor, QIcon
from PyQt5.QtCore import QRectF

from core.extract_data import data_extractor
from models.knn_english.knn_english import KNN_English_model
from models.knn_hindi.knn_hindi import KNN_Hindi_model
from models.lstm.lstm import LSTM_model

import matplotlib.pyplot as plt
from functools import partial
from scipy import signal
from scipy.fft import fft2
from scipy.signal import butter, sosfilt
import numpy as np
from scipy.signal import welch


class CustomLabel(QLabel):
    def __init__(self, *args, padding=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding = padding
        self.setAttribute(Qt.WA_TranslucentBackground)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(
            self.padding // 2,
            self.padding // 2,
            -self.padding // 2,
            -self.padding // 2
        )
        path = QPainterPath()
        radius = 20
        path.addRoundedRect(QRectF(rect), radius, radius)

        pen = QPen(QColor("#000000"), 2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)

        super().paintEvent(event)


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.selected_path = None
        self.model_name = "KNN_English"
        self.model = None
        self.scale = 2

        # words for English models (KNN_English and LSTM_English)
        self.english_words = ["light", "no"]
        # words for Hindi model — hardcoded per label encoder order
        self.hindi_words = ["PAIN", "LIGHT"]

        self.setObjectName("MainWindow")
        self.setWindowTitle("Imagined Word Recognition")
        self.setFixedSize(500, 500)
        res = QDesktopWidget().availableGeometry()
        self.move(
            (res.width() - self.frameSize().width()) // 2,
            (res.height() - self.frameSize().height()) // 2
        )

        self.setStyleSheet("""
            QWidget#MainWindow {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #eaf6ff, stop:1 #b3e0ff);
                font-family: 'Segoe UI', sans-serif;
            }
            QLabel { font-size: 20px; color: #333; }
            QComboBox {
                font-size: 20px;
                padding: 6px 12px;
                border-radius: 8px;
                font-weight: bold;
                background-color: #77b9f7;
                color: black;
                border: 2px solid #000000;
            }
            QComboBox:hover { background-color: #45a1f7; }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 24px;
                border-left: 2px solid black;
            }
            QComboBox::down-arrow {
                image: url(utils/arrow.png);
                width: 12px;
                height: 12px;
            }
            QPushButton {
                font-size: 20px;
                padding: 6px 12px;
                border-radius: 8px;
                font-weight: bold;
                border: 2px solid #000000;
                background-color: #77b9f7;
                color: black;
            }
            QPushButton:hover { background-color: #45a1f7; }
        """)

        self._build_main_page()
        self._build_plot_page()

        self.stacked = QStackedWidget(self)
        self.stacked.addWidget(self.main_page)
        self.stacked.addWidget(self.plot_page)

        root_layout = QVBoxLayout(self)
        root_layout.addWidget(self.stacked)

    # =========================================================
    # MAIN PAGE
    # =========================================================
    def _build_main_page(self):

        self.browse_button = QPushButton("Data")
        self.browse_button.setFixedSize(100 * self.scale, 38 * self.scale)
        self.browse_button.clicked.connect(self.browse_folder)

        self.plot_button = QPushButton("")
        self.plot_button.setFixedSize(38 * self.scale, 38 * self.scale)
        self.plot_button.setIcon(QIcon("utils/plot.png"))
        self.plot_button.setIconSize(self.plot_button.size() * 0.5)
        self.plot_button.clicked.connect(self._connect_plot_button)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setFixedSize(142 * self.scale, 38 * self.scale)
        self.model_combo.addItems(["KNN_English", "KNN_Hindi", "LSTM_English"])
        le = self.model_combo.lineEdit()
        le.setAlignment(Qt.AlignCenter)
        le.setReadOnly(True)
        self.model_combo.currentTextChanged.connect(self.select_model)

        self.start_button = QPushButton("Predict")
        self.start_button.setFixedSize(142 * self.scale, 38 * self.scale)
        self.start_button.clicked.connect(self.on_start)

        self.status_label = QLabel("Status: Idle")
        self.status_label.setAlignment(Qt.AlignCenter)

        self.predicted_label = CustomLabel("Prediction")
        self.predicted_label.setAlignment(Qt.AlignCenter)
        self.predicted_label.setStyleSheet("font-size:50px; color:#77b9f7;")

        top_layout = QHBoxLayout()
        top_layout.setAlignment(Qt.AlignCenter)
        top_layout.addWidget(self.browse_button)
        top_layout.addWidget(self.plot_button)

        model_layout = QHBoxLayout()
        model_layout.addWidget(self.model_combo)

        main_layout = QVBoxLayout()
        main_layout.addStretch()
        main_layout.addWidget(self.predicted_label, alignment=Qt.AlignCenter)
        main_layout.addSpacing(25)
        main_layout.addLayout(top_layout)
        main_layout.addStretch()
        main_layout.addLayout(model_layout)
        main_layout.addStretch()
        main_layout.addWidget(self.start_button, alignment=Qt.AlignCenter)
        main_layout.addSpacing(25)
        main_layout.addWidget(self.status_label, alignment=Qt.AlignCenter)
        main_layout.addSpacing(25)
        main_layout.addStretch()

        self.main_page = QWidget()
        self.main_page.setLayout(main_layout)

    # =========================================================
    # PLOT PAGE
    # =========================================================
    def _build_plot_page(self):
        self.plot_page = QWidget()
        layout = QVBoxLayout(self.plot_page)

        def make_heading(text):
            lbl = QLabel(text)
            font = lbl.font()
            font.setPointSize(14)
            font.setBold(True)
            lbl.setFont(font)
            lbl.setAlignment(Qt.AlignCenter)
            return lbl

        def styled_label(text):
            lbl = QLabel(text)
            font = lbl.font()
            font.setPointSize(10 * self.scale)
            font.setBold(True)
            lbl.setFont(font)
            return lbl

        layout.addWidget(make_heading("Band Pass Filter"))

        self.f_lo = 1
        self.f_hi = 60

        for label_text, default, attr_name in [
            ("Low cutoff:",  1,  'f_lo'),
            ("High cutoff:", 60, 'f_hi'),
        ]:
            h = QHBoxLayout()
            lbl = styled_label(label_text)
            val = QLabel(str(default))
            s = QSlider(Qt.Horizontal)
            s.setRange(0, 60)
            s.setValue(default)
            s.valueChanged.connect(partial(self._on_cutoff_changed, attr_name, val))
            h.addWidget(lbl)
            h.addWidget(s)
            h.addWidget(val)
            layout.addLayout(h)

        layout.addWidget(make_heading("PSD Range"))
        self.psd_start = 0.0
        self.psd_end   = 2.0

        for label_text, default, attr_name in [
            ("PSD start:", 0.0, 'psd_start'),
            ("PSD end:",   2.0, 'psd_end'),
        ]:
            h = QHBoxLayout()
            lbl = styled_label(label_text)
            val = QLabel(f"{default:.2f}")
            s = QSlider(Qt.Horizontal)
            s.setRange(0, 200)
            s.setValue(int(default * 100))
            s.valueChanged.connect(partial(self._on_psd_changed, attr_name, val))
            h.addWidget(lbl)
            h.addWidget(s)
            h.addWidget(val)
            layout.addLayout(h)

        layout.addWidget(make_heading("Notch Filter"))
        self.notch = QCheckBox("Notch Filter")
        font = self.notch.font()
        font.setPointSize(10 * self.scale)
        font.setBold(True)
        self.notch.setFont(font)

        btn_plot = QPushButton("Plot Signal")
        btn_plot.setFixedSize(120 * self.scale, 38 * self.scale)
        layout.addWidget(self.notch, alignment=Qt.AlignCenter)
        layout.addWidget(btn_plot,   alignment=Qt.AlignCenter)
        btn_plot.clicked.connect(self.plot_graph)

        back = QPushButton("Back")
        back.setFixedSize(100 * self.scale, 38 * self.scale)
        back.clicked.connect(lambda: self.stacked.setCurrentWidget(self.main_page))
        layout.addWidget(back, alignment=Qt.AlignCenter)

    # =========================================================
    # SLOT: slider callbacks
    # =========================================================
    def _on_cutoff_changed(self, attr_name, label_widget, new_value):
        setattr(self, attr_name, new_value)
        label_widget.setText(str(new_value))

    def _on_psd_changed(self, attr_name, label_widget, int_value):
        float_value = int_value / 100
        setattr(self, attr_name, float_value)
        label_widget.setText(f"{float_value:.2f}")

    # =========================================================
    # DSP HELPERS (English plot pipeline)
    # =========================================================
    def bandpass_filter(self, data, order=5):
        nyq = 0.5 * 256
        low  = self.f_lo / nyq
        high = self.f_hi / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        filtered = np.empty_like(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[2]):
                filtered[i, :, j] = sosfilt(sos, data[i, :, j])
        return filtered

    def apply_lowpass_fir(self, x, cutoff_hz=50, fs=256.0, numtaps=21, zero_phase=True):
        nyq = 0.5 * fs
        norm_cutoff = cutoff_hz / nyq
        b = signal.firwin(numtaps, cutoff=norm_cutoff, window='hann')
        a = 1
        x_t = x.transpose(0, 2, 1)
        if zero_phase:
            filtered = signal.filtfilt(b, a, x_t, axis=-1)
        else:
            filtered = signal.lfilter(b, a, x_t, axis=-1)
        return filtered.transpose(0, 2, 1)

    def compute_psd(self, filtered_data, fs=256):
        psd_list = []
        nperseg = fs * (self.psd_end - self.psd_start)
        for i in range(filtered_data.shape[0]):
            sample_psd = []
            for j in range(filtered_data.shape[2]):
                f, Pxx = welch(filtered_data[i, :, j], fs=fs, nperseg=nperseg)
                sample_psd.append(Pxx)
            psd_list.append(sample_psd)
        return np.array(psd_list)

    # =========================================================
    # PLOT
    # =========================================================
    def plot_graph(self):
        print(self.data_extractor.data.shape)
        data = self.bandpass_filter(self.data_extractor.data)
        psd  = self.compute_psd(data)

        if self.notch.isChecked():
            data = self.apply_lowpass_fir(data, 50)

        timesteps    = range(data.shape[1])
        num_channels = data.shape[2]

        fig, axes = plt.subplots(nrows=num_channels, ncols=1, figsize=(10, 2 * num_channels))
        if num_channels == 1:
            axes = [axes]

        for i in range(num_channels):
            axes[i].plot(timesteps, data[0, :, i])
            axes[i].set_ylabel(self.data_extractor.channels[i][4:])
            axes[i].tick_params(axis='y', which='both', left=False, labelleft=False)
            axes[i].grid(axis='x')

        axes[-1].set_xlabel("Timesteps")

        fig_psd, ax_psd = plt.subplots(figsize=(10, 4))
        ax_psd.plot(psd[0, 0, :])
        ax_psd.set_title("Power Spectral Density (First Channel)")
        ax_psd.set_xlabel("Frequency Bins")
        ax_psd.set_ylabel("Power")
        ax_psd.grid(True)

        plt.tight_layout()
        plt.show()

    def _connect_plot_button(self):
        if self.model_name == "KNN_Hindi":
            self.update_status("Plot not supported for KNN_Hindi")
            return
        if self.selected_path is None:
            self.update_status("No data path selected")
            return
        self.extract_data()
        self.stacked.setCurrentWidget(self.plot_page)

    # =========================================================
    # BROWSE
    # =========================================================
    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            self.selected_path = folder
            print(f"Selected path: {self.selected_path}")

    # =========================================================
    # STATUS
    # =========================================================
    def update_status(self, state: str):
        self.status_label.setText(f"Status: {state}")
        QApplication.processEvents()

    # =========================================================
    # MODEL SELECTION
    # =========================================================
    def select_model(self, model: str):
        self.model_name = model
        print(f"Selected model: {self.model_name}")

    # =========================================================
    # DISPLAY PREDICTION
    # =========================================================
    def display_predicted_word(self, word: str):
        self.predicted_label.setText(word)
        font = QFont("Arial", 40, QFont.Bold)
        self.predicted_label.setFont(font)
        fm   = QFontMetrics(font)
        rect = fm.boundingRect(word)
        w = rect.width()  + self.predicted_label.padding * 10
        h = rect.height() + self.predicted_label.padding * 10
        self.predicted_label.setFixedSize(w, h)
        self.predicted_label.setStyleSheet("font-size:40px; color:#333;")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        text = self.predicted_label.text()
        if text and text != "Prediction":
            font = QFont("Arial", 40, QFont.Bold)
            self.predicted_label.setFont(font)
            fm   = QFontMetrics(font)
            rect = fm.boundingRect(text)
            w = rect.width()  + self.predicted_label.padding * 10
            h = rect.height() + self.predicted_label.padding * 10
            self.predicted_label.setFixedSize(w, h)
            self.predicted_label.setStyleSheet("font-size:40px; color:#333;")

    # =========================================================
    # PIPELINE
    # =========================================================
    def on_start(self):
        if not self.selected_path:
            self.update_status("No data selected")
            return

        if self.model_name == "KNN_Hindi":
            self._run_hindi_pipeline()
        else:
            self._run_english_pipeline()

        self.update_status("Completed")

    def _run_english_pipeline(self):
        self.extract_data()
        self.load_model()
        self.data_extractor.data = self.model.preprocess_data(self.data_extractor.data)
        print(self.data_extractor.data.shape)
        self.update_status("Predicting")
        preds = self.model.predict(self.data_extractor.data)
        word  = self.english_words[preds]
        print(word)
        self.display_predicted_word(word)

    def _run_hindi_pipeline(self):
        self.update_status("Loading Hindi model")
        self.model = KNN_Hindi_model()
        self.model.load_model()
        self.update_status("Predicting")
        word = self.model.predict_from_folder(self.selected_path)
        self.display_predicted_word(word)

    def extract_data(self):
        self.update_status("Extracting")
        print("Extracting")
        self.data_extractor = data_extractor(self.selected_path)
        self.data_extractor.read_data()
        self.data_extractor.get_data()

    def load_model(self):
        if self.model_name == "KNN_English":
            self.update_status("Loading KNN_English")
            self.model = KNN_English_model()
        elif self.model_name == "LSTM_English":
            self.update_status("Loading LSTM_English")
            self.model = LSTM_model()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())