import torch
import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfilt
import statistics as st

from .models import LSTMModel

_HERE = Path(__file__).parent


class LSTM_model:
    def __init__(self):
        input_size    = 28
        hidden_size_1 = 16
        hidden_size_2 = 8
        num_layers    = 2
        self.fs       = 256

        self.model = LSTMModel(input_size, hidden_size_1, hidden_size_2, num_layers, num_classes=2)
        self.model.load_state_dict(
            torch.load(_HERE / "model.pt", map_location=torch.device('cpu'))
        )

    # ------------------------------------------------------------------
    def _butter_bandpass(self, data, lowcut, highcut, order=5):
        nyq  = 0.5 * self.fs
        low  = lowcut  / nyq
        high = highcut / nyq
        sos  = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sosfilt(sos, data)

    # ------------------------------------------------------------------
    def preprocess_data(self, data):
        print("LSTM preprocess — input shape:", data.shape)
        data = self._butter_bandpass(data, 0.5, 30, order=5)
        return data

    # ------------------------------------------------------------------
    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor(data, dtype=torch.float32)
            output = self.model(tensor)

        per_sample = np.array([np.argmax(p) for p in output])
        print("Per-sample predictions:", per_sample)
        return st.mode(per_sample)