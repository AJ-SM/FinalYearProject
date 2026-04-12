import torch
from .models import LSTMModel   # Replace with your actual model class name
import pickle
import numpy as np
from scipy.signal import butter, sosfilt, sosfreqz
import statistics as st
 
class LSTM_model():
    def __init__(self):
        input_size = 28        
        hidden_size_1 = 16
        hidden_size_2 = 8       
        num_layers = 2          
        self.fs = 256
        self.model = LSTMModel(input_size, hidden_size_1, hidden_size_2, num_layers, 2)
        self.model.load_state_dict(torch.load("LSTM/model.pt", map_location=torch.device('cpu')))

    def butter_bandpass(self,data,lowcut, highcut, order=5):
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        y = sosfilt(sos, data)
        return y

    def preprocess_data(self, data):
        # data = np.random.randn(10, 1280, 28) 
        data = self.butter_bandpass(data, 0.5, 30, order=5)
        return data
    
    def predict(self, data):

        self.model.eval()
        with torch.no_grad():
            data = torch.tensor(data, dtype=torch.float32)
            output = self.model(data)

        predicted = st.mode(np.array([np.argmax(p) for p in output]))
        print(np.array([np.argmax(p) for p in output]))
        return predicted

# test = LSTM_model()
# data = np.random.randn(10, 1280, 28)
# data = test.preprocess_data(data)
# predictions = test.predict(data)
# print(predictions)