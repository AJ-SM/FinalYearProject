import pickle
import numpy as np
from scipy.signal import butter, sosfilt, sosfreqz

class data_preprocessor():
    def __init__(self,path):

        with open(path, "rb") as handle:
            self.data = pickle.load(handle)
        # self.data[:, 1] = self.data[:, 1]
        self.words = np.unique(self.data[:, 0])
        # self.data = np.array([self.data_temp[i] for i in self.words]).astype(np.float32)
        # del self.data

        print(self.data.shape)
        self.bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 100)
        }

        self.fs = 256
        
    def butter_bandpass(self,data,lowcut, highcut, order=5):
        nyq = 0.5 * self.fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        y = sosfilt(sos, data)
        return y

    def splitter(self,words,data,ratio, augment=False , normalise=False):
        num_samples = data.shape[0]
        word_to_index = {word: index for index, word in enumerate(words)}

        np.random.shuffle(data)


        split_index = int(ratio * num_samples)
        train_data = data[:split_index]
        test_data = data[split_index:]

        # print(train_data[:,0])
        train_labels = np.vectorize(word_to_index.get)(train_data[:, 0]) 
        train_data = np.stack(train_data[:, 2])  

        test_labels = np.vectorize(word_to_index.get)(test_data[:, 0])  # First index (word labels)
        test_data = np.stack(test_data[:, 2])  # Second index (1280, 28 data arrays)

        if augment == True:
            train_data = np.vstack([train_data[:,:640,:],train_data[:,640:,:]])
            train_labels = np.hstack([train_labels,train_labels])
            test_labels = np.hstack([test_labels])
            test_data = np.vstack([test_data[:,:640,:]])

        if normalise == True:
            train_data = (train_data-np.min(train_data))/(np.max(train_data)-np.min(train_data))
            test_data = (test_data-np.min(test_data))/(np.max(test_data)-np.min(test_data))

        return train_data,train_labels,test_data,test_labels

    def apply_filter(self, words,low, high):
        filtered_data = np.array([entry for entry in self.data if entry[0] in words])
        filtered_data[:, 2] = [self.butter_bandpass(entry.astype(np.float32), low, high) for entry in filtered_data[:, 2]]
        return filtered_data

 


