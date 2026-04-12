import numpy as np
from scipy import signal
from scipy.fft import fft2
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from scipy.linalg import sqrtm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from collections import Counter
import warnings

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from scipy.linalg import sqrtm
from sklearn.model_selection import train_test_split
from sklearn import metrics
from collections import Counter
import warnings
import time
from scipy.signal import butter, sosfilt, sosfreqz

import statistics as st
class KNN_model():
    def __init__(self, k=10):
        self.k = k


    def preprocess_data(self, data, order=5):
        print(data.shape)
        nyq = 0.5 * 256  
        low = 6 / nyq
        high = 10 / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')

        for i in range(data.shape[0]): 
            for j in range(data.shape[2]): 
                data[i, :, j] = sosfilt(sos, data[i, :, j])

        data = abs(fft2(data.transpose(0,2,1)))

        return data

    def riemannian_distance(self,psd1, psd2):
        N_test = psd1.shape[0]
        N_train = psd2.shape[0]
        
        distances = np.zeros((N_test, N_train))

        for i in range(N_test):
            for j in range(N_train):
                A = psd1[i]
                B = psd2[j]
                distances[i, j] = np.real(np.sqrt(np.trace(A + B - 2 * sqrtm(A @ B))))

        return distances
    
    def knn_classifier(self, distances, train_labels, k=10):
        nearest_neighbors = np.argsort(distances)[:k]
        neighbor_labels = train_labels[nearest_neighbors]
        most_common_label = Counter(neighbor_labels).most_common(1)[0][0]
        return most_common_label   
    
    def predict(self,data=None):
        print(data.shape)
        x_data = np.load("KNN/train_filtered.npy", allow_pickle=True)
        labels_existing = np.load("KNN/labels.npy", allow_pickle=True)
        labels = np.zeros((440,1))
        for i in range(len(labels_existing)):
            if labels_existing[i] == 'light':
                labels[i] = 0
            else:
                labels[i] = 1
        labels = labels.reshape(-1)
        eeg_data = np.concatenate((x_data,data))

        predictions = self.RDCSD(eeg_data,x_data,data,labels)
        predictions = np.array(predictions,dtype=int) 
        print("Predictions:", predictions)

        return st.mode(predictions)


    def load_existing_data(self, data_path):
        data = np.load(data_path, allow_pickle=True)
        return data

    def RDCSD(self,eeg_data,x_data,x_test,labels):
        num_epochs = len(eeg_data)

        def riemannian_distance(psd1, psd2):
            trace_term = np.trace(psd1 + psd2 - 2 * sqrtm(psd1 @ psd2))
            return np.sqrt(trace_term)
        
        print("Calculating CSD matrices...")

        csd_matrices = np.real(np.fft.fft([np.corrcoef(epoch) for epoch in eeg_data], axis=2))
        del eeg_data
        print("CSD Done")

        riemannian_distances = np.zeros((num_epochs, num_epochs))
        upper_i, upper_j = np.triu_indices(num_epochs, k=1)
        for i, j in zip(upper_i, upper_j):
            dist = np.real(riemannian_distance(csd_matrices[i], csd_matrices[j]))
            riemannian_distances[i, j] = dist
            riemannian_distances[j, i] = dist
        print("Riemannian distance calculation done")
        len_train = len(x_data)
        #print(len_train)
        len_test = len(x_test)
        #print(len_test)
        dynamic_test_size = len_test / len_train
        #print(dynamic_test_size)
        X_train, X_test = train_test_split(riemannian_distances, test_size=dynamic_test_size, random_state=None,shuffle=False)
        
        def knn_classifier(train_data, train_labels, test_data, k):
            distances = np.linalg.norm(train_data - test_data, axis=1)
            nearest_neighbors = np.argsort(distances)[:k]
            neighbor_labels = train_labels[nearest_neighbors]
            most_common_label = Counter(neighbor_labels).most_common(1)[0][0]  # Get the most common label
            return most_common_label
        
        predicted_labels = np.zeros(11)
        for test_index in range(11):
            predicted_labels[test_index] = knn_classifier(X_train, labels, X_test[test_index], 6)
            
        return predicted_labels
