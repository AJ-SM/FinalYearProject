import numpy as np
from pathlib import Path
from collections import Counter
from scipy.fft import fft2
from scipy.signal import butter, sosfilt
from scipy.linalg import sqrtm
from sklearn.model_selection import train_test_split
import statistics as st

_HERE = Path(__file__).parent


class KNN_English_model:
    def __init__(self, k=6):
        self.k = k
        self.train_data = np.load(_HERE / "train_filtered.npy", allow_pickle=True)
        self.labels_raw = np.load(_HERE / "labels.npy",         allow_pickle=True)

        # map string labels → int
        self.labels = np.zeros(len(self.labels_raw), dtype=int)
        for i, lbl in enumerate(self.labels_raw):
            self.labels[i] = 0 if lbl == 'light' else 1

    # ------------------------------------------------------------------
    def preprocess_data(self, data, order=5):
        print("KNN_English preprocess — input shape:", data.shape)
        nyq  = 0.5 * 256
        low  = 6  / nyq
        high = 10 / nyq
        sos  = butter(order, [low, high], analog=False, btype='band', output='sos')

        for i in range(data.shape[0]):
            for j in range(data.shape[2]):
                data[i, :, j] = sosfilt(sos, data[i, :, j])

        data = abs(fft2(data.transpose(0, 2, 1)))
        print("KNN_English preprocess — output shape:", data.shape)
        return data

    # ------------------------------------------------------------------
    def _riemannian_distance_matrix(self, eeg_data):
        """Compute full pairwise Riemannian distance matrix via CSD."""
        num_epochs = len(eeg_data)

        def _rd(A, B):
            return np.sqrt(np.real(np.trace(A + B - 2 * sqrtm(A @ B))))

        print("Computing CSD matrices...")
        csd_matrices = np.real(
            np.fft.fft([np.corrcoef(epoch) for epoch in eeg_data], axis=2)
        )

        dist_matrix = np.zeros((num_epochs, num_epochs))
        upper_i, upper_j = np.triu_indices(num_epochs, k=1)
        for i, j in zip(upper_i, upper_j):
            d = _rd(csd_matrices[i], csd_matrices[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

        print("Riemannian distance matrix done.")
        return dist_matrix

    # ------------------------------------------------------------------
    def _knn_classify(self, train_distances, train_labels, test_row, k):
        distances = np.linalg.norm(train_distances - test_row, axis=1)
        neighbors = np.argsort(distances)[:k]
        return Counter(train_labels[neighbors]).most_common(1)[0][0]

    # ------------------------------------------------------------------
    def predict(self, data):
        print("KNN_English predict — input shape:", data.shape)

        # concatenate train + test, compute joint distance matrix
        combined   = np.concatenate((self.train_data, data))
        dist_matrix = self._riemannian_distance_matrix(combined)

        len_train = len(self.train_data)
        len_test  = len(data)
        dynamic_test_size = len_test / len_train

        X_train, X_test = train_test_split(
            dist_matrix,
            test_size=dynamic_test_size,
            random_state=None,
            shuffle=False
        )

        predicted_labels = np.zeros(len_test, dtype=int)
        for idx in range(len_test):
            predicted_labels[idx] = self._knn_classify(
                X_train, self.labels, X_test[idx], self.k
            )

        print("Predictions:", predicted_labels)
        return st.mode(predicted_labels)