import os
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from scipy.linalg import sqrtm
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt
import time

class RiemannKNN_Model:
    def __init__(self, k_range=100):
        self.k_range = k_range
        self.best_k = None
        self.model_data = None

        # ✅ create cache folder (SAFE)
        os.makedirs("cache", exist_ok=True)

    # =========================
    # LOAD DATA
    # =========================
    def load_data(self, main_folder_path, selected_words=["PAIN", "LIGHT"]):
        print("Loading data...")

        data_list = []
        subject_folders = [
            f for f in os.listdir(main_folder_path)
            if os.path.isdir(os.path.join(main_folder_path, f))
        ]

        for subject in subject_folders:
            subject_path = os.path.join(main_folder_path, subject)
            files = [f for f in os.listdir(subject_path) if f.endswith("_labeled_EEG_data.csv")]

            for file in files:
                df = pd.read_csv(os.path.join(subject_path, file))
                df["Subject_ID"] = subject
                data_list.append(df)

        df = pd.concat(data_list, ignore_index=True)
        df = df[df["Label"].isin(selected_words)]

        self.label_encoder = LabelEncoder()
        df["Label"] = self.label_encoder.fit_transform(df["Label"])

        print("Label mapping:",
              dict(zip(self.label_encoder.classes_,
                       self.label_encoder.transform(self.label_encoder.classes_))))

        return df

    # =========================
    # EPOCHING
    # =========================
    def create_epochs(self, df, window_size=256, stride=128):
        print("Creating epochs...")

        eeg_data = df.iloc[:, 1:33].values
        labels = df["Label"].values

        epochs = []
        epoch_labels = []

        for i in range(0, len(eeg_data) - window_size + 1, stride):
            if len(set(labels[i:i + window_size])) == 1:
                epoch = eeg_data[i:i + window_size].T
                epochs.append(epoch)
                epoch_labels.append(labels[i])

        epochs = np.array(epochs)
        epoch_labels = np.array(epoch_labels)

        print("Epochs shape:", epochs.shape)

        return epochs, epoch_labels

    # =========================
    # COVARIANCE (CACHED)
    # =========================
    def compute_covariance(self, epochs):
        cache_path = "cache/cov_matrices.npy"

        if os.path.exists(cache_path):
            print("Loading cached covariance...")
            return np.load(cache_path, allow_pickle=True)

        print("Computing covariance matrices...")
        cov_estimator = LedoitWolf()

        cov_matrices = [
            cov_estimator.fit(epoch.T).covariance_
            for epoch in epochs
        ]

        np.save(cache_path, cov_matrices)
        print("Covariance cached!")

        return cov_matrices

    # =========================
    # RIEMANNIAN DISTANCE
    # =========================
    def riemannian_distance(self, A, B):
        return np.sqrt(np.trace(A + B - 2 * sqrtm(A @ B)).real)

    # =========================
    # DISTANCE MATRIX (CACHED)
    # =========================


    # def compute_distance_matrix(self, cov_matrices):
    #     cache_path = "cache/distance_matrix.npy"

    #     if os.path.exists(cache_path):
    #         print("Loading cached distance matrix...")
    #         return np.load(cache_path, allow_pickle=True)

    #     print("Computing distance matrix...")

    #     num_epochs = len(cov_matrices)
    #     distances = np.zeros((num_epochs, num_epochs))

    #     for i in range(num_epochs):
    #         for j in range(i + 1, num_epochs):
    #             dist = self.riemannian_distance(cov_matrices[i], cov_matrices[j])
    #             distances[i, j] = dist
    #             distances[j, i] = dist

    #     np.save(cache_path, distances)
    #     print("Distance matrix cached!")

    #     return distances


    def compute_distance_matrix(self, cov_matrices):
        cache_path = "cache/distance_matrix.npy"

        if os.path.exists(cache_path):
            print("Loading cached distance matrix...")
            return np.load(cache_path, allow_pickle=True)

        print("Computing distance matrix...")

        num_epochs = len(cov_matrices)
        distances = np.zeros((num_epochs, num_epochs))

        for i in range(num_epochs):
            print(f"Processing row {i}/{num_epochs}")

            for j in range(i + 1, num_epochs):
                dist = self.riemannian_distance(cov_matrices[i], cov_matrices[j])
                distances[i, j] = dist
                distances[j, i] = dist

        np.save(cache_path, distances)
        print("Distance matrix cached!")

        return distances

    # =========================
    # PrePROCESSING 
    # =========================
    # def preprocess_data(self, data):
    #     print("Preprocessing data...")
    #     print("Input shape:", data.shape)

    #     # data shape: (samples, time, channels)

    #     # remove NaNs (important for your warning)
    #     data = np.nan_to_num(data)

    #     # normalize (same style as before)
    #     mean = np.mean(data, axis=1, keepdims=True)
    #     std = np.std(data, axis=1, keepdims=True)

    #     # avoid division by zero
    #     std[std == 0] = 1

    #     data = (data - mean) / std

    #     print("Preprocessed shape:", data.shape)

    #     return data
    def preprocess_data(self, data):
        print("Preprocessing data...")
        print("Input shape:", data.shape)

        # data: (samples, 1280, channels)

        window_size = 256
        stride = 256   # no overlap (safe)

        new_epochs = []

        for sample in data:
            for i in range(0, sample.shape[0] - window_size + 1, stride):
                epoch = sample[i:i+window_size]
                new_epochs.append(epoch)

        new_epochs = np.array(new_epochs)

        print("Converted to epochs:", new_epochs.shape)

        return new_epochs
        
    # =========================
    # KNN CLASSIFIER
    # =========================
    def knn_classifier(self, train_data, train_labels, test_row, k):
        distances = np.linalg.norm(train_data - test_row, axis=1)
        neighbors = np.argsort(distances)[:k]
        labels = train_labels[neighbors]
        return Counter(labels).most_common(1)[0][0]

    # =========================
    # TRAIN
    # =========================
    def train(self, distance_matrix, labels):
        print("Training KNN...")

        X_train, X_test, y_train, y_test = train_test_split(
            distance_matrix, labels, test_size=0.3, random_state=42
        )

        self.X_train = X_train   
        self.y_train = y_train   

        accuracies = []

        for k in range(1, self.k_range):
            preds = [
                self.knn_classifier(X_train, y_train, row, k)
                for row in X_test
            ]
            acc = metrics.accuracy_score(y_test, preds)
            accuracies.append(acc)

        print("training done!")
        self.best_k = np.argmax(accuracies) + 1
        np.save("cache/best_k.npy", self.best_k)
        print(f"Best k: {self.best_k}")

        best_acc = accuracies[self.best_k - 1]

        print(f"Best k: {self.best_k} | Accuracy: {best_acc:.4f}")

        self.model_data = (X_train, X_test, y_train, y_test)

        return accuracies
    

    # =========================
    # EVALUATION
    # =========================
    def evaluate(self):
        print("Evaluating model...")

        X_train, X_test, y_train, y_test = self.model_data

        preds = [
            self.knn_classifier(X_train, y_train, row, self.best_k)
            for row in X_test
        ]

        print("\n--- FINAL METRICS ---")
        print("Accuracy:", metrics.accuracy_score(y_test, preds))
        print("Precision:", metrics.precision_score(y_test, preds))
        print("Recall:", metrics.recall_score(y_test, preds))
        print("F1:", metrics.f1_score(y_test, preds))

        print("\nConfusion Matrix:")
        print(metrics.confusion_matrix(y_test, preds))

        print("\nClassification Report:")
        print(metrics.classification_report(
            y_test, preds,
            target_names=self.label_encoder.classes_
        ))



    # =========================
    # PREDICTION
    # =========================

    # def predict(self, raw_data=None):
    #     print("Predicting from raw EEG...")

    #     # Step 1: convert raw data → epoch
    #     # assuming raw_data shape = (samples, channels)
    #     window_size = 256

    #     if raw_data.shape[0] < window_size:
    #         raise ValueError("Not enough data for one epoch")

    #     # take last window (or you can loop later)
    #     epoch = raw_data[-window_size:].T   # shape → (32, 256)

    #     # Step 2: compute covariance
    #     cov_estimator = LedoitWolf()
    #     cov_new = cov_estimator.fit(epoch.T).covariance_

    #     # Step 3: distance to training
    #     distances = []
    #     for cov_train in self.cached_cov_matrices:
    #         dist = self.riemannian_distance(cov_new, cov_train)
    #         distances.append(dist)

    #     distances = np.array(distances)

    #     # Step 4: KNN
    #     neighbors = np.argsort(distances)[:self.best_k]
    #     labels = self.y_train[neighbors]

    #     prediction = Counter(labels).most_common(1)[0][0]

    #     print("Predicted label:", prediction)
    #     return prediction
    

    def predict(self, data):
        print("Predicting...")
        print("Input shape:", data.shape)

        predictions = []

        # data shape = (samples, time, channels)
        for sample in data:
            # sample shape = (time, channels)

            # Step 1: covariance
            cov_estimator = LedoitWolf()
            cov_new = cov_estimator.fit(sample).covariance_

            # Step 2: distance
            distances = [
                self.riemannian_distance(cov_new, cov_train)
                for cov_train in self.cached_cov_matrices
            ]

            distances = np.array(distances)

            # Step 3: KNN
            neighbors = np.argsort(distances)[:self.best_k]
            labels = self.y_train[neighbors]

            pred = Counter(labels).most_common(1)[0][0]
            predictions.append(pred)

        # final decision (like your old KNN)
        final_pred = Counter(predictions).most_common(1)[0][0]

        print("Predictions:", predictions)
        print("Final prediction:", final_pred)

        return final_pred


    # =========================
    # BUILD MODEL
    # =========================
    def build(self):
        df = self.load_data("C:\\Tanmay\\6thSem\\FYP\\code\\Hindi KNN\\13 Subjects Hindi Preprocessed")
        epochs, labels = self.create_epochs(df)
        np.save("cache/epochs.npy", epochs)
        np.save("cache/labels.npy", labels)

        self.cov_matrices = self.compute_covariance(epochs)
        self.distance_matrix = self.compute_distance_matrix(self.cov_matrices)
        self.train(self.distance_matrix, labels)

    # =========================
    # LOAD MODEL
    # =========================
    def load_model(self):
        
        # self.best_k = int(np.load("cache/best_k.npy"))
        self.best_k = 10
        print(f"Loaded best k: {self.best_k}")

        self.cached_cov_matrices = np.load("cache/cov_matrices.npy", allow_pickle=True)
        self.cached_distance_matrix = np.load("cache/distance_matrix.npy", allow_pickle=True)
        self.cached_labels = np.load("cache/labels.npy", allow_pickle=True)
        print(set(self.cached_labels))

        # ✅ FIX: assign training labels
        self.y_train = self.cached_labels
    
    #########################################################################################################################################
    def predict_from_folder(self, folder_path):
        print("\n=== Predicting from folder (DIRECT) ===")

        # =========================
        # STEP 1: LOAD CSV
        # =========================
        data_path = os.path.join(folder_path, "data.csv")

        df = pd.read_csv(data_path)
        print("Raw CSV shape:", df.shape)

        # =========================
        # STEP 2: REMOVE TIMESTAMP
        # =========================
        data = df.iloc[:, 1:].astype(np.float32).values
        print("After removing timestamp:", data.shape)

        # =========================
        # STEP 3: CLEAN NaNs
        # =========================
        data = np.nan_to_num(data)

        # =========================
        # STEP 4: NORMALIZE
        # =========================
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)

        std[std == 0] = 1
        data = (data - mean) / std

        # =========================
        # STEP 5: CREATE SAMPLES (5 sec)
        # =========================
        window_full = 256 * 5   # 1280
        samples = []

        for i in range(0, len(data) - window_full + 1, window_full):
            samples.append(data[i:i+window_full])

        # ✅ FIX: handle empty case
        if len(samples) == 0:
            raise ValueError("Not enough data to form 5-sec window")

        samples = np.array(samples)
        print("Samples shape:", samples.shape)

        # =========================
        # STEP 6: TO EPOCHS
        # =========================
        samples = self.preprocess_data(samples)

        # =========================
        # STEP 7: PREDICT
        # =========================
        pred = self.predict(samples)

        # =========================
        # STEP 8: MAP LABEL
        # =========================
        word = "PAIN" if pred == 0 else "LIGHT"
        # word = self.label_encoder.inverse_transform([pred])[0]

        print("\nFinal Predicted Word:", word)

        return word
    


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    model = RiemannKNN_Model()

    model.load_model()

    folder_path = r"C:\Tanmay\6thSem\FYP\code\Hindi_KNN\Data\abhay_LIGHT"

    model.predict_from_folder(folder_path)