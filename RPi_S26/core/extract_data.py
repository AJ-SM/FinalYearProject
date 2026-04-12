import numpy as np
import csv
from pathlib import Path


class data_extractor:
    def __init__(self, path):
        self.path             = path
        self.data_path        = str(Path(path) / "data.csv")
        self.timestamps_path  = str(Path(path) / "timestamps.csv")
        self.samples          = 256 * 5   # freq * time = 1280
        self.sfreq            = 256

        self.data       = []
        self.timestamps = []
        self.header     = []

        self.channels = np.array([
            'EEG.Cz',
            'EEG.Fz',  'EEG.Fp1', 'EEG.F7',  'EEG.F3',  'EEG.FC1', 'EEG.C3',  'EEG.FC5',
            'EEG.FT9', 'EEG.T7',  'EEG.P3',  'EEG.P7',
            'EEG.O1',  'EEG.Pz',  'EEG.Oz',  'EEG.O2',  'EEG.P8',  'EEG.P4',  'EEG.CP2', 'EEG.CP6',
            'EEG.T8',  'EEG.FT10','EEG.FC6', 'EEG.C4',  'EEG.FC2', 'EEG.F4',
            'EEG.F8',  'EEG.Fp2'
        ])
        self.reref_channels = np.array([
            "EEG.TP9", "EEG.TP10", "EEG.CP5", "EEG.CP1"
        ])

    def read_data(self):
        self.data = []
        with open(self.data_path, mode='r') as f:
            reader = csv.reader(f)
            for line in reader:
                self.data.append(line)

        # Row 0 is metadata string, row 1 is header, rows 2+ are data
        self.data = np.array(self.data[1:])

        with open(self.timestamps_path, mode='r') as f:
            reader = csv.reader(f)
            self.timestamps = [line for line in reader]

        self.timestamps = np.array(self.timestamps)
        time_stamp_index = np.where(
            np.array(self.timestamps[0]) == "timestamp"
        )[0][0]
        self.timestamps = self.timestamps[1:, time_stamp_index].astype(np.float64)

    def get_data(self):
        timeId          = np.searchsorted(
            np.float64(self.data[1:, 0]),
            np.float64(self.timestamps),
            side="left"
        ) + 1
        channels_id     = np.where(np.in1d(self.data[0], self.channels))[0]
        reref_channels_id = np.where(np.in1d(self.data[0], self.reref_channels))[0]

        self.data       = self.data[1:, channels_id].astype(np.float64)
        self.reref_data = self.data[:, reref_channels_id].astype(np.float64)
        self.reref_data = np.mean(self.reref_data, axis=1).reshape(-1, 1)

        self.data -= self.reref_data
        self.data -= self.data.mean(axis=0, keepdims=True)

        channel_data_list = []
        for id in timeId:
            channel_data_list.append(self.data[id:id + self.samples])

        self.data = np.array(channel_data_list, dtype=np.float32)
        del channel_data_list


if __name__ == "__main__":
    test = data_extractor("data/english/kishore")
    test.read_data()
    test.get_data()
    print(test.data.shape)