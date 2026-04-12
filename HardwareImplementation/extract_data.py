import numpy as np
import csv
from os import walk
import matplotlib.pyplot as plt
class data_extractor():
    def __init__(self, path):
        self.path = path
        self.data_path = path + "/data.csv"
        self.timestamps_path = path + "/timestamps.csv"
        self.samples = 256*5  # freq*time
        self.sfreq = 256

        self.data = []
        self.timestamps = []
        self.header = []

        self.channels = np.array(['EEG.Cz',
                                'EEG.Fz', 'EEG.Fp1' ,'EEG.F7' ,'EEG.F3', 'EEG.FC1', 'EEG.C3', 'EEG.FC5',
                                'EEG.FT9', 'EEG.T7',  'EEG.P3' ,'EEG.P7',
                                'EEG.O1', 'EEG.Pz' ,'EEG.Oz', 'EEG.O2' ,'EEG.P8' ,'EEG.P4', 'EEG.CP2' ,'EEG.CP6',
                                'EEG.T8' ,'EEG.FT10', 'EEG.FC6', 'EEG.C4', 'EEG.FC2', 'EEG.F4',
                                'EEG.F8' ,'EEG.Fp2'])
        self.reref_channels = np.array(["EEG.TP9", "EEG.TP10", "EEG.CP5","EEG.CP1"])

    def read_data(self):
        self.data = []
        with open(self.data_path, mode ='r')as file:
            csvFile = csv.reader(file)
            for lines in csvFile:
                    self.data.append(lines)

        self.data = np.array(self.data[1:])
        # ##print(self.data[0])
        with open(self.timestamps_path, mode ='r')as file:
            csvFile = csv.reader(file)
            self.timestamps = []
            for lines in csvFile:
                    self.timestamps.append(lines)
        self.timestamps = np.array(self.timestamps)
        time_stamp_index = np.where(np.array(self.timestamps[0]) == "timestamp")[0][0]        
        #print(time_stamp_index)
        self.timestamps = self.timestamps[1:,time_stamp_index].astype(np.float64)
        #print(self.timestamps)


    def get_data(self):
        #print(self.data[1:,0])
        #print(self.timestamps)
        timeId = np.searchsorted(np.float64(self.data[1:,0]), np.float64(self.timestamps), side="left")+1
        channels_id = np.where(np.in1d( self.data[0] ,self.channels))[0]
        reref_channels_id = np.where(np.in1d( self.data[0] ,self.reref_channels))[0]    
        #print(channels_id)
        #print(timeId)
        self.data = self.data[1:,channels_id].astype(np.float64)
        #print(self.data.shape)
        self.reref_data = self.data[:,reref_channels_id].astype(np.float64)
        # self.reref_data = self.data[1:, reref_channels_id].astype(np.float32)
        #print("reref_data ", self.reref_data.shape)
        self.reref_data = np.mean(self.reref_data, axis=1).reshape(-1, 1)
        self.data  = self.data - self.reref_data
        self.data -= self.data.mean(axis=0, keepdims=True)
        #print("reeeeerefL ", self.reref_data.shape)
        ChannelDataList = []

        for i, id in enumerate(timeId):
        # #print(id)
            ChannelDataList.append(self.data[id:id+self.samples])
    
        self.data = np.array(ChannelDataList, dtype=np.float32)
        
        # min_val = np.min(self.data)
        # max_val = np.max(self.data)
        # self.data = (self.data - min_val) / (max_val - min_val)

        # #print the max of the first row (or channel) after normalization
        #print("Max of first row after normalization:", np.max(self.data[0]))
        del ChannelDataList
        #print(self.data.shape)
        # for i in range(self.data.shape[2]):
        #     plt.plot(self.data[0, :, i])
        # plt.show()

if __name__ == "__main__":  
    test = data_extractor("Data/kishore")
    # test = data_extractor("Kishore")    
    test.read_data()
    test.get_data()

