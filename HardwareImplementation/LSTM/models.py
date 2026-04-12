import torch 
import torch.nn as nn

    
class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size,hidden_size_2, num_layers=2, num_classes=2):
            super(LSTMModel, self).__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.3, batch_first=True)
            self.fc = nn.Linear(hidden_size, num_classes)
            self.softmax = nn.Softmax(dim=1)
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            
        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            # print("input: ", x.shape)
            out, _ = self.lstm(x, (h0, c0))
            # print("o1: ", out.shape)
            out = self.fc(out[:, -1, :]) 
            # print("o2: ", out.shape)
            out = self.softmax(out)  
            # print("o3: ", out.shape)     
            return out

class LSTMModel2(nn.Module):
        def __init__(self, input_size, hidden_size_1,hidden_size_2, num_layers, num_classes):
            super(LSTMModel2, self).__init__()

            self.lstm1 = nn.LSTM(input_size, hidden_size_1, num_layers, batch_first=True)  
            self.dropout1 = nn.Dropout(0.3)
            self.lstm2 = nn.LSTM(hidden_size_1, hidden_size_2, num_layers, batch_first=True)
            self.dropout2 = nn.Dropout(0.3)

            # self.hidden_size = hidden_size_1
            # self.num_layers = num_layers
            # self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.3,batch_first=True)
            self.fc = nn.Linear(hidden_size_2, num_classes)
            self.softmax = nn.Softmax(dim=1)
            
            
        def forward(self, x):
            # h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            # c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)


            x, _ = self.lstm1(x)
            x = self.dropout1(x)
            x, _ = self.lstm2(x)
            # x = self.dropout2(x)
            out = self.fc(x[:, -1, :]) 

            # # print("input: ", x.shape)
            # out, _ = self.lstm(x, (h0, c0))
            # # print("o1: ", out.shape)
            # out = self.fc(out[:, -1, :]) 
            # # print("o2: ", out.shape)
            out = self.softmax(out)  
            # # print("o3: ", out.shape)     
            return out

