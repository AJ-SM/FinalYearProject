import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size_2, num_layers=2, num_classes=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm    = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.3, batch_first=True)
        self.fc      = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)
        return out


class LSTMModel2(nn.Module):
    """Two-layer stacked LSTM — kept for reference, not used in inference."""
    def __init__(self, input_size, hidden_size_1, hidden_size_2, num_layers, num_classes):
        super(LSTMModel2, self).__init__()
        self.lstm1    = nn.LSTM(input_size,    hidden_size_1, num_layers, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2    = nn.LSTM(hidden_size_1, hidden_size_2, num_layers, batch_first=True)
        self.fc       = nn.Linear(hidden_size_2, num_classes)
        self.softmax  = nn.Softmax(dim=1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x     = self.dropout1(x)
        x, _ = self.lstm2(x)
        out   = self.fc(x[:, -1, :])
        out   = self.softmax(out)
        return out