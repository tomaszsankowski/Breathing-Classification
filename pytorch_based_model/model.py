import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=20, hidden_size=256, num_layers=3, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # Swap the dimensions for LSTM (batch, seq, feature)
        _, (hn, _) = self.lstm(x)
        x = hn[-1]  # Take the last hidden state
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AudioDatasetTrain(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mfcc, label = self.data[idx]
        return torch.tensor(mfcc).float(), torch.tensor(label).long()


class AudioDatasetRealtime(AudioDatasetTrain):
    def __getitem__(self, idx):
        mfcc = self.data[idx]
        return torch.tensor(mfcc).float()
