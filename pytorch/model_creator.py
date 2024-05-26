import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Definiuję model sieci neuronowej
class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(p=0.25)  # Dodajemy Dropout
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.dropout2 = nn.Dropout(p=0.25)  # Dodajemy Dropout
        self.fc1 = nn.Linear(64 * 233, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 3)  # 3 klasy: wydech, wdech, cisza

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = self.dropout1(x)  # Używamy Dropout
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = self.dropout2(x)  # Używamy Dropout
        x = x.view(-1, 64 * 233)  # Zmieniamy 64 * 8 na 64 * 233
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Tworzę klasę Dataset
class AudioDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mfcc, label = self.data[idx]
        mfcc = mfcc.reshape(-1)  # Spłaszczamy dwie ostatnie osie
        mfcc = np.expand_dims(mfcc, axis=0)  # Dodajemy dodatkowy wymiar
        return torch.tensor(mfcc).float(), torch.tensor(label).long()

if __name__ == '__main__':
    # Tworzę instancję modelu
    model = AudioClassifier()

    # Definiuję funkcję straty i optymalizator
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Przygotowuję dane treningowe i testowe
    exhale_dir = '../data/exhale'
    inhale_dir = '../data/inhale'
    silence_dir = '../data/silence'

    # Wczytuję dane
    exhale_files = [os.path.join(exhale_dir, file) for file in os.listdir(exhale_dir)]
    inhale_files = [os.path.join(inhale_dir, file) for file in os.listdir(inhale_dir)]
    silence_files = [os.path.join(silence_dir, file) for file in os.listdir(silence_dir)]

    # Wczytuję dane treningowe
    frame_length = 24000  # Długość ramki w próbkach

    # Wczytuję dane treningowe
    train_data = []
    for label, files in enumerate([exhale_files, inhale_files, silence_files]):
        for file in files:
            y, sr = librosa.load(file, mono=True)
            for i in range(0, len(y), frame_length):
                frame = y[i:i + frame_length]
                if len(frame) == frame_length:  # Ignorujemy ostatnią ramkę, jeśli jest krótsza
                    mfcc = librosa.feature.mfcc(y=frame, sr=sr)
                    train_data.append((mfcc, label))


    # Tworzę instancję klasy Dataset
    train_data, val_data = train_test_split(train_data, test_size=0.2)

    # Tworzę instancje klasy Dataset
    train_dataset = AudioDataset(train_data)
    val_dataset = AudioDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Trening modelu
    num_epochs = 20  # Zwiększamy liczbę epok
    best_val_accuracy = 0.0
    patience = 5  # Ustawiamy wartość cierpliwości dla early stopping
    early_stopping_counter = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch')
        for inputs, labels in progress_bar:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            running_accuracy += accuracy_score(labels.cpu(), predicted.cpu())

            progress_bar.set_postfix(loss=running_loss / len(progress_bar), accuracy=running_accuracy / len(progress_bar))
        print('Train Loss: {:.4f}, Train Accuracy: {:.4f}'.format(running_loss / len(train_loader),
                                                                  running_accuracy / len(train_loader)))

        # Walidacja modelu
        model.eval()
        val_running_loss = 0.0
        val_running_accuracy = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_running_accuracy += accuracy_score(labels.cpu(), predicted.cpu())
        avg_val_loss = val_running_loss / len(val_loader)
        avg_val_accuracy = val_running_accuracy / len(val_loader)
        print('Val Loss: {:.4f}, Val Accuracy: {:.4f}'.format(avg_val_loss, avg_val_accuracy))

        # Early stopping
        if avg_val_accuracy > best_val_accuracy:
            best_val_accuracy = avg_val_accuracy
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping triggered. No improvement in validation accuracy.")
                break

    torch.save(model.state_dict(), 'audio_classifier.pth')
