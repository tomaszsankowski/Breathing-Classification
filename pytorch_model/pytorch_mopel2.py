import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from albumentations import Compose, GaussNoise, ShiftScaleRotate, HueSaturationValue, RandomBrightnessContrast
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import albumentations as A

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def augment_mfcc(mfcc, sr=48000, n_mels=128):
    # Add white noise
    wn = np.random.randn(*mfcc.shape)
    data_wn = mfcc + 0.005 * wn

    # Shifting the sound
    data_roll = np.roll(mfcc, 1600)

    # Stretching the sound
    data_stretch = librosa.effects.time_stretch(mfcc, rate=1.07)

    # Write wav files
    #librosa.output.write_wav('wn.wav', data_wn, sr)
    #librosa.output.write_wav('roll.wav', data_roll, sr)
    #librosa.output.write_wav('stretch.wav', data_stretch, sr)

    return data_wn


class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(p=0.1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1)  # Nowa warstwa konwolucyjna
        self.dropout3 = nn.Dropout(p=0.1)

        self.fc1 = nn.Linear(256 * 233, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #print("1: ",x.shape)
        x = F.max_pool1d(x, 2)
        #print("2: ",x.shape)
        x = self.dropout1(x)

        x = F.relu(self.conv2(x))
        #print("3: ",x.shape)
        x = F.max_pool1d(x, 2)
        x = self.dropout2(x)

        x = F.relu(self.conv3(x))  # Nowa warstwa konwolucyjna
        #print("4: ",x.shape)
        x = F.max_pool1d(x, 2)
        #print("6: ",x.shape)
        x = self.dropout3(x)
        #print("7: ",x.shape)

        x = x.view(-1, 256 * 233)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class AudioDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mfcc, label = self.data[idx]
        mfcc = mfcc.reshape(-1)
        mfcc = np.expand_dims(mfcc, axis=0)
        return torch.tensor(mfcc).float(), torch.tensor(label).long()


if __name__ == '__main__':
    total_time = time.time()
    start_time = time.time()
    print("Creating model...")
    model = AudioClassifier()
    model = model.to(device)
    print("Model created, time: ", time.time() - start_time)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    exhale_dir = '../data/exhale'
    inhale_dir = '../data/inhale'
    silence_dir = '../data/silence'

    exhale_files = [os.path.join(exhale_dir, file) for file in os.listdir(exhale_dir)]
    inhale_files = [os.path.join(inhale_dir, file) for file in os.listdir(inhale_dir)]
    silence_files = [os.path.join(silence_dir, file) for file in os.listdir(silence_dir)]

    frame_length = 48000

    train_data = []
    files_list = [exhale_files, inhale_files, silence_files]
    files_names = ['exhale', 'inhale', 'silence']
    print("Loading data...")
    start_time = time.time()
    exhale_frames_size = 0
    inhale_frames_size = 0
    silence_frames_size = 0
    for label, files in enumerate(files_list):
        for file in files:
            y, sr = librosa.load(file, sr=48000, mono=True)
            for i in range(0, len(y), frame_length):
                frame = y[i:i + frame_length]
                if len(frame) == frame_length:  # Ignorujemy ostatnią ramkę, jeśli jest krótsza
                    mfcc = librosa.feature.mfcc(y=frame, sr=sr)
                    train_data.append((mfcc, label))

        #print(files_names[label], " loaded, size: ", len(train_data), " frames")
        if label == 0:
            exhale_frames_size = len(train_data)
            print("Exhale frames size: ", exhale_frames_size)
        elif label == 1:
            inhale_frames_size = len(train_data) - exhale_frames_size
            print("Inhale frames size: ", inhale_frames_size)
        else:
            silence_frames_size = len(train_data) - exhale_frames_size - inhale_frames_size
            print("Silence frames size: ", silence_frames_size)
    print("Data loaded, time: ", time.time() - start_time)

    train_data, val_data = train_test_split(train_data, test_size=0.2)

    print("Creating datasets...")
    start_time = time.time()
    train_dataset = AudioDataset(train_data)
    val_dataset = AudioDataset(val_data)

    print("Datasets created, time: ", time.time() - start_time)
    print("Creating DataLoaders...")
    start_time = time.time()
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    print("DataLoaders created, time: ", time.time() - start_time)

    num_epochs = 20
    best_val_accuracy = 0.0
    patience = 5
    early_stopping_counter = 0
    print("Training model...")
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch')
        for inputs, labels in progress_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            running_accuracy += accuracy_score(labels.cpu(), predicted.cpu())

            progress_bar.set_postfix(loss=running_loss / len(progress_bar),
                                     accuracy=running_accuracy / len(progress_bar))
        print('Train Loss: {:.4f}, Train Accuracy: {:.4f}'.format(running_loss / len(train_loader),
                                                                  running_accuracy / len(train_loader)))

        model.eval()
        val_running_loss = 0.0
        val_running_accuracy = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
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

    print('Finished Training, time: ', time.time() - start_time)
    print('Saving model...')
    start_time = time.time()
    torch.save(model.state_dict(), 'audio_classifier.pth')
    print("Model saved, time: ", time.time() - start_time)
    print("Finished, Total time: ", time.time() - total_time)
