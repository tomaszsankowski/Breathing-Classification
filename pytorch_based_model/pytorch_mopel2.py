import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import librosa
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from model import AudioClassifier, AudioDatasetTrain as AudioDataset


REFRESH_TIME = 0.25
RATE = 48000
CHUNK_SIZE = int(RATE * REFRESH_TIME)
NUM_EPOCHS = 50
PATIENCE_TIME = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 32

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)

    total_time = time.time()
    start_time = time.time()

    print("Creating model...")
    model = AudioClassifier()
    model = model.to(device)
    print("Model created, time: ", time.time() - start_time)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    exhale_dir = '../data/exhale'
    inhale_dir = '../data/inhale'
    silence_dir = '../data/silence'

    exhale_files = [os.path.join(exhale_dir, file) for file in os.listdir(exhale_dir)]
    inhale_files = [os.path.join(inhale_dir, file) for file in os.listdir(inhale_dir)]
    silence_files = [os.path.join(silence_dir, file) for file in os.listdir(silence_dir)]

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
            y, sr = librosa.load(file, sr=RATE, mono=True)
            for i in range(0, len(y), CHUNK_SIZE):
                frame = y[i:i + CHUNK_SIZE]
                if len(frame) == CHUNK_SIZE:  # Ignore the last frame if it's shorter
                    mfcc = librosa.feature.mfcc(y=frame, sr=sr)
                    train_data.append((mfcc, label))

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
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("DataLoaders created, time: ", time.time() - start_time)

    best_val_accuracy = 0.0
    early_stopping_counter = 0

    print("Training model...")
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        running_accuracy = 0.0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{NUM_EPOCHS}', unit='batch')
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

        scheduler.step()

        # Early stopping
        if avg_val_accuracy > best_val_accuracy:
            best_val_accuracy = avg_val_accuracy
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= PATIENCE_TIME:
                print("Early stopping triggered. No improvement in validation accuracy.")
                break

    print('Finished Training, time: ', time.time() - start_time)
    print('Saving model...')
    start_time = time.time()
    torch.save(model.state_dict(), 'audio_rnn_classifier.pth')
    print("Model saved, time: ", time.time() - start_time)
    print("Finished, Total time: ", time.time() - total_time)
