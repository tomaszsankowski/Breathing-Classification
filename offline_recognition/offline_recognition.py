import wave

import librosa
import numpy as np
import pandas as pd
import pyaudio
import torch
from matplotlib import pyplot as plt

from pytorch_based_model.model import AudioClassifier, AudioDatasetRealtime as AudioDataset

# Constants

REFRESH_TIME = 0.25
N_FOURIER = 512

FORMAT = pyaudio.paInt16

INHALE_COUNTER = 0
EXHALE_COUNTER = 0
SAME_CLASS_IN_ROW_COUNTER = 0
CLASSIFIES_IN_ROW_TO_COUNT = 2  # How many same classifies in row to count it as a real one
PREVIOUS_CLASSIFIED_CLASS = 2  # 0 - Exhale, 1 - Inhale

PREVIOUS_CLASS_BONUS = 0.2

CHANNELS = 2
RATE = 44100
DEVICE_INDEX = 4

CHUNK_SIZE = int(RATE * REFRESH_TIME)

running = True
bonus = 1.15

filename = '2024-07-01_17-58-58'
CLASSIFIER_MODEL_PATH = '../pytorch_based_model/audio_rnn_classifier.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RealTimeAudioClassifier:
    def __init__(self, model_path):
        self.model = AudioClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model = self.model.to(device)
        self.model.eval()
        self.last_prediction = None

    def predict(self, audio_path):
        y, sr = librosa.load(audio_path, sr=RATE, mono=True)
        split_y = [y[i:i+CHUNK_SIZE] for i in range(0, len(y), CHUNK_SIZE)]
        if split_y[-1].shape[0] < CHUNK_SIZE:
            split_y = split_y[:-1]
        predictions = []
        for y in split_y:
            frames = []
            for i in range(0, len(y), CHUNK_SIZE):
                frame = y[i:i + CHUNK_SIZE]
                if len(frame) == CHUNK_SIZE:
                    mfcc = librosa.feature.mfcc(y=frame, sr=sr)
                    frames.append(mfcc)

            frames = AudioDataset(frames)
            frames = torch.utils.data.DataLoader(frames, batch_size=1, shuffle=False)
            for frames in frames:
                frames = frames.to(device)

            outputs = self.model(frames)
            global bonus
            if self.last_prediction is not None:
                print("Before bonus", outputs)
                outputs[0][self.last_prediction] *= bonus
                print("After bonus", outputs)
            _, predicted = torch.max(outputs, 1)
            self.last_prediction = predicted.cpu().numpy()[0]
            predictions.append(predicted.cpu().numpy()[0])
        return predictions


if __name__ == "__main__":
    prediction_path = f'{filename}.wav'
    tags_path = f'{filename}.csv'

    tags = pd.read_csv(tags_path)['tag'].to_numpy()
    tag_time = pd.read_csv(tags_path)['time'].to_numpy()

    wf = wave.open(prediction_path, 'rb')
    classifier = RealTimeAudioClassifier(CLASSIFIER_MODEL_PATH)
    predictions = classifier.predict(prediction_path)

    signal = wf.readframes(-1)
    signal = np.frombuffer(signal, dtype='int16')

    time = np.linspace(0, len(signal) / wf.getframerate(), num=len(signal))

    fig, axs = plt.subplots(2, 1)  # 2 rows, 1 column

    for i in range(len(predictions)):
        color = ''
        if predictions[i] == 0: # exhale
            color = 'green'
        elif predictions[i] == 1: # inhale
            color = 'red'
        else:
            color = 'blue'
        axs[0].plot(time[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE], signal[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE], color=color)
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('Amplitude')
    axs[0].set_title('Waveform of the audio file [CLASSIFICATION]')

    for i in range(len(tags) - 1):
        color = ''
        if tags[i] == 'E':
            color = 'green'
        elif tags[i] == 'I':
            color = 'red'
        else:
            color = 'blue'

        start_index = int(tag_time[i] * wf.getframerate())
        end_index = int(tag_time[i + 1] * wf.getframerate())
        segment_time = np.linspace(tag_time[i], tag_time[i + 1], end_index - start_index)
        axs[1].plot(segment_time, signal[start_index:end_index], color=color)
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Amplitude')
    axs[1].set_title('Waveform of the audio file [TAG]')

    plt.tight_layout()

    plt.show()
