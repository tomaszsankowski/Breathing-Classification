import wave

import librosa
import numpy as np
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

# Model path

CLASSIFIER_MODEL_PATH = 'pytorch_based_model/audio_rnn_classifier.pth'

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
        frames = []
        for i in range(0, len(y), CHUNK_SIZE):
            frame = y[i:i + CHUNK_SIZE]
            if len(frame) == CHUNK_SIZE:  # Ignorujemy ostatnią ramkę, jeśli jest krótsza
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
        return predicted.cpu().numpy()


if __name__ == "__main__":
    wf = wave.open('data/exhale/2024-04-30_11-20-41.wav', 'rb')
    classifier = RealTimeAudioClassifier(CLASSIFIER_MODEL_PATH)
    predictions = classifier.predict('data/exhale/2024-04-30_11-20-41.wav')
    # Wczytanie danych z pliku wav
    signal = wf.readframes(-1)
    signal = np.frombuffer(signal, dtype='int16')

    # Tworzenie osi czasu
    time = np.linspace(0, len(signal) / wf.getframerate(), num=len(signal))

    # Tworzenie wykresu
    plt.figure(figsize=(10, 4))
    plt.plot(time, signal)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Waveform of the audio file')
    plt.show()
