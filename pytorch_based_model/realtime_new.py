import wave
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import time
import pandas as pd
from model import AudioClassifier, AudioDatasetRealtime as AudioDataset
import torch
import librosa

# Constants

REFRESH_TIME = 0.25
N_FOURIER = 512

FORMAT = pyaudio.paInt16

PREVIOUS_CLASS_BONUS = 0.2

CHANNELS = 2
RATE = 44100
DEVICE_INDEX = 4

CHUNK_SIZE = int(RATE * REFRESH_TIME)

running = True
bonus = 1.15

# Model path

CLASSIFIER_MODEL_PATH = 'audio_rnn_classifier.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Audio resource class

class SharedAudioResource:
    buffer = None

    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.buffer_size = int(RATE * REFRESH_TIME)
        self.buffer = np.zeros(self.buffer_size, dtype=np.int16)
        for i in range(self.p.get_device_count()):
            print(self.p.get_device_info_by_index(i))
        self.stream = self.p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE, input=True,
                                  frames_per_buffer=self.buffer_size, input_device_index=DEVICE_INDEX)

    def read(self):
        self.buffer = self.stream.read(self.buffer_size, exception_on_overflow=False)
        return np.frombuffer(self.buffer, dtype=np.int16)

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


# Class for prediction

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


# Plot variables

PLOT_TIME_HISTORY = 5
PLOT_CHUNK_SIZE = int(RATE * REFRESH_TIME)

plotdata = np.zeros((RATE * PLOT_TIME_HISTORY, 1))
x_linspace = np.arange(0, RATE * PLOT_TIME_HISTORY, 1)
predictions = np.zeros((int(PLOT_TIME_HISTORY / REFRESH_TIME), 1))

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(plotdata, color='white')


# Key handler for plot window

def on_key(event):
    global running
    if event.key == ' ':
        plt.close()
        running = False


# Configuration of plot properties and other elements

fig.canvas.manager.set_window_title('Realtime Breath Detector')  # Title
fig.suptitle('Press [SPACE] to stop. Colours meaning: Red - Inhale, Green - Exhale, Blue - Silence')  # Instruction
fig.canvas.mpl_connect('key_press_event', on_key)  # Key handler

ylim = (-500, 500)
facecolor = (0, 0, 0)

ax.set_facecolor(facecolor)
ax.set_ylim(ylim)


# Moving avarage function

def moving_average(data, window_size):
    data_flatten = data.flatten()
    ma = pd.Series(data_flatten).rolling(window=window_size).mean().to_numpy()
    ma[:window_size - 1] = data_flatten[:window_size - 1]  # Leave the first window_size-1 elements unchanged
    return ma.reshape(-1, 1)


# Plot update function

def update_plot(frames, prediction):
    global plotdata, predictions, ax

    # Roll signals and predictions vectors and insert new value at the end

    plotdata = np.roll(plotdata, -len(frames))
    plotdata[-len(frames):] = frames.reshape(-1, 1)

    predictions = np.roll(predictions, -1)
    predictions[-1] = prediction

    # Moving avarage on plotdata (uncomment if needed)

    plotdata = moving_average(plotdata, 50)

    # Clean the plot and plot the new data

    ax.clear()

    for i in range(0, len(predictions)):
        if predictions[i] == 0:  # Exhale
            color = 'green'
        elif predictions[i] == 1:  # Inhale
            color = 'red'
        else:  # Silence
            color = 'blue'
        ax.plot(x_linspace[PLOT_CHUNK_SIZE * i:PLOT_CHUNK_SIZE * (i + 1)],
                plotdata[PLOT_CHUNK_SIZE * i:PLOT_CHUNK_SIZE * (i + 1)], color=color)

    # Set plot properties and show it

    ax.set_facecolor(facecolor)
    ax.set_ylim(ylim)

    plt.draw()
    plt.pause(0.01)


# Main function

if __name__ == "__main__":

    # Initialize microphone and classifier

    audio = SharedAudioResource()

    classifier = RealTimeAudioClassifier(CLASSIFIER_MODEL_PATH)

    # Main loop

    last_prediction = 2
    while running:

        # Set timer to check how long each prediction takes

        start_time = time.time()

        # Collect samples

        buffer = audio.read()

        if buffer is None:
            continue

        # Create wav file to store frames

        wf = wave.open("../temp/temp.wav", 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(buffer))
        wf.close()

        # Make prediction

        prediction = classifier.predict('../temp/temp.wav')

        # Update plot
        update_plot(buffer[::2], prediction)

        # Print time needed for this loop iteration

        print(time.time() - start_time)
    # Close audio

    audio.close()
