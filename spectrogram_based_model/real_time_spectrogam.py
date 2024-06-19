import numpy as np
from tensorflow.keras.models import load_model
from scipy.signal import stft
import pyaudio
import matplotlib.pyplot as plt
import time
import pandas as pd

# Constants

REFRESH_TIME = 0.5
N_FOURIER = 4096

PREVIOUS_CLASS_BONUS = 0.2

CHANNELS = 1
RATE = 44100
DEVICE_INDEX = 4

running = True

# Load the model

model = load_model(f'best_models/mobile_net_model_{N_FOURIER}_{REFRESH_TIME}_small.keras')


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


# Function to create a spectrogram from audio data:

def create_spectrogram(frames):

    # Calculate STFT parameters

    furier_hop = np.floor(RATE * REFRESH_TIME / 224)
    noverlap = N_FOURIER - furier_hop

    # Perform FFT

    stft_data = stft(frames, RATE, nperseg=N_FOURIER, noverlap=noverlap, scaling='spectrum')[2]

    # Take only the first 224x224 part of the spectrogram

    spectrogram_in = stft_data[:224, :224]

    # Return spectrogram as matrix of positive values

    return np.abs(spectrogram_in)


# Function to classify given spectrogram

def classify_realtime_audio(spectrogram_in):
    global last_prediction

    # Prepare input for the model ( change dimensions )

    spectrogram_in = np.expand_dims(spectrogram_in, axis=-1)
    spectrogram_in = np.expand_dims(spectrogram_in, axis=0)

    # Model prefiction

    predictionon = model.predict(spectrogram_in, verbose=0)

    # Add bonus for previous class
    predictionon[0][last_prediction] += PREVIOUS_CLASS_BONUS

    # Get new previous prediction

    last_prediction = np.argmax(predictionon)

    # Print wages for every prediction

    print('Predicted class: ', np.array2string(np.round(predictionon, 4), suppress_small=True))

    # Return predicted class number

    return np.argmax(predictionon)


# Plot variables

PLOT_TIME_HISTORY = 5
PLOT_CHUNK_SIZE = int(RATE*REFRESH_TIME)

plotdata = np.zeros((RATE*PLOT_TIME_HISTORY, 1))
x_linspace = np.arange(0, RATE*PLOT_TIME_HISTORY, 1)
predictions = np.zeros((int(PLOT_TIME_HISTORY/REFRESH_TIME), 1))

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
    ma[:window_size-1] = data_flatten[:window_size-1]  # Leave the first window_size-1 elements unchanged
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
        if predictions[i] == 0:  # Inhale
            color = 'red'
        elif predictions[i] == 1:  # Exhale
            color = 'green'
        else:  # Silence
            color = 'blue'
        ax.plot(x_linspace[PLOT_CHUNK_SIZE*i:PLOT_CHUNK_SIZE*(i+1)], plotdata[PLOT_CHUNK_SIZE*i:PLOT_CHUNK_SIZE*(i+1)], color=color)

    # Set plot properties and show it

    ax.set_facecolor(facecolor)
    ax.set_ylim(ylim)

    plt.draw()
    plt.pause(0.01)


# Main function

if __name__ == "__main__":

    # Initialize microphone

    audio = SharedAudioResource()

    # Main loop

    last_prediction = 2
    while running:

        # Set timer to check how long each prediction takes

        start_time = time.time()

        # Collect samples

        buffer = audio.read()

        if buffer is None:
            continue

        # Create spectrogram

        spectrogram = create_spectrogram(buffer)

        # Make prediction

        prediction = classify_realtime_audio(spectrogram)

        # Update plot

        update_plot(buffer, prediction)

        # Print time needed for this loop iteration

        print(time.time() - start_time)
    # Close audio

    audio.close()
