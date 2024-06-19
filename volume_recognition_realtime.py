import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import time
import pandas as pd
import volume_recognition

REFRESH_TIME = 0.15
CHANNELS = 1
RATE = 44100
DEVICE_INDEX = 5
running = True
VR = volume_recognition.Volume_Recognition()


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


# Plot variables

PLOT_TIME_HISTORY = 5
PLOT_CHUNK_SIZE = int(RATE * REFRESH_TIME)

plotdata = np.zeros((RATE * PLOT_TIME_HISTORY, 1))
x_linspace = np.arange(0, RATE * PLOT_TIME_HISTORY, 1)
predictions = np.zeros((int(PLOT_TIME_HISTORY / REFRESH_TIME), 1))

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(plotdata, color='white')

# Configuration of plot properties and other elements

fig.canvas.manager.set_window_title('Volume recognition')  # Title

ylim = (0, 1000)
facecolor = (0, 0, 0)

ax.set_facecolor(facecolor)
ax.set_ylim(ylim)


def moving_average(data, window_size):
    data_flatten = data.flatten()
    ma = pd.Series(data_flatten).rolling(window=window_size).mean().to_numpy()
    ma[:window_size - 1] = data_flatten[:window_size - 1]  # Leave the first window_size-1 elements unchanged
    return ma.reshape(-1, 1)


# Function to get color based on decision
def get_color(decision):
    if decision == 0:
        return 'blue'
    elif decision == 1:
        return 'red'
    else:
        return 'green'


# Plot update function

def update_plot(frames, decisions):
    global plotdata, ax, x_linspace

    # Roll signals and predictions vectors and insert new value at the end
    plotdata = np.roll(plotdata, -len(frames))
    plotdata[-len(frames):] = frames.reshape(-1, 1)

    # Clean the plot
    ax.clear()

    # Plot each segment with the corresponding color
    for i in range(len(decisions)):
        color = get_color(decisions[i])
        ax.plot(x_linspace[PLOT_CHUNK_SIZE * i:PLOT_CHUNK_SIZE * (i + 1)],
                plotdata[PLOT_CHUNK_SIZE * i:PLOT_CHUNK_SIZE * (i + 1)], color=color)

    ax.set_facecolor(facecolor)
    ax.set_ylim(ylim)
    plt.draw()
    plt.pause(0.01)


# Main function

if __name__ == "__main__":

    # Initialize microphone
    audio = SharedAudioResource()

    # Main loop
    decision_history = []
    noise = 0
    calculated = False

    while running:

        # Collect samples
        buffer = audio.read()
        buffer = abs(buffer)

        if calculated:
            buffer = buffer - noise

        if buffer is None:
            continue

        # Get the decision from volume recognition
        volume_decision = VR.volume_update(buffer)

        # Add the decision to the history
        decision_history.append(volume_decision)
        print(decision_history)
        #manually corrected
        '''for i in range(len(decision_history) -1):
            now = decision_history[i]
            prev = decision_history[i-1]
            next = decision_history[i+1]
            if prev == next:
                if now != prev:
                    decision_history[i] = prev
          '''

        # If history is longer than the number of segments we can display, remove the oldest decision
        if len(decision_history) > int(PLOT_TIME_HISTORY / REFRESH_TIME):
            decision_history.pop(0)
            if not calculated:
                calculated = True
                noise = np.mean(buffer)
                print("Done")


        # Update plot
        update_plot(buffer, decision_history)

    # Close audio
    audio.close()
