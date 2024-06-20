import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import pandas as pd
import volume_recognition

REFRESH_TIME = 0.15
CHANNELS = 1
RATE = 44100
DEVICE_INDEX = 4
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
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(plotdata, color='white')
facecolor = (0, 0, 0)

# Configuration of plot properties and other elements
fig.canvas.manager.set_window_title('Volume recognition')  # Title
fig.suptitle('Red - Inhale, Green - Exhale, Blue - Silence\n press \'space\' to turn on impostor mode \n black background - not ready,  white background - ready,  yellow background - EXP mode')  # Instruction
ylim = (0, 1000)

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


experimental = False
calculated = False

def on_key(event):
    global experimental, facecolor
    if calculated and event.key == ' ':
        experimental = not experimental
        if facecolor == (1, 1, 1):
            facecolor = (0.98, 0.98, 0.59)
        else:
            facecolor = (1,1,1)

fig.canvas.mpl_connect('key_press_event', on_key)  # Key handler

# Main function

if __name__ == "__main__":

    # Initialize microphone
    audio = SharedAudioResource()
    # Main loop
    decision_history = []
    noise = 0

    while running:

        # Collect samples
        buffer = audio.read()
        buffer = abs(buffer)

        if calculated:
            buffer = buffer - (noise * 0.85)

        if buffer is None:
            continue

        # Add the decision to the history
        if not calculated:
            decision_history.append(0)
        else:
            volume_decision = VR.volume_update(buffer)
            decision_history.append(volume_decision)

        # manually corrected
        if calculated and experimental:
            l = len(decision_history) - 1
            left = decision_history[l - 2]
            middle = decision_history[l - 1]
            right = decision_history[l]
            if left == 0 and left == right:
                if middle != left:
                    decision_history[l - 1] = 0
                    VR.calc(0)
                    VR.calc(1)
                    print("\n# Impostor #\n")

        '''for j in range(len(decision_history) - 1):
            now = decision_history[j]
            prev = decision_history[j-1]
            nxt = decision_history[j+1]
            if prev == nxt and prev == 0:
                if now != prev:
                    decision_history[j] = prev
                    VR.calc(1)'''

        # If history is longer than the number of segments we can display, remove the oldest decision
        if len(decision_history) > int(PLOT_TIME_HISTORY / REFRESH_TIME):
            decision_history.pop(0)
            if not calculated:
                calculated = True
                noise = np.mean(buffer)
                facecolor = (1, 1, 1)
                print("Done")

        # Update plot
        update_plot(buffer, decision_history)

    # Close audio
    audio.close()
