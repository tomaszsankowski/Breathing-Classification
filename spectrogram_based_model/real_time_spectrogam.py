import numpy as np
from tensorflow.keras.models import load_model
import pygame
import pygame_gui
from scipy.signal import stft
import pyaudio
import matplotlib.pyplot as plt
import time

# Constants

REFRESH_TIME = 0.25
N_FOURIER = 2048

PREVIOUS_CLASS_BONUS = 0.2

CHANNELS = 1
RATE = 44100
DEVICE_INDEX = 4

running = True

# Load the model

model = load_model(f'models_mobilenet/mobile_net_model_{N_FOURIER}_{REFRESH_TIME}_small.keras')


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


# Function to draw text on screen

def draw_text(v_text, v_pos, v_font, v_screen):
    text_surface, _ = v_font.render(v_text, (255, 255, 255))
    v_screen.blit(text_surface, (v_pos[0] - text_surface.get_width() // 2, v_pos[1] - text_surface.get_height() // 2))


# Function to create a spectrogram from audio data:

def create_spectrogram(frames):
    furier_hop = np.floor(RATE * REFRESH_TIME / 224)
    noverlap = N_FOURIER - furier_hop

    # Perform FFT
    stft_data = stft(frames, RATE, nperseg=N_FOURIER, noverlap=noverlap, scaling='spectrum')[2]

    spectrogram_in = stft_data[:224, :224]

    return np.abs(spectrogram_in)


# Function to classify given spectrogram

def classify_realtime_audio(spectrogram_in):
    global last_prediction

    spectrogram_in = np.expand_dims(spectrogram_in, axis=-1)
    spectrogram_in = np.expand_dims(spectrogram_in, axis=0)

    predictionon = model.predict(spectrogram_in, verbose=0)

    predictionon[0][last_prediction] += PREVIOUS_CLASS_BONUS
    last_prediction = np.argmax(predictionon)

    print('Predicted class: ', np.array2string(np.round(predictionon, 4), suppress_small=True))

    return np.argmax(predictionon)


# Plot variables

PLOT_TIME_HISTORY = 5
PLOT_CHUNK_SIZE = int(RATE*REFRESH_TIME)

plotdata = np.zeros((RATE*PLOT_TIME_HISTORY, 1))
x_linspace = np.arange(0, RATE*PLOT_TIME_HISTORY, 1)
predictions = np.zeros((int(PLOT_TIME_HISTORY/REFRESH_TIME), 1))

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(plotdata, color='white')


def on_key(event):
    global running
    if event.key == ' ':
        plt.close()
        running = False


fig.canvas.manager.set_window_title('Realtime Breath Detector')  # Title
fig.suptitle('Press [SPACE] to stop')  # Subtitle
fig.canvas.mpl_connect('key_press_event', on_key)

ylim = (-1500, 1500)
facecolor = (0, 0, 0)

ax.set_facecolor(facecolor)
ax.set_ylim(ylim)


# Plot update function

def update_plot(frames, prediction):
    global plotdata, predictions, ax

    plotdata = np.roll(plotdata, -len(frames))
    plotdata[-len(frames):] = frames.reshape(-1, 1)

    predictions = np.roll(predictions, -1)
    predictions[-1] = prediction

    ax.clear()  # Usuń poprzednie linie

    for i in range(0, len(predictions)):
        if predictions[i] == 0:  # Inhale
            color = 'red'
        elif predictions[i] == 1:  # Exhale
            color = 'green'
        else:  # Silence
            color = 'blue'
        ax.plot(x_linspace[PLOT_CHUNK_SIZE*i:PLOT_CHUNK_SIZE*(i+1)], plotdata[PLOT_CHUNK_SIZE*i:PLOT_CHUNK_SIZE*(i+1)], color=color)

    ax.set_facecolor(facecolor)
    ax.set_ylim(ylim)

    plt.draw()
    plt.pause(0.01)


# Main function
last_prediction = 2

if __name__ == "__main__":
    # Initialize microphone
    audio = SharedAudioResource()

    # Initialize pygame
    pygame.init()
    WIDTH, HEIGHT = 1366, 768
    manager = pygame_gui.UIManager((WIDTH, HEIGHT))
    FONT_SIZE = 24
    TEXT_POS = (WIDTH // 2, HEIGHT // 2 - 200)
    TEST_POS = (WIDTH // 2, HEIGHT // 2 - 300)
    NOISE_POS = (WIDTH // 2, HEIGHT // 2 + 100)

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.freetype.SysFont(None, FONT_SIZE)
    clock = pygame.time.Clock()

    # Main loop

    while running:
        # Collect samples

        time_delta = clock.tick(60) / 1000.0
        start_time = time.time()
        buffer = audio.read()
        if buffer is None:
            continue
        print(time.time() - start_time)

        # Create spectrogram and make prediction

        spectrogram = create_spectrogram(buffer)

        prediction = classify_realtime_audio(spectrogram)

        # Handle predicted class

        if prediction == 0:
            screen.fill(color="red")
            draw_text(f"Inhale", TEXT_POS, font, screen)
        elif prediction == 1:
            screen.fill(color="green")
            draw_text(f"Exhale", TEXT_POS, font, screen)
        else:
            screen.fill(color="blue")
            draw_text(f"Silence", TEXT_POS, font, screen)

        draw_text("Press SPACE to stop", TEST_POS, font, screen)

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    print("Exiting...")
                    running = False
            manager.process_events(event)

        # Update and draw UI

        update_plot(buffer, prediction)  # Updates signal plot
        manager.update(time_delta)
        manager.draw_ui(screen)
        pygame.display.flip()
        clock.tick(60)

    # Close audio and pygame

    pygame.quit()
    audio.close()
