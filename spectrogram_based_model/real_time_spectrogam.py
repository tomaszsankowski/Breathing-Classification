import numpy as np
from tensorflow.keras.models import load_model
import pyaudio
import queue
import pygame
import pygame_gui
import time
import threading
from matplotlib import pyplot as plt
from scipy.signal import stft

# Load the model
model = load_model('model_4096_05_small_.keras')

REFRESH_TIME = 0.5
N_FOURIER = 4096

AUDIO_CHUNK = 1024
PLOT_CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
DEVICE_INDEX = 4


class SharedAudioResource:
    buffer = None
    pred_aud_buffer = queue.Queue()

    def __init__(self):
        self.p = pyaudio.PyAudio()
        for i in range(self.p.get_device_count()):
            print(self.p.get_device_info_by_index(i))
        self.stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                                  frames_per_buffer=AUDIO_CHUNK, input_device_index=DEVICE_INDEX)
        self.read(AUDIO_CHUNK)

    def read(self, size):
        self.buffer = self.stream.read(size, exception_on_overflow=False)

        # Convert stereo to mono
        data = np.frombuffer(self.buffer, dtype=np.int16)
        mono_data = (data[::2] + data[1::2]) / 2

        self.buffer = mono_data.astype(np.int16)

        return self.buffer

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


def draw_text(text, pos, font, screen):
    text_surface, _ = font.render(text, (255, 255, 255))
    screen.blit(text_surface, (pos[0] - text_surface.get_width() // 2, pos[1] - text_surface.get_height() // 2))


# Function to create a spectrogram from audio datadef create_spectrogram(frames):
def create_spectrogram(frames):
    furier_hop = np.floor(RATE * REFRESH_TIME / 224)
    noverlap = N_FOURIER - furier_hop

    # Perform FFT
    stft_data = stft(frames, RATE, nperseg=N_FOURIER, noverlap=noverlap, scaling='spectrum')[2]

    spectrogram = stft_data[:224, :224]

    return np.abs(spectrogram)


# Function to classify given spectrogram
def classify_realtime_audio(spectrogram):

    spectrogram = np.expand_dims(spectrogram, axis=-1)
    spectrogram = np.expand_dims(spectrogram, axis=0)

    prediction = model.predict(spectrogram)

    prediction = model.predict(spectrogram)

    # Set numpy print options
    np.set_printoptions(suppress=True)

    print(f'Predicted class: {prediction}')

    return np.argmax(prediction)


def pygame_thread(audio):
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

    atten_lim_db_slider = pygame_gui.elements.UIHorizontalSlider(
        relative_rect=pygame.Rect((WIDTH // 2 - 100, HEIGHT // 2 + 200), (200, 20)),
        start_value=10.0,
        value_range=(0.0, 70.0),
        manager=manager
    )
    atten_lim_db = 10

    running = True

    while running:
        time_delta = clock.tick(60) / 1000.0
        start_time = time.time()
        buffer = audio.read(RATE // 2)

        # Model predicion
        spectrogram = create_spectrogram(buffer)
        print(spectrogram.shape)
        prediction = classify_realtime_audio(spectrogram)

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

        print(time.time() - start_time)
        draw_text(f"Noise reduction: {atten_lim_db} ", NOISE_POS, font, screen)

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    print("Exiting")
                    running = False

            if event.type == pygame_gui.UI_HORIZONTAL_SLIDER_MOVED:
                if event.ui_element == atten_lim_db_slider:
                    atten_lim_db = event.value

            manager.process_events(event)
        manager.update(time_delta)
        manager.draw_ui(screen)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    audio = SharedAudioResource()
    pygame_thread_instance = threading.Thread(target=pygame_thread, args=(audio,))
    pygame_thread_instance.start()
    plt.show()
    pygame_thread_instance.join()
    audio.close()
