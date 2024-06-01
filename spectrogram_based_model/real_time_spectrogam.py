import numpy as np
from tensorflow.keras.models import load_model
import pygame
import pygame_gui
import sounddevice as sd
from scipy.signal import stft

# Load the model
model = load_model('models_mobilenet/mobile_net_model_4096_0.5_small.keras')

PREVIOUS_CLASS_BONUS = 0.2

REFRESH_TIME = 0.25
N_FOURIER = 2048

CHANNELS = 1
RATE = 44100
DEVICE_INDEX = 4


class SharedAudioResource:
    def __init__(self):
        self.buffer_size = int(RATE * REFRESH_TIME)
        self.buffer = np.zeros(self.buffer_size, dtype=np.int16)
        self.stream = sd.InputStream(
            device=DEVICE_INDEX,
            samplerate=RATE,
            channels=CHANNELS,
            dtype=np.int16,
            callback=self._callback,
            blocksize=self.buffer_size
        )

    def _callback(self, indata, frames, time, status):
        if status:
            print('Error:', status)
        self.buffer = indata.flatten().copy()

    def read(self):
        return self.buffer

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()


def draw_text(v_text, v_pos, v_font, v_screen):
    text_surface, _ = v_font.render(v_text, (255, 255, 255))
    v_screen.blit(text_surface, (v_pos[0] - text_surface.get_width() // 2, v_pos[1] - text_surface.get_height() // 2))


# Function to create a spectrogram from audio datadef create_spectrogram(frames):
def create_spectrogram(frames):
    furier_hop = np.floor(RATE * REFRESH_TIME / 224)
    noverlap = N_FOURIER - furier_hop

    # Perform FFT
    stft_data = stft(frames, RATE, nperseg=N_FOURIER, noverlap=noverlap, scaling='spectrum')[2]

    spectrogram_in = stft_data[:224, :224]

    return np.abs(spectrogram_in)


# Function to classify given spectrogram
last_class = 2
def classify_realtime_audio(spectrogram_in):
    global last_class

    spectrogram_in = np.expand_dims(spectrogram_in, axis=-1)
    spectrogram_in = np.expand_dims(spectrogram_in, axis=0)

    predictionon = model.predict(spectrogram_in, verbose=0)

    predictionon[0][last_class] += PREVIOUS_CLASS_BONUS
    last_class = np.argmax(predictionon)

    print('Predicted class: ', np.array2string(np.round(predictionon, 4), suppress_small=True))

    return np.argmax(predictionon)


if __name__ == "__main__":
    audio = SharedAudioResource()
    audio.start()
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

    running = True

    while running:
        time_delta = clock.tick(60) / 1000.0
        buffer = audio.read()
        if buffer is None:
            continue

        # Model predicion
        spectrogram = create_spectrogram(buffer)

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

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    print("Exiting...")
                    running = False

            manager.process_events(event)
        manager.update(time_delta)
        manager.draw_ui(screen)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    audio.stop()
