import numpy as np
from tensorflow.keras.models import load_model
import librosa
import matplotlib.pyplot as plt
import pyaudio
import queue
import pygame
import pygame_gui
import time
import os
import threading
import matplotlib.animation as animation
from PIL import Image
from tensorflow.keras.preprocessing import image
import datetime
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import save_img

# Load the model
model = load_model('spectrogram_efficientnet_model.keras')

AUDIO_CHUNK = 1024
PLOT_CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100


class SharedAudioResource:
    buffer = None
    pred_aud_buffer = queue.Queue()

    def __init__(self):
        self.p = pyaudio.PyAudio()
        for i in range(self.p.get_device_count()):
            print(self.p.get_device_info_by_index(i))
        self.stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                                  frames_per_buffer=AUDIO_CHUNK, input_device_index=4)
        self.read(AUDIO_CHUNK)

    def read(self, size):
        self.buffer = self.stream.read(size, exception_on_overflow=False)
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
    # Compute the short-time Fourier transform
    frames = np.frombuffer(frames, dtype=np.int16)
    frames = frames.astype(float)
    D = librosa.stft(frames)
    # Convert an amplitude spectrogram to dB-scaled spectrogram.
    spectrogram_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Save the spectrogram as an image
    plt.imsave('temp.png', spectrogram_db)

    # Open the image file and resize it
    img = Image.open('temp.png').convert('RGB')

    return image.img_to_array(img.resize((224, 224)))


# Function to classify given spectrogram
def classify_realtime_audio(spectrogram):

    spectrogram = np.expand_dims(spectrogram, axis=0)

    filename = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Zapisz spektrogram do pliku przed przewidywaniem
    save_img(f'temp/{filename}.png', spectrogram[0])

    prediction = model.predict(spectrogram)
    print(f'Predicted class: {prediction}')

    predicted_class = np.argmax(prediction)

    # Print the predicted class
    return predicted_class


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


plotdata = np.zeros((RATE * 2, 1))
predictions = np.zeros((RATE * 2, 1))
q = queue.Queue()
ymin = -1500
ymax = 1500
fig, ax = plt.subplots(figsize=(8, 4))
lines, = ax.plot(plotdata, color=(0, 1, 0.29))
ax.set_facecolor((0, 0, 0))
ax.set_ylim(ymin, ymax)
xes = [i for i in range(RATE * 2)]

fill_red = ax.fill_between(xes, ymin, ymax,
                           where=([True if predictions[i][0] == 0 else False for i in range(len(predictions))]),
                           color='red', alpha=0.3)
fill_green = ax.fill_between(xes, ymin, ymax,
                             where=([True if predictions[i][0] == 1 else False for i in range(len(predictions))]),
                             color='green', alpha=0.3)

fill_yellow = ax.fill_between(xes, ymin, ymax,
                              where=(
                                  [True if predictions[i][0] == 2 else False for i in range(len(predictions))]),
                              color='blue', alpha=0.3)


def update_plot(frame):
    global plotdata, predictions, fill_red, fill_green, fill_yellow, xes

    if q.empty():
        data = audio.pred_aud_buffer.get(block=True)
        chunks = np.array_split(data[1], 2)
        for chunk in chunks:
            q.put((data[0], chunk))

    queue_data = q.get()
    frames = np.frombuffer(queue_data[1], dtype=np.int16)
    frames = frames[::2]
    shift = len(frames)

    plotdata = np.roll(plotdata, -shift, axis=0)
    plotdata[-shift:, 0] = frames

    prediction = queue_data[0]
    predictions = np.roll(predictions, -shift, axis=0)
    pred_arr = [prediction for _ in range(shift)]
    predictions[-shift:, 0] = pred_arr

    lines.set_ydata(plotdata)

    fill_red.remove()
    fill_green.remove()
    fill_yellow.remove()

    fill_red = ax.fill_between(xes, ymin, ymax,
                               where=([True if predictions[i][0] == 0 else False for i in range(len(predictions))]),
                               color='red', alpha=0.3)
    fill_green = ax.fill_between(xes, ymin, ymax,
                                 where=([True if predictions[i][0] == 1 else False for i in range(len(predictions))]),
                                 color='green', alpha=0.3)
    fill_yellow = ax.fill_between(xes, ymin, ymax,
                                  where=([True if predictions[i][0] == 2 else False for i in range(len(predictions))]),
                                  color='blue', alpha=0.3)

    return lines, fill_red, fill_green, fill_yellow


if __name__ == "__main__":
    audio = SharedAudioResource()
    pygame_thread_instance = threading.Thread(target=pygame_thread, args=(audio,))
    pygame_thread_instance.start()
    ani = animation.FuncAnimation(fig, update_plot, frames=100, blit=True)
    plt.show()
    pygame_thread_instance.join()
    audio.close()
