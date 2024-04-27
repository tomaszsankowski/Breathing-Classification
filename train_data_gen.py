import datetime
import os
import pygame
import pygame.freetype
import time
import pyaudio
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import threading
import wave

# ###########################################################################################
# Plot is active at recording
# ###########################################################################################
# If there's an issue with the microphone, find the index of the microphone you want to use in the console,
# along with its sampleRate. Then, change the variable RATE below and add the parameter
# input_device_index=INDEX_OF_MICROPHONE
# to
# self.stream = self.p.open(..., input_device_index=INDEX_OF_MICROPHONE)
# ###########################################################################################
AUDIO_CHUNK = 1024
PLOT_CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 48000
##################################
# self.stream = self.p.open(..., input_device_index=INDEX_OF_MICROPHONE)

TEST = False
WAV_EXHALE_PATH = ""
WAV_INHALE_PATH = ""


def changePaths(test: bool):
    global WAV_EXHALE_PATH
    global WAV_INHALE_PATH

    if test:
        WAV_EXHALE_PATH = 'data/test/exhale/'
        WAV_INHALE_PATH = 'data/test/inhale/'
        os.makedirs(os.path.dirname(WAV_EXHALE_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(WAV_INHALE_PATH), exist_ok=True)
    else:
        WAV_EXHALE_PATH = 'data/train/exhale/'
        WAV_INHALE_PATH = 'data/train/inhale/'
        os.makedirs(os.path.dirname(WAV_EXHALE_PATH), exist_ok=True)
        os.makedirs(os.path.dirname(WAV_INHALE_PATH), exist_ok=True)


class SharedAudioResource:
    buffer = None

    def __init__(self):
        self.p = pyaudio.PyAudio()
        for i in range(self.p.get_device_count()):
            print(self.p.get_device_info_by_index(i))
        self.stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                                  frames_per_buffer=AUDIO_CHUNK, input_device_index=6)
        self.read(AUDIO_CHUNK)

    def read(self, size):
        self.buffer = self.stream.read(size, exception_on_overflow=False)
        return self.buffer

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


class Button:
    def __init__(self, pos):
        self.pos = pos
        self.recording = False

    def draw(self, screen):
        color = (255, 0, 0) if self.recording else (0, 255, 0)
        pygame.draw.rect(screen, color, (*self.pos, 100, 50))


def draw_text(text, pos, font, screen):
    text_surface, _ = font.render(text, (255, 255, 255))
    screen.blit(text_surface, (pos[0] - text_surface.get_width() // 2, pos[1] - text_surface.get_height() // 2))


def pygame_thread(audio):
    global TEST
    pygame.init()
    changePaths(TEST)
    WIDTH, HEIGHT = 1366, 768
    FONT_SIZE = 24
    TIMER_POS = (WIDTH // 2, HEIGHT // 2)
    BUTTON_POS = (WIDTH // 2, HEIGHT // 2 + 50)
    TEXT_POS = (WIDTH // 2, HEIGHT // 2 - 200)
    PRESS_POS = (WIDTH // 2, HEIGHT // 2 - 50)
    TEST_POS = (WIDTH // 2, HEIGHT // 2 - 300)

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.freetype.SysFont(None, FONT_SIZE)
    clock = pygame.time.Clock()
    button = Button(BUTTON_POS)

    start_time = None
    w_pressed = True
    p_pressed = False
    running = True
    frames = []
    wf = None
    while running:
        screen.fill((0, 0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    button.recording = not button.recording
                    if button.recording:
                        start_time = time.time()
                        now = datetime.datetime.now()
                        filename = now.strftime('%Y-%m-%d_%H-%M-%S')
                        wav_path = WAV_EXHALE_PATH if p_pressed else WAV_INHALE_PATH
                        filename = wav_path + filename + '.wav'
                        frames = []
                        wf = wave.open(filename, 'wb')
                        wf.setnchannels(CHANNELS)
                        wf.setsampwidth(audio.p.get_sample_size(FORMAT))
                        wf.setframerate(RATE)
                        recording = True
                    else:
                        if wf is not None:
                            wf.writeframes(b''.join(frames))
                            wf.close()
                        recording = False
                elif event.key == pygame.K_w:
                    p_pressed = False
                    w_pressed = True
                elif event.key == pygame.K_e:
                    w_pressed = False
                    p_pressed = True
                elif event.key == pygame.K_t:
                    TEST = not (TEST)
                    changePaths(TEST)

        button.draw(screen)

        if button.recording:
            elapsed_time = time.time() - start_time
            draw_text(f"Recording: {elapsed_time:.2f}s", TIMER_POS, font, screen)
            data = audio.read(AUDIO_CHUNK)
            if recording:
                frames.append(data)
        else:
            draw_text("Press:  W to inhale | E to exhale | T - to change TRAIN/TEST", PRESS_POS, font, screen)
            draw_text("Press Q to start recording", TIMER_POS, font, screen)

        if w_pressed:
            draw_text("Inhale", TEXT_POS, font, screen)
        elif p_pressed:
            draw_text("Exhale", TEXT_POS, font, screen)

        if TEST:
            draw_text('Recording TEST data', TEST_POS, font, screen)
        else:
            draw_text('Recording TRAIN data', TEST_POS, font, screen)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def plot_audio(audio1):
    def animate(i):
        frames = audio1.buffer
        data = np.frombuffer(frames, dtype=np.int16)
        left_channel = data[::2]  # even index: left channel
        right_channel = data[1::2]  # odd index: right channel
        line1.set_ydata(left_channel)
        line2.set_ydata(right_channel)
        return line1, line2,

    fig, axs = plt.subplots(2)
    x = np.arange(0, 2 * PLOT_CHUNK, 2)
    line1, = axs[0].plot(x, np.random.rand(PLOT_CHUNK))
    line2, = axs[1].plot(x, np.random.rand(PLOT_CHUNK))

    axs[0].set_ylim(-1500, 1500)
    axs[0].set_xlim(0, PLOT_CHUNK / 2)
    axs[1].set_ylim(-1500, 1500)
    axs[1].set_xlim(0, PLOT_CHUNK / 2)
    ani = animation.FuncAnimation(fig, animate, frames=100, blit=True)
    plt.show()


if __name__ == "__main__":
    audio = SharedAudioResource()
    pygame_thread_instance = threading.Thread(target=pygame_thread, args=(audio,))
    pygame_thread_instance.start()
    plot_audio(audio)
    pygame_thread_instance.join()
    audio.close()
