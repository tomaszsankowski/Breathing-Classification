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
RATE = 44100
##################################
# self.stream = self.p.open(..., input_device_index=INDEX_OF_MICROPHONE)

WAV_EXHALE_PATH = ""
WAV_INHALE_PATH = ""
WAV_SILENCE_PATH = ""


def initializePaths():
    global WAV_EXHALE_PATH
    global WAV_INHALE_PATH
    global WAV_SILENCE_PATH

    WAV_EXHALE_PATH = 'data/exhale/'
    WAV_INHALE_PATH = 'data/inhale/'
    WAV_SILENCE_PATH = 'data/silence/'
    os.makedirs(os.path.dirname(WAV_EXHALE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(WAV_INHALE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(WAV_SILENCE_PATH), exist_ok=True)


class SharedAudioResource:
    buffer = None

    def __init__(self):
        self.p = pyaudio.PyAudio()
        for i in range(self.p.get_device_count()):
            print(self.p.get_device_info_by_index(i))
        self.stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                                  frames_per_buffer=AUDIO_CHUNK)
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


def pygame_thread(audio):
    pygame.init()
    initializePaths()
    WIDTH, HEIGHT = 1366, 768
    FONT_SIZE = 24
    TIMER_POS = (WIDTH // 2, HEIGHT // 2)
    TEXT_POS = (WIDTH // 2, HEIGHT // 2 - 200)
    PRESS_POS = (WIDTH // 2, HEIGHT // 2 - 50)

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.freetype.SysFont(None, FONT_SIZE)
    clock = pygame.time.Clock()
    recording = False

    def start_recording():
        nonlocal start_time, recording, frames, wf, w_pressed, r_pressed, e_pressed
        start_time = time.time()
        now = datetime.datetime.now()
        filename = now.strftime('%Y-%m-%d_%H-%M-%S')

        if w_pressed:
            wav_path = WAV_INHALE_PATH
        elif e_pressed:
            wav_path = WAV_EXHALE_PATH
        else:
            wav_path = WAV_SILENCE_PATH

        filename = wav_path + filename + '.wav'
        frames = []
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        recording = True

    def save_audio():
        nonlocal wf, frames
        if wf is not None:
            wf.writeframes(b''.join(frames))
            wf.close()

    start_time = None
    w_pressed = False
    e_pressed = False
    r_pressed = False
    running = True
    frames = []
    wf = None
    while running:
        screen.fill((0, 0, 0))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    if recording:
                        save_audio()
                        recording = False
                        w_pressed = False
                        r_pressed = False
                        e_pressed = False
                elif event.key == pygame.K_w:
                    if not w_pressed:
                        if recording:
                            save_audio()
                        w_pressed = True
                        r_pressed = False
                        e_pressed = False
                        start_recording()
                elif event.key == pygame.K_e:
                    if not e_pressed:
                        if recording:
                            save_audio()
                        w_pressed = False
                        r_pressed = False
                        e_pressed = True
                        start_recording()
                elif event.key == pygame.K_r:
                    if not r_pressed:
                        if recording:
                            save_audio()
                        w_pressed = False
                        r_pressed = True
                        e_pressed = False
                        start_recording()

        if recording:
            elapsed_time = time.time() - start_time
            draw_text("Press s to stop recording", PRESS_POS, font, screen)
            draw_text(f"Recording: {elapsed_time:.2f}s", TIMER_POS, font, screen)
            data = audio.read(AUDIO_CHUNK)
            frames.append(data)
        else:
            draw_text("W:   record inhale    |   E:   record exhale", PRESS_POS, font, screen)
            draw_text("R:   record silence", TIMER_POS, font, screen)

        if w_pressed:
            draw_text("Inhale chosen", TEXT_POS, font, screen)
        elif e_pressed:
            draw_text("Exhale chosen", TEXT_POS, font, screen)
        else:
            draw_text("Silence chosen", TEXT_POS, font, screen)

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
