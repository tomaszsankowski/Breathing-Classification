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

# To remove delay from audio plot, change AUDIO_CHUNK and PLOT_CHUNK to e.g. 5000
# then pygame and recorded audio will be delayed

# key SPACE - start/stop recording
# key W - inhale
# key P - exhale

AUDIO_CHUNK = 1024
PLOT_CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
WAV_EXHALE_PATH = 'data/exhale/'
WAV_INHALE_PATH = 'data/inhale/'


class SharedAudioResource:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                                  frames_per_buffer=AUDIO_CHUNK)

    def read(self, size):
        return self.stream.read(size)

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
    pygame.init()

    WIDTH, HEIGHT = 1366, 768
    FONT_SIZE = 24
    TIMER_POS = (WIDTH // 2, HEIGHT // 2)
    BUTTON_POS = (WIDTH // 2, HEIGHT // 2 + 50)
    TEXT_POS = (WIDTH // 2, HEIGHT // 2 - 200)
    PRESS_POS = (WIDTH // 2, HEIGHT // 2 - 50)

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
                if event.key == pygame.K_SPACE:
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
                elif event.key == pygame.K_p:
                    w_pressed = False
                    p_pressed = True

        button.draw(screen)

        if button.recording:
            elapsed_time = time.time() - start_time
            draw_text(f"Recording: {elapsed_time:.2f}s", TIMER_POS, font, screen)
            data = audio.read(AUDIO_CHUNK)
            if recording:
                frames.append(data)
        else:
            draw_text("Press W to record inhale | Press P to record exhale", PRESS_POS, font, screen)
            draw_text("Press SPACE to start recording", TIMER_POS, font, screen)

        if w_pressed:
            draw_text("Inhale", TEXT_POS, font, screen)
        elif p_pressed:
            draw_text("Exhale", TEXT_POS, font, screen)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def plot_audio(audio1):
    def animate(i):
        frames = audio1.read(PLOT_CHUNK)
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
    os.makedirs(os.path.dirname(WAV_EXHALE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(WAV_INHALE_PATH), exist_ok=True)
    audio = SharedAudioResource()
    pygame_thread_instance = threading.Thread(target=pygame_thread, args=(audio,))
    pygame_thread_instance.start()
    #plot_audio(audio)
    pygame_thread_instance.join()
    audio.close()
