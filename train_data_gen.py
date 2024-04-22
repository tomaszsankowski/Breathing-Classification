import csv
import datetime
import pygame
import pygame.freetype
import time
import pyaudio
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import threading
import wave

CHUNK = 10000
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100


class SharedAudioResource:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

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

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.freetype.SysFont(None, FONT_SIZE)
    clock = pygame.time.Clock()
    button = Button(BUTTON_POS)

    start_time = None
    w_pressed = False
    p_pressed = False
    running = True
    frames = []
    data_csv = []
    wf = None
    csvname = ""
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
                        csvname = filename
                        csvname += '.csv'
                        filename += '.wav'
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
                        with open(csvname, 'w', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow(['Time', 'Breath'])
                            writer.writerows(data_csv)
                        recording = False
                elif event.key == pygame.K_w:
                    w_pressed = True
                elif event.key == pygame.K_p:
                    p_pressed = True
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_w:
                    w_pressed = False
                elif event.key == pygame.K_p:
                    p_pressed = False

        button.draw(screen)

        if button.recording:
            elapsed_time = time.time() - start_time
            draw_text(f"Recording: {elapsed_time:.2f}s", TIMER_POS, font, screen)
            data = audio.read(CHUNK)
            if recording:
                frames.append(data)
                value = 0
                if w_pressed:
                    value = 1
                elif p_pressed:
                    value = -1
                data_csv.append([elapsed_time, value])
        else:
            draw_text("Press SPACE to start recording", TIMER_POS, font, screen)

        if w_pressed:
            draw_text("Oddech", TEXT_POS, font, screen)
        elif p_pressed:
            draw_text("Wydech", TEXT_POS, font, screen)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def plot_audio(audio1):
    def animate(i):
        frames = audio1.read(CHUNK)
        data = np.frombuffer(frames, dtype=np.int16)
        left_channel = data[::2]  # even index: left channel
        right_channel = data[1::2]  # odd index: right channel
        line1.set_ydata(left_channel)
        line2.set_ydata(right_channel)
        return line1, line2,

    fig, axs = plt.subplots(2)
    x = np.arange(0, 2 * CHUNK, 2)
    line1, = axs[0].plot(x, np.random.rand(CHUNK))
    line2, = axs[1].plot(x, np.random.rand(CHUNK))

    axs[0].set_ylim(-1500, 1500)
    axs[0].set_xlim(0, CHUNK / 2)
    axs[1].set_ylim(-1500, 1500)
    axs[1].set_xlim(0, CHUNK / 2)
    ani = animation.FuncAnimation(fig, animate, frames=100, blit=True)
    plt.show()


if __name__ == "__main__":
    audio = SharedAudioResource()
    pygame_thread_instance = threading.Thread(target=pygame_thread, args=(audio,))
    pygame_thread_instance.start()
    plot_audio(audio)
    pygame_thread_instance.join()
    audio.close()
