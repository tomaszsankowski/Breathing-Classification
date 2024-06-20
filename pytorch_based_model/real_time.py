import queue
import threading
import time
import wave
from tkinter import *

import librosa
import numpy as np
import pandas as pd
import pyaudio
import pygame
import pygame.freetype
import pygame_gui
import torch
from df.enhance import init_df
import matplotlib as plt
from torch.utils.data import Dataset

from DeepFilterNet.DeepFilterNet.df import enhance
from DeepFilterNet.DeepFilterNet.df.io import load_audio, save_audio
from model import AudioClassifier, AudioDatasetRealtime as AudioDataset

# ###########################################################################################
# If there's an issue with the microphone, find the index of the microphone you want to use in the console,
# along with its sampleRate. Then, change the variable RATE below and add the parameter
# input_device_index=INDEX_OF_MICROPHONE
# to
# self.stream = self.p.open(..., input_device_index=INDEX_OF_MICROPHONE)
# ###########################################################################################
AUDIO_CHUNK = 1024
PLOT_CHUNK = 1024
REFRESH_TIME = 0.25
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 48000
CHUNK_SIZE = int(RATE * REFRESH_TIME)
CLASSIFIER_MODEL_PATH = 'audio_rnn_classifier.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model, df_state, _ = init_df()

bonus = 1.15
noise_reduction = 10
noise_reduction_active = False


class RealTimeAudioClassifier:
    def __init__(self, model_path):
        self.model = AudioClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model = self.model.to(device)
        self.model.eval()
        self.last_prediction = None

    def predict(self, audio_path):
        y, sr = librosa.load(audio_path, sr=RATE, mono=True)

        frames = []
        for i in range(0, len(y), CHUNK_SIZE):
            frame = y[i:i + CHUNK_SIZE]
            if len(frame) == CHUNK_SIZE:  # Ignorujemy ostatnią ramkę, jeśli jest krótsza
                mfcc = librosa.feature.mfcc(y=frame, sr=sr)
                frames.append(mfcc)

        frames = AudioDataset(frames)
        frames = torch.utils.data.DataLoader(frames, batch_size=1, shuffle=False)
        for frames in frames:
            frames = frames.to(device)

        outputs = self.model(frames)
        global bonus
        if self.last_prediction is not None:
            print("Before bonus", outputs)
            outputs[0][self.last_prediction] *= bonus
            print("After bonus", outputs)
        _, predicted = torch.max(outputs, 1)
        self.last_prediction = predicted.cpu().numpy()[0]
        return predicted.cpu().numpy()


class SharedAudioResource:
    buffer = None
    pred_aud_buffer = queue.Queue()

    def __init__(self):
        self.p = pyaudio.PyAudio()
        for i in range(self.p.get_device_count()):
            print(self.p.get_device_info_by_index(i))
        self.stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                                  frames_per_buffer=CHUNK_SIZE, input_device_index=1)
        self.read()

    def read(self):
        self.buffer = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
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
    classifier = RealTimeAudioClassifier(CLASSIFIER_MODEL_PATH)
    global bonus, noise_reduction, noise_reduction_active
    while running:
        time_delta = clock.tick(60) / 1000.0
        start_time = time.time()
        buffer = []
        buffer.append(audio.read())

        start_time = time.time()
        wf = wave.open("../temp/temp.wav", 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(buffer))
        wf.close()
        if noise_reduction_active:
            audio1, _ = load_audio("../temp/temp.wav", sr=df_state.sr())
            enhanced = enhance(model, df_state, audio1, atten_lim_db=noise_reduction)
            save_audio("../temp/temp.wav", enhanced, df_state.sr())
        print(f"Bonus: {bonus}, Noise reduction: {noise_reduction}")

        prediction = classifier.predict('../temp/temp.wav')
        classifier.last_prediction = prediction
        print(prediction)
        print("Time: ", time.time() - start_time)

        audio.pred_aud_buffer.put((prediction, buffer))
        if prediction == 0:
            screen.fill(color="green")
            draw_text(f"Exhale", TEXT_POS, font, screen)
        elif prediction == 1:
            screen.fill(color="red")
            draw_text(f"Inhale", TEXT_POS, font, screen)
        else:
            screen.fill(color="blue")
            draw_text(f"Silence", TEXT_POS, font, screen)
        draw_text("Press SPACE to stop", TEST_POS, font, screen)

        print(time.time() - start_time)
        draw_text(f"Noise reduction: {noise_reduction}, Active: {noise_reduction_active} ", NOISE_POS, font, screen)

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    print("Exiting")
                    running = False

            manager.process_events(event)
        manager.update(time_delta)
        manager.draw_ui(screen)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def tkinker_sliders():
    root = Tk()
    root.title("Sliders")
    root.geometry("600x400")
    root.resizable(False, False)

    def set_bonus(val):
        global bonus
        bonus = float(val)

    def set_noise_reduction(val):
        global noise_reduction
        noise_reduction = float(val)

    bonus_label = Label(root, text="Bonus")
    bonus_label.pack()

    bonus_slider = Scale(root, from_=0.1, to=5, resolution=0.1, orient=HORIZONTAL, command=set_bonus)
    bonus_slider.set(1.15)
    bonus_slider.pack()

    noise_reduction_label = Label(root, text="Noise reduction")
    noise_reduction_label.pack()

    noise_reduction_slider = Scale(root, from_=0, to=100, resolution=0.1, orient=HORIZONTAL,
                                   command=set_noise_reduction)
    noise_reduction_slider.set(10)
    noise_reduction_slider.pack()

    # add button to turn off noise reduction
    def toggle_noise_reduction():
        global noise_reduction_active
        noise_reduction_active = not noise_reduction_active

    noise_reduction_button = Button(root, text="Toggle noise reduction", command=toggle_noise_reduction)
    noise_reduction_button.pack()

    root.mainloop()

PLOT_TIME_HISTORY = 5
PLOT_CHUNK_SIZE = int(RATE * REFRESH_TIME)
plotdata = np.zeros((RATE * PLOT_TIME_HISTORY, 1))
x_linspace = np.arange(0, RATE * PLOT_TIME_HISTORY, 1)
predictions = np.zeros((int(PLOT_TIME_HISTORY / REFRESH_TIME), 1))
def audio_plotter(audio):
    print("Plotting audio")




    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(plotdata, color='white')

    # Key handler for plot window

    def on_key(event):
        global running
        if event.key == ' ':
            plt.close()
            running = False

    # Configuration of plot properties and other elements

    fig.canvas.manager.set_window_title('Realtime Breath Detector')  # Title
    fig.suptitle('Press [SPACE] to stop. Colours meaning: Red - Inhale, Green - Exhale, Blue - Silence')  # Instruction
    fig.canvas.mpl_connect('key_press_event', on_key)  # Key handler

    ylim = (-500, 500)
    facecolor = (0, 0, 0)

    ax.set_facecolor(facecolor)
    ax.set_ylim(ylim)

    # Moving avarage function

    def moving_average(data, window_size):
        data_flatten = data.flatten()
        ma = pd.Series(data_flatten).rolling(window=window_size).mean().to_numpy()
        ma[:window_size - 1] = data_flatten[:window_size - 1]  # Leave the first window_size-1 elements unchanged
        return ma.reshape(-1, 1)

    # Plot update function

    def update_plot(frames, prediction):
        global plotdata, predictions, ax

        # Roll signals and predictions vectors and insert new value at the end
        frames = np.frombuffer(frames, dtype=np.int16)
        plotdata = np.roll(plotdata, -len(frames))
        plotdata[-len(frames):] = frames.reshape(-1, 1)

        predictions = np.roll(predictions, -1)
        predictions[-1] = prediction

        # Moving avarage on plotdata (uncomment if needed)

        plotdata = moving_average(plotdata, 50)

        # Clean the plot and plot the new data

        ax.clear()

        for i in range(0, len(predictions)):
            if predictions[i] == 0:  # Inhale
                color = 'red'
            elif predictions[i] == 1:  # Exhale
                color = 'green'
            else:  # Silence
                color = 'blue'
            ax.plot(x_linspace[PLOT_CHUNK_SIZE * i:PLOT_CHUNK_SIZE * (i + 1)],
                    plotdata[PLOT_CHUNK_SIZE * i:PLOT_CHUNK_SIZE * (i + 1)], color=color)

        # Set plot properties and show it

        ax.set_facecolor(facecolor)
        ax.set_ylim(ylim)

        plt.draw()
        plt.pause(0.01)

    print("Start plotting")
    while True:

        prediction, frames = 0, audio.read()
        print("Updating plot")
        update_plot(frames, prediction)

if __name__ == "__main__":
    audio = SharedAudioResource()

    pygame_thread_instance = threading.Thread(target=pygame_thread, args=(audio,))
    plot_thread_instance = threading.Thread(target=audio_plotter, args=(audio,))

    pygame_thread_instance.start()
    plot_thread_instance.start()
    tkinker_sliders()
    pygame_thread_instance.join()
    plot_thread_instance.join()
    audio.close()
