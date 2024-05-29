import queue
import threading
import time
import wave
from tkinter import *

import librosa
import numpy as np
import pyaudio
import pygame
import pygame.freetype
import pygame_gui
import torch
import torch.nn as nn
import torch.nn.functional as F
from df.enhance import init_df
import pandas as pd
from DeepFilterNet.DeepFilterNet.df import enhance
from DeepFilterNet.DeepFilterNet.df.io import load_audio, save_audio


class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(p=0.1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.dropout2 = nn.Dropout(p=0.1)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=1)  # Nowa warstwa konwolucyjna
        self.dropout3 = nn.Dropout(p=0.1)

        self.fc1 = nn.Linear(256 * 233, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #print("1: ",x.shape)
        x = F.max_pool1d(x, 2)
        #print("2: ",x.shape)
        x = self.dropout1(x)

        x = F.relu(self.conv2(x))
        #print("3: ",x.shape)
        x = F.max_pool1d(x, 2)
        x = self.dropout2(x)

        x = F.relu(self.conv3(x))  # Nowa warstwa konwolucyjna
        #print("4: ",x.shape)
        x = F.max_pool1d(x, 2)
        #print("6: ",x.shape)
        x = self.dropout3(x)
        #print("7: ",x.shape)



        x = x.view(-1, 256 * 233)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


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
CHUNK_SIZE = int(RATE * 0.5) * 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model, df_state, _ = init_df()

bonus = 1.15
noise_reduction = 10
noise_reduction_active = False


class RealTimeAudioClassifier:
    def __init__(self, model_path):
        self.model = AudioClassifier()
        self.model.load_state_dict(torch.load(model_path))
        self.model = self.model.to(device)
        self.model.eval()
        self.last_prediction = None

    def predict(self, audio_path):
        y, sr = librosa.load(audio_path, sr=48000, mono=True)

        frame_length = 48000  # Długość ramki w próbkach
        frames = []
        for i in range(0, len(y), frame_length):
            frame = y[i:i + frame_length]
            if len(frame) == frame_length:  # Ignorujemy ostatnią ramkę, jeśli jest krótsza
                mfcc = librosa.feature.mfcc(y=frame, sr=sr)
                frames.append(mfcc)

        frames = np.array(frames)
        # frames to csv

        frames = frames.reshape(-1)
        frames = np.expand_dims(frames, axis=0)
        frames = torch.tensor(frames).float().to(device)
        outputs = self.model(frames)
        global bonus
        if self.last_prediction is not None:
            print("Before bonus", outputs)
            outputs[0][self.last_prediction] *= bonus
            print("After bonus", outputs)
        _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()


class SharedAudioResource:
    buffer = None
    pred_aud_buffer = queue.Queue()

    def __init__(self):
        self.p = pyaudio.PyAudio()
        for i in range(self.p.get_device_count()):
            print(self.p.get_device_info_by_index(i))
        self.stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                                  frames_per_buffer=CHUNK_SIZE)
        self.read(AUDIO_CHUNK)

    def read(self, size):
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
    classifier = RealTimeAudioClassifier('audio_classifier.pth')
    global bonus, noise_reduction, noise_reduction_active
    while running:
        time_delta = clock.tick(60) / 1000.0
        start_time = time.time()
        buffer = []
        buffer.append(audio.read(AUDIO_CHUNK))

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


if __name__ == "__main__":
    audio = SharedAudioResource()
    pygame_thread_instance = threading.Thread(target=pygame_thread, args=(audio,))
    pygame_thread_instance.start()
    tkinker_sliders()
    pygame_thread_instance.join()
    audio.close()
