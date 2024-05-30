import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft

INHALE_DIR_PATH = 'train-data/inhale'
EXHALE_DIR_PATH = 'train-data/exhale'
SILENCE_DIR_PATH = 'train-data/silence'

folder_paths = [INHALE_DIR_PATH, EXHALE_DIR_PATH, SILENCE_DIR_PATH]

spectrogram_paths = ['spectrograms/inhale_spectrograms',
                     'spectrograms/exhale_spectrograms',
                     'spectrograms/silence_spectrograms', ]

# size of image in pixels is 224x224 because of EfficientNet v2 specifications

segment_length = 0.5  # length of segments in seconds

global_min_inhale = np.inf
global_max_inhale = -np.inf
global_min_exhale = np.inf
global_max_exhale = -np.inf
global_min_silence = np.inf
global_max_silence = -np.inf

# First pass to compute global min and max
for folder_path in folder_paths:
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)

            # Read the audio file
            sample_rate, data = wavfile.read(file_path)

            # Convert stero to mono
            if data.ndim == 2:
                data = data.mean(axis=1)

            if folder_path == INHALE_DIR_PATH:
                global_min_inhale = min(global_min_inhale, data.min())
                global_max_inhale = max(global_max_inhale, data.max())
            elif folder_path == EXHALE_DIR_PATH:
                global_min_exhale = min(global_min_exhale, data.min())
                global_max_exhale = max(global_max_exhale, data.max())
            elif folder_path == SILENCE_DIR_PATH:
                global_min_silence = min(global_min_silence, data.min())
                global_max_silence = max(global_max_silence, data.max())

print(global_min_inhale, global_max_inhale)
print(global_min_exhale, global_max_exhale)
print(global_min_silence, global_max_silence)

spectrogram_max = -np.inf

for folder_path, spectrogram_path in zip(folder_paths, spectrogram_paths):
    os.makedirs(spectrogram_path, exist_ok=True)
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)

            # Read the audio file
            sample_rate, data = wavfile.read(file_path)

            # Convert stero to mono
            if data.ndim == 2:
                data = data.mean(axis=1)

            # Calculate the number of frames in segment_length seconds
            segment_frames = int(segment_length * sample_rate)

            # Split the audio into segments
            segments = [data[i:i + segment_frames] for i in range(0, len(data), segment_frames)]

            spectrogram = np.empty((224, 224))

            for i, segment in enumerate(segments):
                # Skip segment if it's shorter than 0.5 seconds
                if len(segment) < segment_frames:
                    continue

                furier_hop = sample_rate * segment_length / 224
                noverlap = np.floor(4096 - furier_hop)

                # Perform FFT
                freq, time, stft_data = stft(segment, sample_rate, nperseg=4096, noverlap=noverlap, scaling='spectrum')

                # Finally fft_out is matrix [224,224] ready to put into spectrogram
                spectrogram = stft_data[:224, :224]

                spectrogram = np.abs(spectrogram)

                # Unccomment if db-scale:
                # import librosa
                # spectrogram = librosa.amplitude_to_db(np.abs(spectrogram))

                np.save(os.path.join(spectrogram_path, f'{filename.replace(".wav", "")}_{i}.npy'), spectrogram)
                plt.imsave(os.path.join(spectrogram_path, f'{filename.replace(".wav", "")}_{i}.png'), spectrogram,
                           cmap='inferno', vmin=0, vmax=25)
