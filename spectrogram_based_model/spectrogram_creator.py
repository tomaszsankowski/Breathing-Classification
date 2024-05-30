import os
import numpy as np
import matplotlib.pyplot as plt
import librosa

INHALE_DIR_PATH = 'train-data/inhale'
EXHALE_DIR_PATH = 'train-data/exhale'
SILENCE_DIR_PATH = 'train-data/silence'

folder_paths = [INHALE_DIR_PATH, EXHALE_DIR_PATH, SILENCE_DIR_PATH]

spectrogram_paths = ['spectrograms/inhale_spectrograms',
                     'spectrograms/exhale_spectrograms',
                     'spectrograms/silence_spectrograms',]

# size of image in pixels is 224x224 because of EfficientNet v2 specifications

segment_length = 0.25  # length of segments in seconds

for folder_path, spectrogram_path in zip(folder_paths, spectrogram_paths):
    os.makedirs(spectrogram_path, exist_ok=True)
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)

            # Read the audio file
            data, sr = librosa.load(file_path)

            # Calculate the number of frames in segment_length seconds
            segment_frames = int(segment_length * sr)

            # Split the audio into segments
            segments = [data[i:i + segment_frames] for i in range(0, len(data), segment_frames)]

            spectrogram = np.empty((224, 224))

            for i, segment in enumerate(segments):
                # Skip segment if it's shorter than 0.5 seconds
                if len(segment) < segment_frames:
                    continue

                # Perform FFT
                fft_out = np.fft.rfft(segment, n=512)

                # Select the range from 5kHz to 15kHz
                start_index = int(5000 * len(fft_out) / sample_rate)
                end_index = int(15000 * len(fft_out) / sample_rate)
                fft_out = fft_out[start_index:end_index]

                # Resample to 224 points
                fft_out_resampled = np.interp(np.linspace(0, len(fft_out), 224), np.arange(len(fft_out)),
                                              np.abs(fft_out))

                # Add to the spectrogram
                spectrogram[i, :] = fft_out_resampled

                # Save the spectrogram as an image
                plt.imsave(spectrogram_path, f'{filename.replace(".wav", "")}_{i}.png', cmap='inferno')
