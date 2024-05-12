import os
import matplotlib.pyplot as plt
import librosa
from PIL import Image
import numpy as np

INHALE_DIR_PATH = '../data/inhale'
EXHALE_DIR_PATH = '../data/exhale'
SILENCE_DIR_PATH = '../data/silence'

folder_paths = [INHALE_DIR_PATH, EXHALE_DIR_PATH, SILENCE_DIR_PATH]

spectrogram_paths = ['../data/spectrograms/inhale_spectrograms',
                     '../data/spectrograms/exhale_spectrograms',
                     '../data/spectrograms/silence_spectrograms',]

# size of image in pixels is 244x244 because of EfficientNet v2 specifications
image_size = (224, 224)

segment_length = 0.5  # length of segments in seconds

for folder_path, spectrogram_path in zip(folder_paths, spectrogram_paths):
    os.makedirs(spectrogram_path, exist_ok=True)
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)

            y, sr = librosa.load(file_path)

            # Calculate the number of frames in segment_length seconds
            segment_frames = int(segment_length * sr)

            # Split the audio into segments
            segments = [y[i:i + segment_frames] for i in range(0, len(y), segment_frames)]

            for i, segment in enumerate(segments):
                # Skip segment if it's shorter than 0.5 seconds
                if len(segment) < segment_frames:
                    continue

                # Compute the short-time Fourier transform
                D = librosa.stft(segment)
                # Convert an amplitude spectrogram to dB-scaled spectrogram.
                spectrogram_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

                # Save the spectrogram as an image
                plt.imsave('temp.png', spectrogram_db)

                # Open the image file and resize it
                img = Image.open('temp.png')
                img_resized = img.resize(image_size)

                # Save the resized image
                img_resized.save(os.path.join(spectrogram_path, f'{filename.replace(".wav", "")}_{i}.png'))

                # Remove the temporary image file
                os.remove('temp.png')