import os
import matplotlib.pyplot as plt
import librosa
from PIL import Image
import numpy as np

INHALE_DIR_PATH = '../data/inhale'
EXHALE_DIR_PATH = '../data/exhale'
SILENCE_DIR_PATH = '../data/silence'

folder_paths = [INHALE_DIR_PATH, EXHALE_DIR_PATH, SILENCE_DIR_PATH]

spectrogram_paths = ['../data/inhale_mel-spectrograms',
                     '../data/exhale_mel-spectrograms',
                     '../data/silence_mel-spectrograms',]

# size of image in pixels
image_size = (224, 224)

for folder_path, spectrogram_path in zip(folder_paths, spectrogram_paths):
    os.makedirs(spectrogram_path, exist_ok=True)
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)

            y, sr = librosa.load(file_path)

            spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
            spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

            # Save the spectrogram as an image
            plt.imsave('temp.png', spectrogram_db)

            # Open the image file and resize it
            img = Image.open('temp.png')
            img_resized = img.resize(image_size)

            # Save the resized image
            img_resized.save(os.path.join(spectrogram_path, filename.replace('.wav', '.png')))

            # Remove the temporary image file
            os.remove('temp.png')
