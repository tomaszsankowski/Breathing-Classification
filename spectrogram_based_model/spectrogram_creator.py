import os
import matplotlib.pyplot as plt
import librosa
from PIL import Image
import numpy as np

INHALE_DIR_PATH_TEST = '../data/test/inhale'
EXHALE_DIR_PATH_TEST = '../data/test/exhale'
SILENCE_DIR_PATH_TEST = '../data/test/silence'
INHALE_DIR_PATH_TRAIN = '../data/train/inhale'
EXHALE_DIR_PATH_TRAIN = '../data/train/exhale'
SILENCE_DIR_PATH_TRAIN = '../data/train/silence'

folder_paths = [INHALE_DIR_PATH_TEST, EXHALE_DIR_PATH_TEST, SILENCE_DIR_PATH_TEST, INHALE_DIR_PATH_TRAIN,
                EXHALE_DIR_PATH_TRAIN, SILENCE_DIR_PATH_TRAIN]

spectrogram_paths = ['../data/test_mel-spectrograms/inhale_mel-spectrograms', '../data/test_mel-spectrograms/exhale_mel-spectrograms',
                     '../data/test_mel-spectrograms/silence_mel-spectrograms', '../data/train_mel-spectrograms/inhale_mel-spectrograms',
                     '../data/train_mel-spectrograms/exhale_mel-spectrograms', '../data/train_mel-spectrograms/silence_mel-spectrograms']

# Define the size to which you want to scale the images
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

            # Open the image file with PIL and resize it
            img = Image.open('temp.png')
            img_resized = img.resize(image_size)

            # Save the resized image
            img_resized.save(os.path.join(spectrogram_path, filename.replace('.wav', '.png')))

            # Remove the temporary image file
            os.remove('temp.png')
