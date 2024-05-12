import os
import matplotlib.pyplot as plt
import librosa
from PIL import Image

INHALE_DIR_PATH = '../data/inhale'
EXHALE_DIR_PATH = '../data/exhale'
SILENCE_DIR_PATH = '../data/silence'

folder_paths = [INHALE_DIR_PATH, EXHALE_DIR_PATH, SILENCE_DIR_PATH]

mfcc_paths = ['../data/mfcc/inhale_mfcc', '../data/mfcc/exhale_mfcc', '../data/mfcc/silence_mfcc']

# size of image in pixels is 244x244 because of EfficientNet v2 specifications
image_size = (224, 224)

segment_length = 0.5  # length of segments in seconds

for folder_path, mfcc_path in zip(folder_paths, mfcc_paths):
    os.makedirs(mfcc_path, exist_ok=True)
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

                mfcc = librosa.feature.mfcc(y=segment, sr=sr)

                # Save the MFCC as an image
                plt.imsave('temp.png', mfcc)

                # Open the image file and resize it
                img = Image.open('temp.png')
                img_resized = img.resize(image_size)

                # Save the resized image
                img_resized.save(os.path.join(mfcc_path, f'{filename.replace(".wav", "")}_{i}.png'))

                # Remove the temporary image file
                os.remove('temp.png')