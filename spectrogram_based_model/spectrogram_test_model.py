import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

INHALE_PATH = '../data/spectrograms/inhale_spectrograms'
EXHALE_PATH = '../data/spectrograms/exhale_spectrograms'
SILENCE_PATH = '../data/spectrograms/silence_spectrograms'

folder_paths = [INHALE_PATH, EXHALE_PATH, SILENCE_PATH]

# Load the model
model = load_model('spectrogram_efficientnet_model.keras')

# Load the data
images = []
labels = []
for i, folder_path in enumerate(folder_paths):
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            file_path = os.path.join(folder_path, filename)
            img = image.load_img(file_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            images.append(img_array)
            if file_path.startswith(INHALE_PATH):
                labels.append(0)
            elif file_path.startswith(EXHALE_PATH):
                labels.append(1)
            else:
                labels.append(2)

# Select 50 random samples
SAMPLES_TO_CHECK = 250  # 4272

indices = random.sample(range(len(images)), SAMPLES_TO_CHECK)

# Prepare test data
images_test = [images[i] for i in indices]
labels_test = [labels[i] for i in indices]

# Convert lists to numpy arrays
X_test = np.array(images_test)
Y_test = tf.keras.utils.to_categorical(np.array(labels_test))

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, Y_test)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')
