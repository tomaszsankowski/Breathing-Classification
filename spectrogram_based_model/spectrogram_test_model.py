import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

INHALE_PATH = 'spectrograms_large/inhale_spectrograms'
EXHALE_PATH = 'spectrograms_large/exhale_spectrograms'
SILENCE_PATH = 'spectrograms_large/silence_spectrograms'

folder_paths = [INHALE_PATH, EXHALE_PATH, SILENCE_PATH]

# Load the model
model = load_model('models_mobilenet/mobile_net_model_4096_0.5_small.keras')

# Load the data
spectrograms = []
labels = []
for i, folder_path in enumerate(folder_paths):
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            spectrogram = np.load(file_path)
            spectrograms.append(spectrogram)
            labels.append(i)

# Convert lists to numpy arrays
X_test = np.array(spectrograms)
Y_test = tf.keras.utils.to_categorical(np.array(labels))

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, Y_test)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')
