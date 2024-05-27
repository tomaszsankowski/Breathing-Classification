

# IMPORTANT

# TRAIN NETWORK ONLY USING GOOGLE COALB
# IT WILL TRAIN YEARS ON YOUR COMPUTER

import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout

INHALE_PATH = '../data/spectrograms/inhale_spectrograms'
EXHALE_PATH = '../data/spectrograms/exhale_spectrograms'
SILENCE_PATH = '../data/spectrograms/silence_spectrograms'

folder_paths = [INHALE_PATH, EXHALE_PATH, SILENCE_PATH]
if tf.config.list_physical_devices('GPU'):
    print("GPU is available")
else:
    print("GPU is not available")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

images = []
class_labels = ['inhale', 'exhale', 'silence']
labels = []
indices_train = []
indices_test = []
indice = 0
for i, folder_path in enumerate(folder_paths):
    sample_count = 0
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
            if sample_count < 900:
                indices_train.append(indice)
            else:
                indices_test.append(indice)
            sample_count += 1
            indice += 1


# Prepare training and test data
images_train = [images[i] for i in indices_train]
labels_train = [labels[i] for i in indices_train]

images_test = [images[i] for i in indices_test]
labels_test = [labels[i] for i in indices_test]

# Convert lists to numpy arrays
X_train = np.array(images_train)
Y_train = tf.keras.utils.to_categorical(np.array(labels_train))

X_test = np.array(images_test)
Y_test = tf.keras.utils.to_categorical(np.array(labels_test))

base_model = EfficientNetV2B0(weights='imagenet', input_shape=(224, 224, 3), include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(len(np.unique(labels)), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train model
model.fit(X_train, Y_train, epochs=10, validation_split=0.2)

# Save the model
model.save('spectrogram_efficientnet_model.keras')

# Test the model for test data

loss, accuracy = model.evaluate(X_test, Y_test)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')