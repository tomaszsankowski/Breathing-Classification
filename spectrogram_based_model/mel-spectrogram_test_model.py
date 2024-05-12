from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing import image
# Load files

test_folder_paths = ['data/test/inhale_mel-spectrograms', 'data/test/exhale_mel-spectrograms', 'data/test/silence_mel-spectrograms']

# TODO : After finishing EfficientNet model, test the model on test data

test_images = []
test_labels = []
for i, folder_path in enumerate(test_folder_paths):
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            file_path = os.path.join(folder_path, filename)
            img = image.load_img(file_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            test_images.append(img_array)
            test_labels.append(i)

X_test = np.array(test_images)
y_test = tf.keras.utils.to_categorical(test_labels)


# Load the model
model = load_model('model/efficientnet_model.h5')

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')
