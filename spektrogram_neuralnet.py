import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

folder_paths = ['data/test/inhale_mel-spectrograms', 'data/test/exhale_mel-spectrograms', 'data/test/silence_mel-spectrograms',
                'data/train/inhale_mel-spectrograms', 'data/train/exhale_mel-spectrograms', 'data/train/silence_mel-spectrograms']

images = []
class_labels = ['inhale', 'exhale', 'silence']
labels = []
for i, folder_path in enumerate(folder_paths):
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            file_path = os.path.join(folder_path, filename)
            img = image.load_img(file_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            images.append(img_array)
            if file_path.startswith("data/train/inhale") or file_path.startswith("data/test/inhale"):
                labels.append(0)
            elif file_path.startswith("data/train/exhale") or file_path.startswith("data/test/exhale"):
                labels.append(1)
            else:
                labels.append(2)

X = np.array(images)
y = tf.keras.utils.to_categorical(labels)

base_model = EfficientNetB0(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(len(np.unique(labels)), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# train model
model.fit(X, y, epochs=10, validation_split=0.2)
# Save the model

model.save('model/efficientnet_model')

# Test the model for test data

test_folder_paths = ['data/test/inhale_mel-spectrograms', 'data/test/exhale_mel-spectrograms', 'data/test/silence_mel-spectrograms']

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

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')
