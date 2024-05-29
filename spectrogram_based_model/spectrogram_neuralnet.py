import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Ścieżki do katalogów ze spektrogramami
INHALE_PATH = '/content/dataset/spectrograms/inhale_spectrograms'
EXHALE_PATH = '/content/dataset/spectrograms/exhale_spectrograms'
SILENCE_PATH = '/content/dataset/spectrograms/silence_spectrograms'

folder_paths = [INHALE_PATH, EXHALE_PATH, SILENCE_PATH]

# Inicjalizacja list do przechowywania danych i etykiet
images = []
class_labels = ['inhale', 'exhale', 'silence']
labels = []

# Wczytywanie obrazów i etykiet
for i, folder_path in enumerate(folder_paths):
    ctr = 0
    for filename in os.listdir(folder_path):
        if ctr < 700 and filename.endswith('.png'):
            file_path = os.path.join(folder_path, filename)
            img = image.load_img(file_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            images.append(img_array)
            labels.append(i)
            ctr += 1

# Konwersja list do tablic numpy
images = np.array(images)
labels = to_categorical(np.array(labels))

# Podział danych na zestawy treningowy i testowy
X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Użycie EfficientNetV2B0 jako bazowego modelu
base_model = EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Dodanie własnych warstw
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
output_layer = layers.Dense(3, activation='softmax')(x)

# Definiowanie pełnego modelu
model = models.Model(inputs=base_model.input, outputs=output_layer)

# Kompilacja modelu
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Wyświetlenie podsumowania modelu
model.summary()

# Trenowanie modelu
model.fit(X_train, Y_train, epochs=10, validation_split=0.2)

# Zapisanie modelu
model.save('spectrogram_efficientnet_model.keras')

# Testowanie modelu na danych testowych
loss, accuracy = model.evaluate(X_test, Y_test)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')
