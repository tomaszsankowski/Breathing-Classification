import os
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dropout

# Paths to the directories with spectrograms
INHALE_PATH = 'spectrograms/inhale_spectrograms'
EXHALE_PATH = 'spectrograms/exhale_spectrograms'
SILENCE_PATH = 'spectrograms/silence_spectrograms'

folder_paths = [INHALE_PATH, EXHALE_PATH, SILENCE_PATH]

# Initialize lists to store data and labels
spectrograms = []
class_labels = ['inhale', 'exhale', 'silence']
labels = []

# Load spectrograms and labels
for i, folder_path in enumerate(folder_paths):
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            spectrogram = np.load(file_path)
            spectrograms.append(spectrogram)
            labels.append(i)

# Convert lists to numpy arrays
spectrograms = np.array(spectrograms)
spectrograms = np.expand_dims(spectrograms, axis=-1)  # Add channel dimension
labels = to_categorical(np.array(labels))

# Split data into training, validation, and test sets
X_train, X_test, Y_train, Y_test = train_test_split(spectrograms, labels, test_size=0.2, random_state=42)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

# For debugging purposes, print the shapes of the data
print("X_train shape:", X_train.shape)
print("X_val shape:", X_val.shape)
print("X_test shape:", X_test.shape)

# Use MobileNetV2 as the base model
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
# base_model = EfficientNetV2B0(weights='imagenet', input_shape=(224, 224, 3), include_top=False)
base_model.trainable = False

# Add a convolutional layer to change the input shape to (224, 224, 3)
input_layer = layers.Input(shape=(224, 224, 1))
x = layers.Concatenate()([input_layer, input_layer, input_layer])  # Replicate to have 3 channels
x = base_model(x)

# Add a MaxPooling2D layer
x = layers.MaxPooling2D(pool_size=(2, 2))(x)

# Add the rest of the model
x = layers.Flatten()(x)

# Dodanie warstwy Dropout
x = Dropout(0.5)(x)

# Dodanie regularyzacji L2 do warstwy wyjściowej modelu
output_layer = layers.Dense(3, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(x)


# Define the full model
model = models.Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(X_train, Y_train,
          validation_data=(X_val, Y_val),
          epochs=50,
          batch_size=32,
          callbacks=[early_stopping])

# Save the model
model.save('spectrogram_mobilenetv2_model.keras')

# Test the model on the test data
loss, accuracy = model.evaluate(X_test, Y_test)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')
