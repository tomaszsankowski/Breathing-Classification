import os
import numpy as np
# Its normal that tensorflow.keras.models is red underlined, I don't know why is that, but it will work if you run the script
from tensorflow.keras.models import load_model
from scipy.io import wavfile
from scipy.signal import stft
from result_analysis_neural_network import start_analysis
'''
Co robi program?
Podajsz ścieżki do folderów z nagraniami wdechu, wydechu i ciszy
Następnie podajesz folder, w którym znajduje się model, który chcesz przetestować
Oraz podajesz jego parametry czyli n_fourier i segment_length
Dzięki temu program sam stworzy stringa który zawierać będzie ścieżkę do modelu
Najpierw każde nagranie jest dzielone na segmenty o długości segment_length sekund
Z każdego nagrania zapisywany jest spektrogram, który następnie jest dodawany do listy odpowiadającej jego klasie
W tym momencie masz 3 listy z spektrogramami, które należy przetestować na modelu
Następnie iterujesz po każdej liście i dla każdego spektrogramu przewidujesz jego klasę
Na końcu masz macierz pomyłek, która pokazuje ile spektrogramów z każdej klasy zostało poprawnie zaklasyfikowanych
A ile z nich zostało zaklasyfikowanych jako inna klasa
'''

'''
GLOBAL VARIABLES TO CHANGE TO TEST DIFFERENT SPECTROGRAM MODELS
'''

# Paths to inhale, exhale and silence audio files

INHALE_DIR_PATH = './spectogram_test_data/inhale'
EXHALE_DIR_PATH = './spectogram_test_data/exhale'
SILENCE_DIR_PATH = './spectogram_test_data/silence'

# Choosen model variables

n_fourier = 512  # Number of fourier points
segment_length = 0.5  # Length of audio file to be analyzed in seconds
directory = '../spectrogram_based_model/best_models'  # Directory where the model is stored; alternatively: directory = 'models_mobilenet'

# Concatenate the model path

MODEL_PATH = f'{directory}/mobile_net_model_{n_fourier}_{segment_length}_small.keras'

'''
END OF GLOBAL VARIABLES
'''

folder_paths = [INHALE_DIR_PATH, EXHALE_DIR_PATH, SILENCE_DIR_PATH]

# Vector of spectrograms to be classified by the model for every class

X_test_inhale = []
X_test_exhale = []
X_test_silence = []

# Iterate through the folders and read all the audio files

for folder_path in folder_paths:
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            # Get the file path

            file_path = os.path.join(folder_path, filename)

            # Read the audio file

            sample_rate, data = wavfile.read(file_path)

            # Convert stereo to mono

            if data.ndim == 2:
                data = data.mean(axis=1)

            # Calculate the number of frames in every segment

            segment_frames = int(segment_length * sample_rate)

            # Split the frames audio into segments of length <segment_length> seconds

            segments = [data[i:i + segment_frames] for i in range(0, len(data), segment_frames)]

            # Preallocate the spectrogram array (224x224 beacuse of neural network requirements)

            spectrogram = np.empty((224, 224))

            # Iterate through every segment, create spectrogram and perform model prediction

            for segment in segments:

                # Skip segment if it's shorter than <segment_length> seconds

                if len(segment) < segment_frames:
                    continue

                # Variables for stft function

                furier_hop = np.floor(sample_rate * segment_length / 224)
                noverlap = n_fourier - furier_hop

                # Perform STFT

                freq, time, stft_data = stft(segment, sample_rate, nperseg=n_fourier, noverlap=noverlap, scaling='spectrum')

                spectrogram = np.abs(stft_data[:224, :224])

                # Append the spectrogram to the corresponding class vector

                if folder_path == INHALE_DIR_PATH:
                    X_test_inhale.append(spectrogram)
                elif folder_path == EXHALE_DIR_PATH:
                    X_test_exhale.append(spectrogram)
                elif folder_path == SILENCE_DIR_PATH:
                    X_test_silence.append(spectrogram)


# Evaluate the model on the test data

# Load model

model = load_model(MODEL_PATH)

# Matrix to store every prediction of the model
# Rows represents accual class
# Columns represents predicted class

confusion_matrix = np.zeros((3, 3))

# Iterate through every class and perform model prediction

predictions = []
for i, X_test in enumerate([X_test_inhale, X_test_exhale, X_test_silence]):
    for spectrogram in X_test:
        # Convert the spectrogram to numpy array

        spectrogram = np.array(spectrogram)

        # Add a new axis to the spectrogram to match the input shape of the model

        spectrogram = spectrogram[np.newaxis, :, :, np.newaxis]

        # Perform the prediction

        prediction = model.predict(spectrogram, verbose=0)
        predictions.append(prediction)
        #print(prediction.__class__)

        # Get the class index with the highest probability

        predicted_class = np.argmax(prediction)

        # Update the confusion matrix

        confusion_matrix[i, predicted_class] += 1

# Afterward print the confusion matrix

print(f'\t\t\t\t\t\t\tPredicted class\tPredicted class\t Predicted class')
print(f'\t\t\t\t\t\t\tInhale\t\t\tExhale\t\t\tSilence')
print(f'Actual class\tInhale\t\t{confusion_matrix[0, 0]}\t\t\t\t{confusion_matrix[0, 1]}\t\t\t\t{confusion_matrix[0, 2]}')
print(f'Actual class\tExhale\t\t{confusion_matrix[1, 0]}\t\t\t\t{confusion_matrix[1, 1]}\t\t\t\t{confusion_matrix[1, 2]}')
print(f'Actual class\tSilence\t\t{confusion_matrix[2, 0]}\t\t\t\t{confusion_matrix[2, 1]}\t\t\t\t{confusion_matrix[2, 2]}')

print(predictions)
#print(predictions[1].__class__)
start_analysis(predictions, confusion_matrix)