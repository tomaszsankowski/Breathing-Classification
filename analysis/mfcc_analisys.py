import os
import numpy as np
from pytorch_based_model.model import AudioClassifier, AudioDatasetRealtime as AudioDataset
from scipy.io import wavfile
from scipy.signal import stft
import torch
import librosa

'''
Tutaj zmieniasz tylko sciezke do danych testowych
I ewentualnie dlugosc analizowanej probki ale piter i tak to jakos usrednia
wiec nie powinno to miec az takiego znaczenie chyba
'''

'''
GLOBAL VARIABLES TO CHANGE TO TEST APPROACHES
'''

# Paths to inhale, exhale and silence audio files

INHALE_DIR_PATH = '../spectrogram_based_model/train-data/inhale'
EXHALE_DIR_PATH = '../spectrogram_based_model/train-data/exhale'
SILENCE_DIR_PATH = '../spectrogram_based_model/train-data/silence'

# Choosen model variables

segment_length = 0.5  # Length of audio file to be analyzed in seconds
sample_rate = 44100

CLASSIFIER_MODEL_PATH = '../pytorch_based_model/audio_rnn_classifier.pth'  # I think only one is working

'''
END OF GLOBAL VARIABLES
'''

folder_paths = [INHALE_DIR_PATH, EXHALE_DIR_PATH, SILENCE_DIR_PATH]

# Vector of spectrograms to be classified by the model for every class

X_test_inhale = []
X_test_exhale = []
X_test_silence = []


# Evaluate the model on the test data

# Load model

model = AudioClassifier()

model.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# Matrix to store every prediction of the model
# Rows represents accual class
# Columns represents predicted class

confusion_matrix = np.zeros((3, 3))

# Iterate through the folders and read all the audio files
for folder_path in folder_paths:
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            # Get the file path

            file_path = os.path.join(folder_path, filename)

            # Load audio file

            y, sr = librosa.load(file_path, sr=44100, mono=True)

            # Calculate the number of frames in every segment

            segment_frames = int(segment_length * sr)

            # Split the frames audio into segments of length <segment_length> seconds

            frames = []
            for i in range(0, len(y), segment_frames):
                frame = y[i:i + segment_frames]
                if len(frame) == segment_frames:  # Ignorujemy ostatnią ramkę, jeśli jest krótsza
                    mfcc = librosa.feature.mfcc(y=frame, sr=sr)
                    frames.append(mfcc)

            # Prepare data for the model

            frames = AudioDataset(frames)
            frames = torch.utils.data.DataLoader(frames, batch_size=1, shuffle=False)
            for frames in frames:
                frames = frames.to('cpu')

            # Predict the class of the segment

            outputs = model(frames)
            _, predicted = torch.max(outputs, 1)
            prediction = predicted.cpu().numpy()[0]

            # In this model 0 = exhale and 1 = inhale so lets change that

            if prediction == 0:
                prediction = 1
            elif prediction == 1:
                prediction = 0

            # Update the confusion matrix

            if folder_path == INHALE_DIR_PATH:
                confusion_matrix[0, prediction] += 1
            elif folder_path == EXHALE_DIR_PATH:
                confusion_matrix[1, prediction] += 1
            else:
                confusion_matrix[2, prediction] += 1

# Afterward print the confusion matrix

print(f'\t\t\t\t\t\t\tPredicted class\tPredicted class\t Predicted class')
print(f'\t\t\t\t\t\t\tInhale\t\t\tExhale\t\t\tSilence')
print(
    f'Actual class\tInhale\t\t{confusion_matrix[0, 0]}\t\t\t\t{confusion_matrix[0, 1]}\t\t\t\t{confusion_matrix[0, 2]}')
print(
    f'Actual class\tExhale\t\t{confusion_matrix[1, 0]}\t\t\t\t{confusion_matrix[1, 1]}\t\t\t\t{confusion_matrix[1, 2]}')
print(
    f'Actual class\tSilence\t\t{confusion_matrix[2, 0]}\t\t\t\t{confusion_matrix[2, 1]}\t\t\t\t{confusion_matrix[2, 2]}')
