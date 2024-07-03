import wave
import math
import librosa
import numpy as np
import pandas as pd
import pyaudio
import torch
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from pytorch_based_model.model import AudioClassifier, AudioDatasetRealtime as AudioDataset
from scipy.signal import stft
from vggish_based_model.model import vggish_postprocess, vggish_params, vggish_slim, vggish_input
import tensorflow.compat.v1 as tf
from df.enhance import init_df
import joblib

# Constants

vggish_checkpoint_path = '../vggish_based_model/model/vggish_model.ckpt'
CLASS_MODEL_PATH = '../vggish_based_model/model/trained_model_rf.pkl'
VGGISH_PARAMS_PATH = '../vggish_based_model/model/vggish_pca_params.npz'
pproc = vggish_postprocess.Postprocessor(VGGISH_PARAMS_PATH)
model, df_state, _ = init_df()

REFRESH_TIME = 0.25
N_FOURIER = 2048

FORMAT = pyaudio.paInt16

CHANNELS = 2
RATE = 44100
DEVICE_INDEX = 4

CHUNK_SIZE = int(RATE * REFRESH_TIME)

running = True

filename = '2024-07-03_12-23-20'
CLASSIFIER_MODEL_PATH = '../pytorch_based_model/audio_rnn_classifier.pth'

model_spectrogams = load_model(
    f'../spectrogram_based_model/best_models/mobile_net_model_{N_FOURIER}_{REFRESH_TIME}_small.keras')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

'''                                 Mfcc classifier class                  '''


class RealTimeAudioClassifier:
    def __init__(self, model_path):
        self.model = AudioClassifier()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model = self.model.to(device)
        self.model.eval()
        self.last_prediction = None

    def predict(self, audio_path):
        y, sr = librosa.load(audio_path, sr=RATE, mono=True)
        split_y = [y[i:i + CHUNK_SIZE] for i in range(0, len(y), CHUNK_SIZE)]
        if split_y[-1].shape[0] < CHUNK_SIZE:
            split_y = split_y[:-1]
        predictions = []
        for y in split_y:
            frames = []
            for i in range(0, len(y), CHUNK_SIZE):
                frame = y[i:i + CHUNK_SIZE]
                if len(frame) == CHUNK_SIZE:
                    mfcc = librosa.feature.mfcc(y=frame, sr=sr)
                    frames.append(mfcc)

            frames = AudioDataset(frames)
            frames = torch.utils.data.DataLoader(frames, batch_size=1, shuffle=False)
            for frames in frames:
                frames = frames.to(device)

            outputs = self.model(frames)
            _, predicted = torch.max(outputs, 1)
            self.last_prediction = predicted.cpu().numpy()[0]
            predictions.append(predicted.cpu().numpy()[0])
        return predictions


'''                         Main function                             '''

if __name__ == "__main__":
    prediction_path = f'{filename}.wav'
    tags_path = f'{filename}.csv'

    tags = pd.read_csv(tags_path)['tag'].to_numpy()
    tag_time = pd.read_csv(tags_path)['time'].to_numpy()

    wf = wave.open(prediction_path, 'rb')

    signal = wf.readframes(-1)

    # Length of recording in seconds

    length_in_seconds = math.floor(len(signal) / wf.getframerate()) / 4  # 4 because of stereo + 2 bytes per sample

    # Number of 0.25s sectors

    num_frames = math.floor(length_in_seconds / REFRESH_TIME)

    # Count number of frames to read

    frames_to_read = num_frames * CHUNK_SIZE

    signal = np.frombuffer(signal[:frames_to_read * 4], dtype='int16')

    time = np.linspace(0, len(signal) / wf.getframerate(), num=len(signal))

    fig, axs = plt.subplots(4, 1)  # 2 rows, 1 column

    '''                    HUMAN LABELED                   '''

    for i in range(len(tags) - 1):
        color = ''
        if tags[i] == 'E':
            color = 'green'
        elif tags[i] == 'I':
            color = 'red'
        else:
            color = 'blue'

        start_index = int(tag_time[i] * wf.getframerate())
        end_index = int(tag_time[i + 1] * wf.getframerate())
        segment_time = np.linspace(tag_time[i], tag_time[i + 1], end_index - start_index)
        axs[0].plot(segment_time, signal[start_index:end_index], color=color)

    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('Amplitude')
    axs[0].set_title('Ręcznie oznaczone segmenty')

    '''                         MFCC                       '''

    classifier = RealTimeAudioClassifier(CLASSIFIER_MODEL_PATH)
    predictions_mfcc = classifier.predict(prediction_path)

    for i in range(len(predictions_mfcc)):
        color = ''
        if predictions_mfcc[i] == 0:  # exhale
            color = 'green'
        elif predictions_mfcc[i] == 1:  # inhale
            color = 'red'
        else:
            color = 'blue'
        axs[1].plot(time[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE], signal[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE], color=color)
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Amplitude')
    axs[1].set_title('Segmenty przewidziane przez model oparty na mfcc')

    '''                        VGGish                      '''

    rf_classifier = joblib.load(CLASS_MODEL_PATH)

    with tf.Graph().as_default(), tf.Session() as sess:
        # Define VGGish

        embeddings = vggish_slim.define_vggish_slim()

        # Initialize all variables in the model, then load the VGGish checkpoint

        sess.run(tf.global_variables_initializer())
        vggish_slim.load_vggish_slim_checkpoint(sess, vggish_checkpoint_path)

        # Get the input tensor

        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)

        # Number of samples per 0.25s

        samples_per_quarter_second = int(wf.getframerate() * 0.25 * 2)

        # Dividing recording into 0.25s parts

        split_signal = [signal[i:i + samples_per_quarter_second] for i in
                        range(0, len(signal), samples_per_quarter_second)]

        for i, recording in enumerate(split_signal):
            if len(recording) < samples_per_quarter_second:
                continue

            # Write 0.25s part to a file

            buffer = [recording, recording, recording, recording]

            wf = wave.open("temp.wav", 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(buffer))
            wf.close()

            # Prepare input for the model

            input_batch = vggish_input.wavfile_to_examples('temp.wav')

            # Calculate embeddings

            embedding_batch = np.array(sess.run(embeddings, feed_dict={features_tensor: input_batch}))

            postprocessed_batch = pproc.postprocess(embedding_batch)

            df = pd.DataFrame(postprocessed_batch)  # 128 features vector

            prediction = rf_classifier.predict(df)

            color = ''
            if prediction == 0:  # Inhale
                color = 'red'
            elif prediction == 1:  # Exhale
                color = 'green'
            else:
                color = 'blue'
            axs[2].plot(time[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE], signal[i * CHUNK_SIZE:(i + 1) * CHUNK_SIZE],
                        color=color)

        axs[2].set_xlabel('Time [s]')
        axs[2].set_ylabel('Amplitude')
        axs[2].set_title('Segmenty przewidziane przez model oparty na modelu VGGish')

    '''                     SPECTROGRAMS                    '''

    # Convert stereo to mono

    mono_signal = np.mean(signal.reshape(-1, 2), axis=1)

    # Number of samples per 0.25s

    samples_per_quarter_second = int(wf.getframerate() * 0.25)

    # Dividing recording into 0.25s parts

    split_signal = [mono_signal[i:i + samples_per_quarter_second] for i in
                    range(0, len(mono_signal), samples_per_quarter_second)]

    for i, recording in enumerate(split_signal):
        if len(recording) < samples_per_quarter_second:
            continue

        # Create spectrogram
        # Calculate STFT parameters

        furier_hop = np.floor(RATE * REFRESH_TIME / 224)
        noverlap = N_FOURIER - furier_hop

        # Perform FFT

        stft_data = stft(recording, RATE, nperseg=N_FOURIER, noverlap=noverlap, scaling='spectrum')[2]

        # Take only the first 224x224 part of the spectrogram

        spectrogram_in = stft_data[:224, :224]

        spectrogram_in = np.abs(spectrogram_in)

        # Prepare input for the model ( change dimensions )

        spectrogram_in = np.expand_dims(spectrogram_in, axis=-1)
        spectrogram_in = np.expand_dims(spectrogram_in, axis=0)

        # Model prefiction

        predictions = model_spectrogams.predict(spectrogram_in, verbose=0)

        prediction = np.argmax(predictions)

        color = ''
        if prediction == 0:  # Inhale
            color = 'red'
        elif prediction == 1:  # Exhale
            color = 'green'
        else:
            color = 'blue'  # Silence
        axs[3].plot(time[i * samples_per_quarter_second:(i + 1) * samples_per_quarter_second],
                    signal[i * samples_per_quarter_second:(i + 1) * samples_per_quarter_second], color=color)
    axs[3].set_xlabel('Time [s]')
    axs[3].set_ylabel('Amplitude')
    axs[3].set_title('Segmenty przewidziane przez model oparty na analizie spektralnej')

    plt.tight_layout()

    plt.show()
