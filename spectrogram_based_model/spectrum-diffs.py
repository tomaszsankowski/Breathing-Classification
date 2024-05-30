import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft


def plot_average_spectrum(folder_path, name, segment_length=0.25):
    # Initialize an empty list to store the spectra
    spectra = []

    frequencies = None

    # Loop over all files in the folder
    for filename in os.listdir(folder_path):

        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)

            # Read audio
            sample_rate, data = wavfile.read(file_path)

            # Convert stereo to mono if necessary
            if data.ndim == 2:
                data = data.mean(axis=1)

            # Calculate the number of frames in segment_length seconds
            segment_frames = int(segment_length * sample_rate)

            # Split the audio into segments
            segments = [data[i:i + segment_frames] for i in range(0, len(data), segment_frames)]

            # Initialize an empty list to store the spectra for this recording
            recording_spectra = []

            for segment in segments:
                # Skip segment if it's shorter than segment_length seconds
                if len(segment) < segment_frames:
                    continue

                furier_hop = sample_rate * segment_length / 224
                noverlap = 1024 - np.floor(furier_hop)

                # Perform FFT
                frequencies, times, Zxx = stft(segment, fs=sample_rate, nperseg=1024, noverlap=int(noverlap))

                # Take absolute value to get magnitude
                magnitudes = np.abs(Zxx)

                # Add the spectrum to the list
                recording_spectra.append(np.mean(magnitudes, axis=1))

            # Calculate the average spectrum for this recording and add it to the list
            spectra.append(np.mean(recording_spectra, axis=0))

    # Calculate the average spectrum across all recordings
    average_spectrum = np.mean(spectra, axis=0)

    # Plot the average spectrum
    plt.plot(frequencies[:128], average_spectrum[:128])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title(f'Average Spectrum of {name}')
    plt.ylim(0, 20)
    plt.show()


# Plot average inhale, exhale and silence files
plot_average_spectrum('train-data/inhale', 'inhale')
plot_average_spectrum('train-data/exhale', 'exhale')
plot_average_spectrum('train-data/silence', 'silence')
