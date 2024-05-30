import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


def plot_spectrum(file_path, name):
    # Read audio
    sample_rate, data = wavfile.read(file_path)

    # Fast Fourier Transformation
    fft_out = np.fft.rfft(data)
    frequencies = np.abs(fft_out)

    frequencies = frequencies[:22000]

    # Plot results
    plt.plot(frequencies)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title(f'Spectrum of {name}')
    plt.ylim([0, 800])
    plt.show()


# Plot inhale, exhale and silence files
plot_spectrum('train-data/inhale/2024-05-30_13-07-50.wav', 'inhale')
plot_spectrum('train-data/exhale/2024-05-30_13-09-24.wav', 'exhale')
plot_spectrum('train-data/silence/2024-05-30_13-06-36.wav', 'silence')
