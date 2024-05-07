import noisereduce as nr
import librosa
import numpy as np
import soundfile as sf
import time
from pedalboard import *

from matplotlib import pyplot as plt

# Load the audio file
audio_file_path = 'audio_file.wav'
y, sr = librosa.load(audio_file_path)

# Apply noise reduction
start_time = time.time()
reduced_noise = nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=1)
print("Time taken for noise reduction:", time.time() - start_time)

board = Pedalboard([
    NoiseGate(threshold_db=-30, ratio=1.5, release_ms=250),
    Compressor(threshold_db=-16, ratio=2.5),
    LowShelfFilter(cutoff_frequency_hz=400, gain_db=10, q=1),
    Gain(gain_db=10)
])
effected = board(reduced_noise, sr)

# Write the noise-reduced audio to a new file
reduced_noise_path = 'reduced_noise.wav'
sf.write(reduced_noise_path, effected, sr)
print("Noise reduction completed. Output saved to:", reduced_noise_path)

fig, ax = plt.subplots(2, 1, figsize=(15,8))
ax[0].set_title("Original signal")
ax[0].plot(np.array(y))
ax[1].set_title("Reduced noise signal")
ax[1].plot(effected)
plt.show()