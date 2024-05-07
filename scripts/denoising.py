import numpy as np
from scipy.io import wavfile


noise_file_path = '../data/test/inhale/2024-05-06_23-46-40.wav'
audio_file_path = '../data/test/inhale/2024-05-06_23-44-34.wav'

rate_noise, noise_data = wavfile.read(noise_file_path)
rate_audio, audio_data = wavfile.read(audio_file_path)

assert rate_noise == rate_audio, "Częstotliwość próbkowania obu nagrania musi być taka sama"

difference_length = len(audio_data) - len(noise_data)

if difference_length > 0:
    noise_data = np.pad(noise_data, (0, difference_length), mode='constant')

noise_mean = np.mean(noise_data)
audio_mean = np.mean(audio_data)
difference_mean = audio_mean - noise_mean

odszyfrowane_nagranie = audio_data - difference_mean

wavfile.write("odszyfrowane_nagranie.wav", rate_audio, odszyfrowane_nagranie)
