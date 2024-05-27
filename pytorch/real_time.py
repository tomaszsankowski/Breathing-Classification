import sounddevice as sd
import numpy as np
import torch
import librosa
from pytorch.model_creator import AudioClassifier

# Wczytanie wytrenowanego modelu
model = AudioClassifier()
model.load_state_dict(torch.load('audio_classifier.pth'))
model.eval()

