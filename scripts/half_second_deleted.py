import os
from pydub import AudioSegment

# script deletes first 300ms from every recording in given directories (deleting keyboard sounds)

INHALE_DIR_PATH_TEST = '../data/test/inhale'
EXHALE_DIR_PATH_TEST = '../data/test/exhale'
SILENCE_DIR_PATH_TEST = '../data/test/silence'
INHALE_DIR_PATH_TRAIN = '../data/train/inhale'
EXHALE_DIR_PATH_TRAIN = '../data/train/exhale'
SILENCE_DIR_PATH_TRAIN = '../data/train/silence'

folder_paths = [INHALE_DIR_PATH_TEST, EXHALE_DIR_PATH_TEST, SILENCE_DIR_PATH_TEST, INHALE_DIR_PATH_TRAIN,
                EXHALE_DIR_PATH_TRAIN, SILENCE_DIR_PATH_TRAIN]

for folder_path in folder_paths:
    for filename in os.listdir(folder_path):
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)

            audio = AudioSegment.from_wav(file_path)

            audio = audio[300:-300]

            audio.export(file_path, format="wav")
