import os
from pydub import AudioSegment

# script deletes recordings that are shorter than 1 second from given directories

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

            if len(audio) < 1000:
                os.remove(file_path)
                print(f"Removed file: {filename}")
