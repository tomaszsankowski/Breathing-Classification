import time
import os
import numpy as np
import tensorflow.compat.v1 as tf
from model import vggish_input, vggish_params, vggish_slim
import pandas as pd

##################################################
TEST = True
VGGISH_CHECKPOINT_PATH = 'model/vggish_model.ckpt'
##################################################

if TEST:
    CSV_PATH = 'data/test/csv/'
    INHALE_DIR_PATH = 'data/test/inhale'
    EXHALE_DIR_PATH = 'data/test/exhale'
    SILENCE_DIR_PATH = 'data/test/silence'
else:
    CSV_PATH = 'data/train/csv/'
    INHALE_DIR_PATH = 'data/train/inhale'
    EXHALE_DIR_PATH = 'data/train/exhale'
    SILENCE_DIR_PATH = 'data/train/silence'

start_time = time.time()
os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
paths = [INHALE_DIR_PATH, EXHALE_DIR_PATH, SILENCE_DIR_PATH]
for path in paths:
    all_embeddings = []
    print("Converting:", path)
    for filename in os.listdir(path):
        if filename.endswith('.wav'):
            one_time = time.time()
            # Full path to the sound file
            breathing_sound_file_path = os.path.join(path, filename)

            # Load the breathing sound as sound waves
            breathing_waveform = vggish_input.wavfile_to_examples(breathing_sound_file_path)
            with tf.Graph().as_default(), tf.Session() as sess:
                # Define VGGish
                embeddings = vggish_slim.define_vggish_slim()

                # Initialize all variables in the model, then load the VGGish checkpoint
                sess.run(tf.global_variables_initializer())
                vggish_slim.load_vggish_slim_checkpoint(sess, VGGISH_CHECKPOINT_PATH)

                # Get the input tensor
                features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)

                # Transform sound waves into log mel spectrogram and pass to the VGGish model to get embeddings
                [embedding_batch] = np.array(sess.run([embeddings], feed_dict={features_tensor: breathing_waveform}))
                all_embeddings.append(embedding_batch)
                print("Size", len(embedding_batch))
                print("Time", time.time() - one_time)

    df = pd.DataFrame(np.concatenate(all_embeddings))
    if path == EXHALE_DIR_PATH:
        print('Saving csv ' + EXHALE_DIR_PATH)
        file_path = CSV_PATH + 'exhale.csv'
        df.to_csv(file_path, index=False)
    elif path == INHALE_DIR_PATH:
        print('Saving csv' + INHALE_DIR_PATH)
        file_path = CSV_PATH + 'inhale.csv'
        df.to_csv(file_path, index=False)
    else:
        print('Saving csv' + SILENCE_DIR_PATH)
        file_path = CSV_PATH + 'silence.csv'
        df.to_csv(file_path, index=False)

print("End time:", time.time() - start_time)
