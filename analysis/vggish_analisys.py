import os
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
from vggish_based_model.model import vggish_postprocess, vggish_params, vggish_slim, vggish_input
import joblib

##################################################
VGGISH_CHECKPOINT_PATH = '../vggish_based_model/model/vggish_model.ckpt'
VGGISH_PARAMS_PATH = '../vggish_based_model/model/vggish_pca_params.npz'
pproc = vggish_postprocess.Postprocessor(VGGISH_PARAMS_PATH)
CLASS_MODEL_PATH = '../vggish_based_model/model/trained_model_rf.pkl'
rf_classifier = joblib.load(CLASS_MODEL_PATH)
##################################################

'''
Jak działa program?
iteruje po kazdym nagraniu z danego folderu, wczytuje je jako fale dzwiekowe, przetwarza na mel spectrogram, a nastepnie
wrzuca do folderu VGGish.
VGGish dzieli nagranie
'''


'''
GLOBAL VARIABLES TO CHANGE TO TEST DIFFERENT SPECTROGRAM MODELS
'''

# Paths to inhale, exhale and silence audio files

INHALE_DIR_PATH = '../spectrogram_based_model/train-data/inhale'
EXHALE_DIR_PATH = '../spectrogram_based_model/train-data/exhale'
SILENCE_DIR_PATH = '../spectrogram_based_model/train-data/silence'

folder_paths = [INHALE_DIR_PATH, EXHALE_DIR_PATH, SILENCE_DIR_PATH]
'''
END OF GLOBAL VARIABLES
'''

# Vectors of embeddings to be classified by the model for every class

X_test_inhale = []
X_test_exhale = []
X_test_silence = []

for path in folder_paths:
    for filename in os.listdir(path):
        if filename.endswith('.wav'):
            # Full path to the sound file

            breathing_sound_file_path = os.path.join(path, filename)

            # Load the breathing sound as sound waves

            breathing_waveform = vggish_input.wavfile_to_examples(breathing_sound_file_path)

            with tf.Graph().as_default(), tf.compat.v1.Session() as sess:

                # Define VGGish

                embeddings = vggish_slim.define_vggish_slim()

                # Initialize all variables in the model, then load the VGGish checkpoint

                sess.run(tf.global_variables_initializer())
                vggish_slim.load_vggish_slim_checkpoint(sess, VGGISH_CHECKPOINT_PATH)

                # Get the input tensor

                features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)

                # Transform sound waves into log mel spectrogram and pass to the VGGish model to get embeddings

                try:
                    [embedding_batch] = np.array(
                        sess.run([embeddings], feed_dict={features_tensor: breathing_waveform}))
                except Exception as e:
                    print("Error:", e)
                    continue
                postprocessed_batch = pproc.postprocess(embedding_batch)
                if path == INHALE_DIR_PATH:
                    X_test_inhale.extend(postprocessed_batch)
                elif path == EXHALE_DIR_PATH:
                    X_test_exhale.extend(postprocessed_batch)
                else:
                    X_test_silence.extend(postprocessed_batch)

# Create confusion matrix


confusion_matrix = np.zeros((3, 3))

# Iterate through every embedding and classify it

for i, X_test in enumerate([X_test_inhale, X_test_exhale, X_test_silence]):
    for embedding in X_test:
        # Perform model prediction

        prediction = rf_classifier.predict(embedding.reshape(1, -1))

        # Update the confusion matrix

        confusion_matrix[i, prediction] += 1

# Afterward print the confusion matrix

print(f'\t\t\t\t\t\t\tPredicted class\tPredicted class\t Predicted class')
print(f'\t\t\t\t\t\t\tInhale\t\t\tExhale\t\t\tSilence')
print(f'Actual class\tInhale\t\t{confusion_matrix[0, 0]}\t\t\t\t{confusion_matrix[0, 1]}\t\t\t\t{confusion_matrix[0, 2]}')
print(f'Actual class\tExhale\t\t{confusion_matrix[1, 0]}\t\t\t\t{confusion_matrix[1, 1]}\t\t\t\t{confusion_matrix[1, 2]}')
print(f'Actual class\tSilence\t\t{confusion_matrix[2, 0]}\t\t\t\t{confusion_matrix[2, 1]}\t\t\t\t{confusion_matrix[2, 2]}')
