import time
import os
import numpy as np
import tensorflow.compat.v1 as tf
from model import vggish_input, vggish_params, vggish_slim
import matplotlib.pyplot as plt

# Path to the sound file with the breathing recording
breaths_dir_path = 'breaths'
start_time = time.time()
# Path to the VGGish checkpoint
vggish_checkpoint_path = 'model/vggish_model.ckpt'


# List to store the embeddings from all files
all_embeddings = []
for filename in os.listdir(breaths_dir_path):
    if filename.endswith('.wav'):
        one_time = time.time()
        # Full path to the sound file
        breathing_sound_file_path = os.path.join(breaths_dir_path, filename)

        # Load the breathing sound as sound waves
        breathing_waveform = vggish_input.wavfile_to_examples(breathing_sound_file_path)
        with tf.Graph().as_default(), tf.Session() as sess:
            # Define VGGish
            embeddings = vggish_slim.define_vggish_slim()

            # Initialize all variables in the model, then load the VGGish checkpoint
            sess.run(tf.global_variables_initializer())
            vggish_slim.load_vggish_slim_checkpoint(sess, vggish_checkpoint_path)

            # Get the input tensor
            features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)

            # Transform sound waves into log mel spectrogram and pass to the VGGish model to get embeddings
            [embedding_batch] = np.array(sess.run([embeddings], feed_dict={features_tensor: breathing_waveform}))
            all_embeddings.append(embedding_batch)
            print("Size", len(embedding_batch))
            print("Time", time.time() - one_time)


print("End time:", time.time() - start_time)


# Now the embeddings (embedding_batch) can be used for classification
