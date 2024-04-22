import numpy as np
import tensorflow.compat.v1 as tf
from model import vggish_input, vggish_params, vggish_slim
import matplotlib.pyplot as plt

# Path to the sound file with the breathing recording
breathing_sound_file_path = 'breaths/MLY9E8W-breathing.mp3'

# Path to the VGGish checkpoint
vggish_checkpoint_path = 'model/vggish_model.ckpt'

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

    print(embedding_batch)
    print(len(embedding_batch))
    plt.figure(figsize=(10, 4))
    plt.imshow(embedding_batch, aspect='auto', cmap='jet')
    plt.title('VGGish Embeddings')
    plt.colorbar(format='%+2.0f dB')
    plt.tight_layout()

    plt.show()

# Now the embeddings (embedding_batch) can be used for classification
