import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import time
import pandas as pd
from vggish_based_model.model import vggish_postprocess, vggish_params, vggish_slim, vggish_input
from df.enhance import init_df, enhance, load_audio, save_audio
import joblib
import tensorflow.compat.v1 as tf
import wave
import subprocess

"""

INSTRUCTION

Firtly start the program. In console it will write all possible input devices. Change 
DEVICE_INDEX constant to the microphone's index that you want to use. You can also change sample rate constant,
but it is advisable to leave it as it is (44100 Hz). Then close program (press spacebar) and run it again 
with changed constants.

If you don't want to calibrate microphone, you want to do this manually or you are
using microphone plugged to USB port you can set CALIBRATE_MICROPHONE to False.

Calibration works only for input devices connected to minijack port or built-in in laptop.
Program will calibrate device that is set as 'sysdefault'

Before starting program, please set your microphone volume to max manually and don't breathe
untill message on program window stop showing 'Dont breathe! Calibrating microphone...'.

If program will classify silence as other classes it is probably because microphone sensitivity is not
set correctly. Try running program again to calibrate it again or try to adjust sensitivity manually.

You can press 'r' to reset inhale and exhale counters.

Have fun!

"""

# Constants

CALIBRATE_MICROPHONE = True
SILENCES_IN_ROW_TO_END_CALIBRATION = 5

DUPLICATE = 4  # 1 = 1s refresh time, 2 = 0.5s refresh time, 4 = 0.25 refresh time, etc.
REFRESH_TIME = 1/DUPLICATE

INHALE_COUNTER = 0
EXHALE_COUNTER = 0
SAME_CLASS_IN_ROW_COUNTER = 0
CLASSIFIES_IN_ROW_TO_COUNT = 2  # How many same classifies in row to count it as a real one
PREVIOUS_CLASSIFIED_CLASS = 2  # 0 - Inhale, 1 - Exhale

CHANNELS = 2
RATE = 48000
DEVICE_INDEX = 4

running = True

# Load the models

vggish_checkpoint_path = 'model/vggish_model.ckpt'
CLASS_MODEL_PATH = 'model/trained_model_rf.pkl'
VGGISH_PARAMS_PATH = 'model/vggish_pca_params.npz'

pproc = vggish_postprocess.Postprocessor(VGGISH_PARAMS_PATH)
model, df_state, _ = init_df()

noise_reduction = 10  # Noise reduction in dB


# Audio resource class

class SharedAudioResource:
    buffer = None

    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.buffer_size = int(RATE * REFRESH_TIME)
        self.buffer = np.zeros(self.buffer_size, dtype=np.int16)
        for i in range(self.p.get_device_count()):
            print(self.p.get_device_info_by_index(i))
        self.stream = self.p.open(format=pyaudio.paInt16, channels=CHANNELS, rate=RATE, input=True,
                                  frames_per_buffer=self.buffer_size, input_device_index=DEVICE_INDEX)

    def read(self):
        self.buffer = self.stream.read(self.buffer_size, exception_on_overflow=False)
        return self.buffer

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


# Plot variables

PLOT_TIME_HISTORY = 5
PLOT_CHUNK_SIZE = int(RATE*REFRESH_TIME)

plotdata = np.zeros((RATE*PLOT_TIME_HISTORY, 1))
x_linspace = np.arange(0, RATE*PLOT_TIME_HISTORY, 1)
predictions = np.zeros((int(PLOT_TIME_HISTORY/REFRESH_TIME), 1))

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(plotdata, color='white')


# Key handler for plot window

def on_key(event):
    global running
    if event.key == ' ':
        plt.close()
        running = False
    elif event.key == 'r':
        global INHALE_COUNTER, EXHALE_COUNTER
        INHALE_COUNTER = 0
        EXHALE_COUNTER = 0


# Configuration of plot properties and other elements

fig.canvas.manager.set_window_title('Realtime Breath Detector ( Press [SPACE] to stop, [R] to reset counter )')  # Title
fig.suptitle(f'Inhales: {INHALE_COUNTER}  Exhales: {EXHALE_COUNTER}        Colours meaning: Red - Inhale, Green - Exhale, Blue - Silence')  # Instruction
fig.canvas.mpl_connect('key_press_event', on_key)  # Key handler

ylim = (-500, 500)
facecolor = (0, 0, 0)

ax.set_facecolor(facecolor)
ax.set_ylim(ylim)


# Plot update function

def update_plot(frames, prediction, is_calibrating=False):
    global plotdata, predictions, ax

    # Roll signals and predictions vectors and insert new value at the end

    plotdata = np.roll(plotdata, -len(frames))
    plotdata[-len(frames):] = frames.reshape(-1, 1)

    predictions = np.roll(predictions, -1)
    predictions[-1] = prediction

    # Clean the plot and plot the new data

    ax.clear()

    for i in range(0, len(predictions)):
        if predictions[i] == 0:  # Inhale
            color = 'red'
        elif predictions[i] == 1:  # Exhale
            color = 'green'
        else:  # Silence
            color = 'blue'
        ax.plot(x_linspace[PLOT_CHUNK_SIZE*i:PLOT_CHUNK_SIZE*(i+1)], plotdata[PLOT_CHUNK_SIZE*i:PLOT_CHUNK_SIZE*(i+1)], color=color)

    # Set plot properties and show it

    ax.set_facecolor(facecolor)
    ax.set_ylim(ylim)

    if is_calibrating:
        fig.suptitle(f'Dont breathe! Calibrating microphone...')  # Instruction
    else:
        fig.suptitle(
            f'Inhales: {INHALE_COUNTER}  Exhales: {EXHALE_COUNTER}        Colours meaning: Red - Inhale, Green - Exhale, Blue - Silence')  # Instruction

    plt.draw()
    plt.pause(0.01)


# Main function

if __name__ == "__main__":

    # Initialize microphone

    audio = SharedAudioResource()

    # Initialize RandomForest

    rf_classifier = joblib.load(CLASS_MODEL_PATH)

    with tf.Graph().as_default(), tf.Session() as sess:
        # Define VGGish

        embeddings = vggish_slim.define_vggish_slim()

        # Initialize all variables in the model, then load the VGGish checkpoint

        sess.run(tf.global_variables_initializer())
        vggish_slim.load_vggish_slim_checkpoint(sess, vggish_checkpoint_path)

        # Get the input tensor

        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)

        if CALIBRATE_MICROPHONE:
            # Microphone calibration

            silences_in_row = 0

            # Decrease volume untill we get 5 silences in a row

            while silences_in_row < SILENCES_IN_ROW_TO_END_CALIBRATION and running:

                # Read audio

                frames = audio.read()

                if frames is None:
                    continue

                # Duplicating samples, to extend recording to 1s (VGGish model needs 1 second of sound)

                buffer = [frames]

                for i in range(DUPLICATE-1):
                    buffer.append(frames)

                # Create wav file for VGGish

                wf = wave.open("temp.wav", 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(audio.p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(buffer))
                wf.close()
                audio1, _ = load_audio("temp.wav", sr=df_state.sr())

                # VGGish feature extraction

                input_batch = vggish_input.wavfile_to_examples("temp.wav")

                embedding_batch = np.array(sess.run(embeddings, feed_dict={features_tensor: input_batch}))

                postprocessed_batch = pproc.postprocess(embedding_batch)

                df = pd.DataFrame(postprocessed_batch)  # 128 features vector

                # Random Forest prediction from VGGish embeddings

                prediction = rf_classifier.predict(df)

                # Update plot with is_calibrating flag on

                update_plot((np.frombuffer(frames, dtype=np.int16))[::2], prediction, True)

                if prediction == 2:
                    silences_in_row += 1
                else:
                    silences_in_row = 0

                    # Decrease microphone volume by 5%

                    subprocess.run(["amixer", "sset", "Capture", "5%-"])

        # Main loop

        while running:
            # Set timer to check how long each prediction takes

            start_time = time.time()

            # Collect samples

            frames = audio.read()

            # Duplicating samples, to extend recording to 1s (VGGish model needs 1 second of sound)

            buffer = [frames]

            for i in range(DUPLICATE - 1):
                buffer.append(frames)

            # Noice reduce

            wf = wave.open("temp.wav", 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(buffer))
            wf.close()
            audio1, _ = load_audio("temp.wav", sr=df_state.sr())

            # VGGish feature extraction

            input_batch = vggish_input.wavfile_to_examples("temp.wav")

            embedding_batch = np.array(sess.run(embeddings, feed_dict={features_tensor: input_batch}))

            postprocessed_batch = pproc.postprocess(embedding_batch)

            df = pd.DataFrame(postprocessed_batch)  # 128 features vector

            # Random Forest prediction from VGGish embeddings

            prediction = rf_classifier.predict(df)

            # Increase same class classififications in row

            if prediction != PREVIOUS_CLASSIFIED_CLASS:
                SAME_CLASS_IN_ROW_COUNTER = 0
            else:
                SAME_CLASS_IN_ROW_COUNTER += 1

            # If we classified enough same classes in row, we can count it as a real one

            if SAME_CLASS_IN_ROW_COUNTER == CLASSIFIES_IN_ROW_TO_COUNT:
                if prediction == 0:
                    INHALE_COUNTER += 1
                elif prediction == 1:
                    EXHALE_COUNTER += 1

            # Update previous classified class

            PREVIOUS_CLASSIFIED_CLASS = prediction

            # Update plot

            plot_frames = np.frombuffer(frames, dtype=np.int16)

            update_plot(plot_frames[::2], prediction[0], False)

            # Print time needed for this loop iteration

            print(time.time() - start_time)

    # Close audio

    audio.close()
