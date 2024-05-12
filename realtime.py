import time
import wave
import joblib
import pygame
import pygame.freetype
import pyaudio
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import threading
import numpy as np
import tensorflow.compat.v1 as tf
from model import vggish_input, vggish_params, vggish_slim, vggish_postprocess
import pandas as pd
from df.enhance import enhance, init_df, load_audio, save_audio
from df.utils import download_file
# ###########################################################################################
# If there's an issue with the microphone, find the index of the microphone you want to use in the console,
# along with its sampleRate. Then, change the variable RATE below and add the parameter
# input_device_index=INDEX_OF_MICROPHONE
# to
# self.stream = self.p.open(..., input_device_index=INDEX_OF_MICROPHONE)
# ###########################################################################################
AUDIO_CHUNK = 1024
PLOT_CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 48000

vggish_checkpoint_path = 'model/vggish_model.ckpt'
CLASS_MODEL_PATH = 'model/trained_model_rf.pkl'
VGGISH_PARAMS_PATH = 'model/vggish_pca_params.npz'

pproc = vggish_postprocess.Postprocessor(VGGISH_PARAMS_PATH)
model, df_state, _ = init_df()


class SharedAudioResource:
    buffer = None

    def __init__(self):
        self.p = pyaudio.PyAudio()
        for i in range(self.p.get_device_count()):
            print(self.p.get_device_info_by_index(i))
        self.stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True,
                                  frames_per_buffer=AUDIO_CHUNK)
        self.read(AUDIO_CHUNK)

    def read(self, size):
        self.buffer = self.stream.read(size, exception_on_overflow=False)
        return self.buffer

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()


def draw_text(text, pos, font, screen):
    text_surface, _ = font.render(text, (255, 255, 255))
    screen.blit(text_surface, (pos[0] - text_surface.get_width() // 2, pos[1] - text_surface.get_height() // 2))


def pygame_thread(audio):
    pygame.init()
    WIDTH, HEIGHT = 1366, 768
    FONT_SIZE = 24
    TEXT_POS = (WIDTH // 2, HEIGHT // 2 - 200)
    TEST_POS = (WIDTH // 2, HEIGHT // 2 - 300)

    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.freetype.SysFont(None, FONT_SIZE)
    clock = pygame.time.Clock()

    running = True
    rf_classifier = joblib.load(CLASS_MODEL_PATH)
    with tf.Graph().as_default(), tf.Session() as sess:
        # Define VGGish
        embeddings = vggish_slim.define_vggish_slim()

        # Initialize all variables in the model, then load the VGGish checkpoint
        sess.run(tf.global_variables_initializer())
        vggish_slim.load_vggish_slim_checkpoint(sess, vggish_checkpoint_path)

        # Get the input tensor
        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        prediction_arr = []
        while running:
            screen.fill((0, 0, 0))
            draw_text("Press SPACE to stop", TEST_POS, font, screen)
            start_time = time.time()
            buffer = []
            for i in range(0, (RATE // AUDIO_CHUNK)):
                buffer.append(audio.read(AUDIO_CHUNK))

            wf = wave.open("temp/temp.wav", 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(audio.p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(buffer))
            wf.close()
            audio1, _ = load_audio("temp/temp.wav", sr=df_state.sr())
            enhanced = enhance(model, df_state, audio1, atten_lim_db=10.0)
            save_audio("temp/temp.wav", enhanced, df_state.sr())
            breathing_waveform = vggish_input.wavfile_to_examples("temp/temp.wav")

            embedding_batch = np.array(sess.run(embeddings, feed_dict={features_tensor: breathing_waveform}))
            postprocessed_batch = pproc.postprocess(embedding_batch)
            df = pd.DataFrame(postprocessed_batch)

            prediction = rf_classifier.predict(df)
            if len(prediction_arr) == 5:
                prediction_arr = []
            if prediction[0] == 0:
                prediction_arr.append('Inhale')
            elif prediction[0] == 1:
                prediction_arr.append('Exhale')
            else:
                prediction_arr.append('Silence')

            draw_text(f"Prediction {prediction_arr}", TEXT_POS, font, screen)
            print(time.time() - start_time)
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        print("Exiting")
                        running = False
            pygame.display.flip()
            clock.tick(60)

    pygame.quit()


def plot_audio(audio1):
    def animate(i):
        frames = audio1.buffer
        data = np.frombuffer(frames, dtype=np.int16)
        left_channel = data[::2]  # even index: left channel
        right_channel = data[1::2]  # odd index: right channel
        line1.set_ydata(left_channel)
        line2.set_ydata(right_channel)
        return line1, line2,

    fig, axs = plt.subplots(2)
    x = np.arange(0, 2 * PLOT_CHUNK, 2)
    line1, = axs[0].plot(x, np.random.rand(PLOT_CHUNK))
    line2, = axs[1].plot(x, np.random.rand(PLOT_CHUNK))

    axs[0].set_ylim(-1000, 1000)
    axs[0].set_xlim(0, PLOT_CHUNK / 2)
    axs[1].set_ylim(-1000, 1000)
    axs[1].set_xlim(0, PLOT_CHUNK / 2)
    ani = animation.FuncAnimation(fig, animate, frames=100, blit=True)
    plt.show()


if __name__ == "__main__":
    audio = SharedAudioResource()
    pygame_thread_instance = threading.Thread(target=pygame_thread, args=(audio,))
    pygame_thread_instance.start()
    plot_audio(audio)
    pygame_thread_instance.join()
    audio.close()
