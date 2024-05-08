import time

from df.enhance import enhance, init_df, load_audio, save_audio
from df.utils import download_file

if __name__ == "__main__":
    # Load default model
    model, df_state, _ = init_df()
    parameters = model.state_dict()

    # Download and open some audio file. You use your audio files here
    audio_path = "moje.wav"
    audio, _ = load_audio(audio_path, sr=df_state.sr())
    # Denoise the audio
    start_time = time.time()
    enhanced = enhance(model, df_state, audio, atten_lim_db=25.0)
    print("Time:", time.time() - start_time)
    # Save for listening
    save_audio("enhanced.wav", enhanced, df_state.sr())