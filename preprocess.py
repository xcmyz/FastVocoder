import os
import numpy as np
import data.audio as audio

from tqdm import tqdm


def preprocess(data_path_file, save_path):
    os.makedirs(save_path, exist_ok=True)
    audio_index = []
    mel_index = []
    with open(data_path_file, "r") as f:
        lines = f.readlines()
        for i in tqdm(range(len(lines))):
            filename = lines[i]
            try:
                wav_filepath = os.path.join(filename[:-1])
                wav_filename = filename[:-1].split("/")[-1]
                mel_filepath = os.path.join(save_path, f"{wav_filename}.mel.npy")
                y = audio.load_wav(wav_filepath, encode=False)
                new_wav_filepath = os.path.join(save_path, f"{wav_filename}.npy")
                mel = audio.melspectrogram(y)
                np.save(mel_filepath, mel)
                np.save(new_wav_filepath, y)
                audio_index.append(new_wav_filepath)
                mel_index.append(mel_filepath)
            except Exception as e:
                print(f"ERROR: {str(e)}")
    return audio_index, mel_index


if __name__ == "__main__":
    preprocess(os.path.join("dataset", "ljspeech.txt"), os.path.join("dataset", "processed"))
