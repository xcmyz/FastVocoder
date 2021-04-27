import argparse
import os
import numpy as np
import data.audio as audio

from data.utils import pad
from tqdm import tqdm
from functools import partial
from concurrent.futures import ProcessPoolExecutor

CPU_NUM = 8
MULTI_PROCESS = True


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


def kernel(wav_filepath, mel_filepath, new_wav_filepath):
    try:
        y = audio.load_wav(wav_filepath, encode=False)
        mel = audio.melspectrogram(y)
        np.save(mel_filepath, mel)
        np.save(new_wav_filepath, y)
    except Exception:
        pass


def preprocess_multiprocessing(data_path_file, save_path):
    executor = ProcessPoolExecutor(max_workers=CPU_NUM)
    futures = []
    os.makedirs(save_path, exist_ok=True)
    audio_index = []
    mel_index = []

    with open(data_path_file, "r") as f:
        lines = f.readlines()
        for i in tqdm(range(len(lines))):
            filename = lines[i]
            wav_filepath = os.path.join(filename[:-1])
            wav_filename = filename[:-1].split("/")[-1]
            mel_filepath = os.path.join(save_path, f"{wav_filename}.mel.npy")
            new_wav_filepath = os.path.join(save_path, f"{wav_filename}.npy")
            audio_index.append(new_wav_filepath)
            mel_index.append(mel_filepath)
            futures.append(executor.submit(partial(kernel, wav_filepath, mel_filepath, new_wav_filepath)))
    [future.result() for future in tqdm(futures)]
    return audio_index, mel_index


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=os.path.join("dataset", "ljspeech.txt"))
    parser.add_argument('--save_path', type=str, default=os.path.join("dataset", "processed"))
    parser.add_argument('--audio_index_path', type=str, default=os.path.join("dataset", "audio_index_path.txt"))
    parser.add_argument('--mel_index_path', type=str, default=os.path.join("dataset", "mel_index_path.txt"))
    args = parser.parse_args()
    audio_index, mel_index = 0, 0
    if MULTI_PROCESS:
        audio_index, mel_index = preprocess_multiprocessing(args.data_path, args.save_path)
    else:
        audio_index, mel_index = preprocess(args.data_path, args.save_path)
    with open(args.audio_index_path, "w", encoding="utf-8") as f:
        for path in audio_index:
            f.write(path + "\n")

    with open(args.mel_index_path, "w", encoding="utf-8") as f:
        for path in mel_index:
            f.write(path + "\n")
