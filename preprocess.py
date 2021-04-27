import argparse
import os
import numpy as np
import data.audio as audio
import hparams as hp
import random

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


def write_file(audio_index, mel_index, index_list, file_name, audio_index_path, mel_index_path):
    with open(os.path.join(audio_index_path, file_name), "w", encoding="utf-8") as f:
        for index in index_list:
            f.write(audio_index[index] + "\n")
    with open(os.path.join(mel_index_path, file_name), "w", encoding="utf-8") as f:
        for index in index_list:
            f.write(mel_index[index] + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=os.path.join("dataset", "ljspeech.txt"))
    parser.add_argument('--save_path', type=str, default=os.path.join("dataset", "processed"))
    parser.add_argument('--audio_index_path', type=str, default=os.path.join("dataset", "audio"))
    parser.add_argument('--mel_index_path', type=str, default=os.path.join("dataset", "mel"))
    args = parser.parse_args()
    audio_index, mel_index = 0, 0
    if MULTI_PROCESS:
        audio_index, mel_index = preprocess_multiprocessing(args.data_path, args.save_path)
    else:
        audio_index, mel_index = preprocess(args.data_path, args.save_path)

    os.makedirs(args.audio_index_path)
    os.makedirs(args.mel_index_path)
    assert len(audio_index) >= (hp.train_size + hp.valid_size + hp.eval_size)
    index_list = [i for i in range(hp.train_size + hp.valid_size + hp.eval_size)]
    random.shuffle(index_list)
    index_list_train = index_list[0:hp.train_size]
    index_list_valid = index_list[hp.train_size:hp.train_size + hp.valid_size]
    index_list_eval = index_list[hp.train_size + hp.valid_size:hp.train_size + hp.valid_size + hp.eval_size]
    write_file(audio_index, mel_index, index_list_train, "train", args.audio_index_path, args.mel_index_path)
    write_file(audio_index, mel_index, index_list_valid, "valid", args.audio_index_path, args.mel_index_path)
    write_file(audio_index, mel_index, index_list_eval, "eval", args.audio_index_path, args.mel_index_path)
