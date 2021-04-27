import os
import time
import torch
import random
import hparams
import numpy as np

from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from data.utils import pad_1D_tensor, pad_2D_tensor, parse_path_file

random.seed(str(time.time()))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data_to_buffer(audio_index_path_file, mel_index_path_file):
    audio_index = parse_path_file(audio_index_path_file)
    mel_index = parse_path_file(mel_index_path_file)
    buffer = []
    start = time.perf_counter()
    min_length = 1e10
    assert len(audio_index) == len(mel_index)
    length_dataset = len(audio_index)
    if hparams.test_size != 0:
        length_dataset = hparams.test_size
    if hparams.test_size == 0 and hparams.train_size != 0:
        length_dataset = hparams.train_size
    for i in tqdm(range(length_dataset)):
        mel = np.load(mel_index[i])
        mel = torch.from_numpy(mel)
        wav = np.load(audio_index[i])
        wav = torch.from_numpy(wav)
        if mel.size(0) < min_length:
            min_length = mel.size(0)
        buffer.append({"mel": mel, "wav": wav})
    end = time.perf_counter()
    print(f"Cost {int(end-start)}s loading all data into buffer.")
    print("Min length of mel spec in dataset is {min_length}.")
    return buffer


class BufferDataset(Dataset):
    def __init__(self, buffer):
        self.buffer = buffer
        self.length_dataset = len(self.buffer)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        data = self.buffer[idx]
        len_data = data["mel"].size(0)
        start_index = random.randint(0, len_data - hparams.fixed_length - 1)
        end_index = start_index + hparams.fixed_length
        wav_start_index = start_index * hparams.hop_size
        wav_end_index = end_index * hparams.hop_size
        buffer_cut = {
            "mel": data["mel"][start_index:end_index, :],
            "wav": data["wav"][wav_start_index:wav_end_index]
        }
        return buffer_cut


def reprocess_tensor(batch, cut_list):
    mels = [batch[i]["mel"] for i in cut_list]
    wavs = [batch[i]["wav"] for i in cut_list]
    wavs = pad_1D_tensor(wavs)
    mels = pad_2D_tensor(mels)
    return {"mel": mels, "wav": wavs}


def collate_fn_tensor(batch):
    len_arr = np.array([d["mel"].size(0) for d in batch])
    index_arr = np.argsort(-len_arr)
    batchsize = len(batch)
    real_batchsize = batchsize // hparams.batch_expand_size
    cut_list = []
    for i in range(hparams.batch_expand_size):
        cut_list.append(index_arr[i * real_batchsize:(i+1) * real_batchsize])
    output = []
    for i in range(hparams.batch_expand_size):
        output.append(reprocess_tensor(batch, cut_list[i]))
    return output
