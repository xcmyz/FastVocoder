import os
import math
import time
import audio
import torch
import random
import hparams
import numpy as np

from tqdm import tqdm
from model_melgan import MelGANGenerator
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from utils import process_text, pad_1D, pad_2D
from utils import pad_1D_tensor, pad_2D_tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_data_to_buffer():
    buffer = list()
    text = process_text(os.path.join("data", "train.txt"))
    basis_signal_weight = np.load(os.path.join("data", "basis_signal_weight.npy"))
    basis_signal_weight = torch.from_numpy(basis_signal_weight)
    model = MelGANGenerator(basis_signal_weight)

    start = time.perf_counter()
    min_length = 1000000000
    length_dataset = len(text)
    if hparams.test_size != 0:
        length_dataset = hparams.test_size
    for i in tqdm(range(length_dataset)):
        mel = np.load(os.path.join(hparams.mel_ground_truth, "ljspeech-mel-%05d.npy" % (i+1)))
        mel = torch.from_numpy(mel)
        weight = np.load(os.path.join("weight", str(i)+".npy"))
        weight = torch.from_numpy(weight).transpose(0, 1)

        with torch.no_grad():
            wav = model.test_basis_signal(weight.unsqueeze(0))
        wav = wav.detach().numpy()[0]
        wav = torch.from_numpy(wav)

        if mel.size(0) < min_length:
            min_length = mel.size(0)
        buffer.append({"weight": weight, "mel": mel, "wav": wav})

    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end-start))
    print("min length of dataset is {:d}.".format(min_length))

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
        weight_start_index = start_index * hparams.expand_size
        weight_end_index = end_index * hparams.expand_size
        wav_start_index = weight_start_index * (hparams.L // 2)
        wav_end_index = weight_end_index * (hparams.L // 2)
        buffer_cut = {
            "weight": data["weight"][weight_start_index:weight_end_index, :],
            "mel": data["mel"][start_index:end_index, :],
            "wav": data["wav"][wav_start_index:wav_end_index]
        }

        return buffer_cut


def reprocess_tensor(batch, cut_list):
    mels = [batch[i]["mel"] for i in cut_list]
    weights = [batch[i]["weight"] for i in cut_list]
    wavs = [batch[i]["wav"] for i in cut_list]

    length_weight = []
    for weight in weights:
        length_weight.append(weight.size(0))
    length_weight = torch.Tensor(length_weight)

    wavs = pad_1D_tensor(wavs)
    mels = pad_2D_tensor(mels)
    weights = pad_2D_tensor(weights)

    return {"mel": mels, "weight": weights, "wav": wavs, "length": length_weight}


def collate_fn_tensor(batch):
    len_arr = np.array([d["mel"].size(0) for d in batch])
    index_arr = np.argsort(-len_arr)
    batchsize = len(batch)
    real_batchsize = batchsize // hparams.batch_expand_size

    cut_list = list()
    for i in range(hparams.batch_expand_size):
        cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])

    output = list()
    for i in range(hparams.batch_expand_size):
        output.append(reprocess_tensor(batch, cut_list[i]))

    return output


if __name__ == "__main__":
    # TEST
    get_data_to_buffer()
