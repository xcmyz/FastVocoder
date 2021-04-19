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


class PreLoadDataset(Dataset):
    def __init__(self):
        text = process_text(os.path.join("data", "train.txt"))
        self.length_dataset = len(text)

        basis_signal_weight = np.load(os.path.join("data", "basis_signal_weight.npy"))
        basis_signal_weight = torch.from_numpy(basis_signal_weight)
        self.model = MelGANGenerator(basis_signal_weight)
        self.wav_buffer = {}
        self.mel_buffer = {}

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        mel = 0
        if idx in self.mel_buffer:
            mel = self.mel_buffer[idx]
        else:
            mel = np.load(os.path.join(hparams.mel_ground_truth, "ljspeech-mel-%05d.npy" % (idx+1)))
            mel = torch.from_numpy(mel)
            self.mel_buffer[idx] = mel

        wav = 0
        if idx in self.wav_buffer:
            wav = self.wav_buffer[idx]
        else:
            weight = np.load(os.path.join("weight", str(idx)+".npy"))
            weight = torch.from_numpy(weight).transpose(0, 1)
            with torch.no_grad():
                wav = self.model.test_basis_signal(weight.unsqueeze(0))
            wav = wav.detach().numpy()[0]
            wav = torch.from_numpy(wav)
            self.wav_buffer[idx] = wav

        data = {
            "weight": os.path.join("weight", str(idx)+".npy"),
            "mel": mel,
            "wav": wav
        }
        len_data = data["mel"].size(0)
        start_index = random.randint(0, len_data - hparams.fixed_length - 1)
        end_index = start_index + hparams.fixed_length
        weight_start_index = start_index * hparams.expand_size
        weight_end_index = end_index * hparams.expand_size
        wav_start_index = weight_start_index * (hparams.L // 2)
        wav_end_index = weight_end_index * (hparams.L // 2)
        buffer_cut = {
            "weight_start_index": weight_start_index,
            "weight_end_index": weight_end_index,
            "weight": data["weight"],
            "mel": data["mel"][start_index:end_index, :],
            "wav": data["wav"][wav_start_index:wav_end_index]
        }

        return buffer_cut


def reprocess_tensor(batch, cut_list):
    mels = [batch[i]["mel"] for i in cut_list]
    wavs = [batch[i]["wav"] for i in cut_list]

    weights = [batch[i]["weight"] for i in cut_list]
    weight_start_indexs = [batch[i]["weight_start_index"] for i in cut_list]
    weight_end_indexs = [batch[i]["weight_end_index"] for i in cut_list]

    wavs = pad_1D_tensor(wavs)
    mels = pad_2D_tensor(mels)

    return {"mel": mels, "wav": wavs, "weight": weights, "weight_start_index": weight_start_indexs, "weight_end_index": weight_end_indexs}


def collate_fn_tensor(batch):
    if False:
        len_arr = np.array([d["mel"].size(0) for d in batch])
        index_arr = np.argsort(-len_arr)
        batchsize = len(batch)
        real_batchsize = batchsize // hparams.batch_expand_size

        cut_list = list()
        for i in range(hparams.batch_expand_size):
            cut_list.append(index_arr[i * real_batchsize:(i + 1) * real_batchsize])

        output = list()
        for i in range(hparams.batch_expand_size):
            output.append(reprocess_tensor(batch, cut_list[i]))

        return output
    else:
        index_arr = [i for i in range(len(batch))]
        batchsize = len(batch)
        real_batchsize = batchsize // hparams.batch_expand_size

        cut_list = list()
        for i in range(hparams.batch_expand_size):
            cut_list.append(index_arr[i * real_batchsize:(i + 1) * real_batchsize])

        output = list()
        for i in range(hparams.batch_expand_size):
            output.append(reprocess_tensor(batch, cut_list[i]))

        return output
