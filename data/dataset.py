import os
import time
import torch
import pickle
import random
import hparams
import numpy as np

from tqdm import tqdm
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from data.utils import pad_1D_tensor, pad_2D_tensor, parse_path_file

random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data_to_buffer(audio_index_path_file, mel_index_path_file, logger, feature_savepath="features.bin"):
    if os.path.exists(feature_savepath):
        logger.info("Loading from bin...")
        with open(feature_savepath, "rb") as f:
            buffer = pickle.load(f)
        return buffer

    logger.info("Firstly loading...")
    audio_index = parse_path_file(audio_index_path_file)
    mel_index = parse_path_file(mel_index_path_file)
    buffer = []
    start = time.perf_counter()
    min_length = 1e10
    assert len(audio_index) == len(mel_index)
    length_dataset = len(audio_index)
    if hparams.test_size != 0 and hparams.test_size < length_dataset:
        length_dataset = hparams.test_size
    for i in tqdm(range(length_dataset)):
        # assert mel_index[i].split("/")[-1].split(".")[0] == audio_index[i].split("/")[-1].split(".")[0]  # check ljspeech
        mel = np.load(mel_index[i]).T
        mel = torch.from_numpy(mel)
        wav = np.load(audio_index[i])
        wav = torch.from_numpy(wav)
        if mel.size(0) < min_length:
            min_length = mel.size(0)
        buffer.append({"mel": mel, "wav": wav})
    end = time.perf_counter()
    logger.info(f"Cost {int(end-start)}s loading all data into buffer.")
    logger.info(f"Min length of mel spec in dataset is {min_length}.")

    with open(feature_savepath, "wb") as f:
        pickle.dump(buffer, f)

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


class WeightDataset(Dataset):
    def __init__(self, audio_index_file_path, mel_index_file_path, L):
        self.audio_index = parse_path_file(audio_index_file_path)
        self.mel_index = parse_path_file(mel_index_file_path)
        assert (len(self.audio_index) == len(self.mel_index))
        self.L = L

    def __len__(self):
        return len(self.audio_index)

    def __getitem__(self, idx):
        mel = np.load(self.mel_index[idx]).T
        mel = torch.from_numpy(mel)
        wav = np.load(self.audio_index[idx])
        wav = torch.from_numpy(wav)
        weight_path = os.path.join("Basis-MelGAN-dataset", "weight", self.audio_index[idx].split("/")[-1])
        weight = np.load(weight_path)
        weight = torch.from_numpy(weight)
        data = {"mel": mel, "wav": wav, "weight": weight}

        len_data = data["mel"].size(0)
        start_index = random.randint(0, len_data - hparams.fixed_length - 1)
        end_index = start_index + hparams.fixed_length
        weight_start_index = start_index * (self.L // 2)
        weight_end_index = end_index * (self.L // 2)
        wav_start_index = start_index * hparams.hop_size
        wav_end_index = end_index * hparams.hop_size
        buffer_cut = {
            "mel": data["mel"][start_index:end_index, :],
            "wav": data["wav"][wav_start_index:wav_end_index],
            "weight": data["weight"][weight_start_index:weight_end_index, :]
        }
        return buffer_cut


def reprocess_tensor(batch, cut_list):
    mels = [batch[i]["mel"] for i in cut_list]
    wavs = [batch[i]["wav"] for i in cut_list]
    wavs = pad_1D_tensor(wavs)
    mels = pad_2D_tensor(mels)

    if "weight" in batch[0]:
        weights = [batch[i]["weight"] for i in cut_list]
        weights = pad_2D_tensor(weights)
        return {"mel": mels, "wav": wavs, "weight": weights}
    else:
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


def collate_fn_tensor_valid(batch):
    return reprocess_tensor(batch, [0])
