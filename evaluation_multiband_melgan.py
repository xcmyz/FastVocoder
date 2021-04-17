import torch
import torch.nn as nn

import os
import re
import time
import h5py
import utils
import audio
import ctypes
import random
import shutil
import librosa
import argparse

import numpy as np
import hparams as hp

from pqmf import PQMF
from model_multiband_melgan import MelGANGenerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_wav(path):
    audio = librosa.core.load(path, sr=hp.sample_rate)[0]
    audio = audio / max(0.01, np.max(np.abs(audio)))
    return audio


def logmelfilterbank(audio,
                     sampling_rate=22050,
                     fft_size=1024,
                     hop_size=256,
                     win_length=None,
                     window="hann",
                     num_mels=80,
                     fmin=80,
                     fmax=7600,
                     eps=1e-10,
                     ):
    """Compute log-Mel filterbank feature.

    Args:
        audio (ndarray): Audio signal (T,).
        sampling_rate (int): Sampling rate.
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length. If set to None, it will be the same as fft_size.
        window (str): Window function type.
        num_mels (int): Number of mel basis.
        fmin (int): Minimum frequency in mel basis calculation.
        fmax (int): Maximum frequency in mel basis calculation.
        eps (float): Epsilon value to avoid inf in log calculation.

    Returns:
        ndarray: Log Mel filterbank feature (#frames, num_mels).

    """
    # get amplitude spectrogram
    x_stft = librosa.stft(audio, n_fft=fft_size, hop_length=hop_size,
                          win_length=win_length, window=window, pad_mode="reflect")
    spc = np.abs(x_stft).T  # (#frames, #bins)

    # get mel basis
    fmin = 0 if fmin is None else fmin
    fmax = sampling_rate / 2 if fmax is None else fmax
    mel_basis = librosa.filters.mel(sampling_rate, fft_size, num_mels, fmin, fmax)

    return np.log10(np.maximum(eps, np.dot(spc, mel_basis.T)))


def get_DNN():
    checkpoint_path = os.path.join(hp.checkpoint_path, "multiband_melgan")
    model = MelGANGenerator(out_channels=4, channels=384, upsample_scales=[8, 4, 2], stacks=4).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model']['generator'])
    model.pqmf = PQMF().to(device)
    model.eval()
    model.remove_weight_norm()
    return model


def get_file_list():
    mel_file_list = [os.path.join("mels.test_mb", filename) for filename in os.listdir(os.path.join("mels.test_mb"))]
    return mel_file_list


def synthesize(model, mel):
    mel = torch.stack([torch.from_numpy(mel.T)]).float().to(device)
    with torch.no_grad():
        source_vocoder = model.inference(mel)
    return source_vocoder[0][0].cpu().numpy()


def test_rtf(model, mel):
    with torch.no_grad():
        model.inference(mel)


def get_wav_duration(path):
    wav = audio.load_wav(path)
    wav_len = len(wav) / hp.sample_rate
    return wav_len


if __name__ == "__main__":
    # Test
    model = get_DNN()
    print("number of parameter:", utils.get_param_num(model))
    mel_file_list = get_file_list()
    save_path = os.path.join("result", "multiband_melgan")
    os.makedirs(save_path, exist_ok=True)

    all_duration_time = 0.
    s = time.perf_counter()
    for i, mel_file_name in enumerate(mel_file_list):
        mel = np.load(mel_file_name)
        wav_vocoder = synthesize(model, mel)
        save_file_name = os.path.join(save_path, str(i) + "_mb.wav")
        save_file_name = os.path.join(save_path, str(i) + ".wav")
        audio.save_wav(wav_vocoder, save_file_name)
        all_duration_time += get_wav_duration(save_file_name)
        print("Done", i)
    e = time.perf_counter()
    print("use time: {:2f}".format(e - s))

    mels = []
    lengths = []
    device_for_rtf = torch.device("cpu")
    model = model.to(device_for_rtf)
    for i, mel_file_name in enumerate(mel_file_list):
        mel = np.load(mel_file_name)
        length = torch.Tensor([mel.shape[0] * hp.expand_size]).long().to(device_for_rtf)
        mel = torch.stack([torch.from_numpy(mel.T)]).float().to(device_for_rtf)
        lengths.append(length)
        mels.append(mel)

    start_time = time.perf_counter()
    round_time = 10
    for cnt in range(round_time):
        for i in range(len(mels)):
            test_rtf(model, mels[i])
        print(cnt)
    end_time = time.perf_counter()

    print("RTF:", (end_time - start_time) / (all_duration_time * float(round_time)))
