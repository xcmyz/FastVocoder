import torch
import torch.nn as nn

import os
import re
import time
import audio
import ctypes
import random
import shutil
import argparse
import audio_tool

import numpy as np
import hparams as hp

from model_melgan import MelGANGenerator
from evaluation import synthesize, get_DNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def add_noise(wav, quantization_channel):
    noise = torch.empty_like(wav.float()).normal_(0, 0.50 / quantization_channel)
    return noise


def trans_weight(model, weight):
    with torch.no_grad():
        source = model.test_basis_signal(weight)
    return source[0].cpu().numpy()


if __name__ == "__main__":
    # Test
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=0)
    parser.add_argument('--weightpath', type=str, default="")
    parser.add_argument('--wavpath', type=str, default="")
    parser.add_argument('--melpath', type=str, default="")
    args = parser.parse_args()

    model = get_DNN(step=args.step)
    if args.weightpath != "":
        weight = np.load(args.weightpath)
        weight = torch.stack([torch.from_numpy(weight.T)]).float().to(device)
        wav = trans_weight(model, weight)
        audio.save_wav(wav, os.path.join("test_vocoder.wav"))
    elif args.wavpath != "":
        wav = torch.from_numpy(audio.load_wav(args.wavpath)).float()
        noi = add_noise(wav, quantization_channel=int(np.sqrt(2 ** 8)))
        mix = wav + noi
        audio.save_wav(mix.numpy(), "mix.wav")
        mel = audio_tool.tools.get_mel("mix.wav").numpy().astype(np.float32).T
        wav_ = synthesize(model, mel)
        audio.save_wav(wav_, os.path.join("test_vocoder.wav"))
    elif args.melpath != "":
        mel = np.load(args.melpath)
        wav = synthesize(model, mel)
        audio.save_wav(wav, os.path.join("test_vocoder.wav"))
