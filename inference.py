import torch
import torch.nn as nn

import os
import re
import time
import utils
import audio
import ctypes
import random
import shutil
import argparse

import numpy as np
import hparams as hp

from discriminator import PDiscriminator
from discriminator import PMDiscriminator
from discriminator import Discriminator as WPDiscriminator
from discriminator import MelDiscriminator as MDiscriminator
from discriminator import WavDiscriminator as WDiscriminator
from discriminator import NewDiscriminator as WMDiscriminator

from model_melgan import MelGANGenerator as MelGANGenerator_light
from model_melgan_large import MelGANGenerator as MelGANGenerator_large

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_DNN(step=0, checkpoint='', discriminator_type='WM', generator_type="light"):
    checkpoint_path = ""
    if step != 0:
        checkpoint_path = "checkpoint_" + str(step) + ".pth.tar"
    else:
        checkpoint_path = checkpoint
    basis_signal_weight = np.load(os.path.join("data", "basis_signal_weight.npy"))
    basis_signal_weight = torch.from_numpy(basis_signal_weight)
    model = 0
    if generator_type == "light":
        model = MelGANGenerator_light(basis_signal_weight).to(device)
    elif generator_type == "large":
        model = MelGANGenerator_large(basis_signal_weight).to(device)
    model.load_state_dict(torch.load(os.path.join(hp.checkpoint_path, checkpoint_path), map_location=torch.device(device))['model'])

    checkpoint = torch.load(os.path.join(hp.checkpoint_path, checkpoint_path), map_location=torch.device(device))
    discriminator = 0
    if discriminator_type == "WP":
        discriminator = WPDiscriminator().to(device)
    elif discriminator_type == "WM":
        discriminator = WMDiscriminator().to(device)
    elif discriminator_type == "M":
        discriminator = MDiscriminator().to(device)
    elif discriminator_type == "W":
        discriminator = WDiscriminator().to(device)
    elif discriminator_type == "P":
        discriminator = PDiscriminator().to(device)
    elif discriminator_type == "PM":
        discriminator = PMDiscriminator().to(device)

    discriminator.load_state_dict(checkpoint['discriminator'])
    optimizer = torch.optim.Adam(model.melgan.parameters(), lr=1e-4, eps=1.0e-6, weight_decay=0.0)
    optimizer.load_state_dict(checkpoint['optimizer'])
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=5e-5, eps=1.0e-6, weight_decay=0.0)
    discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])
    os.makedirs("checkpoint", exist_ok=True)
    torch.save(
        {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'discriminator': discriminator.state_dict(),
            'discriminator_optimizer': discriminator_optimizer.state_dict()
        },
        os.path.join("checkpoint", f"checkpoint_{step}.pth.tar"),
        _use_new_zipfile_serialization=False)

    model.eval()
    model.remove_weight_norm()
    return model


def get_file_list():
    mel_file_list = [os.path.join("mels.test", filename) for filename in os.listdir(os.path.join("mels.test"))]
    return mel_file_list


def synthesize(model, mel):
    mel = torch.stack([torch.from_numpy(mel.T)]).float().to(device)
    with torch.no_grad():
        source_vocoder, _ = model(mel)
    return source_vocoder[0].cpu().numpy()


def test_rtf(model, mel):
    with torch.no_grad():
        model(mel)


def get_wav_duration(path):
    wav = audio.load_wav(path)
    wav_len = len(wav) / hp.sample_rate
    return wav_len


if __name__ == "__main__":
    # Test
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, default='')
    parser.add_argument('--savepath', type=str, default='')
    parser.add_argument('--discriminator', type=str, default='WM')
    parser.add_argument('--generator', type=str, default="light")
    parser.add_argument('--step', type=int, default=0)
    args = parser.parse_args()

    model = get_DNN(step=args.step, discriminator_type=args.discriminator, generator_type=args.generator)
    mel = np.load(args.filepath).T
    wav_vocoder = synthesize(model, mel)
    audio.save_wav(wav_vocoder, args.savepath)
