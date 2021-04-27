import torch
import os
import argparse
import numpy as np
import hparams as hp

from data.utils import parse_path_file
from model.generator.melgan import MelGANGenerator
from synthesize import Synthesizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_data(audio_index_path, mel_index_path, index_list):
    audio_index = parse_path_file(audio_index_path)
    mel_index = parse_path_file(mel_index_path)
    audio_list = []
    mel_list = []
    for index in index_list:
        audio_list.append(np.load(audio_index[index]))
        mel_list.append(torch.from_numpy(np.load(mel_index[index])))
    return audio_list, mel_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--audio_index_path', type=str, default=os.path.join("dataset", "audio", "eval"))
    parser.add_argument('--mel_index_path', type=str, default=os.path.join("dataset", "mel", "eval"))
    args = parser.parse_args()

    synthesizer = Synthesizer(args.checkpoint_path)
    audio_list, mel_list = load_data(args.audio_index_path, args.mel_index_path, [0, 1, 2, 3, 4, 5])
