import torch
import os
import argparse
import random
import yaml
import numpy as np
import hparams as hp

from data.audio import save_wav, inv_mel_spectrogram
from model.generator import MelGANGenerator
from model.generator import MultiBandHiFiGANGenerator
from model.generator import HiFiGANGenerator
from model.generator import BasisMelGANGenerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def publish_model(checkpoint_path, config_path, model_name, save_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    print(f"Loading Model of {model_name}...")
    if model_name == "melgan":
        model = MelGANGenerator(in_channels=config["in_channels"],
                                out_channels=config["out_channels"],
                                kernel_size=config["kernel_size"],
                                channels=config["channels"],
                                upsample_scales=config["upsample_scales"],
                                stack_kernel_size=config["stack_kernel_size"],
                                stacks=config["stacks"],
                                use_weight_norm=config["use_weight_norm"],
                                use_causal_conv=config["use_causal_conv"]).to(device)
    elif model_name == "hifigan":
        model = HiFiGANGenerator(resblock_kernel_sizes=config["resblock_kernel_sizes"],
                                 upsample_rates=config["upsample_rates"],
                                 upsample_initial_channel=config["upsample_initial_channel"],
                                 resblock_type=config["resblock_type"],
                                 upsample_kernel_sizes=config["upsample_kernel_sizes"],
                                 resblock_dilation_sizes=config["resblock_dilation_sizes"],
                                 transposedconv=config["transposedconv"],
                                 bias=config["bias"]).to(device)
    elif model_name == "multiband-hifigan":
        model = MultiBandHiFiGANGenerator(resblock_kernel_sizes=config["resblock_kernel_sizes"],
                                          upsample_rates=config["upsample_rates"],
                                          upsample_initial_channel=config["upsample_initial_channel"],
                                          resblock_type=config["resblock_type"],
                                          upsample_kernel_sizes=config["upsample_kernel_sizes"],
                                          resblock_dilation_sizes=config["resblock_dilation_sizes"],
                                          transposedconv=config["transposedconv"],
                                          bias=config["bias"]).to(device)
    elif model_name == "basis-melgan":
        basis_signal_weight = torch.zeros(config["L"], config["out_channels"]).float()
        model = BasisMelGANGenerator(basis_signal_weight=basis_signal_weight,
                                     L=config["L"],
                                     in_channels=config["in_channels"],
                                     out_channels=config["out_channels"],
                                     kernel_size=config["kernel_size"],
                                     channels=config["channels"],
                                     upsample_scales=config["upsample_scales"],
                                     stack_kernel_size=config["stack_kernel_size"],
                                     stacks=config["stacks"],
                                     use_weight_norm=config["use_weight_norm"],
                                     use_causal_conv=config["use_causal_conv"],
                                     transposedconv=config["transposedconv"]).to(device)
    else:
        raise Exception("no model find!")
    model.load_state_dict(torch.load(os.path.join(checkpoint_path), map_location=torch.device(device))['model'])
    if model_name == "basis-melgan":
        with torch.no_grad():
            bias = model.inference(torch.zeros(30000, 80))  # support up to synthesize 300s waveform
            pattern = bias.cpu().numpy()
            published_dict = {
                'model': model.state_dict(),
                'pattern': pattern
            }
            torch.save(published_dict, save_path)
    model.eval()
    model.remove_weight_norm()
    return


def run_publisher():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument("--model_name", type=str, help="melgan, hifigan and multiband-hifigan.")
    parser.add_argument("--config", type=str, help="path to model configuration file")
    parser.add_argument("--save_path", type=str, help="path to save published model")
    args = parser.parse_args()
    publish_model(args.checkpoint_path, args.config, args.model_name, args.save_path)
