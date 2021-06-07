import torch
import os
import argparse
import yaml
import numpy as np
import hparams as hp

from data.audio import save_wav, inv_mel_spectrogram
from model.generator import MelGANGenerator
from model.generator import MultiBandHiFiGANGenerator
from model.generator import HiFiGANGenerator
from model.generator import BasisMelGANGenerator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Synthesizer:
    def __init__(self, checkpoint_path, config_path, model_name) -> None:
        self.model = self.load_model(checkpoint_path, config_path, model_name)

    def load_model(self, checkpoint_path, config_path, model_name):
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
            model = BasisMelGANGenerator(L=config["L"],
                                         in_channels=config["in_channels"],
                                         out_channels=config["out_channels"],
                                         kernel_size=config["kernel_size"],
                                         channels=config["channels"],
                                         upsample_scales=config["upsample_scales"],
                                         stack_kernel_size=config["stack_kernel_size"],
                                         stacks=config["stacks"],
                                         use_weight_norm=config["use_weight_norm"],
                                         use_causal_conv=config["use_causal_conv"]).to(device)
        else:
            raise Exception("no model find!")
        model.load_state_dict(
            torch.load(os.path.join(checkpoint_path),
                       map_location=torch.device(device))['model'])
        model.eval()
        model.remove_weight_norm()
        return model

    def synthesize(self, mel):
        with torch.no_grad():
            zero_mel = torch.zeros_like(torch.from_numpy(mel).float())
            bias = self.model.inference(zero_mel)
            est_source = self.model.inference(mel)
            est_source_remove_bias = est_source - bias
        return est_source, est_source_remove_bias, bias


def run_synthesizer():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str)
    parser.add_argument('--mel_path', type=str)
    parser.add_argument('--wav_path', type=str)
    parser.add_argument("--model_name", type=str, help="melgan, hifigan and multiband-hifigan.")
    parser.add_argument("--config", type=str, help="path to model configuration file")
    args = parser.parse_args()

    synthesizer = Synthesizer(args.checkpoint_path, args.config, args.model_name)
    mel = np.load(args.mel_path)
    gl_wav = inv_mel_spectrogram(mel)
    est_source, est_source_remove_bias, bias = synthesizer.synthesize(mel.T)
    est_source, est_source_remove_bias, bias = est_source.cpu().numpy(), est_source_remove_bias.cpu().numpy(), bias.cpu().numpy()
    save_wav(est_source, args.wav_path, hp.sample_rate, rescale_out=hp.rescale_out)
    save_wav(est_source_remove_bias, args.wav_path[:-3] + "remove.wav", hp.sample_rate, rescale_out=hp.rescale_out)
    save_wav(bias, args.wav_path[:-3] + "bias.wav", hp.sample_rate, rescale_out=hp.rescale_out)
    save_wav(gl_wav, args.wav_path[:-3] + "gl.wav", hp.sample_rate, rescale_out=hp.rescale_out)
