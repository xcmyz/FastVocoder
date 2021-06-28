import torch
import os
import argparse
import yaml
import time
import numpy as np
import hparams as hp

from data.audio import save_wav, inv_mel_spectrogram
from model.generator import MelGANGenerator
from model.generator import MultiBandHiFiGANGenerator
from model.generator import HiFiGANGenerator
from model.generator import BasisMelGANGenerator

USE_PATTERN = True
TEST_RTF = True

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
        model_dict = torch.load(os.path.join(checkpoint_path), map_location=torch.device(device))
        if model_name == "basis-melgan":
            self.pattern = model_dict["pattern"]
            self.L = config["L"]
        model.load_state_dict(model_dict['model'])
        model.eval()
        model.remove_weight_norm()
        return model

    def synthesize(self, mel):
        # only support basis-melgan
        with torch.no_grad():
            est_source = self.model.inference(mel)[:-(self.L // 2)]
            if USE_PATTERN:
                est_source = est_source - self.pattern[:est_source.size(0)]
            else:
                bias = self.model.inference(torch.zeros_like(torch.from_numpy(mel)).float())[:-(self.L // 2)]
                est_source = est_source - bias
        return est_source

    def test_rtf(self, mel):
        with torch.no_grad():
            self.model.inference(mel)


def run_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str)
    parser.add_argument("--file_path", type=str)
    parser.add_argument("--model_name", type=str, help="melgan, hifigan and multiband-hifigan.")
    parser.add_argument("--config", type=str, help="path to model configuration file")
    args = parser.parse_args()

    synthesizer = Synthesizer(args.checkpoint_path, args.config, args.model_name)
    mels = []
    duration = 0.0
    list_files = os.listdir(args.file_path)
    for file in list_files:
        mel = np.load(os.path.join(args.file_path, file)).T
        mels.append(mel)
        duration += (mel.shape[0] * hp.hop_size) / hp.sample_rate
    print(f"duration is {duration}s.")

    if args.model_name == "basis-melgan":
        for mel, filename in zip(mels, list_files):
            est_source = synthesizer.synthesize(mel)
            save_wav(est_source.numpy(), os.path.join(args.file_path, f"{filename}.wav"), sample_rate=hp.sample_rate)

    if TEST_RTF:
        s = time.perf_counter()
        for _ in range(10):
            for mel in mels:
                synthesizer.test_rtf(mel)
        e = time.perf_counter()
        cost = e - s
        print(f"cost time: {cost}s.")
        rtf = cost / (10.0 * duration)
        print(f"rtf is {rtf}.")


if __name__ == "__main__":
    run_test()
