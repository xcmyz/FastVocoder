# -*- coding: utf-8 -*-

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""MelGAN Modules."""


import torch
import torch.nn.functional as F

import logging
import numpy as np

from layers import CausalConv1d
from layers import CausalConvTranspose1d
from layers import ResidualStack

from distutils.version import LooseVersion
from model_discriminator import MelGANMultiScaleDiscriminator
from hifigan_discriminator import DiscriminatorP

is_pytorch_17plus = LooseVersion(torch.__version__) >= LooseVersion("1.7")


def stft(x, fft_size, hop_size, win_length, window):
    """Perform STFT and convert to magnitude spectrogram.

    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.

    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).

    """
    if is_pytorch_17plus:
        x_stft = torch.stft(x, fft_size, hop_size, win_length, window, return_complex=False)
    else:
        x_stft = torch.stft(x, fft_size, hop_size, win_length, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]
    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    spectrum = torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7))
    return spectrum


class STFTDiscriminator(torch.nn.Module):

    def __init__(
        self,
        fft_size=1024,
        shift_size=120,
        win_length=600,
        window="hann_window",
        out_channels=1,
        kernel_sizes=[5, 3],
        channels=64,
        max_downsample_channels=1024,
        bias=True,
        downsample_scales=[4, 4],
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.2},
        pad="ReflectionPad1d",
        pad_params={},
    ):
        super(STFTDiscriminator, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        # NOTE(kan-bayashi): Use register_buffer to fix #223
        self.register_buffer("window", getattr(torch, window)(win_length))

        self.layers = torch.nn.ModuleList()

        # check kernel size is valid
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1
        assert kernel_sizes[1] % 2 == 1

        # add first layer
        self.layers += [
            torch.nn.Sequential(
                getattr(torch.nn, pad)((np.prod(kernel_sizes) - 1) // 2, **pad_params),
                torch.nn.Conv1d(fft_size // 2 + 1, channels, np.prod(kernel_sizes), bias=bias),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]

        # add downsample layers
        in_chs = channels
        for downsample_scale in downsample_scales:
            out_chs = min(in_chs * downsample_scale, max_downsample_channels)
            self.layers += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chs, out_chs,
                        kernel_size=downsample_scale * 6 + 1,
                        stride=downsample_scale,
                        padding=downsample_scale * 3,
                        groups=in_chs // 4,
                        bias=bias,
                    ),
                    getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                )
            ]
            in_chs = out_chs

        # add final layers
        out_chs = min(in_chs * 2, max_downsample_channels)
        self.layers += [
            torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_chs, out_chs, kernel_sizes[0],
                    padding=(kernel_sizes[0] - 1) // 2,
                    bias=bias,
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]
        self.layers += [
            torch.nn.Conv1d(
                out_chs, out_channels, kernel_sizes[1],
                padding=(kernel_sizes[1] - 1) // 2,
                bias=bias,
            ),
        ]

        # apply weight norm
        self.apply_weight_norm()

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def forward(self, x):
        x = stft(x, self.fft_size, self.shift_size, self.win_length, self.window)
        outs = []
        for f in self.layers:
            x = f(x)
            outs += [x]
        return outs


class MultiResolutionSTFTDiscriminator(torch.nn.Module):
    def __init__(
        self,
        fft_sizes=[2048, 1024, 512],
        hop_sizes=[240, 120, 50],
        win_lengths=[1200, 600, 240],
        window="hann_window",
        downsample_pooling="AvgPool1d",
        # follow the official implementation setting
        downsample_pooling_params={
            "kernel_size": 4,
            "stride": 2,
            "padding": 1,
            "count_include_pad": False,
        }
    ):
        """Initialize Multi resolution STFT loss module.

        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.

        """
        super(MultiResolutionSTFTDiscriminator, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.stft_discriminator = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_discriminator += [STFTDiscriminator(fft_size=fs, shift_size=ss, win_length=wl, window=window)]

    def forward(self, x):
        x = x.squeeze(1)

        outs = []
        for f in self.stft_discriminator:
            outs += [f(x)]
        return outs


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = torch.nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y):
        y_d_rs = []
        for _, d in enumerate(self.discriminators):
            y_d_r, fmap = d(y)
            y_d_rs.append(fmap + [y_d_r.unsqueeze(1)])
        return y_d_rs


class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.wav_discriminator = MelGANMultiScaleDiscriminator()
        self.period_discriminator = MultiPeriodDiscriminator()

    def forward(self, x):
        outs_1 = self.wav_discriminator(x)
        outs_2 = self.period_discriminator(x)
        return outs_1 + outs_2


class NewDiscriminator(torch.nn.Module):
    def __init__(self):
        super(NewDiscriminator, self).__init__()
        self.wav_discriminator = MelGANMultiScaleDiscriminator()
        self.mel_discriminator = MultiResolutionSTFTDiscriminator()

    def forward(self, x):
        outs_1 = self.wav_discriminator(x)
        outs_2 = self.mel_discriminator(x)
        return outs_1 + outs_2


class PMDiscriminator(torch.nn.Module):
    def __init__(self):
        super(PMDiscriminator, self).__init__()
        self.period_discriminator = MultiPeriodDiscriminator()
        self.mel_discriminator = MultiResolutionSTFTDiscriminator()

    def forward(self, x):
        outs_1 = self.period_discriminator(x)
        outs_2 = self.mel_discriminator(x)
        return outs_1 + outs_2


class MelDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MelDiscriminator, self).__init__()
        self.mel_discriminator = MultiResolutionSTFTDiscriminator()

    def forward(self, x):
        outs = self.mel_discriminator(x)
        return outs


class WavDiscriminator(torch.nn.Module):
    def __init__(self):
        super(WavDiscriminator, self).__init__()
        self.wav_discriminator = MelGANMultiScaleDiscriminator()

    def forward(self, x):
        outs = self.wav_discriminator(x)
        return outs


class PDiscriminator(torch.nn.Module):
    def __init__(self):
        super(PDiscriminator, self).__init__()
        self.period_discriminator = MultiPeriodDiscriminator()

    def forward(self, x):
        outs = self.period_discriminator(x)
        return outs


if __name__ == "__main__":
    model_test = MultiResolutionSTFTDiscriminator()
    y = torch.randn(2, 12345)
    outs = model_test(y)
    # print(len(outs[0]))
    # for fm in outs[0]:
    #     print(fm.size())
    # for out in outs:
    #     print(out[-1].size())
    # print(outs[0][-1].size())
    # print(model_test)

    melgan_discriminator_test = MelGANMultiScaleDiscriminator()
    outs_1 = melgan_discriminator_test(y.unsqueeze(1))
    # for out in outs_1:
    #     print(out[-1].size())

    # outs_2 = outs + outs_1
    # for out in outs_2:
    #     print(out[-1].size())

    discriminator_test = Discriminator()
    outs_test = discriminator_test(y.unsqueeze(1))
    for out in outs_test:
        print(out[-1].size())

    # test_hifigan_discriminator = MultiPeriodDiscriminator()
    # print(test_hifigan_discriminator)
    # outs = test_hifigan_discriminator(y.unsqueeze(1))
    # for out in outs:
    #     print(out.size())
