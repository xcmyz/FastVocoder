# -*- coding: utf-8 -*-

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""MelGAN Modules."""

import torch
import logging
import hparams as hp
import numpy as np

from .modules import ResidualStack
from .modules import BasisSignalLayer, LastLinear
from .modules import UpsampleLayer


class BasisMelGANGenerator(torch.nn.Module):
    """MelGAN generator module."""

    def __init__(self,
                 basis_signal_weight,
                 L=30,
                 in_channels=80,
                 out_channels=256,
                 kernel_size=7,
                 channels=[256, 256, 256],
                 bias=True,
                 upsample_scales=[4, 4],
                 stack_kernel_size=3,
                 stacks=3,
                 nonlinear_activation="LeakyReLU",
                 nonlinear_activation_params={"negative_slope": 0.2},
                 pad="ReflectionPad1d",
                 pad_params={},
                 use_final_nonlinear_activation=True,
                 use_weight_norm=True,
                 use_causal_conv=False,
                 transposedconv=True
                 ):
        """Initialize MelGANGenerator module.
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size of initial and final conv layer.
            channels (int): Initial number of channels for conv layer.
            bias (bool): Whether to add bias parameter in convolution layers.
            upsample_scales (list): List of upsampling scales.
            stack_kernel_size (int): Kernel size of dilated conv layers in residual stack.
            stacks (int): Number of stacks in a single residual stack.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            pad (str): Padding function module name before dilated convolution layer.
            pad_params (dict): Hyperparameters for padding function.
            use_final_nonlinear_activation (torch.nn.Module): Activation function for the final layer.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_causal_conv (bool): Whether to use causal convolution.
        """

        super(BasisMelGANGenerator, self).__init__()

        # check hyper parameters is valid
        # assert channels >= np.prod(upsample_scales)
        # assert channels % (2 ** len(upsample_scales)) == 0
        if not use_causal_conv:
            assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."

        # add initial layer
        layers = []
        layers += [
            getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params),
            torch.nn.Conv1d(in_channels, channels[0], kernel_size, bias=bias),
        ]

        for i, upsample_scale in enumerate(upsample_scales):
            # add upsampling layer
            layers += [getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)]
            layers += [
                UpsampleLayer(channels[i],
                              channels[i + 1],
                              upsample_rate=upsample_scale,
                              kernel_size=(upsample_scale * 2 + 1),
                              stride=1,
                              padding=upsample_scale,
                              bias=bias) if transposedconv == False else
                torch.nn.ConvTranspose1d(
                    channels[i],
                    channels[i + 1],
                    upsample_scale * 2,
                    stride=upsample_scale,
                    padding=upsample_scale // 2 + upsample_scale % 2,
                    output_padding=upsample_scale % 2,
                    bias=bias,
                )
            ]

            # add residual stack
            for j in range(stacks):
                layers += [
                    ResidualStack(
                        kernel_size=stack_kernel_size,
                        channels=channels[i + 1],
                        dilation=stack_kernel_size ** j,
                        bias=bias,
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                        pad=pad,
                        pad_params=pad_params,
                        use_causal_conv=use_causal_conv,
                    )
                ]

        # add final layer
        layers += [LastLinear(channels[-1], out_channels, bias=bias)]

        if use_final_nonlinear_activation:
            layers += [torch.nn.ReLU()]

        # define the model as a single function
        self.melgan = torch.nn.Sequential(*layers)

        # basis signal
        self.L = L
        self.basis_signal = BasisSignalLayer(basis_signal_weight, L=L)

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # reset parameters
        self.reset_parameters()

        # initialize pqmf for inference
        self.pqmf = None

    def forward(self, c):
        """Calculate forward propagation.
        Args:
            c (Tensor): Input tensor (B, channels, T).
        Returns:
            Tensor: Output tensor (B, 1, T ** prod(upsample_scales)).
        """

        weight = self.melgan(c)
        weight = weight.contiguous().transpose(1, 2)
        est_source = self.basis_signal(weight)
        est_source = est_source[:, :weight.size(1) * (self.L // 2)]
        return est_source, weight

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""
        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def reset_parameters(self):
        """Reset parameters.
        This initialization follows official implementation manner.
        https://github.com/descriptinc/melgan-neurips/blob/master/mel2wav/modules.py
        """
        def _reset_parameters(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.ConvTranspose1d):
                m.weight.data.normal_(0.0, 0.02)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)

    def inference(self, c):
        """Perform inference.
        Args:
            c (Union[Tensor, ndarray]): Input tensor (T, in_channels).
        Returns:
            Tensor: Output tensor (T ** prod(upsample_scales), out_channels).
        """
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float).to(next(self.parameters()).device)
        weight = self.melgan(c.transpose(1, 0).unsqueeze(0))
        weight = weight.contiguous().transpose(1, 2)
        c = self.basis_signal(weight)
        return c.squeeze()

    def test(self, weight):
        wav_test = self.basis_signal(weight)
        return wav_test
