# -*- coding: utf-8 -*-

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""MelGAN Modules."""

import torch
import utils
import logging

import hparams as hp
import numpy as np

from layers import CausalConv1d
from layers import CausalConvTranspose1d
from layers import ResidualStack
from layers import Conv1d1x1

from modules import BasisSignalLayer, LastLinear


class MelGANGenerator(torch.nn.Module):
    """MelGAN generator module."""

    def __init__(self,
                 basis_signal_weight, L=32,
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
        super(MelGANGenerator, self).__init__()

        # check hyper parameters is valid
        # assert channels >= np.prod(upsample_scales)
        # assert channels % (2 ** len(upsample_scales)) == 0
        if not use_causal_conv:
            assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."

        # add initial layer
        layers = []
        if not use_causal_conv:
            layers += [
                getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params),
                torch.nn.Conv1d(in_channels, channels[0], kernel_size, bias=bias),
            ]
        else:
            layers += [
                CausalConv1d(in_channels, channels, kernel_size,
                             bias=bias, pad=pad, pad_params=pad_params),
            ]

        for i, upsample_scale in enumerate(upsample_scales):
            # add upsampling layer
            layers += [getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)]
            if not use_causal_conv:
                layers += [
                    torch.nn.ConvTranspose1d(
                        channels[i],
                        channels[i+1],
                        upsample_scale * 2,
                        stride=upsample_scale,
                        padding=upsample_scale // 2 + upsample_scale % 2,
                        output_padding=upsample_scale % 2,
                        bias=bias,
                    )
                ]
            else:
                layers += [
                    CausalConvTranspose1d(
                        channels // (2 ** i),
                        channels // (2 ** (i + 1)),
                        upsample_scale * 2,
                        stride=upsample_scale,
                        bias=bias,
                    )
                ]

            # add residual stack
            for j in range(stacks):
                layers += [
                    ResidualStack(
                        kernel_size=stack_kernel_size,
                        channels=channels[i+1],
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
        layers += [LastLinear(channels[-1], out_channels)]

        if use_final_nonlinear_activation:
            layers += [torch.nn.ReLU()]

        # define the model as a single function
        self.melgan = torch.nn.Sequential(*layers)

        # basis signal
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
        # est_source = est_source * (1. / max(0.01, torch.max(torch.abs(est_source))))
        return est_source, weight

    def test_basis_signal(self, weight):
        return self.basis_signal(weight)

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
        c = self.melgan(c.transpose(1, 0).unsqueeze(0))
        if self.pqmf is not None:
            c = self.pqmf.synthesis(c)
        return c.squeeze(0).transpose(1, 0)


if __name__ == "__main__":
    # TEST
    basis_signal_weight = torch.randn(hp.L, 256)
    model_test = MelGANGenerator(basis_signal_weight)
    print(model_test)
    print("number of parameter:", utils.get_param_num(model_test))

    c_test = torch.randn(2, 80, 100)
    wav_test, weight_test = model_test(c_test)
    print("wav_test size:", wav_test.size())
    print("weight_test size:", weight_test.size())
