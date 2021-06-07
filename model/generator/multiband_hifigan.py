# from https://github.com/jik876/hifi-gan

import torch
import torch.nn.functional as F
import torch.nn as nn
import logging

from torch.nn import Conv1d, ConvTranspose1d
from .modules import get_padding, UpsampleLayer
from .modules import ResBlock1, ResBlock2, LRELU_SLOPE
from .pqmf import PQMF


class MultiBandHiFiGANGenerator(torch.nn.Module):
    '''
    MultiBand-HiFiGAN model encountered the problem of strong checkerboard artifacts -- the generated audio
    has interference at a specific frequency. I used temporal nearest interpolation layer and tested
    some different upsample rate and kernel sizes. It still didn't completely solve the problem,
    but the parameter provided by this repo has made checkerboard artifacts relatively weak. 
    In order to completely solve this problem when generating, a trick is adopted. You can refer to the code.
    '''

    def __init__(self,
                 resblock_kernel_sizes=[3, 7, 11],
                 upsample_rates=[10, 6],
                 upsample_initial_channel=256,
                 resblock_type="1",
                 upsample_kernel_sizes=[20, 12],
                 resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                 transposedconv=True,
                 bias=True):
        super(MultiBandHiFiGANGenerator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(80, upsample_initial_channel, 7, 1, padding=3, bias=bias)
        resblock = ResBlock1 if resblock_type == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                UpsampleLayer(upsample_initial_channel // (2 ** i),
                              upsample_initial_channel // (2 ** (i + 1)),
                              upsample_rate=u,
                              kernel_size=k,
                              stride=1,
                              padding=k // 2,
                              bias=bias) if transposedconv == False else
                ConvTranspose1d(upsample_initial_channel // (2 ** i),
                                upsample_initial_channel // (2 ** (i + 1)),
                                k, u,
                                padding=(u // 2 + u % 2),
                                output_padding=u % 2,
                                bias=bias))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d, bias=bias))

        self.conv_post = Conv1d(ch, 4, 7, 1, padding=3, bias=bias)  # 4 band

        self.pqmf = PQMF()  # 4 band
        # apply weight norm
        self.apply_weight_norm()
        # reset parameters
        self.reset_parameters()

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
                m.weight.data.normal_(0.0, 0.01)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)

    def forward(self, x):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def inference(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float).to(next(self.parameters()).device)
        x = x.transpose(1, 0).unsqueeze(0)
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        x = self.pqmf.synthesis(x)
        return x.squeeze()
