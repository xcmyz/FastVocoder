# from https://github.com/jik876/hifi-gan

import torch
import torch.nn.functional as F
import torch.nn as nn

from torch.nn import Conv1d, ConvTranspose1d
from .modules import get_padding
from .pqmf import PQMF
import logging

LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__()
        self.convs1 = nn.ModuleList([
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                   padding=get_padding(kernel_size, dilation[0])),
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                   padding=get_padding(kernel_size, dilation[1])),
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[2],
                   padding=get_padding(kernel_size, dilation[2]))
        ])

        self.convs2 = nn.ModuleList([
            Conv1d(channels, channels, kernel_size, 1, dilation=1,
                   padding=get_padding(kernel_size, 1)),
            Conv1d(channels, channels, kernel_size, 1, dilation=1,
                   padding=get_padding(kernel_size, 1)),
            Conv1d(channels, channels, kernel_size, 1, dilation=1,
                   padding=get_padding(kernel_size, 1))
        ])

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x


class ResBlock2(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__()
        self.convs = nn.ModuleList([
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[0],
                   padding=get_padding(kernel_size, dilation[0])),
            Conv1d(channels, channels, kernel_size, 1, dilation=dilation[1],
                   padding=get_padding(kernel_size, dilation[1]))
        ])

    def forward(self, x):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x


class MultiBandHiFiGANGenerator(torch.nn.Module):
    def __init__(self,
                 resblock_kernel_sizes=[3, 7, 11],
                 upsample_rates=[6, 5, 2],
                 upsample_initial_channel=512,
                 resblock_type="1",
                 upsample_kernel_sizes=[16, 15, 4],
                 resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]]):
        super(MultiBandHiFiGANGenerator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(80, upsample_initial_channel, 7, 1, padding=3)
        resblock = ResBlock1 if resblock_type == '1' else ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(ConvTranspose1d(upsample_initial_channel // (2 ** i),
                            upsample_initial_channel // (2 ** (i + 1)),
                            k, u, padding=(u // 2 + u % 2), output_padding=u % 2))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 4, 7, 1, padding=3)  # 4 band

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
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
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
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        x = self.pqmf.synthesis(x)
        return x.squeeze()
