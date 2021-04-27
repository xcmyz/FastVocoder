import torch.nn as nn

from .mfd import MultiResolutionSTFTDiscriminator
from .msd import MelGANMultiScaleDiscriminator
from .mpd import MultiPeriodDiscriminator


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.msd = MelGANMultiScaleDiscriminator()
        self.mfd = MultiResolutionSTFTDiscriminator()

    def forward(self, x):
        outs_1 = self.msd(x)
        outs_2 = self.mfd(x)
        return outs_1 + outs_2
