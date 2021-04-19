import torch
import torch.nn as nn

import os
import hparams as hp

from stft_loss import MultiResolutionSTFTLoss


class DNNLoss(nn.Module):
    def __init__(self):
        super(DNNLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.stft_loss = MultiResolutionSTFTLoss()

    def forward(self, est_weight, weight, est_source, wav):
        weight.requires_grad = False
        wav.requires_grad = False

        weight_loss = self.l1_loss(est_weight, weight)

        min_len = min(est_source.size(1), wav.size(1))
        est_source = est_source[:, :min_len]
        wav = wav[:, :min_len]
        sc_loss, mag_loss = self.stft_loss(est_source, wav)
        stft_loss = sc_loss + mag_loss
        return weight_loss, stft_loss


if __name__ == "__main__":
    # TEST
    print("testing...")
