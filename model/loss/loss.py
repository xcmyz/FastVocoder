import torch
import torch.nn as nn

from .stft_loss import MultiResolutionSTFTLoss


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.stft_loss = MultiResolutionSTFTLoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, est_source, wav, est_weight=None, weight=None, pqmf=None):
        weight_loss = None
        if pqmf is not None:
            est_source_sub_band = est_source
            wav_full_band = wav
            wav_sub_band = pqmf.analysis(wav_full_band.unsqueeze(1))
            est_source_full_band = pqmf.synthesis(est_source_sub_band)[:, 0, :]
            est_source_sub_band = est_source_sub_band.view(-1, est_source_sub_band.size(2))
            wav_sub_band = wav_sub_band.view(-1, wav_sub_band.size(2))
            assert est_source_sub_band.size(0) == wav_sub_band.size(0)
            assert est_source_sub_band.size(1) == wav_sub_band.size(1)

            wav_full_band.requires_grad = False
            wav_sub_band.requires_grad = False
            sc_loss_sub_band, mag_loss_sub_band = self.stft_loss(est_source_sub_band, wav_sub_band)
            stft_loss_sub_band = sc_loss_sub_band + mag_loss_sub_band
            sc_loss_full_band, mag_loss_full_band = self.stft_loss(est_source_full_band, wav_full_band)
            stft_loss_full_band = sc_loss_full_band + mag_loss_full_band
            stft_loss = (stft_loss_sub_band + stft_loss_full_band) / 2.
            return stft_loss, weight_loss

        wav.requires_grad = False
        assert est_source.size(1) == wav.size(1)
        sc_loss, mag_loss = self.stft_loss(est_source, wav)
        stft_loss = sc_loss + mag_loss

        if est_weight is not None and weight is not None:
            weight_loss = self.l1_loss(est_weight, weight)

        return stft_loss, weight_loss
