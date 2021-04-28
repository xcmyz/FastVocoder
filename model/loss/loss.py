import torch.nn as nn

from .stft_loss import MultiResolutionSTFTLoss


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.stft_loss = MultiResolutionSTFTLoss()

    def forward(self, est_source, wav, pqmf=None):
        if pqmf is not None:
            wav = pqmf.analysis(wav.unsqueeze(1))
            print(wav.size(), est_source.size())
            assert est_source.size(2) == wav.size(2)
            wav.requires_grad = False
            sc_loss, mag_loss = self.stft_loss(est_source, wav)
            stft_loss = sc_loss + mag_loss
            return stft_loss
        wav.requires_grad = False
        assert est_source.size(1) == wav.size(1)
        sc_loss, mag_loss = self.stft_loss(est_source, wav)
        stft_loss = sc_loss + mag_loss
        return stft_loss
