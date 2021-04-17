import os
import h5py
import numpy as np

import torch
import audio
import hparams as hp

import matplotlib.pyplot as plt

from discriminator import PDiscriminator
from discriminator import PMDiscriminator
from discriminator import Discriminator as WPDiscriminator
from discriminator import MelDiscriminator as MDiscriminator
from discriminator import WavDiscriminator as WDiscriminator

# from discriminator import WMDiscriminator
from discriminator import NewDiscriminator

from model_melgan import MelGANGenerator as MelGANGenerator_light
from model_melgan_large import MelGANGenerator as MelGANGenerator_large

from ptflops import get_model_complexity_info
from thop import profile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_hdf5(hdf5_name):
    hdf5_file = h5py.File(hdf5_name, "r")
    hdf5_data = hdf5_file["feats"][()]
    hdf5_file.close()
    return hdf5_data


def transfer(step=1000000, generator_type="light"):
    checkpoint_path = "checkpoint_" + str(step) + ".pth.tar"
    basis_signal_weight = np.load(os.path.join("data", "basis_signal_weight.npy"))
    basis_signal_weight = torch.from_numpy(basis_signal_weight)
    model = 0
    if generator_type == "light":
        model = MelGANGenerator_light(basis_signal_weight).to(device)
    elif generator_type == "large":
        model = MelGANGenerator_large(basis_signal_weight).to(device)
    model.load_state_dict(torch.load(os.path.join(hp.checkpoint_path, checkpoint_path), map_location=torch.device(device))['model'])

    checkpoint = torch.load(os.path.join(hp.checkpoint_path, checkpoint_path), map_location=torch.device(device))
    discriminator = NewDiscriminator().to(device)
    # new_discriminator = WMDiscriminator().to(device)

    discriminator.load_state_dict(checkpoint['discriminator'])
    optimizer = torch.optim.Adam(model.melgan.parameters(), lr=1e-4, eps=1.0e-6, weight_decay=0.0)
    optimizer.load_state_dict(checkpoint['optimizer'])
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=5e-5, eps=1.0e-6, weight_decay=0.0)
    discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])
    # new_discriminator.wav_discriminator = discriminator.wav_discriminator
    # new_discriminator.mel_discriminator = discriminator.stft_discriminator
    os.makedirs("checkpoint", exist_ok=True)
    torch.save(
        {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'discriminator': discriminator.state_dict(),
            'discriminator_optimizer': discriminator_optimizer.state_dict()
        },
        os.path.join("checkpoint", f"checkpoint_{step}.pth.tar"),
        _use_new_zipfile_serialization=False)

    model.eval()
    model.remove_weight_norm()
    return model


if __name__ == "__main__":
    print("test")

    # for filename in os.listdir("mels.test_mb"):
    #     data = read_hdf5(os.path.join("mels.test_mb", filename))
    #     os.remove(os.path.join("mels.test_mb", filename))
    #     np.save(os.path.join("mels.test_mb", filename[:-3]), data)

    # transfer()

    # y1 = audio.load_wav(os.path.join("test_basis_signal.wav"))
    # y2 = audio.load_wav(os.path.join("test.wav"))
    # min_len = min(y1.shape[0], y2.shape[0])
    # y1 = y1[:min_len]
    # y2 = y2[:min_len]
    # x = np.array([i for i in range(y2.shape[0])])
    # plt.plot(x, y1, c="b")
    # plt.plot(x, y2, c="r")
    # plt.show()

    basis_signal_weight = np.load(os.path.join("data", "basis_signal_weight.npy"))
    basis_signal_weight = torch.from_numpy(basis_signal_weight)
    model_light = MelGANGenerator_light(basis_signal_weight).to(device)
    model_large = MelGANGenerator_large(basis_signal_weight).to(device)

    macs, params = get_model_complexity_info(model_light, (80, 100), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # macs, params = profile(model_light, inputs=(torch.randn(1, 80, 100), ))
    # print('{:<30}  {:<8}'.format('## Computational complexity: ', macs))

    macs, params = get_model_complexity_info(model_large, (80, 100), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # macs, params = profile(model_large, inputs=(torch.randn(1, 80, 100), ))
    # print('{:<30}  {:<8}'.format('## Computational complexity: ', macs))
