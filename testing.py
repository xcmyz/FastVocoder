import os
import torch
import utils
import audio
import audio_tool
import numpy as np

from model_melgan import MelGANGenerator
from stft_loss import MultiResolutionSTFTLoss
from test import add_noise


if __name__ == "__main__":
    # --------------------------------- TEST --------------------------------- #

    # visual.plot_data([torch.eye(256, 256) for _ in range(2)], 0)

    # cb = modules.ConvDecoder(256, 3000)
    # print(utils.get_param_num(cb))
    # x = torch.randn(2, 1234, 256)
    # pos = torch.Tensor([[(i+1) for i in range(1234)]for _ in range(2)]).long()
    # print(cb(x, pos, 1234)[0].size())

    # ce = modules.ConvEncoder(256, 300)
    # print(utils.get_param_num(ce))
    # x = torch.Tensor([[i for i in range(123)] for _ in range(2)]).long()
    # pos = torch.Tensor([[(i+1) for i in range(123)] for _ in range(2)]).long()
    # print(ce(x, pos)[0].size())

    # te = tacotron_encoder.Encoder(256)
    # test_input = torch.randn(2, 123, 256)
    # print(te(test_input).size())

    # cd = modules.ConvDecoder(512, 270, 3000, 4, 3, 3, 3, 3, 1, 1)
    # test_input = torch.randn(2, 123, 512)
    # mel_pos = torch.Tensor(
    #     [[(i+1) for i in range(123)] for _ in range(2)]).long()
    # mel_view_pos = torch.Tensor(
    #     [[(i+1) for i in range(12)] for _ in range(2)]).long()
    # o, _ = cd(test_input, mel_pos, mel_view_pos)
    # for _o_ in o:
    #     print(_o_.size())

    # test_model = model.ConvSpeech()
    # test_character = torch.Tensor([[16, 54, 78, 234], [12, 32, 146, 0]]).long()
    # test_src_pos = torch.Tensor([[1, 2, 3, 4], [1, 2, 3, 0]]).long()
    # test_mel_pos = torch.Tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #                              [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]).long()
    # test_mel_max_length = 10
    # test_length_target = torch.Tensor([[1, 2, 3, 4], [1, 2, 6, 0]]).int()
    # mel_outputs, weight_output, coef_output, duration_predictor_output \
    #     = test_model(test_character,
    #                  test_src_pos,
    #                  mel_pos=test_mel_pos,
    #                  mel_max_length=test_mel_max_length,
    #                  length_target=test_length_target)

    # print("length of mel_outputs:", len(mel_outputs))
    # print("size of mel_output:", mel_outputs[0].size())
    # print("size of weight_output:", weight_output.size())
    # print("size of coef_output:", coef_output.size())

    # test_model = model.ParallelWaveGANGenerator()
    # x = torch.randn(2, 80, 1234)
    # c = torch.randn(2, 80, 1234)
    # print(test_model(x, c).size())

    print("testing...")
    # wav = audio.load_wav(os.path.join(
    #     "data", "LJSpeech-1.1", "wavs", "LJ001-0003.wav"))
    # print(wav.shape)
    # L = wav.shape[0]
    # wav = wav[:(L - 10) // 10 * 10]
    # print(wav.shape)
    # melspec = audio.melspectrogram(wav)
    # print(melspec.shape)
    # wav_ = audio.inv_mel_spectrogram(melspec)
    # audio.save_wav(wav_, "test_1.wav")

    # min_value = 2**32
    # listfile = os.listdir(os.path.join("data", "LJSpeech-1.1", "wavs"))
    # for i, filename in enumerate(listfile):
    #     wav = audio.load_wav(os.path.join(
    #         "data", "LJSpeech-1.1", "wavs", filename))
    #     if wav.shape[0] // 10 < min_value:
    #         min_value = wav.shape[0] // 10
    #     if ((i + 1) % 1000) == 0:
    #         print("Done", i + 1)
    # print(min_value)
    # min length: 888

    # wav = audio.load_wav(os.path.join(
    #     "data", "LJSpeech-1.1", "wavs", "LJ001-0001.wav"))
    # mel = audio.melspectrogram(wav)
    # print(mel.shape)
    # wav_ = audio.inv_mel_spectrogram(mel)
    # audio.save_wav(wav_, "test_1.wav")

    # wav = audio.load_wav(os.path.join("data", "LJSpeech-1.1", "wavs_8k", "LJ001-0001.wav"))
    # print(wav.shape)
    # mel = audio_tool.tools.get_mel(os.path.join("data", "LJSpeech-1.1", "wavs_8k", "LJ001-0001.wav"))
    # print(mel.size())
    # audio_tool.tools.inv_mel_spec(mel, "test.wav")
    # mel = audio.melspectrogram(audio.load_wav(os.path.join("data", "LJSpeech-1.1", "wavs_8k", "LJ001-0001.wav")))
    # wav_ = audio.inv_mel_spectrogram(mel)
    # audio.save_wav(wav_, "test.wav")

    basis_signal_weight = np.load(os.path.join("data", "basis_signal_weight.npy"))
    basis_signal_weight = torch.from_numpy(basis_signal_weight)
    model = MelGANGenerator(basis_signal_weight)
    test_weight = torch.stack([torch.from_numpy(np.load(os.path.join("weight", "1.npy")).T)])
    wav = model.test_basis_signal(test_weight)
    wav = wav.detach().numpy()[0]
    audio.save_wav(wav, "test_basis_signal.wav")

    stft_loss = MultiResolutionSTFTLoss()
    test_x = torch.randn(2, 12321)
    test_y = torch.randn(2, 12321)
    sc_loss, mag_loss = stft_loss(test_x, test_y)
    print(sc_loss, mag_loss)

    # mel = audio_tool.tools.get_mel(os.path.join("data", "LJSpeech-1.1", "wavs", "LJ001-0001.wav"))
    # print(mel.size())
    # audio_tool.tools.inv_mel_spec(mel, "test.wav")

    # wav_16k = audio.load_wav(os.path.join("data", "LJSpeech-1.1", "wavs", "LJ001-0001.wav"))
    # audio.save_wav(wav_16k, "test_16k.wav")

    for i in range(1, 6 + 1):
        wav = np.load(os.path.join("mels", "ljspeech-wav-0000{:d}.npy".format(i)))
        wav = torch.from_numpy(wav).float()
        noi = add_noise(wav, quantization_channel=int(np.sqrt(2 ** 16)))
        mix = wav + noi
        audio.save_wav(mix.numpy(), "ljspeech-mix-0000{:d}.wav".format(i))
        mel = audio_tool.tools.get_mel("ljspeech-mix-0000{:d}.wav".format(i)).numpy().astype(np.float32).T
        np.save("ljspeech-mix-0000{:d}.npy".format(i), mel)

    # --------------------------------- TEST --------------------------------- #
