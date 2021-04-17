import os
import torch
import scipy
import audio
import numpy as np
import pandas as pd
import hparams as hp
import seaborn as sns

from scipy import signal
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator


if __name__ == "__main__":
    # frequency_response = []
    # sorted_frequency_response = []
    # peak_magnitude = []
    # w = 0
    # basis_signal = np.load(os.path.join("data", "basis_signal_weight.npy"))
    # for i in range(basis_signal.shape[1]):
    #     one = basis_signal[:, i]
    #     # w, h = signal.freqz(one)
    #     h = scipy.fft(one)
    #     frequency_response.append(abs(h))
    #     peak_magnitude.append(np.argmax(h))
    # index = np.argsort(np.array(peak_magnitude))
    # for i in index:
    #     sorted_frequency_response.append(frequency_response[i])
    # sorted_frequency_response = torch.Tensor(sorted_frequency_response).numpy().T

    magnitude = []
    sorted_magnitude = []
    peak_magnitude = []
    w = 0
    basis_signal = np.load(os.path.join("data", "basis_signal_weight.npy"))
    for i in range(basis_signal.shape[1]):
        one = basis_signal[:, i]
        fft_result = np.fft.fft(one)
        fft_result = abs(fft_result)[len(abs(fft_result)) // 2:]
        magnitude.append(fft_result)
        peak_magnitude.append(np.argmax(fft_result))
    index = np.argsort(-np.array(peak_magnitude))
    for i in index:
        sorted_magnitude.append(magnitude[i])
    sorted_magnitude = torch.Tensor(sorted_magnitude).numpy().T
    hz = int(hp.sample_rate / (2 * sorted_magnitude.shape[0]))
    data = pd.DataFrame(sorted_magnitude,
                        index=list(reversed([(i * hz) for i in range(sorted_magnitude.shape[0])])),
                        columns=[i for i in range(sorted_magnitude.shape[1])])
    plt.figure(figsize=(16, 5))
    sns.heatmap(data=data)
    plt.title('FFT')
    plt.xlabel("Filter index")
    plt.ylabel("Frequency (Hz)")
    plt.show()
