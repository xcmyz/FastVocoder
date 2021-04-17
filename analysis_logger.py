import os
import matplotlib.pyplot as plt

N = 15000

stft_loss_stft = []
with open(os.path.join("logger", "stft_loss_stft.txt"), "r", encoding="utf-8") as f:
    for loss in f.readlines():
        stft_loss_stft.append(float(loss))
    stft_loss_stft = stft_loss_stft[:N]

stft_loss_wav = []
with open(os.path.join("logger", "stft_loss_wav.txt"), "r", encoding="utf-8") as f:
    for loss in f.readlines():
        stft_loss_wav.append(float(loss))
    stft_loss_wav = stft_loss_wav[:N]

plt.plot([i for i in range(N)], stft_loss_stft, linewidth=0.3,)
plt.plot([i for i in range(N)], stft_loss_wav, linewidth=0.3)
plt.show()
