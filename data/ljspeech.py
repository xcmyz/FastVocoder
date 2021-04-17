import os
import audio
import audio_tool
import hparams
import numpy as np

from tqdm import tqdm
# from functools import partial
# from concurrent.futures import ProcessPoolExecutor


def build_from_path(in_dir, out_dir):
    index = 1
    # executor = ProcessPoolExecutor(max_workers=2)
    futures = []

    with open(os.path.join(in_dir, 'metadata.csv'), encoding='utf-8') as f:
        lines = f.readlines()
        if hparams.test_size != 0:
            lines = lines[:max(hparams.preload_test_size, hparams.test_size)]
        for i in tqdm(range(len(lines))):
            line = lines[i]
            parts = line.strip().split('|')
            # wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % parts[0])
            wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % parts[0])
            text = parts[2]
            # futures.append(executor.submit(partial(_process_utterance, out_dir, index, wav_path, text)))
            futures.append(_process_utterance(out_dir, index, wav_path, text))
            index = index + 1

    # return [future.result() for future in tqdm(futures)]
    return futures


def _process_utterance(out_dir, index, wav_path, text):
    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio_tool.tools.get_mel(wav_path).numpy().astype(np.float32)
    wav = audio.load_wav(wav_path)
    # mel_spectrogram = audio.melspectrogram(wav)

    # Write the spectrograms to disk:
    mel_filename = 'ljspeech-mel-%05d.npy' % index
    np.save(os.path.join(out_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

    wav_filename = 'ljspeech-wav-%05d.npy' % index
    wav = audio.load_wav(wav_path)
    np.save(os.path.join(out_dir, wav_filename), wav, allow_pickle=False)

    return text
