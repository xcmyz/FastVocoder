# Fast (GAN Based Neural) Vocoder (Building...)
[Chinese README](/README_CN.md)

## Discription

Building. Include MelGAN, HifiGAN and Multiband-HifiGAN, maybe [NHV](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/3188.pdf) in the future.

## Usage

- Prepare data: ``` bash preprocess.sh <wav path file> processed/ dataset/audio dataset/mel ```
- Train:
    ```
    bash train.sh \
        <GPU ids> \
        /path/to/audio/train \
        /path/to/audio/valid \
        /path/to/mel/train \
        /path/to/mel/valid \
        <model name> \
        <if multi band> \
        <if use scheduler>
    ```

## Acknowledgment

- [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)
- [HiFi-GAN](https://github.com/jik876/hifi-gan)
