# Fast (GAN Based Neural) Vocoder
[Chinese README](/resource/README_CN.md)

## Todo

- [ ] Support [Basis-MelGAN](https://blog.xcmyz.xyz/demo/)
- [ ] Support NHV

## Discription

Include MelGAN, HifiGAN and Multiband-HifiGAN, maybe include [NHV](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/3188.pdf) in the future. Developed on [BiaoBei dataset](https://www.data-baker.com/#/data/index/source), you can modify `conf` and `hparams.py` to fit your own dataset and model.

## Demo
[Demo README](/resource/demo/README.md)

## Usage

- Prepare data
    - write path of wav data in a file, for example: ``` cd dataset && python3 biaobei.py ```
    - ``` bash preprocess.sh <wav path file> <path to save processed data> dataset/audio dataset/mel ```
    - for example: ``` bash preprocess.sh dataset/BZNSYP.txt processed dataset/audio dataset/mel ```
- Train
    - command:
    ```
    bash train.sh \
        <GPU ids> \
        /path/to/audio/train \
        /path/to/audio/valid \
        /path/to/mel/train \
        /path/to/mel/valid \
        <model name> \
        <if multi band> \
        <if use scheduler> \
        <path to configuration file>
    ```
    - for example:
    ```
    bash train.sh \
    0 \
    dataset/audio/train \
    dataset/audio/valid \
    dataset/mel/train \
    dataset/mel/valid \
    hifigan \
    0 0 0 \
    conf/hifigan/light.yaml
    ```
- Train from checkpoint
    - command:
    ```
    bash train.sh \
        <GPU ids> \
        /path/to/audio/train \
        /path/to/audio/valid \
        /path/to/mel/train \
        /path/to/mel/valid \
        <model name> \
        <if multi band> \
        <if use scheduler> \
        <path to configuration file> \
        /path/to/checkpoint \
        <step of checkpoint>
    ```
- Synthesize
    - command:
    ```
    bash synthesize.sh \
        /path/to/checkpoint \
        /path/to/mel \
        /path/for/saving/wav \
        <model name> \
        /path/to/configuration/file
    ```

## Acknowledgments

- [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN)
- [HiFi-GAN](https://github.com/jik876/hifi-gan)