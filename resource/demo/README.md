# DEMO

## HiFiGAN (large)

- step: 355k
- training command: ``` bash train.sh 0 dataset/audio/train dataset/audio/valid dataset/mel/train dataset/mel/valid hifigan conf/hifigan/large.yaml 0 1 ```
- demo: `./resource/demo/0.hifigan.large.355000.wav`

## HiFiGAN (light)

- step: 705k
- training command: ``` bash train.sh 0 dataset/audio/train dataset/audio/valid dataset/mel/train dataset/mel/valid hifigan conf/hifigan/light.yaml 0 0```
- demo: `./resource/demo/0.hifigan.light.705000.wav`

## MultiBand-HiFiGAN (large)

- step: 945k
- training command: ``` bash train.sh 0 dataset/audio/train dataset/audio/valid dataset/mel/train dataset/mel/valid multiband-hifigan conf/multiband-hifigan/large.yaml 0 0 ```
- demo: `./resource/demo/0.multiband.hifigan.large.945000.wav`

## MultiBand-HiFiGAN (light)

- step: 845k
- training command: ``` bash train.sh 0 dataset/audio/train dataset/audio/valid dataset/mel/train dataset/mel/valid multiband-hifigan conf/multiband-hifigan/light.yaml 0 0 ```
- demo: `./resource/demo/0.multiband.hifigan.light.845000.wav`

## Basis-MelGAN

- step: 815k
- training command: ``` bash train.sh 0 dataset/audio/train dataset/audio/valid dataset/mel/train dataset/mel/valid basis-melgan conf/basis-melgan/light.yaml 0 0 ```
- demo: `./resource/demo/0.basis.melgan.light.815000.remove.wav` (LJSpeech demo: `./resource/demo/ljspeech`)
