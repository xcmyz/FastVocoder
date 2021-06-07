# Fast (GAN Based Neural) Vocoder

收录MelGAN，HiFi-GAN和Multi-Band HiFi-GAN，Discriminator包含MSD，MPD和MFD（from universal MelGAN）

## 使用说明

1. 数据准备
    - 准备好你所有wav数据的路径，写入一个txt文件，一行写一个wav文件路径
    - 运行``` bash preprocess.sh <上述文件的路径> <处理完后的数据保存的路径>  <audio_index_path> <mel_index_path>```
        - ```audio_index_path```：preprocess结束后会分别生成audio和mel的用于‘train’，‘valid’和‘eval’的数据路径文件，audio的‘train’，‘valid’和‘eval’写入```<audio_index_path>```
        - ```mel_index_path```：mel的‘train’，‘valid’和‘eval’写入```<mel_index_path>```
    - 修改`hparams.py`中的23-25行改变你想要的‘train’，‘valid’和‘eval’大小，程序会随机从数据中选取所需数量的数据
2. 训练
    - 运行：
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
    - 例子：
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
3. 从checkpoint开始训练
    - 运行：
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
4. 生成样本
    - 运行：
    ```
    bash synthesize.sh \
        /path/to/checkpoint \
        /path/to/mel \
        /path/for/saving/wav \
        <model name> \
        /path/to/configuration/file
    ```

## 备注

1. 对于biaobei数据集，repo中的参数是已经经过调参的，可以跑一跑。
2. MultiBand-HiFiGAN模型遇到了较强的checkerboard artifacts的问题（生成音频在特定频率出现干扰），使用temporal nearest interpolation layer并且测试了大量不同的upsample size和kernel size，依然没有彻底解决问题，但是repo中提供的参数已经让checkerboard artifacts比较弱了（强的checkerboard artifacts会导致GAN训练的时候崩掉），Synthesize的时候为解决掉这个问题，采用了一个小技巧，大家可以参考代码。
3. 欢迎贡献代码，欢迎互相交流！