# GAN Based Neural Vocoder

收录HiFi-GAN，MelGAN和Multi-Band HiFi-GAN，Discriminator包含MSD，MPD和MFD（from universal MelGAN）

## Train

1. 数据准备
    - 准备好你所有wav数据的路径，写入一个txt文件，一行写一个wav文件路径
    - 运行``` bash preprocess.sh <上述文件的路径> <处理完后的数据保存的路径>  <audio_index_path> <mel_index_path>```
        - ```audio_index_path```：preprocess结束后会分别生成audio和mel的用于‘train’，‘valid’和‘eval’的数据路径文件，audio的‘train’，‘valid’和‘eval’写入```<audio_index_path>```
        - ```mel_index_path```：mel的‘train’，‘valid’和‘eval’写入```<mel_index_path>```
    - 修改`hparams.py`中的23-25行改变你想要的‘train’，‘valid’和‘eval’大小，程序会随机从数据中选取所需数量的数据
2. 训练
    - 运行``` CUDA_VISIBLE_DEVICES=<GPU id> python3 train.py --audio_index_path <audio_index_path> --mel_index_path <mel_index_path>```
    - 如果你想从保存的路径开始，运行
    ```
    CUDA_VISIBLE_DEVICES=<GPU id> python3 train.py \
        --audio_index_path <audio_index_path> \
        --mel_index_path <mel_index_path> \
        --checkpoint_path <checkpoint path> \
        --restore_step <starting step from your checkpoint>
    ```
3. 默认模式
    - 下载`ljspeech`数据集至`dataset`文件夹
    - 直接运行``` bash preprocess.sh ```
    - 运行``` CUDA_VISIBLE_DEVICES=<GPU id> python3 train.py ```即可开始训练
