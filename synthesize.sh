checkpoint_path=$1
mel_path=$2
wav_path=$3

CUDA_VISIBLE_DEVICES=-1 python3 synthesize.py \
    --checkpoint $checkpoint_path \
    --mel_path $mel_path \
    --wav_path $wav_path
