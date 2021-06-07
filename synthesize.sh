checkpoint=$1
mel_path=$2
wav_path=$3
model_name=$4
config=$5

export MODE=synthesize

CUDA_VISIBLE_DEVICES=-1 python3 bin/launcher.py \
    --checkpoint $checkpoint \
    --mel_path $mel_path \
    --wav_path $wav_path \
    --model_name $model_name \
    --config $config
