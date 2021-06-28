checkpoint=$1
model_name=$2
config=$3

export MODE=test

CUDA_VISIBLE_DEVICES=-1 python3 bin/launcher.py \
    --checkpoint $checkpoint \
    --model_name $model_name \
    --config $config
