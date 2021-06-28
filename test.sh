checkpoint=$1
filelist=$2
model_name=$3
config=$4

export MODE=test

CUDA_VISIBLE_DEVICES=-1 python3 bin/launcher.py \
    --checkpoint $checkpoint \
    --model_name $model_name \
    --config $config \
    --file_path $filelist
