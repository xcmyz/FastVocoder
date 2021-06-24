checkpoint=$1
model_name=$2
config=$3
save_path=$4

export MODE=publish

CUDA_VISIBLE_DEVICES=-1 python3 bin/launcher.py \
    --checkpoint $checkpoint \
    --model_name $model_name \
    --config $config \
    --save_path $save_path
