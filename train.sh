GPU=$1
dataset_audio=$2
dataset_audio_valid=$3
dataset_mel=$4
dataset_mel_valid=$5
model_name=$6
config=$7
use_scheduler=$8
mixprecision=${9:-'0'}
checkpoint_path=${10:-'""'}
restore_step=${11:-'0'}
if [ "$mixprecision" -eq "1" ]; then
    echo "mix precision training"
fi

export MODE=train

CUDA_VISIBLE_DEVICES=$GPU python3 bin/launcher.py \
    --audio_index_path $dataset_audio \
    --mel_index_path $dataset_mel \
    --audio_index_valid_path $dataset_audio_valid \
    --mel_index_valid_path $dataset_mel_valid \
    --model_name $model_name \
    --config $config \
    --use_scheduler $use_scheduler \
    --mixprecision $mixprecision \
    --checkpoint_path $checkpoint_path\
    --restore_step $restore_step
