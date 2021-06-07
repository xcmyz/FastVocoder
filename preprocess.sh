data_path=$1
save_path=$2
audio_index_path=$3
mel_index_path=$4

export MODE=preprocess

python3 bin/launcher.py \
    --data_path $data_path \
    --save_path $save_path \
    --audio_index_path $audio_index_path \
    --mel_index_path $mel_index_path
