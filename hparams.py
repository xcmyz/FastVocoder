import os


# Mel
num_mels = 80
num_freq = 1025
sample_rate = 22050
frame_length_ms = 6.00
frame_shift_ms = 1.25
preemphasis_enable = False
preemphasis = 0.97
fmin = 40
min_level_db = -100
ref_level_db = 20
signal_normalization = True
griffin_lim_iters = 60
power = 1.5


# Model
L = 32
weight_dim = 256
dropout = 0.0
fixed_length = 80
lambda_adv = 1.0
expand_size = 16

use_feature_map_loss = False


# Train
test_size = 2000
preload_test_size = 13000

checkpoint_path = os.path.join("model_new")
logger_path = os.path.join("logger")
mel_ground_truth = os.path.join("mels")
alignment_path = os.path.join("alignments")
tasnet_path = os.path.join("model_tasnet", "checkpoint_270000.pth.tar")

batch_size = 32
epochs = 100000
n_warm_up_step = 4000
discriminator_train_start_steps = 300000

learning_rate = 1e-3
weight_decay = 1e-6
grad_clip_thresh = 1.0

save_step = 5000
log_step = 5
clear_time = 20

batch_expand_size = 8
