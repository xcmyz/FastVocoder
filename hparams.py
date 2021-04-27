import os

# Mel
num_mels = 80
num_freq = 1025
frame_length_ms = 50
frame_shift_ms = 10
fmin = 40
hop_size = 240
sample_rate = 24000
min_level_db = -100
ref_level_db = 20
preemphasize = True
preemphasis = 0.97
rescale_out = 0.4
signal_normalization = True
griffin_lim_iters = 60
power = 1.5


# Train
test_size = 0  # for testing training process
train_size = 12900
valid_size = 100

epochs = 100000
batch_size = 32
batch_expand_size = 8
discriminator_train_start_steps = 100000
n_warm_up_step = 0

use_feature_map_loss = False

learning_rate = 1e-4
learning_rate_discriminator = 5e-5
weight_decay = 1e-6
grad_clip_thresh = 1.0

log_step = 5
clear_time = 20
save_step = 5000

checkpoint_path = os.path.join("checkpoint")
logger_path = os.path.join("logger")

fixed_length = 70
lambda_adv = 1.
