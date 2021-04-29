import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import random
import os
import time
import hparams as hp
import numpy as np

from datetime import datetime
from model.loss.loss import Loss
from optimizer import RAdam, ScheduledOptim
from model.generator import MelGANGenerator
from model.generator import HiFiGANGenerator
from model.generator import MultiBandHiFiGANGenerator
from model.discriminator import Discriminator
from model.generator.pqmf import PQMF

from data.dataset import BufferDataset, DataLoader
from data.dataset import load_data_to_buffer, collate_fn_tensor
from data.utils import get_param_num

random.seed(str(time.time()))
MODEL_NAME = "hifigan"
MULTI_BAND = False
if MODEL_NAME == "multiband-hifigan":
    MULTI_BAND = True


def trainer(model, discriminator,
            optimizer, discriminator_optimizer,
            scheduled_optim, discriminator_sche_optim,
            vocoder_loss,
            mel, wav,
            epoch, current_step, total_step,
            time_list, Start,
            current_checkpoint_path, current_logger_path,
            pqmf=None):
    # Start
    start_time = time.perf_counter()

    # Init
    scheduled_optim.zero_grad()
    discriminator_sche_optim.zero_grad()

    # Generator Forward
    est_source = model(mel)

    # Cal Loss
    total_loss = 0.
    stft_loss = vocoder_loss(est_source, wav, pqmf=pqmf)
    total_loss = total_loss + stft_loss

    # Adversarial
    a_l = 0.
    f_l = 0.
    if current_step > hp.discriminator_train_start_steps:
        if pqmf is not None:
            est_source = pqmf.synthesis(est_source)[:, 0, :]
        est_p = discriminator(est_source.unsqueeze(1))

        # for multi-scale discriminator
        adversarial_loss = 0.0
        for ii in range(len(est_p)):
            adversarial_loss += nn.BCEWithLogitsLoss()(est_p[ii][-1], est_p[ii][-1].new_ones(est_p[ii][-1].size()))
        adversarial_loss /= float(len(est_p))
        a_l = adversarial_loss.item()
        adversarial_loss = hp.lambda_adv * adversarial_loss
        total_loss = total_loss + adversarial_loss

        if hp.use_feature_map_loss:
            # feature matching loss
            # no need to track gradients
            with torch.no_grad():
                p = discriminator(wav.unsqueeze(1))
            feature_map_loss = 0.0
            for ii in range(len(est_p)):
                for jj in range(len(est_p[ii]) - 1):
                    feature_map_loss += nn.L1Loss()(est_p[ii][jj], p[ii][jj].detach())
            feature_map_loss /= (float(len(est_p)) * float(len(est_p[0]) - 1))
            f_l = feature_map_loss.item()
            total_loss = total_loss + feature_map_loss

    # Backward
    total_loss.backward()

    # Clipping gradients to avoid gradient explosion
    nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip_thresh)

    # Update weights
    if args.frozen_learning_rate:
        scheduled_optim.step_and_update_lr_frozen(args.learning_rate_frozen)
    else:
        scheduled_optim.step_and_update_lr()

    #######################
    #    Discriminator    #
    #######################
    d_l = 0.
    if current_step > hp.discriminator_train_start_steps:
        # re-compute y_ which leads better quality
        with torch.no_grad():
            est_source_for_d = model(mel)
            if pqmf is not None:
                est_source_for_d = pqmf.synthesis(est_source_for_d)[:, 0, :]

        # discriminator loss
        p = discriminator(wav.unsqueeze(1))
        est_p_for_d = discriminator(est_source_for_d.unsqueeze(1).detach())

        # for multi-scale discriminator
        real_loss = 0.0
        fake_loss = 0.0
        for ii in range(len(p)):
            real_loss += nn.BCEWithLogitsLoss()(p[ii][-1], p[ii][-1].new_ones(p[ii][-1].size()))
            fake_loss += nn.BCEWithLogitsLoss()(est_p_for_d[ii][-1], est_p_for_d[ii][-1].new_zeros(est_p_for_d[ii][-1].size()))
        real_loss /= float(len(p))
        fake_loss /= float(len(p))
        discriminator_loss = real_loss + fake_loss
        d_l = discriminator_loss.item()

        # Backward
        discriminator_loss.backward()

        # Clipping gradients to avoid gradient explosion
        nn.utils.clip_grad_norm_(discriminator.parameters(), hp.grad_clip_thresh)

        # Update weights
        if args.frozen_learning_rate:
            discriminator_sche_optim.step_and_update_lr_frozen(args.learning_rate_discriminator_frozen)
        else:
            discriminator_sche_optim.step_and_update_lr()

    # Logger
    t_l = total_loss.item()
    s_l = stft_loss.item()

    with open(os.path.join(current_logger_path, "total_loss.txt"), "a") as f_total_loss:
        f_total_loss.write(str(t_l)+"\n")
    with open(os.path.join(current_logger_path, "stft_loss.txt"), "a") as f_stft_loss:
        f_stft_loss.write(str(s_l)+"\n")

    # Print
    if current_step % hp.log_step == 0:
        Now = time.perf_counter()

        str1 = f"Epoch [{epoch + 1}/{hp.epochs}], Step [{current_step}/{total_step}]:"
        str2 = "STFT Loss: {:.6f}, Total Loss: {:.6f};".format(s_l, t_l)
        str3 = "Adversarial Loss: {:.6f}, Discriminator Loss: {:.6f}, Feature Map Loss: {:.6f};".format(a_l, d_l, f_l)
        str4 = "Current Learning Rate is {:.6f}, discriminator Learning Rate is {:.6f};".format(scheduled_optim.get_learning_rate(), discriminator_sche_optim.get_learning_rate())
        str5 = "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format((Now - Start), (total_step - current_step) * np.mean(time_list))

        print()
        print(str1)
        print(str2)
        print(str3)
        print(str4)
        print(str5)

        with open(os.path.join(current_logger_path, "logger.txt"), "a") as f_logger:
            f_logger.write(str1 + "\n")
            f_logger.write(str2 + "\n")
            f_logger.write(str3 + "\n")
            f_logger.write(str4 + "\n")
            f_logger.write(str5 + "\n")
            f_logger.write("\n")

    if current_step % hp.save_step == 0:
        torch.save(
            {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'discriminator': discriminator.state_dict(),
                'discriminator_optimizer': discriminator_optimizer.state_dict()
            },
            os.path.join(current_checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
        print("save model at step %d ..." % current_step)

    end_time = time.perf_counter()
    time_list = np.append(time_list, end_time - start_time)
    if len(time_list) == hp.clear_time:
        temp_value = np.mean(time_list)
        time_list = np.delete(time_list, [i for i in range(len(time_list))], axis=None)
        time_list = np.append(time_list, temp_value)
    return time_list


def main(args):
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define model
    print(f"Loading Model of {MODEL_NAME}...")
    if MODEL_NAME == "melgan":
        model = MelGANGenerator().to(device)
    elif MODEL_NAME == "hifigan":
        model = HiFiGANGenerator().to(device)
    elif MODEL_NAME == "multiband-hifigan":
        model = MultiBandHiFiGANGenerator().to(device)
    else:
        raise Exception("no model find!")
    pqmf = None
    if MULTI_BAND:
        print("Define PQMF")
        pqmf = PQMF().to(device)
    print("model is", model)
    discriminator = Discriminator().to(device)

    print("Model Has Been Defined")
    num_param = get_param_num(model)
    print('Number of TTS Parameters:', num_param)

    # Get buffer
    print("Load data to buffer")
    buffer = load_data_to_buffer(args.audio_index_path, args.mel_index_path)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate_frozen, eps=1.0e-6, weight_decay=0.0)
    scheduled_optim = ScheduledOptim(optimizer, 256, hp.n_warm_up_step, args.restore_step)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate_discriminator_frozen, eps=1.0e-6, weight_decay=0.0)
    discriminator_sche_optim = ScheduledOptim(discriminator_optimizer, 256, hp.n_warm_up_step, args.restore_step)
    vocoder_loss = Loss().to(device)
    print("Defined Optimizer and Loss Function.")

    # Load checkpoint if exists
    os.makedirs(hp.checkpoint_path, exist_ok=True)
    current_checkpoint_path = str(datetime.now()).replace(" ", "-").replace(":", "-").replace(".", "-")
    current_checkpoint_path = os.path.join(hp.checkpoint_path, current_checkpoint_path)
    try:
        checkpoint = torch.load(os.path.join(args.checkpoint_path), map_location=torch.device(device))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'discriminator' in checkpoint:
            print("loading discriminator")
            discriminator.load_state_dict(checkpoint['discriminator'])
            discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])
        os.makedirs(current_checkpoint_path, exist_ok=True)
        print("\n---Model Restored at Step %d---\n" % args.restore_step)
    except:
        print("\n---Start New Training---\n")
        os.makedirs(current_checkpoint_path, exist_ok=True)

    # Init logger
    os.makedirs(hp.logger_path, exist_ok=True)
    current_logger_path = str(datetime.now()).replace(" ", "-").replace(":", "-").replace(".", "-")
    current_logger_path = os.path.join(hp.logger_path, current_logger_path)
    os.makedirs(current_logger_path, exist_ok=True)

    # Get dataset
    dataset = BufferDataset(buffer)

    # Get Training Loader
    training_loader = DataLoader(dataset,
                                 batch_size=hp.batch_expand_size * hp.batch_size,
                                 shuffle=True,
                                 collate_fn=collate_fn_tensor,
                                 drop_last=True,
                                 num_workers=4,
                                 prefetch_factor=2,
                                 pin_memory=True)
    total_step = hp.epochs * len(training_loader) * hp.batch_expand_size

    # Define Some Information
    time_list = np.array([])
    Start = time.perf_counter()

    # Training
    model = model.train()

    for epoch in range(hp.epochs):
        for i, batchs in enumerate(training_loader):
            preload_data = None

            # real batch start here
            for j, db in enumerate(batchs):
                current_step = i * hp.batch_expand_size + j + args.restore_step + epoch * len(training_loader) * hp.batch_expand_size + 1

                # Get Data
                clock_1_s = time.perf_counter()
                mel, wav = 0, 0
                if preload_data == None:
                    mel = db["mel"].float().to(device)
                    wav = db["wav"].float().to(device)
                    mel = mel.contiguous().transpose(1, 2)
                else:
                    mel = preload_data["mel"]
                    wav = preload_data["wav"]
                clock_1_e = time.perf_counter()
                time_used_1 = round(clock_1_e - clock_1_s, 5)
                # print(f"loading data: {time_used_1}")

                # Training
                clock_2_s = time.perf_counter()
                time_list = trainer(
                    model, discriminator,
                    optimizer, discriminator_optimizer,
                    scheduled_optim, discriminator_sche_optim,
                    vocoder_loss,
                    mel, wav,
                    epoch, current_step, total_step,
                    time_list, Start,
                    current_checkpoint_path, current_logger_path,
                    pqmf=pqmf)
                clock_2_e = time.perf_counter()
                time_used_2 = round(clock_2_e - clock_2_s, 5)
                # print(f"training: {time_used_2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_index_path', type=str, default=os.path.join("dataset", "audio", "train"))
    parser.add_argument('--mel_index_path', type=str, default=os.path.join("dataset", "mel", "train"))
    parser.add_argument('--checkpoint_path', type=str, default="")
    parser.add_argument('--restore_step', type=int, default=0)
    parser.add_argument('--frozen_learning_rate', type=bool, default=True)
    parser.add_argument("--learning_rate_frozen", type=float, default=hp.learning_rate)
    parser.add_argument("--learning_rate_discriminator_frozen", type=float, default=hp.learning_rate_discriminator)
    args = parser.parse_args()
    main(args)
