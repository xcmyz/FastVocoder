import torch
import numpy as np
import hparams as hp
import torch.nn as nn
import torch.nn.functional as F

import os
import time
import math
import audio
import utils
import random
import ctypes
import argparse

from loss import DNNLoss
from multiprocessing import cpu_count

from optimizer import RAdam
from optimizers import ScheduledOptim

from model_melgan import MelGANGenerator as MelGANGenerator_light
from model_melgan_large import MelGANGenerator as MelGANGenerator_large
from discriminator import Discriminator

from dataset import BufferDataset, DataLoader
from dataset import get_data_to_buffer, collate_fn_tensor

from dataset_non_weight_buffer import NonWeightBufferDataset

random.seed(str(time.time()))


def trainer(model, discriminator,
            optimizer, discriminator_optimizer, basis_signal_optimizer,
            scheduled_optim, discriminator_sche_optim,
            vocoder_loss,
            mel, weight, wav,
            epoch, current_step, total_step,
            time_list, Start):
    # Start
    start_time = time.perf_counter()

    # Init
    basis_signal_optimizer.zero_grad()
    scheduled_optim.zero_grad()
    discriminator_sche_optim.zero_grad()

    # Generator Forward
    est_source, est_weight = model(mel)

    est_weight_average = est_weight.sum() / (est_weight.size(0) * est_weight.size(1) * est_weight.size(2))
    weight_average = weight.sum() / (weight.size(0) * weight.size(1) * weight.size(2))
    str0 = "est_weight average value: {:.6f}, weight average value: {:.6f}.".format(est_weight_average, weight_average)

    # Cal Loss
    total_loss = 0.
    weight_loss, stft_loss = vocoder_loss(est_weight, weight, est_source, wav)
    if current_step < hp.discriminator_train_start_steps:
        total_loss = total_loss + weight_loss + stft_loss
    else:
        total_loss = total_loss + stft_loss

    # Adversarial
    a_l = 0.
    f_l = 0.
    if current_step > hp.discriminator_train_start_steps:
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
            est_source_for_d, _ = model(mel)

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
    w_l = weight_loss.item()
    s_l = stft_loss.item()

    with open(os.path.join("logger", "total_loss.txt"), "a") as f_total_loss:
        f_total_loss.write(str(t_l)+"\n")
    with open(os.path.join("logger", "weight_loss.txt"), "a") as f_weight_loss:
        f_weight_loss.write(str(w_l)+"\n")
    with open(os.path.join("logger", "stft_loss.txt"), "a") as f_stft_loss:
        f_stft_loss.write(str(s_l)+"\n")

    # Print
    if current_step % hp.log_step == 0:
        Now = time.perf_counter()

        str1 = "Epoch [{}/{}], Step [{}/{}]:".format(epoch + 1, hp.epochs, current_step, total_step)
        str2 = "Weight Loss: {:.6f}, STFT Loss: {:.6f}, Total Loss: {:.6f};".format(w_l, s_l, t_l)
        str3 = "Adversarial Loss: {:.6f}, Discriminator Loss: {:.6f}, Feature Map Loss: {:.6f};".format(a_l, d_l, f_l)
        str4 = "Current Learning Rate is {:.6f}, discriminator Learning Rate is {:.6f};".format(scheduled_optim.get_learning_rate(), discriminator_sche_optim.get_learning_rate())
        str5 = "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format((Now - Start), (total_step - current_step) * np.mean(time_list))

        print()
        print(str0)
        print(str1)
        print(str2)
        print(str3)
        print(str4)
        print(str5)

        with open(os.path.join("logger", "logger.txt"), "a") as f_logger:
            f_logger.write(str0 + "\n")
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
            os.path.join(hp.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
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
    print("Use MelGAN Generator")
    basis_signal_weight = np.load(os.path.join("data", "basis_signal_weight.npy"))
    basis_signal_weight = torch.from_numpy(basis_signal_weight)
    model = 0
    if args.generator == "light":
        model = MelGANGenerator_light(basis_signal_weight).to(device)
    elif args.generator == "large":
        model = MelGANGenerator_large(basis_signal_weight).to(device)
    discriminator = Discriminator().to(device)

    print("Model Has Been Defined")
    num_param = utils.get_param_num(model)
    print('Number of TTS Parameters:', num_param)

    # Get buffer
    if args.dataset == "buffer":
        print("Load data to buffer")
        buffer = get_data_to_buffer()

    # Optimizer and loss
    basis_signal_optimizer = torch.optim.Adam(model.basis_signal.parameters())

    optimizer = torch.optim.Adam(model.melgan.parameters(), lr=args.learning_rate_frozen, eps=1.0e-6, weight_decay=0.0)
    scheduled_optim = ScheduledOptim(optimizer, hp.weight_dim, hp.n_warm_up_step, args.restore_step)

    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.learning_rate_discriminator_frozen, eps=1.0e-6, weight_decay=0.0)
    discriminator_sche_optim = ScheduledOptim(discriminator_optimizer, hp.weight_dim, hp.n_warm_up_step, args.restore_step)

    vocoder_loss = DNNLoss().to(device)
    print("Defined Optimizer and Loss Function.")

    # Load checkpoint if exists
    try:
        checkpoint = torch.load(os.path.join(hp.checkpoint_path, 'checkpoint_%d.pth.tar' % args.restore_step), map_location=torch.device(device))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'discriminator' in checkpoint:
            print("loading discriminator")
            discriminator.load_state_dict(checkpoint['discriminator'])
            discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])
        print("\n---Model Restored at Step %d---\n" % args.restore_step)
    except:
        print("\n---Start New Training---\n")
        if not os.path.exists(hp.checkpoint_path):
            os.mkdir(hp.checkpoint_path)

    # Init logger
    if not os.path.exists(hp.logger_path):
        os.mkdir(hp.logger_path)

    # Get dataset
    dataset = 0
    if args.dataset == "buffer":
        dataset = BufferDataset(buffer)
    else:
        dataset = NonWeightBufferDataset()

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
            number_of_batch = len(batchs)
            preload_data = None

            # real batch start here
            for j, db in enumerate(batchs):
                current_step = i * hp.batch_expand_size + j + args.restore_step + epoch * len(training_loader) * hp.batch_expand_size + 1

                # Get Data
                clock_1_s = time.perf_counter()
                mel, weight, wav = 0, 0, 0
                if preload_data == None:
                    mel = db["mel"].float().to(device)
                    weight = db["weight"].float().to(device)
                    wav = db["wav"].float().to(device)
                    mel = mel.contiguous().transpose(1, 2)
                else:
                    mel = preload_data["mel"]
                    weight = preload_data["weight"]
                    wav = preload_data["wav"]
                clock_1_e = time.perf_counter()
                time_used_1 = round(clock_1_e - clock_1_s, 5)
                # print(f"loading data: {time_used_1}")

                # Training
                clock_2_s = time.perf_counter()
                time_list = trainer(
                    model, discriminator,
                    optimizer, discriminator_optimizer, basis_signal_optimizer,
                    scheduled_optim, discriminator_sche_optim,
                    vocoder_loss,
                    mel, weight, wav,
                    epoch, current_step, total_step,
                    time_list, Start)
                clock_2_e = time.perf_counter()
                time_used_2 = round(clock_2_e - clock_2_s, 5)
                # print(f"training: {time_used_2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int, default=0)
    parser.add_argument('--frozen_learning_rate', type=bool, default=True)
    parser.add_argument("--learning_rate_frozen", type=float, default=1e-4)
    parser.add_argument("--learning_rate_discriminator_frozen", type=float, default=5e-5)
    parser.add_argument("--dataset", type=str, default="buffer")
    parser.add_argument("--generator", type=str, default="light")
    args = parser.parse_args()
    main(args)
