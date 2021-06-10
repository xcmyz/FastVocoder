import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
import random
import os
import time
import logging
import tensorboardX
import yaml
import hparams as hp
import numpy as np

from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime
from model.loss.loss import Loss
from model.generator import MelGANGenerator
from model.generator import HiFiGANGenerator
from model.generator import MultiBandHiFiGANGenerator
from model.generator.basis_melgan import BasisMelGANGenerator
from model.discriminator import Discriminator
from model.generator.pqmf import PQMF

from data.dataset import BufferDataset, OriginalDataset, DataLoader
from data.dataset import load_data_to_buffer, collate_fn_tensor, collate_fn_tensor_valid
from data.utils import get_param_num

from tensorboardX import SummaryWriter

random.seed(0)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


try:
    import apex
    from apex import amp
except:
    logger.info("Cannot load Apex (Using apex to accelerate training.)")


def trainer(model, discriminator,
            optimizer, discriminator_optimizer,
            scheduler, discriminator_scheduler,
            vocoder_loss,
            mel, wav,
            epoch, current_step, total_step,
            time_list, Start,
            current_checkpoint_path, current_logger_path, tensorboard_writer,
            pqmf=None, mixprecision=0):
    # Start
    start_time = time.perf_counter()

    # Init
    optimizer.zero_grad()
    discriminator_optimizer.zero_grad()

    # Generator Forward
    est_source = model(mel)

    # Cal Loss
    total_loss = 0.
    stft_loss = vocoder_loss(est_source, wav, pqmf=pqmf)
    s_l = stft_loss.item()
    stft_loss = hp.lambda_stft * stft_loss
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
            adversarial_loss += nn.MSELoss()(est_p[ii][-1], est_p[ii][-1].new_ones(est_p[ii][-1].size()))
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
            feature_map_loss = hp.lambda_fm * feature_map_loss
            total_loss = total_loss + feature_map_loss

    # Backward
    if mixprecision:
        with amp.scale_loss(total_loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        total_loss.backward()

    # Clipping gradients to avoid gradient explosion
    if mixprecision:
        nn.utils.clip_grad_norm_(amp.master_params(optimizer), hp.grad_clip_thresh)
    else:
        nn.utils.clip_grad_norm_(model.parameters(), hp.grad_clip_thresh)

    # Update weights
    optimizer.step()
    if scheduler is not None:
        scheduler.step()

    #######################
    #    Discriminator    #
    #######################
    d_l = 0.
    if current_step > hp.discriminator_train_start_steps:
        # add zero grad module
        discriminator_optimizer.zero_grad()

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
            real_loss += nn.MSELoss()(p[ii][-1], p[ii][-1].new_ones(p[ii][-1].size()))
            fake_loss += nn.MSELoss()(est_p_for_d[ii][-1], est_p_for_d[ii][-1].new_zeros(est_p_for_d[ii][-1].size()))
        real_loss /= float(len(p))
        fake_loss /= float(len(p))
        discriminator_loss = real_loss + fake_loss
        d_l = discriminator_loss.item()

        # Backward
        if mixprecision:
            with amp.scale_loss(discriminator_loss, discriminator_optimizer) as discriminator_scaled_loss:
                discriminator_scaled_loss.backward()
        else:
            discriminator_loss.backward()

        # Clipping gradients to avoid gradient explosion
        if mixprecision:
            nn.utils.clip_grad_norm_(amp.master_params(discriminator_optimizer), hp.grad_clip_thresh)
        else:
            nn.utils.clip_grad_norm_(discriminator.parameters(), hp.grad_clip_thresh)

        # Update weights
        discriminator_optimizer.step()
        if discriminator_scheduler is not None:
            discriminator_scheduler.step()

    # Logger
    t_l = total_loss.item()
    with open(os.path.join(current_logger_path, "total_loss.txt"), "a") as f_total_loss:
        f_total_loss.write(str(t_l)+"\n")
    with open(os.path.join(current_logger_path, "stft_loss.txt"), "a") as f_stft_loss:
        f_stft_loss.write(str(s_l)+"\n")

    # Logging
    if current_step % hp.log_step == 0:
        Now = time.perf_counter()

        str1 = f"Epoch [{epoch + 1}/{hp.epochs}], Step [{current_step}/{total_step}]:"
        str2 = "STFT Loss: {:.6f}, Total Loss: {:.6f};".format(s_l, t_l)
        str3 = "Adversarial Loss: {:.6f}, Discriminator Loss: {:.6f}, Feature Map Loss: {:.6f};".format(a_l, d_l, f_l)
        if scheduler is None:
            str4 = "Current Learning Rate is {:.6f}, discriminator Learning Rate is {:.6f};".format(hp.learning_rate, hp.learning_rate_discriminator)
        else:
            str4 = "Current Learning Rate is {:.6f}, discriminator Learning Rate is {:.6f};".format(scheduler.get_last_lr()[-1], discriminator_scheduler.get_last_lr()[-1])
        str5 = "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format((Now - Start), (total_step - current_step) * np.mean(time_list))

        logger.info("\n")
        logger.info(str1)
        logger.info(str2)
        logger.info(str3)
        logger.info(str4)
        logger.info(str5)

        with open(os.path.join(current_logger_path, "logger.txt"), "a") as f_logger:
            f_logger.write(str1 + "\n")
            f_logger.write(str2 + "\n")
            f_logger.write(str3 + "\n")
            f_logger.write(str4 + "\n")
            f_logger.write(str5 + "\n")
            f_logger.write("\n")

        tensorboard_writer.add_scalar('total_loss', t_l, global_step=current_step)
        tensorboard_writer.add_scalar('stft_loss', s_l, global_step=current_step)
        tensorboard_writer.add_scalar('adversarial_loss', a_l, global_step=current_step)
        tensorboard_writer.add_scalar('discriminator_loss', d_l, global_step=current_step)
        tensorboard_writer.add_scalar('feature_map_loss', f_l, global_step=current_step)
        if scheduler is not None:
            tensorboard_writer.add_scalar('learning_rate', scheduler.get_last_lr()[-1], global_step=current_step)

    if current_step % hp.save_step == 0:
        checkpoint_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'discriminator': discriminator.state_dict(),
            'discriminator_optimizer': discriminator_optimizer.state_dict()
        }
        if mixprecision:
            checkpoint_dict.update({'amp': amp.state_dict()})
        torch.save(
            checkpoint_dict,
            os.path.join(current_checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
        logger.info("save model at step %d ..." % current_step)

    end_time = time.perf_counter()
    time_list = np.append(time_list, end_time - start_time)
    if len(time_list) == hp.clear_time:
        temp_value = np.mean(time_list)
        time_list = np.delete(time_list, [i for i in range(len(time_list))], axis=None)
        time_list = np.append(time_list, temp_value)
    return time_list


def run(args):
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define model
    logger.info(f"Loading Model of {args.model_name}...")
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    hp.lambda_stft = config["lamda_stft"]

    if args.model_name == "melgan":
        model = MelGANGenerator(in_channels=config["in_channels"],
                                out_channels=config["out_channels"],
                                kernel_size=config["kernel_size"],
                                channels=config["channels"],
                                upsample_scales=config["upsample_scales"],
                                stack_kernel_size=config["stack_kernel_size"],
                                stacks=config["stacks"],
                                use_weight_norm=config["use_weight_norm"],
                                use_causal_conv=config["use_causal_conv"]).to(device)
    elif args.model_name == "hifigan":
        model = HiFiGANGenerator(resblock_kernel_sizes=config["resblock_kernel_sizes"],
                                 upsample_rates=config["upsample_rates"],
                                 upsample_initial_channel=config["upsample_initial_channel"],
                                 resblock_type=config["resblock_type"],
                                 upsample_kernel_sizes=config["upsample_kernel_sizes"],
                                 resblock_dilation_sizes=config["resblock_dilation_sizes"],
                                 transposedconv=config["transposedconv"],
                                 bias=config["bias"]).to(device)
    elif args.model_name == "multiband-hifigan":
        model = MultiBandHiFiGANGenerator(resblock_kernel_sizes=config["resblock_kernel_sizes"],
                                          upsample_rates=config["upsample_rates"],
                                          upsample_initial_channel=config["upsample_initial_channel"],
                                          resblock_type=config["resblock_type"],
                                          upsample_kernel_sizes=config["upsample_kernel_sizes"],
                                          resblock_dilation_sizes=config["resblock_dilation_sizes"],
                                          transposedconv=config["transposedconv"],
                                          bias=config["bias"]).to(device)
    elif args.model_name == "basis-melgan":
        basis_signal_weight = np.load(os.path.join("Basis-MelGAN-dataset", "basis_signal_weight.npy"))
        basis_signal_weight = torch.from_numpy(basis_signal_weight)
        model = BasisMelGANGenerator(basis_signal_weight=basis_signal_weight,
                                     L=config["L"],
                                     in_channels=config["in_channels"],
                                     out_channels=config["out_channels"],
                                     kernel_size=config["kernel_size"],
                                     channels=config["channels"],
                                     upsample_scales=config["upsample_scales"],
                                     stack_kernel_size=config["stack_kernel_size"],
                                     stacks=config["stacks"],
                                     use_weight_norm=config["use_weight_norm"],
                                     use_causal_conv=config["use_causal_conv"]).to(device)
    else:
        raise Exception("no model find!")
    pqmf = None
    if config["multiband"] == True:
        logger.info("Define PQMF")
        pqmf = PQMF().to(device)
    logger.info(f"model is {str(model)}")
    discriminator = Discriminator().to(device)

    logger.info("Model Has Been Defined")
    num_param = get_param_num(model)
    logger.info(f'Number of TTS Parameters: {num_param}')

    # Optimizer and loss
    if not args.mixprecision:
        if args.model_name == "basis-melgan":
            optimizer = Adam(model.melgan.parameters(), lr=args.learning_rate, eps=1.0e-6, weight_decay=0.0)
            # freeze basis signal layer
            basis_signal_optimizer = Adam(model.basis_signal.parameters())
        else:
            optimizer = Adam(model.parameters(), lr=args.learning_rate, eps=1.0e-6, weight_decay=0.0)
        discriminator_optimizer = Adam(discriminator.parameters(), lr=args.learning_rate_discriminator, eps=1.0e-6, weight_decay=0.0)
    else:
        if args.model_name == "basis-melgan":
            raise Exception("basis melgan don't support amp!")
        optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=args.learning_rate)
        discriminator_optimizer = apex.optimizers.FusedAdam(discriminator.parameters(), lr=args.learning_rate_discriminator)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1", keep_batchnorm_fp32=None)
        discriminator, discriminator_optimizer = amp.initialize(discriminator, discriminator_optimizer, opt_level="O1")
        logger.info("Start mix precision training...")

    if args.use_scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=2500, eta_min=args.learning_rate / 10.)
        discriminator_scheduler = CosineAnnealingLR(discriminator_optimizer,
                                                    T_max=2500,
                                                    eta_min=args.learning_rate_discriminator / 10.)
    else:
        scheduler = None
        discriminator_scheduler = None
    vocoder_loss = Loss().to(device)
    logger.info("Defined Optimizer and Loss Function.")

    # Load checkpoint if exists
    os.makedirs(hp.checkpoint_path, exist_ok=True)
    current_checkpoint_path = str(datetime.now()).replace(" ", "-").replace(":", "-").replace(".", "-")
    current_checkpoint_path = os.path.join(hp.checkpoint_path, current_checkpoint_path)
    try:
        checkpoint = torch.load(os.path.join(args.checkpoint_path), map_location=torch.device(device))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'discriminator' in checkpoint:
            logger.info("loading discriminator")
            discriminator.load_state_dict(checkpoint['discriminator'])
            discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer'])
        os.makedirs(current_checkpoint_path, exist_ok=True)
        if args.mixprecision:
            amp.load_state_dict(checkpoint['amp'])
        logger.info("\n---Model Restored at Step %d---\n" % args.restore_step)
    except:
        logger.info("\n---Start New Training---\n")
        os.makedirs(current_checkpoint_path, exist_ok=True)

    # Init logger
    os.makedirs(hp.logger_path, exist_ok=True)
    current_logger_path = str(datetime.now()).replace(" ", "-").replace(":", "-").replace(".", "-")
    writer = SummaryWriter(os.path.join(hp.tensorboard_path, current_logger_path))
    current_logger_path = os.path.join(hp.logger_path, current_logger_path)
    os.makedirs(current_logger_path, exist_ok=True)

    # Get buffer
    if args.model_name != "basis-melgan":
        logger.info("Load data to buffer")
        buffer = load_data_to_buffer(args.audio_index_path, args.mel_index_path, logger, feature_savepath="features_train.bin")
        logger.info("Load valid data to buffer")
        valid_buffer = load_data_to_buffer(args.audio_index_valid_path, args.mel_index_valid_path, logger, feature_savepath="features_valid.bin")

    # Get dataset
    if args.model_name == "basis-melgan":
        dataset = OriginalDataset(args.audio_index_path,
                                  args.mel_index_path,
                                  args.weight_index_path,
                                  config["L"])
        valid_dataset = OriginalDataset(args.audio_index_valid_path,
                                        args.mel_index_valid_path,
                                        args.weight_index_valid_path,
                                        config["L"])
    else:
        dataset = BufferDataset(buffer)
        valid_dataset = BufferDataset(valid_buffer)

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

                # Training
                clock_2_s = time.perf_counter()
                time_list = trainer(
                    model, discriminator,
                    optimizer, discriminator_optimizer,
                    scheduler, discriminator_scheduler,
                    vocoder_loss,
                    mel, wav,
                    epoch, current_step, total_step,
                    time_list, Start,
                    current_checkpoint_path, current_logger_path, writer,
                    pqmf=pqmf,
                    mixprecision=args.mixprecision)
                clock_2_e = time.perf_counter()
                time_used_2 = round(clock_2_e - clock_2_s, 5)

                if current_step % hp.valid_step == 0:
                    logger.info("Start valid...")
                    valid_loader = DataLoader(valid_dataset, batch_size=1,
                                              shuffle=True,
                                              collate_fn=collate_fn_tensor_valid,
                                              num_workers=0)
                    valid_loss_all = 0.
                    for ii, valid_batch in enumerate(valid_loader):
                        valid_mel = valid_batch["mel"].float().to(device)
                        valid_mel = valid_mel.contiguous().transpose(1, 2)
                        valid_wav = valid_batch["wav"].float().to(device)
                        with torch.no_grad():
                            valid_est_source = model(valid_mel)
                            valid_stft_loss = vocoder_loss(valid_est_source, valid_wav, pqmf=pqmf)
                            valid_loss_all += valid_stft_loss.item()
                        if ii == hp.valid_num:
                            break
                    writer.add_scalar('valid_stft_loss', valid_loss_all / float(hp.valid_num), global_step=current_step)

    writer.export_scalars_to_json(os.path.join("all_scalars.json"))
    writer.close()
    return


def run_train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_index_path", type=str, default=os.path.join("dataset", "audio", "train"))
    parser.add_argument("--mel_index_path", type=str, default=os.path.join("dataset", "mel", "train"))
    parser.add_argument("--weight_index_path", type=str, default=os.path.join("dataset", "weight", "train"))

    parser.add_argument("--audio_index_valid_path", type=str, default=os.path.join("dataset", "audio", "valid"))
    parser.add_argument("--mel_index_valid_path", type=str, default=os.path.join("dataset", "mel", "valid"))
    parser.add_argument("--weight_index_valid_path", type=str, default=os.path.join("dataset", "weight", "valid"))

    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--restore_step", type=int, default=0)

    parser.add_argument("--learning_rate", type=float, default=hp.learning_rate)
    parser.add_argument("--learning_rate_discriminator", type=float, default=hp.learning_rate_discriminator)

    parser.add_argument("--model_name", type=str, help="melgan, hifigan and multiband-hifigan.")
    parser.add_argument("--config", type=str, help="path to model configuration file")

    parser.add_argument("--use_scheduler", type=int, default=0)
    parser.add_argument("--mixprecision", type=int, default=0)

    args = parser.parse_args()
    run(args)
