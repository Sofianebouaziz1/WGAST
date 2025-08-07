import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
import os
sys.path.append(os.path.abspath('..'))  # go up to root directory

from model.WGAST import *
from data_loader.data import PatchSet, get_pair_path_with_masks
from data_loader.utils import *


import shutil
from timeit import default_timer as timer
from datetime import datetime
import numpy as np
import pandas as pd
import math

from math import exp
import random
from tqdm import tqdm

def gaussian(window_size, sigma):
    # Create a 1D Gaussian distribution centered at the middle of the window
    gauss = torch.Tensor([
        exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    
    # Normalize so that the sum of all values is 1 (i.e., it's a proper probability distribution)
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        max_val = 255 if torch.max(img1) > 128 else 1
        min_val = -1 if torch.min(img1) < -0.5 else 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average,
                       full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models,
    # not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output

class Experiment(object):
    def __init__(self, option):
        # Set device to GPU if available, otherwise use CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set image size from options
        self.image_size = option.image_size

        # Create and manage directories for saving models and logs
        self.save_dir = option.save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.train_dir = self.save_dir / 'train'
        self.train_dir.mkdir(exist_ok=True)
        self.history = self.train_dir / 'history.csv'
        self.test_dir = self.save_dir / 'test'
        self.test_dir.mkdir(exist_ok=True)
        self.best = self.train_dir / 'best.pth'
        self.last_g = self.train_dir / 'generator.pth'
        self.last_pd = self.train_dir / 'nlayerdiscriminator.pth'

        # Model configuration flags
        self.ifAdaIN = option.ifAdaIN  # Whether to use Adaptive Instance Normalization
        self.ifAttention = option.ifAttention  # Whether to use Attention Mechanism
        self.ifTwoInput = option.ifTwoInput  # Whether to use two input channels

        self.a = option.a  # Custom parameter a
        self.b = option.b  # Custom parameter b
        self.c = option.c  # Custom parameter c
        self.d = option.d  # Custom parameter d

        # Initialize logger
        self.logger = get_logger()
        self.logger.info('Model initialization')

        # Initialize generator and discriminator models
        self.generator = CombinFeatureGenerator(ifAdaIN=self.ifAdaIN, ifAttention=self.ifAttention, ifTwoInput=self.ifTwoInput).to(self.device)
        self.nlayerdiscriminator = NLayerDiscriminator(input_nc=2, getIntermFeat=True).to(self.device)

        # Define loss function for the discriminator
        self.pd_loss = GANLoss().to(self.device)

        # Handle multiple GPUs if available
        device_ids = [i for i in range(option.ngpu)]
        if option.cuda and option.ngpu > 1:
            self.generator = nn.DataParallel(self.generator, device_ids)
            self.nlayerdiscriminator = nn.DataParallel(self.nlayerdiscriminator, device_ids)

        # Set up optimizers for both generator and discriminator
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=option.lr)
        self.pd_optimizer = optim.Adam(self.nlayerdiscriminator.parameters(), lr=option.lr)

        # Learning rate scheduler (currently fixed at 1.0)
        def lambda_rule(epoch):
            lr_l = 1.0
            return lr_l

        self.g_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer, lr_lambda=lambda_rule)
        self.pd_scheduler = torch.optim.lr_scheduler.LambdaLR(self.pd_optimizer, lr_lambda=lambda_rule)

        # Log the number of trainable parameters in each model
        n_params = sum(p.numel() for p in self.generator.parameters() if p.requires_grad)
        self.logger.info(f'There are {n_params} trainable parameters for generator.')

        n_params = sum(p.numel() for p in self.nlayerdiscriminator.parameters() if p.requires_grad)
        self.logger.info(f'There are {n_params} trainable parameters for nlayerdiscriminator.')

        # Uncomment to log model architectures (disabled for now)
        # self.logger.info(str(self.generator))
        # self.logger.info(str(self.nlayerdiscriminator))

            
    def gaussian_kernel(self, sigma=1.0, channels=1):
        """
        Creates a 2D Gaussian kernel using the given sigma.

        Args:
            sigma (float): Standard deviation of the Gaussian.
            channels (int): Number of input channels.

        Returns:
            kernel (torch.Tensor): Gaussian kernel of shape [channels, 1, k, k]
        """
        # Compute kernel size: 2 * ceil(3*sigma) + 1
        kernel_size = int(2 * math.ceil(3 * sigma) + 1)
        ax = torch.arange(kernel_size) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel = kernel / kernel.sum()
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        kernel = kernel.repeat(channels, 1, 1, 1)
        return kernel, kernel_size

    def apply_gaussian_blur(self, input_tensor, sigma=1.0):
        """
        Applies Gaussian blur to a tensor (with gradient support).

        Args:
            input_tensor (torch.Tensor): Tensor of shape [B, C, H, W]
            sigma (float): Standard deviation of the Gaussian.

        Returns:
            torch.Tensor: Blurred tensor of shape [B, C, H, W]
        """
        B, C, H, W = input_tensor.shape
        kernel, kernel_size = self.gaussian_kernel(sigma=sigma, channels=C)
        kernel = kernel.to(input_tensor.device)

        padding = kernel_size // 2
        # Apply reflect padding manually (4 values: left, right, top, bottom)
        input_padded = input_padded = F.pad(input_tensor, [padding]*4, mode='replicate')

        # Convolve with groups=C to apply the filter per channel
        output = F.conv2d(input_padded, kernel, groups=C, padding=0)
        return output
    
    def train_on_epoch(self, n_epoch, data_loader):
        # Adjust learning rates
        self.g_scheduler.step()
        self.pd_scheduler.step()

        # Set models to training mode
        self.generator.train()
        self.nlayerdiscriminator.train()

        # Initialize loss trackers
        epg_loss = AverageMeter()  # Tracks generator loss
        eppd_loss = AverageMeter()  # Tracks discriminator loss
        epg_error = AverageMeter()  # Tracks mean squared error (MSE)
        # Log epoch start
        self.logger.info(f'Epoch[{n_epoch}] - {datetime.now()}')

        # Iterate over the dataset
        for idx, data in enumerate(tqdm(data_loader, desc="Processing")):            

            # Load and move input data to device (GPU/CPU)
            images, masks = data
            images = [im.to(self.device) for im in images]
            masks = [im.to(self.device) for im in masks]

            # Separate inputs and target
            inputs, target = images[:-1], images[-1:]
            
            # ----------------------
            # (1) Generate prediction
            # ----------------------
            prediction = self.generator(inputs)

            # ----------------------
            # (2) Weakly supervised learning
            # ----------------------
            LST_landsat_t2 = target[0][:, :1, :, :]

            smoothed = self.apply_gaussian_blur(prediction.clone(), sigma=1.0)
            prediction_interpolated = F.avg_pool2d(smoothed, kernel_size=3, stride=3)
            if prediction_interpolated.shape != LST_landsat_t2.shape:
                prediction_interpolated = F.interpolate(prediction_interpolated, size=LST_landsat_t2.shape[-2:], mode='bicubic', align_corners=False)


            LST_MODIS_t2_interpolated =F.avg_pool2d(inputs[3], kernel_size=3, stride=3)
            if LST_MODIS_t2_interpolated.shape != LST_landsat_t2.shape:
                LST_MODIS_t2_interpolated = F.interpolate(LST_MODIS_t2_interpolated, size=LST_landsat_t2.shape[-2:], mode='bicubic', align_corners=False)

            # Get discriminator outputs for fake and real images
            pred_fake = self.nlayerdiscriminator(torch.cat((prediction_interpolated, LST_MODIS_t2_interpolated), dim=1))
            pred_real1 = self.nlayerdiscriminator(torch.cat((LST_landsat_t2, LST_MODIS_t2_interpolated), dim=1))

            # ----------------------
            # (3) Update Discriminator
            # ----------------------
            # Compute discriminator loss
            pd_loss = (self.pd_loss(pred_fake, False) + self.pd_loss(pred_real1, True)) * 0.5

            # Backpropagate and update discriminator
            self.pd_optimizer.zero_grad()
            pd_loss.backward()
            self.pd_optimizer.step()
            torch.cuda.empty_cache() 

            # Update loss tracker
            eppd_loss.update(pd_loss.item())

            # ----------------------
            # (4) Update Generator
            # ----------------------
            prediction = self.generator(inputs)

            smoothed = self.apply_gaussian_blur(prediction.clone(), sigma=1.0)
            prediction_interpolated = F.avg_pool2d(smoothed, kernel_size=3, stride=3)
            if prediction_interpolated.shape != LST_landsat_t2.shape:
                prediction_interpolated = F.interpolate(prediction_interpolated, size=LST_landsat_t2.shape[-2:], mode='bicubic', align_corners=False)

            # Get discriminator outputs for fake and real images
            pred_fake = self.nlayerdiscriminator(torch.cat((prediction_interpolated, LST_MODIS_t2_interpolated), dim=1))

            # Compute adversarial loss
            loss_G_GAN = self.pd_loss(pred_fake, True) * self.a

            # Compute L1 loss with additional perceptual losses
            loss_G_l1 = (F.l1_loss(prediction_interpolated, LST_landsat_t2) * self.b +
                        (1.0 - msssim(prediction_interpolated, LST_landsat_t2, normalize=True)) * self.d +
                        (1.0 - torch.mean(F.cosine_similarity(prediction_interpolated, LST_landsat_t2, 1))) * self.c)

            # Total generator loss
            g_loss = loss_G_l1 + loss_G_GAN

            # Backpropagate and update generator
            self.g_optimizer.zero_grad()
            g_loss.backward()
            self.g_optimizer.step()
            torch.cuda.empty_cache() 

            # Update loss tracker
            epg_loss.update(g_loss.item())

            # Compute mean squared error
            mse = F.mse_loss(prediction_interpolated, LST_landsat_t2).item()
            epg_error.update(mse)

        # Log epoch completion time
        self.logger.info(f'Epoch[{n_epoch}] - {datetime.now()}')

        # Save model checkpoints
        save_checkpoint(self.generator, self.g_optimizer, self.last_g)
        save_checkpoint(self.nlayerdiscriminator, self.pd_optimizer, self.last_pd)

        # Return average losses
        return epg_loss.avg, eppd_loss.avg, epg_error.avg



    def train(self, train_dir, patch_size, patch_stride, batch_size,
            num_workers=0, epochs=50, resume=True):
        last_epoch = -1  # Initialize last epoch as -1
        least_error = float('inf')  # Set least validation error to infinity

        # Resume training if enabled and history file exists
        if resume and self.history.exists():
            df = pd.read_csv(self.history)  # Load training history
            last_epoch = int(df.iloc[-1]['epoch'])  # Get last completed epoch
            least_error = df['train_g_loss'].min()
    
            # Load latest saved model checkpoints
            load_checkpoint(self.last_g, self.generator, optimizer=self.g_optimizer)
            load_checkpoint(self.last_pd, self.nlayerdiscriminator, optimizer=self.pd_optimizer)

        start_epoch = last_epoch + 1  # Determine the starting epoch

        # Load trainin  data
        self.logger.info('Loading data...')
        train_set = PatchSet(train_dir, self.image_size, patch_size, patch_stride)  # Training dataset

        # Create data loaders for training and 
        train_loader = DataLoader(train_set, batch_size= batch_size, shuffle=True,
                                num_workers=1, drop_last=True)
        
        print("There are", len(train_set), "samples for training.")  # Log dataset size

        # Start training process
        self.logger.info('Training...')
        for epoch in range(start_epoch, epochs + start_epoch):
            # Log current learning rates
            self.logger.info(f"Learning rate for Generator: {self.g_optimizer.param_groups[0]['lr']}")
            self.logger.info(f"Learning rate for Discriminator: {self.pd_optimizer.param_groups[0]['lr']}")

            # Train the generator and discriminator for one epoch
            train_g_loss, train_pd_loss, train_g_error = self.train_on_epoch(epoch, train_loader)

            # Save training results to history file
            csv_header = ['epoch', 'train_g_loss', 'train_pd_loss', 'train_g_error']
            csv_values = [epoch, train_g_loss, train_pd_loss, train_g_error]
            log_csv(self.history, csv_values, header=csv_header)
            
            if  train_g_loss < least_error :
                least_error = train_g_loss
                shutil.copy(str(self.last_g), str(self.best))  # Save best generator model



    @torch.no_grad()
    def test(self, test_dir, patch_size, num_workers=0):
        print("*****************")
        self.generator.eval()
        load_checkpoint(self.best, model=self.generator)
        self.logger.info('Testing...')

        image_dirs = [p for p in test_dir.glob('*') if p.is_dir()]
        pairs = [get_pair_path_with_masks(d) for d in image_dirs]
        image_paths = [[p[0] for p in pair] for pair in pairs]

        # Patch stride (overlap control)
        patch_stride = [8 for _ in patch_size]

        rows = int((self.image_size[1] - patch_size[1]) / patch_stride[1]) + 1
        cols = int((self.image_size[0] - patch_size[0]) / patch_stride[0]) + 1
        
        test_set = PatchSet(test_dir, self.image_size, patch_size, patch_stride=patch_stride)
        test_loader = DataLoader(test_set, batch_size=1, num_workers=num_workers)
        n_blocks = len(test_loader)/len(image_paths)

        # Scaling for final image shape
        scaled_patch_size = tuple(i * 3 for i in patch_size)
        scaled_image_size = tuple(i * 3 for i in self.image_size)

        im_count = 0
        t_start = timer()

        patches = []

        print_indice = 0

        for data in test_loader:
            name = image_paths[im_count][-1].name.replace("Landsat", "Sentinel")
            if (print_indice ==0):
                print("Start test for image : ", name)
                print_indice = 1

            t_start = timer()  # Track time per batch

            images, masks = data
            images = [im.to(self.device) for im in images]
            masks = [im.to(self.device) for im in masks]

            inputs, target = images[:-1], images[-1:]
            prediction = self.generator(inputs)
            prediction = self.apply_gaussian_blur(prediction, sigma=1.0)

            patches.append(prediction.cpu().numpy())

            # If all patches for one image are collected
            if len(patches) == n_blocks:
                sum_buffer = np.zeros((NUM_BANDS, *scaled_image_size), dtype=np.float32)
                weight_buffer = np.zeros((1, *scaled_image_size), dtype=np.float32)

                block_count = 0
                for i in range(rows):
                    row_start = i * (patch_stride[1] * 3)  # vertical stride scaled by 3

                    for j in range(cols):
                        col_start = j * (patch_stride[0] * 3)  # horizontal stride scaled by 3

                        # Determine row/col starts and ends with conditional border exclusion
                        x1 = col_start if col_start == 0 else col_start + 1
                        y1 = row_start if row_start == 0 else row_start + 1

                        col_end_raw = col_start + scaled_patch_size[0]
                        row_end_raw = row_start + scaled_patch_size[1]

                        x2 = col_end_raw if col_end_raw == scaled_image_size[0]  else col_end_raw - 1
                        y2 = row_end_raw if row_end_raw == scaled_image_size[1]  else row_end_raw - 1

                        # Crop the patch only if it's not touching the edges
                        crop_left = 0 if col_start == 0 else 1
                        crop_right = None if col_end_raw == scaled_image_size[0] else -1
                        crop_top = 0 if row_start == 0 else 1
                        crop_bottom = None if row_end_raw == scaled_image_size[1] else -1

                        patch = patches[block_count][0][:, crop_left:crop_right, crop_top:crop_bottom]

                        # Update buffers
                        sum_buffer[:, x1:x2, y1:y2] += patch
                        weight_buffer[:, x1:x2, y1:y2] += 1

                        block_count += 1
                #patches.clear()
                # Normalize overlapping regions
                result = sum_buffer / (weight_buffer)

                # Save the full predicted image
                prototype = str(image_paths[im_count][2])
                save_array_as_tif(result, test_dir/ name, prototype=prototype)

                im_count += 1
                patches = []
                print_indice = 0
                t_end = timer()
                self.logger.info(f'Time cost: {t_end - t_start}s')

                print("End test for image : ", name)
                print("*****************************************************")
