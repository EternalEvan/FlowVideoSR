# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import cv2
import glob
import os
import torch
import requests
import numpy as np
from os import path as osp
from collections import OrderedDict
from torch.utils.data import DataLoader

from models.network_vrt import VRT as net
from models.network_vrt import SimpleVRT as simplenet
from utils import utils_image as util
from utils.utils_video import img2tensor
from data.dataset_video_test import VideoRecurrentTestDataset, VideoTestVimeo90KDataset, \
    SingleVideoRecurrentTestDataset, VFI_DAVIS, VFI_UCF101, VFI_Vid4
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import math
from PIL import Image
import lpips
from torchvision.transforms.functional import normalize

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='001_VRT_videosr_bi_REDS_6frames', help='tasks: 001 to 008')
    parser.add_argument('--sigma', type=int, default=0, help='noise level for denoising: 10, 20, 30, 40, 50')
    parser.add_argument('--folder_lq', type=str, default='testsets/REDS4/sharp_bicubic',
                        help='input low-quality test video folder')
    parser.add_argument('--folder_gt', type=str, default=None)
    parser.add_argument('--folder_pred', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=10)
    args = parser.parse_args()

    
    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    test_results['lpips'] = []


    gt_root = args.folder_gt
    pred_root = args.folder_pred

    clips = sorted(os.listdir(gt_root))
    for clip in clips:

        psnr_scores = []
        ssim_scores = []
        psnry_scores = []
        ssimy_scores = []
        lpips_scores = []

        # gt_files = sorted(os.listdir(args.folder_gt))
        # pred_files = sorted(os.listdir(args.folder_pred))
        gt_files = sorted(os.listdir(os.path.join(gt_root, clip)))
        pred_files = sorted(os.listdir(os.path.join(pred_root, clip)))
        assert len(gt_files) == len(pred_files), "Number of files in gt and pred folders must be the same"
        num_batches = len(gt_files) // args.batch_size

        # lpips_model = lpips.LPIPS(net="alex")
        lpips_model = lpips.LPIPS(net="vgg")
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

        for gt_file, pred_file in zip(gt_files, pred_files):
            # gt_img = cv2.imread(os.path.join(args.folder_gt, gt_file))
            # pred_img = cv2.imread(os.path.join(args.folder_pred, pred_file))
            
            # gt_img = np.clip(gt_img.astype(np.float32) / 255., 0, 1)
            # pred_img = np.clip(pred_img.astype(np.float32) / 255., 0, 1)

            # gt_img = cv2.imread(os.path.join(args.folder_gt, gt_file)).astype(np.float32) / 255.
            gt_img = cv2.imread(os.path.join(gt_root, clip, gt_file)).astype(np.float32) / 255.

            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
            gt_img = torch.from_numpy(gt_img.transpose(2, 0, 1))
            gt_img = gt_img.float()
            gt_img = gt_img.data.cpu().clamp_(0,1).numpy()
            gt_img = np.transpose(gt_img[[2, 1, 0], :, :], (1, 2, 0))
            gt_img = (gt_img * 255.0).round().astype(np.uint8)
            gt_img = np.squeeze(gt_img)

            try:
                # pred_img = cv2.imread(os.path.join(args.folder_pred, pred_file)).astype(np.float32) / 255.
                pred_img = cv2.imread(os.path.join(pred_root, clip, pred_file)).astype(np.float32) / 255.

            except Exception as e:
                print(f"Error reading file: {pred_file}, Error: {e}")

            pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)
            pred_img = torch.from_numpy(pred_img.transpose(2, 0, 1))
            pred_img = pred_img.float()
            pred_img = pred_img.data.cpu().clamp_(0,1).numpy()
            pred_img = np.transpose(pred_img[[2, 1, 0], :, :], (1, 2, 0))
            pred_img = (pred_img * 255.0).round().astype(np.uint8)
            pred_img = np.squeeze(pred_img)

            # gt_Image = Image.open(os.path.join(args.folder_gt, gt_file))
            # pred_Image = Image.open(os.path.join(args.folder_pred, pred_file))
            # gt_Image_tensor = torch.tensor(np.array(gt_Image)).permute(2, 0, 1).unsqueeze(0).float() / 255.
            # pred_Image_tensor = torch.tensor(np.array(pred_Image)).permute(2, 0, 1).unsqueeze(0).float() / 255.
            # lpips_scores.append(lpips_model(gt_Image_tensor, pred_Image_tensor).item())

            # gt_Image = cv2.imread(os.path.join(args.folder_gt, gt_file), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            gt_Image = cv2.imread(os.path.join(gt_root, clip, gt_file), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

            # pred_Image = cv2.imread(os.path.join(args.folder_pred, pred_file), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
            pred_Image = cv2.imread(os.path.join(pred_root, clip, pred_file), cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

            gt_Image, pred_Image = img2tensor([gt_Image, pred_Image], bgr2rgb=True, float32=True)
            normalize(gt_Image, mean, std, inplace=True)
            normalize(pred_Image, mean, std, inplace=True)

            lpips_scores.append(lpips_model(pred_Image.unsqueeze(0), gt_Image.unsqueeze(0)).item())



            psnr_value = util.calculate_psnr(pred_img, gt_img, border=0)
            ssim_value = util.calculate_ssim(pred_img, gt_img, border=0)

            psnr_scores.append(psnr_value)
            ssim_scores.append(ssim_value)

            gt_img = util.bgr2ycbcr(gt_img.astype(np.float32) / 255.) * 255.
            pred_img = util.bgr2ycbcr(pred_img.astype(np.float32) / 255.) * 255.

            psnry_scores.append(util.calculate_psnr(pred_img, gt_img, border=0))
            ssimy_scores.append(util.calculate_ssim(pred_img, gt_img, border=0))

        avg_psnr = np.mean(psnr_scores)
        avg_ssim = np.mean(ssim_scores)
        avg_psnry = np.mean(psnry_scores)
        avg_ssimy = np.mean(ssimy_scores)
        avg_lpips = np.mean(lpips_scores)

        print(f"Now calculating clip {clip}")
        print(f"Average PSNR = {avg_psnr}, Average SSIM = {avg_ssim}")

        print(f"Average PSNR_y = {avg_psnry}, Average SSIM_y = {avg_ssimy}")

        print(f"Average LPIPS = {avg_lpips}")
        test_results['psnr'].append(avg_psnr)
        test_results['ssim'].append(avg_ssim)
        test_results['psnr_y'].append(avg_psnry)
        test_results['ssim_y'].append(avg_ssimy)
        test_results['lpips'].append(avg_lpips)
    
    print(f"Overall test results :")
    print(f"Average PSNR = {np.mean(test_results['psnr'])}, Average SSIM = {np.mean(test_results['ssim'])}")

    print(f"Average PSNR_y = {np.mean(test_results['psnr_y'])}, Average SSIM_y = {np.mean(test_results['ssim_y'])}")

    print(f"Average LPIPS = {np.mean(test_results['lpips'])}")

if __name__ == "__main__":
    main()