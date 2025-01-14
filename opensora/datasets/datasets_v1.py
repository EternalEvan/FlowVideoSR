# REDS for CogVideoX

import os
from glob import glob
from os import path as osp
import numpy as np
import torch
from PIL import ImageFile
import torch.utils.data as data
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader

from opensora.registry import DATASETS
import opensora.utils.video_utils as utils_video
from .read_video import read_video
from .utils import VID_EXTENSIONS, get_transforms_image, get_transforms_video, read_file, temporal_random_crop
from PIL import Image
import torch.nn.functional as F
ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_FPS = 120




@DATASETS.register_module()
class VideoRecurrentTestDataset_v1(data.Dataset):
    """Video test dataset for recurrent architectures, which takes LR video
    frames as input and output corresponding HR video frames. Modified from
    https://github.com/xinntao/BasicSR/blob/master/basicsr/data/reds_dataset.py

    Supported datasets: Vid4, REDS4, REDSofficial.
    More generally, it supports testing dataset with following structures:

    dataroot
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── subfolder1
        ├── frame000
        ├── frame001
        ├── ...
    ├── ...

    For testing datasets, there is no need to prepare LMDB files.

    Args:
        opt (dict): Config for train dataset. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            io_backend (dict): IO backend type and other kwarg.
            cache_data (bool): Whether to cache testing datasets.
            name (str): Dataset name.
            meta_info_file (str): The path to the file storing the list of test
                folders. If not provided, all the folders in the dataroot will
                be used.
            num_frame (int): Window size for input frames.
            padding (str): Padding mode.
    """

    def __init__(self, opt):
        super(VideoRecurrentTestDataset_v1, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.data_info = {'lq_path': [], 'gt_path': [], 'folder': [], 'idx': [], 'border': []}
        self.num_frames = opt['num_frames']
        self.image_size = opt['image_size']
        self.dataset_name = opt['name']
        self.mode = opt['mode']
        self.crop_type = opt['crop_type']
        
        if self.dataset_name == 'REDS4':
            self.skip_list = ['000', '011', '015', '020']
            # self.skip_list = ['020']

        else:
            self.skip_list = []
        # print(f'self.lq_root = {self.lq_root}')
        # print(f'self.gt_root = {self.gt_root}')
        self.imgs_lq, self.imgs_gt = {}, {}
        if 'meta_info_file' in opt:
            print(f'meta_info_file exists')
            with open(opt['meta_info_file'], 'r') as fin:
                subfolders = [line.split(' ')[0] for line in fin]
                subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
                subfolders_gt = [osp.join(self.gt_root, key) for key in subfolders]
        else:
            subfolders_lq = sorted(glob(osp.join(self.lq_root, '*')))
            subfolders_gt = sorted(glob(osp.join(self.gt_root, '*')))
        # print("subfolders_lq = ", subfolders_lq)
        # print("subfolders_gt = ", subfolders_gt)
        # import pdb; pdb.set_trace()

        for subfolder_lq, subfolder_gt in zip(subfolders_lq, subfolders_gt):
            # get frame list for lq and gt
            subfolder_name = osp.basename(subfolder_lq)
            if self.mode == "train":
                if subfolder_name in self.skip_list:
                    continue
            else:
                if subfolder_name not in self.skip_list:
                    continue
            img_paths_lq = sorted(list(utils_video.scandir(subfolder_lq, full_path=True)))
            img_paths_gt = sorted(list(utils_video.scandir(subfolder_gt, full_path=True)))
            # print(f'img_paths_lq = {img_paths_lq}')
            # print(f'img_paths_gt = {img_paths_gt}')
            # import pdb; pdb.set_trace()
            max_idx = len(img_paths_lq)
            assert max_idx == len(img_paths_gt), (f'Different number of images in lq ({max_idx})'
                                                  f' and gt folders ({len(img_paths_gt)})')

            self.data_info['lq_path'].extend(img_paths_lq)
            self.data_info['gt_path'].extend(img_paths_gt)
            self.data_info['folder'].extend([subfolder_name] * max_idx)
            # print(f'subfolder_name = {subfolder_name}')
            # print(f'max_idx = {max_idx}')
            for i in range(max_idx):
                self.data_info['idx'].append(f'{i}/{max_idx}')
            border_l = [0] * max_idx
            for i in range(self.opt['num_frames'] // 2):
                border_l[i] = 1
                border_l[max_idx - i - 1] = 1
            self.data_info['border'].extend(border_l)

            # cache data or save the frame list
            if self.cache_data:
                print(f'Cache {subfolder_name} for VideoTestDataset...')
                self.imgs_lq[subfolder_name] = utils_video.read_img_seq(img_paths_lq)
                self.imgs_gt[subfolder_name] = utils_video.read_img_seq(img_paths_gt)
                # self.imgs_lq[subfolder_name], self.imgs_gt[subfolder_name] = utils_video.random_read_paired_img_seq(img_paths_lq, img_paths_gt, num_frames=self.num_frames)
            else:
                self.imgs_lq[subfolder_name] = img_paths_lq
                self.imgs_gt[subfolder_name] = img_paths_gt

        # Find unique folder strings
        # print(f"self.data_info_folder = {self.data_info['folder']}")
        self.folders = sorted(list(set(self.data_info['folder'])))
        # print(f'self.folders = {self.folders}')
        
        self.sigma = opt['sigma'] / 255. if 'sigma' in opt else 0 # for non-blind video denoising


    def __len__(self):
        return len(self.folders)
    
    def getitem(self, index):
        folder = self.folders[index]

        if self.sigma:
        # for non-blind video denoising
            if self.cache_data:
                imgs_gt = self.imgs_gt[folder]
            else:
                imgs_gt = utils_video.read_img_seq(self.imgs_gt[folder])

            torch.manual_seed(0)
            noise_level = torch.ones((1, 1, 1, 1)) * self.sigma
            noise = torch.normal(mean=0, std=noise_level.expand_as(imgs_gt))
            imgs_lq = imgs_gt + noise
            t, _, h, w = imgs_lq.shape
            imgs_lq = torch.cat([imgs_lq, noise_level.expand(t, 1, h, w)], 1)
        else:
        # for video sr and deblurring
            if self.cache_data:
                imgs_lq = self.imgs_lq[folder]
                imgs_gt = self.imgs_gt[folder]
            else:
                # print(f'self.imgs_gtfolder = {self.imgs_gt[folder]}')
                # print(f'self.imgs_lqfolder = {self.imgs_lq[folder]}')

                # import pdb; pdb.set_trace()
                # imgs_lq = utils_video.read_img_seq(self.imgs_lq[folder])
                # imgs_gt = utils_video.read_img_seq(self.imgs_gt[folder])
                # print(f'{self.imgs_lq[folder]}')
                # print(f'{self.imgs_gt[folder]}')
                imgs_lq, imgs_gt = utils_video.random_read_paired_img_seq(self.imgs_gt[folder], self.imgs_lq[folder], num_frame=self.num_frames)
        # print()


        _, _, h, w = imgs_gt.shape
        # print(f'h = {h}, w = {w}')
        # import pdb; pdb.set_trace()
        # 使用插值方法将 imgs_lq 调整到与 imgs_gt 相同的尺寸
        # imgs_lq = F.interpolate(imgs_lq, size=(h, w), mode='bicubic', align_corners=False)
        ######################################### 有问题,不能直接增大尺寸

        # # # # transform
        if self.crop_type == "random": 
            imgs = torch.stack([imgs_lq, imgs_gt], dim=0)
            top = torch.randint(0, imgs.shape[-2] - self.image_size[0] + 1, (1,)).item()
            left = torch.randint(0, imgs.shape[-1] - self.image_size[1] + 1, (1,)).item()
            imgs_crop = imgs[:, :, :, top:top+self.image_size[0], left: left+self.image_size[1]]
            [imgs_lq, imgs_gt] = imgs_crop
        elif self.crop_type == "center":
            imgs = torch.stack([imgs_lq, imgs_gt], dim=0)
            center_y = imgs.shape[-2] // 2
            center_x = imgs.shape[-1] // 2
            top = center_y - self.image_size[0] // 2
            left = center_x - self.image_size[1] // 2
            imgs_crop = imgs[:, :, :, top:top+self.image_size[0], left:left+self.image_size[1]]
            [imgs_lq, imgs_gt] = imgs_crop


        
        imgs_lq = 2 * imgs_lq - 1
        imgs_gt = 2 * imgs_gt - 1
        imgs_lq = imgs_lq.permute(1, 0, 2, 3)
        imgs_gt = imgs_gt.permute(1, 0, 2, 3)
        # print(f'imgs_lq.shape = {imgs_lq.shape}, imgs_gt.shape = {imgs_gt.shape}')
        # import pdb;pdb.set_trace()
        return {
            'L': imgs_lq,
            'H': imgs_gt,
            'folder': folder,
            'lq_path': self.imgs_lq[folder],
            'fps': 24,
            'height': self.image_size[1],
            'width': self.image_size[0],
            'num_frames': self.num_frames,
        }

    def __getitem__(self, index):
        return self.getitem(index)
