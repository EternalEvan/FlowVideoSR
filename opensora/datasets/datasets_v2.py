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

import cv2
import random
def get_random_frames(video_path, num_frames):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < num_frames:
        print(f"视频总帧数 ({total_frames}) 小于请求的帧数 ({num_frames})。返回整个视频的所有帧。")
        num_frames = total_frames  # 将请求帧数调整为视频总帧数
    
    start_frame = random.randint(0, total_frames - num_frames) if total_frames >= num_frames else 0
    
    frames = []
    for i in range(start_frame, start_frame + num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    
    cap.release()
    return frames

def get_synced_frames(lq_video_path, gt_video_path, num_frames):
    # 读取LQ视频的帧
    lq_frames = get_random_frames(lq_video_path, num_frames)
    if not lq_frames:
        raise ValueError("Failed to read frames from the LQ video.")

    # 打开GT视频文件
    gt_cap = cv2.VideoCapture(gt_video_path)

    # 确保GT视频的总帧数至少与LQ视频相同
    total_frames = int(gt_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if len(lq_frames) > total_frames:
        raise ValueError("GT video does not have enough frames to match LQ video.")

    # 获取LQ视频的起始帧位置
    start_frame = int(gt_cap.get(cv2.CAP_PROP_POS_FRAMES))

    gt_frames = []
    for i in range(start_frame, start_frame + num_frames):
        gt_cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = gt_cap.read()
        if ret:
            gt_frames.append(frame)
        else:
            break

    # 释放GT视频文件
    gt_cap.release()

    # 转换为NumPy数组（张量）
    lq_frames = np.array(lq_frames)  # (num_frames, 高度, 宽度, 通道数)
    gt_frames = np.array(gt_frames)  # (num_frames, 高度, 宽度, 通道数)

    # 转换为PyTorch张量
    lq_frames_tensor = torch.tensor(lq_frames, dtype=torch.float32)
    gt_frames_tensor = torch.tensor(gt_frames, dtype=torch.float32)

    # 转置为(num_frames, 通道数, 高度, 宽度)
    lq_frames_tensor = lq_frames_tensor.permute(0, 3, 1, 2)
    gt_frames_tensor = gt_frames_tensor.permute(0, 3, 1, 2)

    # 标准化到[0, 1]范围（如果需要）
    lq_frames_tensor /= 255.0
    gt_frames_tensor /= 255.0

    return lq_frames_tensor, gt_frames_tensor



@DATASETS.register_module()
class VideoRecurrentTestDataset_v2(data.Dataset):
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
        super(VideoRecurrentTestDataset_v2, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.data_info = {'lq_path': [], 'gt_path': [], 'folder': [], 'idx': [], 'border': []}
        self.num_frames = opt['num_frames']
        self.image_size = opt['image_size']
        self.dataset_name = opt['name']
        self.mode = opt['mode']
        self.img_paths_gt = []
        self.img_paths_lq = []
        if self.dataset_name == 'REDS4':
            self.skip_list = ['000', '011', '015', '020']
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
        for subfolder_lq, subfolder_gt in zip(subfolders_lq, subfolders_gt):
            # get frame list for lq and gt
            subfolder_name = osp.basename(subfolder_lq)
            if self.mode == "train":
                if subfolder_name in self.skip_list:
                    continue
            else:
                if subfolder_name not in self.skip_list:
                    continue
            img_paths_lq = sorted(list(utils_video.scandir(subfolder_lq, recursive=True, full_path=True)))
            img_paths_gt = sorted(list(utils_video.scandir(subfolder_gt, recursive=True, full_path=True)))
            max_idx = len(img_paths_lq)
            assert max_idx == len(img_paths_gt), (f'Different number of images in lq ({max_idx})'
                                                  f' and gt folders ({len(img_paths_gt)})')

            self.img_paths_gt.extend(img_paths_gt)
            self.img_paths_lq.extend(img_paths_lq)
        
        self.sigma = opt['sigma'] / 255. if 'sigma' in opt else 0 # for non-blind video denoising
        # print(f'self.img_paths_gt = {self.img_paths_gt}')
        # print(f'self.img_paths_lq = {self.img_paths_lq}')
        # import pdb; pdb.set_trace()
    def __len__(self):
        return len(self.img_paths_gt)
    
    def getitem(self, index):
        img_path_gt = self.img_paths_gt[index]
        img_path_lq = self.img_paths_lq[index]
        # print(f'img_path_gt = {img_path_gt}')

        imgs_lq, imgs_gt = get_synced_frames(img_path_lq, img_path_gt, self.num_frames)
        # print(f'imgs_gt.shape = {imgs_gt.shape}')



        _, _, h, w = imgs_gt.shape



        # # # transform
        imgs = torch.stack([imgs_lq, imgs_gt], dim=0)
        # # print(f'imgs.shape = {imgs.shape}')

            
        top = torch.randint(0, imgs.shape[-2] - self.image_size[0] + 1, (1,)).item()
        left = torch.randint(0, imgs.shape[-1] - self.image_size[1] + 1, (1,)).item()
        imgs_crop = imgs[:, :, :, top:top+self.image_size[0], left: left+self.image_size[1]]
        [imgs_lq, imgs_gt] = imgs_crop



        
        imgs_lq = 2 * imgs_lq - 1
        imgs_gt = 2 * imgs_gt - 1
        # print(f'imgs_gt.shape = {imgs_gt.shape}')
        return {
            'L': imgs_lq,
            'H': imgs_gt,
            # 'folder': folder,
            'lq_path': img_path_lq, #self.imgs_lq[folder],
            'fps': 24,
            'height': self.image_size[1],
            'width': self.image_size[0],
            'num_frames': self.num_frames,
        }

    def __getitem__(self, index):
        return self.getitem(index)
