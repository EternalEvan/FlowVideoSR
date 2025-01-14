import os
from glob import glob
from os import path as osp
import numpy as np
import torch
from PIL import ImageFile, Image
import torch.utils.data as data
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader

from opensora.registry import DATASETS
import opensora.utils.video_utils as utils_video

from .read_video import read_video
from .utils import VID_EXTENSIONS, get_transforms_image, get_transforms_video, read_file, temporal_random_crop
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_FPS = 120


from typing import Dict, Sequence
import math
import time

from .realesrgan_utils import circular_lowpass_kernel, random_mixed_kernels, augment, load_file_list #  random_crop_arr, center_crop_arr,


@DATASETS.register_module()
class VideoRecurrentDataset(data.Dataset):


    # Supported datasets: YouHQ


    def __init__(self, opt):
        super(VideoRecurrentDataset, self).__init__()
        self.opt = opt
        # self.cache_data = opt['cache_data']
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.num_frames = opt['num_frames']
        self.image_size = opt['image_size']
        self.crop_type = opt['crop_type']
        self.video_paths_lq = []
        self.video_paths_gt = []

        self.blur_kernel_size = 21
        self.kernel_list = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']

        self.kernel_prob = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.sinc_prob = 0.1
        self.blur_sigma = [0.2, 3]
        self.betag_range = [0.5, 4]
        self.betap_range = [1, 2]

        self.blur_kernel_size2 = 21
        self.kernel_list2 = ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso']
        self.kernel_prob2 = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
        self.sinc_prob2 = 0.1
        self.blur_sigma2 = [0.2, 1.5]
        self.betag_range2 = [0.5, 4]
        self.betap_range2 = [1, 2]

        self.final_sinc_prob = 0.8

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        self.pulse_tensor = torch.zeros(21, 21).float()
        self.pulse_tensor[10, 10] = 1

        self.use_hflip = False
        self.use_rot = False




        self.imgs_lq, self.imgs_gt = {}, {}
        if 'meta_info_file' in opt:
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
            self.video_paths_lq.extend(sorted(list(utils_video.scandir(subfolder_lq, recursive=True, full_path=True))))
            self.video_paths_gt.extend(sorted(list(utils_video.scandir(subfolder_gt, recursive=True, full_path=True))))
            # print("self.video_paths_lq = ", self.video_paths_lq)
            # print("self.video_paths_gt = ", self.video_paths_gt)
 

        assert len(self.video_paths_lq) == len(self.video_paths_gt)


    def __len__(self):
        return len(self.video_paths_lq)
    
    def getitem(self, index):
        video_path_gt = self.video_paths_gt[index]

        cap = cv2.VideoCapture(self.folders[index])
        if not cap.isOpened(): 
            print(f"Unable to open the video file: {video_path}")
            return None
                    
        frames = []
        frame_count = 0
        
        while frame_count < self.num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            frame_count += 1
    
        cap.release()

        imgs_gt = np.array(frames).transpose(3,0,1,2) # [C, T, H, W]
        imgs_gt = (imgs_gt / 255.0)

        imgs_lq = self.esrgan_degradation_numpy(imgs_gt)

        
    






        # # 获取视频的帧数
        # num_frames_gt = len(imgs_gt)
        # assert num_frames_lq == num_frames_gt

        # # 如果视频帧数少于所需的帧数，则调整 self.num_frames 以匹配实际帧数
        # if num_frames_lq < self.num_frames or num_frames_gt < self.num_frames:
        #     self.num_frames = min(num_frames_lq, num_frames_gt)
        #     imgs_lq = imgs_lq[:self.num_frames]
        #     imgs_gt = imgs_gt[:self.num_frames]
        
        # # 随机选择起始帧
        # start_frame = random.randint(0, len(imgs_lq) - self.num_frames)

        # # 选择截取的帧
        # imgs_lq = imgs_lq[start_frame:start_frame + self.num_frames]
        # imgs_gt = imgs_gt[start_frame:start_frame + self.num_frames]

        # # 空间上随机裁剪
        # height, width = imgs_lq.shape[2], imgs_lq.shape[3]

        # if height > self.image_size[1] and width > self.image_size[0]:
        #     top = random.randint(0, height - self.image_size[1])
        #     left = random.randint(0, width - self.image_size[0])
            
        #     imgs_lq = imgs_lq[:, :, top:top + self.image_size[1], left:left + self.image_size[0]]
        #     imgs_gt = imgs_gt[:, :, top:top + self.image_size[1], left:left + self.image_size[0]]
        # else:
        #     # 如果视频帧的高度或宽度小于所需的裁剪尺寸
        #     # 这时候裁剪区域会是整个视频帧的区域
        #     top = 0
        #     left = 0
        #     imgs_lq = imgs_lq[:, :, top:top + self.image_size[1], left:left + self.image_size[0]]
        #     imgs_gt = imgs_gt[:, :, top:top + self.image_size[1], left:left + self.image_size[0]]



        # imgs_lq = 2 * imgs_lq - 1
        # imgs_gt = 2 * imgs_gt - 1

        # return {
        #     'L': imgs_lq,
        #     'H': imgs_gt,

        #     'fps': 24,
        #     'height': self.image_size[1],
        #     'width': self.image_size[0],
        #     'num_frames': self.num_frames,
        # }

    


    def __getitem__(self, index):
        return self.getitem(index)



    def esrgan_degradation_numpy(self, imgs_gt):
        C, T, H, W = imgs_gt.shape
        lq_list = []

        if self.crop_type == "random":
            video_array = random_crop_arr(video_array, self.out_size)
        elif self.crop_type == "center":
            video_array = center_crop_arr(video_array, self.out_size)
        # self.crop_type is "none"
        else:
            assert video_array.shape[2:4] == (self.out_size, self.out_size)
        video_array = np.transpose(video_array, (1, 2, 3, 0)) # [T, H, W, C]

        lq_list = []
        for i in range(T):
            img_hq = video_array[i, :, :, :]

            # -------------------- Do augmentation for training: flip, rotation -------------------- #
            img_hq = augment(img_hq, self.use_hflip, self.use_rot)

            # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
            kernel_size = random.choice(self.kernel_range)
            if np.random.uniform() < self.sinc_prob:
                # this sinc filter setting is for kernels ranging from [7, 21]
                if kernel_size < 13:
                    omega_c = np.random.uniform(np.pi / 3, np.pi)
                else:
                    omega_c = np.random.uniform(np.pi / 5, np.pi)
                kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            else:
                kernel = random_mixed_kernels(
                    self.kernel_list,
                    self.kernel_prob,
                    kernel_size,
                    self.blur_sigma,
                    self.blur_sigma, [-math.pi, math.pi],
                    self.betag_range,
                    self.betap_range,
                    noise_range=None
                )
            # pad kernel
            pad_size = (21 - kernel_size) // 2
            kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

            # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
            kernel_size = random.choice(self.kernel_range)
            if np.random.uniform() < self.sinc_prob2:
                if kernel_size < 13:
                    omega_c = np.random.uniform(np.pi / 3, np.pi)
                else:
                    omega_c = np.random.uniform(np.pi / 5, np.pi)
                kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
            else:
                kernel2 = random_mixed_kernels(
                    self.kernel_list2,
                    self.kernel_prob2,
                    kernel_size,
                    self.blur_sigma2,
                    self.blur_sigma2, [-math.pi, math.pi],
                    self.betag_range2,
                    self.betap_range2,
                    noise_range=None
                )

            # pad kernel
            pad_size = (21 - kernel_size) // 2
            kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))


            # ------------------------------------- the final sinc kernel ------------------------------------- #
            if np.random.uniform() < self.final_sinc_prob:
                kernel_size = random.choice(self.kernel_range)
                omega_c = np.random.uniform(np.pi / 3, np.pi)
                sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
                sinc_kernel = torch.FloatTensor(sinc_kernel)
            else:
                sinc_kernel = self.pulse_tensor

            # [0, 1], BGR to RGB, HWC to CHW
            img_hq = torch.from_numpy(
                img_hq.transpose(2, 0, 1).copy()
            ).float()
            kernel = torch.FloatTensor(kernel)
            kernel2 = torch.FloatTensor(kernel2)

            




def random_crop_arr(video_array, out_size):
    """
    Randomly crop the video frames in spatial dimensions.

    Args:
        video_array (np.ndarray): Input video array of shape [C, T, H, W].
        out_size (int): The size of the output crop (out_size x out_size).
    
    Returns:
        np.ndarray: Cropped video array of shape [C, T, out_size, out_size].
    """
    _, _, h, w = video_array.shape
    top = np.random.randint(0, h - out_size + 1)
    left = np.random.randint(0, w - out_size + 1)
    return video_array[:, :, top:top + out_size, left:left + out_size]

def center_crop_arr(video_array, out_size):
    """
    Center crop the video frames in spatial dimensions.

    Args:
        video_array (np.ndarray): Input video array of shape [C, T, H, W].
        out_size (int): The size of the output crop (out_size x out_size).
    
    Returns:
        np.ndarray: Cropped video array of shape [C, T, out_size, out_size].
    """
    _, _, h, w = video_array.shape
    top = (h - out_size) // 2
    left = (w - out_size) // 2
    return video_array[:, :, top:top + out_size, left:left + out_size]

# 示例代码，用于基于crop_type进行裁剪
if self.crop_type == "random":
    video_array = random_crop_arr(video_array, self.out_size)
elif self.crop_type == "center":
    video_array = center_crop_arr(video_array, self.out_size)
# self.crop_type is "none"
else:
    assert video_array.shape[2:4] == (self.out_size, self.out_size)









