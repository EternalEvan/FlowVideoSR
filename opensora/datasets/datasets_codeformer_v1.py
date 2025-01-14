# mix REDS and YouHQ for CogVideoX

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

import cv2
from typing import Dict, Sequence
import math
import time



@DATASETS.register_module()
class VideoRecurrentCodeFormerDataset_V1(data.Dataset):


    # Supported datasets: YouHQ


    def __init__(self, opt):
        super(VideoRecurrentCodeFormerDataset, self).__init__()
        self.opt = opt
        # self.cache_data = opt['cache_data']
        # self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.gt_root = opt['dataroot_gt']

        self.num_frames = opt['num_frames']
        self.image_size = opt['image_size']
        self.crop_type = opt['crop_type']
        # self.video_paths_lq = []
        self.video_paths_gt = []



        if 'mix_rate' in opt:
            self.mix_rate = opt['mix_rate']
        else:
            self.mix_rate = 0.3



        self.imgs_lq, self.imgs_gt = {}, {}
        if 'meta_info_file' in opt:
            with open(opt['meta_info_file'], 'r') as fin:
                subfolders = [line.split(' ')[0] for line in fin]
                # subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
                subfolders_gt = [osp.join(self.gt_root, key) for key in subfolders]
        else:
            # subfolders_lq = sorted(glob(osp.join(self.lq_root, '*')))
            subfolders_gt = sorted(glob(osp.join(self.gt_root, '*')))
        # print("subfolders_lq = ", subfolders_lq)
        # print("subfolders_gt = ", subfolders_gt)
        # for subfolder_lq, subfolder_gt in zip(subfolders_lq, subfolders_gt):
        for subfolder_gt in subfolders_gt:

            # get frame list for lq and gt
            subfolder_name = osp.basename(subfolder_gt)
            # self.video_paths_lq.extend(sorted(list(utils_video.scandir(subfolder_lq, recursive=True, full_path=True))))
            self.video_paths_gt.extend(sorted(list(utils_video.scandir(subfolder_gt, recursive=True, full_path=True))))
            # print("self.video_paths_lq = ", self.video_paths_lq)
            # print("self.video_paths_gt = ", self.video_paths_gt)
 

        # assert len(self.video_paths_lq) == len(self.video_paths_gt)


    def __len__(self):
        return len(self.video_paths_gt)
    
    def getitem(self, index):
        video_path_gt = self.video_paths_gt[index]

        cap = cv2.VideoCapture(video_path_gt)
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
        # print(f'frame_count = {frame_count}')
        cap.release()

        imgs_gt = np.array(frames).transpose(3,0,1,2) # [C, T, H, W]
        imgs_gt = (imgs_gt / 255.0)

        imgs_lq = process_numpy(imgs_gt)

        imgs_lq = 2 * imgs_lq - 1
        imgs_gt = 2 * imgs_gt - 1

        # print(f'imgs_lq.shape ={imgs_lq.shape}')
        # print(f'imgs_gt.shape ={imgs_gt.shape}')
        # num_frames_gt = imgs_gt.shape(1)
        # if num_frames_gt < self.num_frames:
        #     self.num_frames = min(num_frames_gt, self.num_frames)
        #     imgs_lq = imgs_lq[:, :self.num_frames, :, :]
        #     imgs_hq = imgs_hq[:, :self.num_frames, :, :]
        
        # # 随机选择起始帧
        # start_frame = random.randint(0, num_frames_gt - self.num_frames)

        # # 选择截取的帧
        # imgs_lq = imgs_lq[:, start_frame:start_frame + self.num_frames, :, :]
        # imgs_gt = imgs_gt[:, start_frame:start_frame + self.num_frames, :, :]


        # 空间上随机裁剪
        height, width = imgs_lq.shape[2], imgs_lq.shape[3]
        # print(f'height, width = {height, width}')
        if self.crop_type != 'None':
            if height > self.image_size[1] and width > self.image_size[0]:
                top = random.randint(0, height - self.image_size[1])
                left = random.randint(0, width - self.image_size[0])
                
                imgs_lq = imgs_lq[:, :, top:top + self.image_size[1], left:left + self.image_size[0]]
                imgs_gt = imgs_gt[:, :, top:top + self.image_size[1], left:left + self.image_size[0]]
            else:
                # 如果视频帧的高度或宽度小于所需的裁剪尺寸
                # 这时候裁剪区域会是整个视频帧的区域
                top = 0
                left = 0
                imgs_lq = imgs_lq[:, :, top:top + self.image_size[1], left:left + self.image_size[0]]
                imgs_gt = imgs_gt[:, :, top:top + self.image_size[1], left:left + self.image_size[0]]
        height, width = imgs_lq.shape[2], imgs_lq.shape[3]
        # print(f'height, width = {height, width}')
        # print(f'imgs_lq.shape - {imgs_lq.shape}')
        # print(f'imgs_gt.shape = {imgs_gt.shape}')
        return {
            'L': imgs_lq,
            'H': imgs_gt,
            'fps': 30,
            'height': self.image_size[1],
            'width': self.image_size[0],
            'num_frames': self.num_frames,
        }


    


    def __getitem__(self, index):
        return self.getitem(index)



    
def gen_lq_img(img_gt,kernel=None,scale=None,noise_range=None,jpeg_range=None):
        
        h, w, _ = img_gt.shape
        # blur
        img_blurred = cv2.filter2D(img_gt, -1, kernel)
        # downsample
        img_lq = cv2.resize(img_gt, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)
        # noise
        if noise_range is not None:
            img_lq = random_add_gaussian_noise(img_lq,noise_range)
        # jpeg compression
        if jpeg_range is not None:
            img_lq = random_add_jpg_compression(img_lq, jpeg_range)
        
        # resize to original size
        img_lq = cv2.resize(img_lq, (w, h), interpolation=cv2.INTER_LINEAR)
        img_lq = img_lq.transpose(2,0,1)
        return img_lq
    
def process_numpy(imgs_gt):
    C, T, H, W = imgs_gt.shape
    lq_list = []
    
    blur_kernel_size= 41
    kernel_list=['iso', 'aniso']
    kernel_prob= [0.5, 0.5]
    blur_sigma= [0.1, 10]
    downsample_range=[0.8, 8]
    scale = np.random.uniform(downsample_range[0], downsample_range[1])
    kernel = random_mixed_kernels(
            kernel_list,
            kernel_prob,
            blur_kernel_size,
            blur_sigma,
            [-math.pi, math.pi],
            noise_range=None
        )
    noise_range=[0, 20]
    jpeg_range=[60, 100]
 
    for t in range(T):
        img = imgs_gt[:, t, :, :]
        lq_img = gen_lq_img(img.transpose(1, 2, 0).astype(np.float32), kernel=kernel, scale=scale, noise_range=noise_range, jpeg_range=jpeg_range)
        lq_list.append(lq_img)
    lq_tensor = torch.tensor(np.array(lq_list)).permute(1, 0, 2, 3)
    #write_video(lq_tensor)
    return lq_tensor

def random_mixed_kernels(kernel_list,
                         kernel_prob,
                         kernel_size=21,
                         sigma_x_range=(0.6, 5),
                         sigma_y_range=(0.6, 5),
                         rotation_range=(-math.pi, math.pi),
                         betag_range=(0.5, 8),
                         betap_range=(0.5, 8),
                         noise_range=None):
  
    kernel_type = random.choices(kernel_list, kernel_prob)[0]
    if kernel_type == 'iso':
        kernel = random_bivariate_Gaussian(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, noise_range=noise_range, isotropic=True)
    elif kernel_type == 'aniso':
        kernel = random_bivariate_Gaussian(
            kernel_size, sigma_x_range, sigma_y_range, rotation_range, noise_range=noise_range, isotropic=False)
    return kernel

def random_bivariate_Gaussian(kernel_size,
                              sigma_x_range,
                              sigma_y_range,
                              rotation_range,
                              noise_range=None,
                              isotropic=True):

    assert kernel_size % 2 == 1, 'Kernel size must be an odd number.'
    assert sigma_x_range[0] < sigma_x_range[1], 'Wrong sigma_x_range.'
    sigma_x = np.random.uniform(sigma_x_range[0], sigma_x_range[1])
    if isotropic is False:
        assert sigma_y_range[0] < sigma_y_range[1], 'Wrong sigma_y_range.'
        assert rotation_range[0] < rotation_range[1], 'Wrong rotation_range.'
        sigma_y = np.random.uniform(sigma_y_range[0], sigma_y_range[1])
        rotation = np.random.uniform(rotation_range[0], rotation_range[1])
    else:
        sigma_y = sigma_x
        rotation = 0

    kernel = bivariate_Gaussian(kernel_size, sigma_x, sigma_y, rotation, isotropic=isotropic)

    # add multiplicative noise
    if noise_range is not None:
        assert noise_range[0] < noise_range[1], 'Wrong noise range.'
        noise = np.random.uniform(noise_range[0], noise_range[1], size=kernel.shape)
        kernel = kernel * noise
    kernel = kernel / np.sum(kernel)
    return kernel

def bivariate_Gaussian(kernel_size, sig_x, sig_y, theta, grid=None, isotropic=True):
    if grid is None:
        grid, _, _ = mesh_grid(kernel_size)
    if isotropic:
        sigma_matrix = np.array([[sig_x**2, 0], [0, sig_x**2]])
    else:
        sigma_matrix = sigma_matrix2(sig_x, sig_y, theta)
    kernel = pdf2(sigma_matrix, grid)
    kernel = kernel / np.sum(kernel)
    return kernel

def mesh_grid(kernel_size):
    ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    xy = np.hstack((xx.reshape((kernel_size * kernel_size, 1)), yy.reshape(kernel_size * kernel_size,
                                                                           1))).reshape(kernel_size, kernel_size, 2)
    return xy, xx, yy

def sigma_matrix2(sig_x, sig_y, theta):
    d_matrix = np.array([[sig_x**2, 0], [0, sig_y**2]])
    u_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.dot(u_matrix, np.dot(d_matrix, u_matrix.T))

def pdf2(sigma_matrix, grid):
    inverse_sigma = np.linalg.inv(sigma_matrix)
    kernel = np.exp(-0.5 * np.sum(np.dot(grid, inverse_sigma) * grid, 2))
    return kernel

def random_add_gaussian_noise(img, sigma_range=(0, 1.0), gray_prob=0, clip=True, rounds=False):
    noise = random_generate_gaussian_noise(img, sigma_range, gray_prob)
    out = img + noise
    if clip and rounds:
        out = np.clip((out * 255.0).round(), 0, 255) / 255.
    elif clip:
        out = np.clip(out, 0, 1)
    elif rounds:
        out = (out * 255.0).round() / 255.
    return out

def random_add_jpg_compression(img, quality_range=(90, 100)):
    quality = np.random.uniform(quality_range[0], quality_range[1])
    return add_jpg_compression(img, int(quality))

def add_jpg_compression(img, quality=90):
    img = np.clip(img, 0, 1)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encimg = cv2.imencode('.jpg', img * 255., encode_param)
    img = np.float32(cv2.imdecode(encimg, 1)) / 255.
    return img

def random_generate_gaussian_noise(img, sigma_range=(0, 10), gray_prob=0):
    sigma = np.random.uniform(sigma_range[0], sigma_range[1])
    if np.random.uniform() < gray_prob:
        gray_noise = True
    else:
        gray_noise = False
    return generate_gaussian_noise(img, sigma, gray_noise)

def generate_gaussian_noise(img, sigma=10, gray_noise=False):
    if gray_noise:
        noise = np.float32(np.random.randn(*(img.shape[0:2]))) * sigma / 255.
        noise = np.expand_dims(noise, axis=2).repeat(3, axis=2)
    else:
        noise = np.float32(np.random.randn(*(img.shape))) * sigma / 255.
    return noise


def write_video(images, video_path='output_video.mp4', fps=30):
    T =30 # 总帧数
    C, H, W = 3, 1080, 1920  # 通道数，图像高度和宽度

    images = (images*255).numpy().astype(np.uint8)

    # 转换 tensor 形状为 (T, H, W, C)
    images = images.transpose(1, 2, 3, 0)

    # 编码器和视频保存路径
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_path = 'output_video.mp4'

    # 初始化视频写入对象
    out = cv2.VideoWriter(video_path, fourcc, fps, (W, H))

    # 将每帧图片写入视频
    for t in range(T):
        frame = images[t]  # 获取第 t 帧
        out.write(frame)

    # 释放视频写入对象
    out.release()

    print(f"视频保存至: {video_path}")