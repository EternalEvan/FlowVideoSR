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
from pathlib import Path

import math
import random
from scipy import special
from scipy.stats import multivariate_normal

@DATASETS.register_module()

class VideoTextDataset(torch.utils.data.Dataset):
    """load video according to the csv file.

    Args:
        target_video_len (int): the number of video frames will be load.
        align_transform (callable): Align different videos in a specified size.
        temporal_sample (callable): Sample the target length of a video.
    """

    def __init__(
        self,
        data_path=None,
        num_frames=16,
        frame_interval=1,
        image_size=(256, 256),
        transform_name="center",
    ):
        self.data_path = data_path
        self.data = read_file(data_path)
        self.get_text = "text" in self.data.columns
        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.image_size = image_size
        self.transforms = {
            "image": get_transforms_image(transform_name, image_size),
            "video": get_transforms_video(transform_name, image_size),
        }

    def _print_data_number(self):
        num_videos = 0
        num_images = 0
        for path in self.data["path"]:
            if self.get_type(path) == "video":
                num_videos += 1
            else:
                num_images += 1
        print(f"Dataset contains {num_videos} videos and {num_images} images.")

    def get_type(self, path):
        ext = os.path.splitext(path)[-1].lower()
        if ext.lower() in VID_EXTENSIONS:
            return "video"
        else:
            assert ext.lower() in IMG_EXTENSIONS, f"Unsupported file format: {ext}"
            return "image"

    def getitem(self, index):
        sample = self.data.iloc[index]
        path = sample["path"]
        file_type = self.get_type(path)

        if file_type == "video":
            # loading
            vframes, vinfo = read_video(path, backend="av")
            video_fps = vinfo["video_fps"] if "video_fps" in vinfo else 24

            # Sampling video frames
            video = temporal_random_crop(vframes, self.num_frames, self.frame_interval)

            # transform
            transform = self.transforms["video"]
            video = transform(video)  # T C H W
        else:
            # loading
            image = pil_loader(path)
            video_fps = IMG_FPS

            # transform
            transform = self.transforms["image"]
            image = transform(image)

            # repeat
            video = image.unsqueeze(0).repeat(self.num_frames, 1, 1, 1)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)

        ret = {"video": video, "fps": video_fps}
        if self.get_text:
            ret["text"] = sample["text"]

        '''
        video
        fps
        text
        '''
        return ret

    def __getitem__(self, index):
        for _ in range(10):
            try:
                return self.getitem(index)
            except Exception as e:
                path = self.data.iloc[index]["path"]
                print(f"data {path}: {e}")
                index = np.random.randint(len(self))
        raise RuntimeError("Too many bad data.")

    def __len__(self):
        return len(self.data)


@DATASETS.register_module()
class VariableVideoTextDataset(VideoTextDataset):
    def __init__(
        self,
        data_path=None,
        num_frames=None,
        frame_interval=1,
        image_size=(None, None),
        transform_name=None,
        dummy_text_feature=False,
    ):
        super().__init__(data_path, num_frames, frame_interval, image_size, transform_name=None)
        self.transform_name = transform_name
        self.data["id"] = np.arange(len(self.data))
        self.dummy_text_feature = dummy_text_feature

    def get_data_info(self, index):
        T = self.data.iloc[index]["num_frames"]
        H = self.data.iloc[index]["height"]
        W = self.data.iloc[index]["width"]
        return T, H, W

    def getitem(self, index):
        # a hack to pass in the (time, height, width) info from sampler
        index, num_frames, height, width = [int(val) for val in index.split("-")]

        sample = self.data.iloc[index]
        path = sample["path"]
        file_type = self.get_type(path)
        ar = height / width

        video_fps = 24  # default fps
        if file_type == "video":
            # loading
            vframes, vinfo = read_video(path, backend="av")
            video_fps = vinfo["video_fps"] if "video_fps" in vinfo else 24

            # Sampling video frames
            video = temporal_random_crop(vframes, num_frames, self.frame_interval)
            video = video.clone()
            del vframes

            video_fps = video_fps // self.frame_interval

            # transform
            transform = get_transforms_video(self.transform_name, (height, width))
            video = transform(video)  # T C H W
        else:
            # loading
            image = pil_loader(path)
            video_fps = IMG_FPS

            # transform
            transform = get_transforms_image(self.transform_name, (height, width))
            image = transform(image)

            # repeat
            video = image.unsqueeze(0)

        # TCHW -> CTHW
        video = video.permute(1, 0, 2, 3)
        ret = {
            "video": video,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "ar": ar,
            "fps": video_fps,
        }
        if self.get_text:
            ret["text"] = sample["text"]
        if self.dummy_text_feature:
            text_len = 50
            ret["text"] = torch.zeros((1, text_len, 1152))
            ret["mask"] = text_len
        return ret

    def __getitem__(self, index):
        return self.getitem(index)


@DATASETS.register_module()
class BatchFeatureDataset(torch.utils.data.Dataset):
    """
    The dataset is composed of multiple .bin files.
    Each .bin file is a list of batch data (like a buffer). All .bin files have the same length.
    In each training iteration, one batch is fetched from the current buffer.
    Once a buffer is consumed, load another one.
    Avoid loading the same .bin on two difference GPUs, i.e., one .bin is assigned to one GPU only.
    """

    def __init__(self, data_path=None):
        self.path_list = sorted(glob(data_path + "/**/*.bin"))

        self._len_buffer = len(torch.load(self.path_list[0]))
        self._num_buffers = len(self.path_list)
        self.num_samples = self.len_buffer * len(self.path_list)

        self.cur_file_idx = -1
        self.cur_buffer = None

    @property
    def num_buffers(self):
        return self._num_buffers

    @property
    def len_buffer(self):
        return self._len_buffer

    def _load_buffer(self, idx):
        file_idx = idx // self.len_buffer
        if file_idx != self.cur_file_idx:
            self.cur_file_idx = file_idx
            self.cur_buffer = torch.load(self.path_list[file_idx])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        self._load_buffer(idx)

        batch = self.cur_buffer[idx % self.len_buffer]  # dict; keys are {'x', 'fps'} and text related

        ret = {
            "video": batch["x"],
            "text": batch["y"],
            "mask": batch["mask"],
            "fps": batch["fps"],
            "height": batch["height"],
            "width": batch["width"],
            "num_frames": batch["num_frames"],
        }
        return ret

# @DATASETS.register_module()
# class VideoRecurrentTestDataset(data.Dataset):
#     """Video test dataset for recurrent architectures, which takes LR video
#     frames as input and output corresponding HR video frames. Modified from
#     https://github.com/xinntao/BasicSR/blob/master/basicsr/data/reds_dataset.py

#     Supported datasets: Vid4, REDS4, REDSofficial.
#     More generally, it supports testing dataset with following structures:

#     dataroot
#     ├── subfolder1
#         ├── frame000
#         ├── frame001
#         ├── ...
#     ├── subfolder1
#         ├── frame000
#         ├── frame001
#         ├── ...
#     ├── ...

#     For testing datasets, there is no need to prepare LMDB files.

#     Args:
#         opt (dict): Config for train dataset. It contains the following keys:
#             dataroot_gt (str): Data root path for gt.
#             dataroot_lq (str): Data root path for lq.
#             io_backend (dict): IO backend type and other kwarg.
#             cache_data (bool): Whether to cache testing datasets.
#             name (str): Dataset name.
#             meta_info_file (str): The path to the file storing the list of test
#                 folders. If not provided, all the folders in the dataroot will
#                 be used.
#             num_frame (int): Window size for input frames.
#             padding (str): Padding mode.
#     """

#     def __init__(self, opt):
#         super(VideoRecurrentTestDataset, self).__init__()
#         self.opt = opt
#         self.cache_data = opt['cache_data']
#         self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
#         self.data_info = {'lq_path': [], 'gt_path': [], 'folder': [], 'idx': [], 'border': []}
#         self.num_frames = opt['num_frames']
#         self.image_size = opt['image_size']

#         self.imgs_lq, self.imgs_gt = {}, {}
#         if 'meta_info_file' in opt:
#             with open(opt['meta_info_file'], 'r') as fin:
#                 subfolders = [line.split(' ')[0] for line in fin]
#                 subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
#                 subfolders_gt = [osp.join(self.gt_root, key) for key in subfolders]
#         else:
#             subfolders_lq = sorted(glob(osp.join(self.lq_root, '*')))
#             subfolders_gt = sorted(glob(osp.join(self.gt_root, '*')))
#         for subfolder_lq, subfolder_gt in zip(subfolders_lq, subfolders_gt):
#             # get frame list for lq and gt
#             subfolder_name = osp.basename(subfolder_lq)
#             img_paths_lq = sorted(list(utils_video.scandir(subfolder_lq, full_path=True)))
#             img_paths_gt = sorted(list(utils_video.scandir(subfolder_gt, full_path=True)))
#             max_idx = len(img_paths_lq)
#             assert max_idx == len(img_paths_gt), (f'Different number of images in lq ({max_idx})'
#                                                   f' and gt folders ({len(img_paths_gt)})')

#             self.data_info['lq_path'].extend(img_paths_lq)
#             self.data_info['gt_path'].extend(img_paths_gt)
#             self.data_info['folder'].extend([subfolder_name] * max_idx)
#             # print(f'subfolder_name = {subfolder_name}')
#             # print(f'max_idx = {max_idx}')
#             for i in range(max_idx):
#                 self.data_info['idx'].append(f'{i}/{max_idx}')
#             border_l = [0] * max_idx
#             for i in range(self.opt['num_frames'] // 2):
#                 border_l[i] = 1
#                 border_l[max_idx - i - 1] = 1
#             self.data_info['border'].extend(border_l)

#             # cache data or save the frame list
#             if self.cache_data:
#                 print(f'Cache {subfolder_name} for VideoTestDataset...')
#                 self.imgs_lq[subfolder_name] = utils_video.read_img_seq(img_paths_lq)
#                 self.imgs_gt[subfolder_name] = utils_video.read_img_seq(img_paths_gt)
#             else:
#                 self.imgs_lq[subfolder_name] = img_paths_lq
#                 self.imgs_gt[subfolder_name] = img_paths_gt

#         # Find unique folder strings
#         # print(f"self.data_info_folder = {self.data_info['folder']}")
#         self.folders = sorted(list(set(self.data_info['folder'])))
#         # print(f'self.folders = {self.folders}')
        
#         self.sigma = opt['sigma'] / 255. if 'sigma' in opt else 0 # for non-blind video denoising


#     def __len__(self):
#         return len(self.folders)
    
#     def getitem(self, index):
#         folder = self.folders[index]

#         if self.sigma:
#         # for non-blind video denoising
#             if self.cache_data:
#                 imgs_gt = self.imgs_gt[folder]
#             else:
#                 imgs_gt = utils_video.read_img_seq(self.imgs_gt[folder])

#             torch.manual_seed(0)
#             noise_level = torch.ones((1, 1, 1, 1)) * self.sigma
#             noise = torch.normal(mean=0, std=noise_level.expand_as(imgs_gt))
#             imgs_lq = imgs_gt + noise
#             t, _, h, w = imgs_lq.shape
#             imgs_lq = torch.cat([imgs_lq, noise_level.expand(t, 1, h, w)], 1)
#         else:
#         # for video sr and deblurring
#             if self.cache_data:
#                 imgs_lq = self.imgs_lq[folder]
#                 imgs_gt = self.imgs_gt[folder]
#             else:
#                 # print(f'self.imgs_gtfolder = {self.imgs_gt[folder]}')
#                 # import pdb; pdb.set_trace()
#                 imgs_lq = utils_video.read_img_seq(self.imgs_lq[folder])
#                 imgs_gt = utils_video.read_img_seq(self.imgs_gt[folder])
#         # # transform
#         # imgs_lq = temporal_random_crop(imgs_lq, self.num_frames, 1)
#         # imgs_gt = temporal_random_crop(imgs_gt,self.num_frames,1)
#         # transform = get_transforms_video("resize_crop",(self.image_size[0], self.image_size[1])) 
#         # imgs_lq = transform(imgs_lq)
#         # imgs_gt = transform(imgs_gt)
#         _, _, h, w = imgs_gt.shape

#         # 使用插值方法将 imgs_lq 调整到与 imgs_gt 相同的尺寸
#         imgs_lq = F.interpolate(imgs_lq, size=(h, w), mode='bicubic', align_corners=False)

        
#         imgs_lq = 2 * imgs_lq - 1
#         imgs_gt = 2 * imgs_gt - 1  # [-1, 1]

#         return {
#             'L': imgs_lq,
#             'H': imgs_gt,
#             'folder': folder,
#             'lq_path': self.imgs_lq[folder],
#             'fps': 24,
#             'height': self.image_size[1],
#             'width': self.image_size[0],
#             'num_frames': self.num_frames,
#         }

#     def __getitem__(self, index):
#         return self.getitem(index)
@DATASETS.register_module()
class VideoRecurrentTestDataset(data.Dataset):

    def __init__(self, opt):
        super(VideoRecurrentTestDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.gt_root= opt['dataroot_gt']
        self.data_info = {'gt_path': [], 'folder': [], 'idx': [], 'border': []}
        self.num_frames = opt['num_frames']
        self.image_size = opt['image_size']

        self.imgs_lq, self.imgs_gt = {}, {}
        self.folders = []
        subfolders_gt = sorted(glob(osp.join(self.gt_root, '*')))
        
        for subfolder_gt in subfolders_gt:
            name1 = osp.basename(subfolder_gt)
            for subsubfolder_gt in Path(subfolder_gt).iterdir():
                name2 = osp.basename(subsubfolder_gt)
               # print(f'subsubfolder_gt = {subsubfolder_gt}')
           
                img_paths_gt = sorted(list(utils_video.scandir(subsubfolder_gt, full_path=True)))
                #print(f'img_paths_gt = {img_paths_gt}')
        


                # cache data or save the frame list
                # if self.cache_data:
                #     print(f'Cache {name1} for VideoTestDataset...')
                #     # self.imgs_lq[subfolder_name] = utils_video.read_img_seq(img_paths_lq)
                #     self.imgs_gt[name1+"_"+name2] = utils_video.read_img_seq(img_paths_gt)
                # else:
                #     # self.imgs_lq[subfolder_name] = img_paths_lq
                self.imgs_gt[name1+"_"+name2] = img_paths_gt
                self.folders.extend(img_paths_gt)
        

        # Find unique folder strings
        # print(f"self.data_info_folder = {self.data_info['folder']}")
        # self.folders = sorted(list(set(self.data_info['folder'])))
       # print(f'self.folders = {len(self.folders)}')
        
        self.sigma = opt['sigma'] / 255. if 'sigma' in opt else 0 # for non-blind video denoising


    def __len__(self):
        return len(self.folders)
    
    def getitem(self, index):
        #folder = self.folders[index]

        if self.sigma:
        # for non-blind video denoising
            # if self.cache_data:
            #     imgs_gt = self.imgs_gt[folder]
            # else:
            #     imgs_gt = utils_video.read_img_seq(self.imgs_gt[folder])
            print(f'imgs_gt = {imgs_gt}')   
            torch.manual_seed(0)
            noise_level = torch.ones((1, 1, 1, 1)) * self.sigma
            noise = torch.normal(mean=0, std=noise_level.expand_as(imgs_gt))
            imgs_lq = imgs_gt + noisey
            t, _, h, w = imgs_lq.shape
            imgs_lq = torch.cat([imgs_lq, noise_level.expand(t, 1, h, w)], 1)
        else:
        # for video sr and deblurring
            if self.cache_data:
                imgs_lq = self.imgs_lq[folder]
                imgs_gt = self.imgs_gt[folder]
            else:
                # print(f'self.imgs_gtfolder = {self.imgs_gt[folder]}')
                # import pdb; pdb.set_trace()
                # imgs_lq = utils_video.read_img_seq(self.imgs_lq[folder])
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
                
                if len(frames) < self.num_frames:
                    print(f"Insufficient video frames: Only {len(frames)} frames")
                    return None
            
                # video_array = torch.tensor(frames, dtype=torch.float32) # [30, 1080, 1920, 3]
                imgs_gt = np.array(frames).transpose(3,0,1,2) # ([3, 30, 1080, 1920])
                imgs_gt = (imgs_gt / 255.0)

                imgs_lq = process_numpy(imgs_gt)

        
        imgs_lq = 2 * imgs_lq - 1
        imgs_gt = 2 * imgs_gt - 1

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