# import os
# import glob
# import torch
# from os import path as osp
# import torch.utils.data as data
# from torchvision import transforms
# from PIL import Image
# from opensora.registry import DATASETS

# import opensora.utils.video_utils as utils_video

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
#         self.num_frame = opt['num_frame']
#         self.image_size = opt['image_size']
#         self.imgs_lq, self.imgs_gt = {}, {}
#         if 'meta_info_file' in opt:
#             with open(opt['meta_info_file'], 'r') as fin:
#                 subfolders = [line.split(' ')[0] for line in fin]
#                 subfolders_lq = [osp.join(self.lq_root, key) for key in subfolders]
#                 subfolders_gt = [osp.join(self.gt_root, key) for key in subfolders]
#         else:
#             subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, '*')))
#             subfolders_gt = sorted(glob.glob(osp.join(self.gt_root, '*')))
#         # print("subfolders_lq = ", subfolders_lq)
#         # print("subfolders_gt = ", subfolders_gt)
#         for subfolder_lq, subfolder_gt in zip(subfolders_lq, subfolders_gt):
#             # get frame list for lq and gt
#             subfolder_name = osp.basename(subfolder_lq)
#             img_paths_lq = sorted(list(utils_video.scandir(subfolder_lq, full_path=True)))
#             img_paths_gt = sorted(list(utils_video.scandir(subfolder_gt, full_path=True)))
#             # print("img_paths_lq = ", img_paths_lq)
#             # print("img_paths_gt = ", img_paths_gt)
#             max_idx = len(img_paths_lq)
#             assert max_idx == len(img_paths_gt), (f'Different number of images in lq ({max_idx})'
#                                                   f' and gt folders ({len(img_paths_gt)})')

#             self.data_info['lq_path'].extend(img_paths_lq)
#             self.data_info['gt_path'].extend(img_paths_gt)
#             self.data_info['folder'].extend([subfolder_name] * max_idx)
#             for i in range(max_idx):
#                 self.data_info['idx'].append(f'{i}/{max_idx}')
#             border_l = [0] * max_idx
#             for i in range(self.opt['num_frame'] // 2):
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
#         self.folders = sorted(list(set(self.data_info['folder'])))
#         self.sigma = opt['sigma'] / 255. if 'sigma' in opt else 0 # for non-blind video denoising

#     def __getitem__(self, index):
#         print("get item")
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
#             print("yessss")
#         else:
#             print("2222222yessss")
#         # for video sr and deblurring
#             if self.cache_data:
#                 imgs_lq = self.imgs_lq[folder]
#                 imgs_gt = self.imgs_gt[folder]
#             else:
#                 imgs_lq = utils_video.read_img_seq(self.imgs_lq[folder],require_mod_crop=False, scale=0.5)
#                 imgs_gt = utils_video.read_img_seq(self.imgs_gt[folder],require_mod_crop=False, scale=0.5)
        
#         return {
#             'L': imgs_lq,
#             'H': imgs_gt,
#             'folder': folder,
#             'lq_path': self.imgs_lq[folder],
#             'fps': 24,
#         }

#     def __len__(self):
#         return len(self.folders)
