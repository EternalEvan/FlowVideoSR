# 使用VideoRecurrentTestDataset_v1
# 使用concate的条件注入方法
# 使用更新后的REDS数据集
# VRT作为first stage model


# Dataset settings


dataset = dict(
    type = "VideoRecurrentTestDataset_v1",
    opt = dict(
        name = "REDS4", 
        cache_data=None,
        dataroot_gt='/mnt/nfs/train_REDS4/train_sharp',#'/mnt/nfs/train51/train_sharp',#"/mnt/nfs/YouHQ-Train",
        dataroot_lq="/home/whl/workspace/Open-Sora/VRT/REDS_VRT/001_VRT_videosr_bi_REDS_6frames",
        num_frames= 29,
        mode="train",
        image_size= (448, 448),
        crop_type="random",
    )
)

# webvid
bucket_config = {  # 20s/it
    "144p": {1: (1.0, 475), 51: (1.0, 51), 102: (1.0, 27), 204: (1.0, 13), 408: (1.0, 6)},
    # ---
    "256": {1: (1.0, 297), 51: (0.5, 20), 102: (0.5, 10), 204: (0.5, 5), 408: ((0.5, 0.5), 2)},
    "240p": {1: (1.0, 297), 51: (0.5, 20), 102: (0.5, 10), 204: (0.5, 5), 408: ((0.5, 0.4), 2)},
    # ---
    "360p": {1: (1.0, 141), 51: (0.5, 8), 102: (0.5, 4), 204: (0.5, 2), 408: ((0.5, 0.3), 1)},
    "512": {1: (1.0, 141), 51: (0.5, 8), 102: (0.5, 4), 204: (0.5, 2), 408: ((0.5, 0.2), 1)},
    # ---
    "480p": {1: (1.0, 89), 51: (0.5, 5), 102: (0.5, 3), 204: ((0.5, 0.5), 1), 408: (0.0, None)},
    # ---
    "720p": {1: (0.3, 36), 51: (0.2, 2), 102: (0.1, 1), 204: (0.0, None)},
    "1024": {1: (0.3, 36), 51: (0.1, 2), 102: (0.1, 1), 204: (0.0, None)},
    # ---
    "1080p": {1: (0.1, 5)},
    # ---
    "2048": {1: (0.05, 5)},
}

grad_checkpoint = True

# Acceleration settings
num_workers = 8
num_bucket_build_workers = 16
dtype = "bf16"
plugin = "zero2"

# Model settings
model = dict(
    type = "CogVideo-2b",
)
model_path = '/mnt/nfs/CogVideoX-2b/'

scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    sample_method="logit-normal",
)

# Mask settings
# 25%
mask_ratios = {
    "random": 0.01,
    "intepolate": 0.002,
    "quarter_random": 0.002,
    "quarter_head": 0.002,
    "quarter_tail": 0.002,
    "quarter_head_tail": 0.002,
    "image_random": 0.0,
    "image_head": 0.22,
    "image_tail": 0.005,
    "image_head_tail": 0.005,
}

# Log settings
seed = 42
outputs = "CogVideoX_sr_REDS_v2"
wandb = False
epochs = 1000
log_every = 2#20
ckpt_every = 1000

# optimization settings

# load = "/home/whl/workspace/Open-Sora/CogVideoX_sr/012-CogVideo-2b/epoch1-global_step50000"

grad_clip = 1.0
lr = 1e-4
ema_decay = 0.99
adam_eps = 1e-15
warmup_steps = 1000

batch_size=1