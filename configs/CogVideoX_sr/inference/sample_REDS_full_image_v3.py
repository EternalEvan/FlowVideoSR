# 使用VideoRecurrentTestDataset_v1
# 使用concate的条件注入方法
# 使用更新后的REDS数据集
# lora controlnet

resolution = "240p"
aspect_ratio = "9:16"
num_frames = 51
fps = 24
frame_interval = 1
save_fps = 24

save_dir = "./samples_REDS_v3/CogVideoX_17frames_nocache_full_step5000/samples"
save_dir1 = "./samples_REDS_v3/CogVideoX_17frames_nocache_full_step5000/lq"
save_dir2 = "./samples_REDS_v3/CogVideoX_17frames_nocache_full_step5000/hq"


seed = 42
batch_size = 1
multi_resolution = "STDiT2"
dtype = "bf16"
condition_frame_length = 5
align = 5



dataset = dict(
    type = "VideoRecurrentTestDataset_v1",
    opt = dict(
        name = "REDS4", 
        cache_data=None,
        dataroot_gt='/mnt/nfs/train_REDS4/train_sharp',#'/mnt/nfs/train51/train_sharp',#"/mnt/nfs/YouHQ-Train",
        dataroot_lq="/home/whl/workspace/Open-Sora/VRT/REDS_VRT/001_VRT_videosr_bi_REDS_6frames",
        num_frames= 100,
        mode="test",
        image_size= (1280, 720),
        crop_type="None",
    )
)

model_path = "/mnt/nfs/CogVideoX-2b/"

load = "/home/whl/workspace/Open-Sora/CogVideoX_sr_REDS_v3/004-CogVideo-2b/epoch18-global_step5000"

scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=30,
    cfg_scale=7.0,
)

aes = 6.5
flow = None
