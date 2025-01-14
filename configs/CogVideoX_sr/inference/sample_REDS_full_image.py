resolution = "240p"
aspect_ratio = "9:16"
num_frames = 51
fps = 24
frame_interval = 1
save_fps = 24

save_dir = "./samples_REDS/CogVideoX_29frames_nocache_full_step30000/samples"
save_dir1 = "./samples_REDS/CogVideoX_29frames_nocache_full_step30000/lq"
save_dir2 = "./samples_REDS/CogVideoX_29frames_nocache_full_step30000/hq"

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
        dataroot_lq="/mnt/nfs/train_REDS4/train_sharp_BICUBIC",
        num_frames= 29,
        mode="test",
        image_size= (1280, 720),
        crop_type="None",
    )
)

model_path = "/mnt/nfs/CogVideoX-2b/"

# load = "/home/whl/workspace/Open-Sora/CogVideoX_sr/002-CogVideo-2b/epoch0-global_step19000"
# load = "/home/whl/workspace/Open-Sora/CogVideoX_sr/009-CogVideo-2b/epoch0-global_step3000"
# load = "/home/whl/workspace/Open-Sora/CogVideoX_sr/012-CogVideo-2b/epoch0-global_step6000"
# load = "/home/whl/workspace/Open-Sora/CogVideoX_sr/012-CogVideo-2b/epoch0-global_step10000"
# load = "/home/whl/workspace/Open-Sora/CogVideoX_sr/012-CogVideo-2b/epoch0-global_step30000"
# load = "/home/whl/workspace/Open-Sora/CogVideoX_sr/012-CogVideo-2b/epoch0-global_step4000"
# load = "/home/whl/workspace/Open-Sora/CogVideoX_sr/012-CogVideo-2b/epoch0-global_step20000"
# load = "/home/whl/workspace/Open-Sora/CogVideoX_sr/012-CogVideo-2b/epoch0-global_step31000"
# load = "/home/whl/workspace/Open-Sora/CogVideoX_sr_REDS/011-CogVideo-2b/epoch3-global_step1000"
# load = "/home/whl/workspace/Open-Sora/CogVideoX_sr_REDS/011-CogVideo-2b/epoch60-global_step16000"
# load = "/home/whl/workspace/Open-Sora/CogVideoX_sr_REDS/011-CogVideo-2b/epoch86-global_step23000"
load = "/home/whl/workspace/Open-Sora/CogVideoX_sr_REDS/011-CogVideo-2b/epoch112-global_step30000"

scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=30,
    cfg_scale=7.0,
)

aes = 6.5
flow = None
