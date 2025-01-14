resolution = "240p"
aspect_ratio = "9:16"
num_frames = 51
fps = 24
frame_interval = 1
save_fps = 24

save_dir = "./samples_YouHQ/CogVideoX_29frames_nocache_step78000/samples"
save_dir1 = "./samples_YouHQ/CogVideoX_29frames_nocache_step78000/lq"
save_dir2 = "./samples_YouHQ/CogVideoX_29frames_nocache_step78000/hq"

seed = 42
batch_size = 1
multi_resolution = "STDiT2"
dtype = "bf16"
condition_frame_length = 5
align = 5


dataset = dict(
    type = "VideoRecurrentCodeFormerDataset",
    opt = dict(
        name = "test_dataset", 
        dataroot_gt='/mnt/nfs/YouHQ40-Test-Video',#'/mnt/nfs/train51/train_sharp',#"/mnt/nfs/YouHQ-Train",
       # dataroot_lq="/mnt/nfs/train51/train_sharp_BICUBIC",
        num_frames= 29,
        image_size= (448, 448),
        crop_type="center",
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
# load = "/home/whl/workspace/Open-Sora/CogVideoX_sr/012-CogVideo-2b/epoch1-global_step50000"
load = "/home/whl/workspace/Open-Sora/CogVideoX_sr/012-CogVideo-2b/epoch2-global_step78000"

scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=30,
    cfg_scale=7.0,
)

aes = 6.5
flow = None
