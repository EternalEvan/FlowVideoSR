resolution = "240p"
aspect_ratio = "9:16"
num_frames = 51
fps = 24
frame_interval = 1
save_fps = 24

save_dir = "./samples/CogVideoX/samples"
save_dir1 = "./samples/CogVideoX/lq"
save_dir2 = "./samples/CogVideoX/hq"

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
        dataroot_gt='/mnt/nfs/train_REDS4/train_sharp',#'/mnt/nfs/train51/train_sharp',#"/mnt/nfs/YouHQ-Train",
        dataroot_lq="/mnt/nfs/train_REDS4/train_sharp_BICUBIC",
        cache_data=False,
        io_backend={ "type": "disk" },
        num_frames= 17,
        image_size= (1920, 1080),
        mode="test",
    )
)

model_path = "/mnt/nfs/CogVideoX-2b/"

# load = "/home/whl/workspace/Open-Sora/CogVideoX_sr/001-CogVideo-2b/epoch0-global_step4000"

scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=30,
    cfg_scale=7.0,
)

aes = 6.5
flow = None
