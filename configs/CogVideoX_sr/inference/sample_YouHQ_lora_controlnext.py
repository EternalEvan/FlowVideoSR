resolution = "240p"
aspect_ratio = "9:16"
num_frames = 51
fps = 24
frame_interval = 1
save_fps = 24

save_dir = "./samples_YouHQ/CogVideoX_lora_controlnext_9frames_nocache_full_step60000-5step-mid/samples"
save_dir1 = "./samples_YouHQ/CogVideoX_lora_controlnext_9frames_nocache_full_step60000-5step-mid/lq"
save_dir2 = "./samples_YouHQ/CogVideoX_lora_controlnext_9frames_nocache_full_step60000-5step-mid/hq"

seed = 42
batch_size = 1
multi_resolution = "STDiT2"
dtype = "bf16"
condition_frame_length = 5
align = 5

# dataset = dict(
#     type = "VideoRecurrentTestDataset_v1",
#     opt = dict(
#         name = "REDS4", 
#         cache_data=None,
#         dataroot_gt='/mnt/nfs/train_REDS4/train_sharp',#'/mnt/nfs/train51/train_sharp',#"/mnt/nfs/YouHQ-Train",
#         dataroot_lq="/home/whl/workspace/Open-Sora/VRT/REDS_VRT/001_VRT_videosr_bi_REDS_6frames",
#         num_frames= 100,
#         mode="test",
#         image_size= (1280, 720),
#         crop_type="None",
#     )
# )

dataset = dict(
    type = "VideoRecurrentCodeFormerVideoDataset",
    opt = dict(
        name = "test_dataset", 
        dataroot_gt='/mnt/nfs/YouHQ40-Test-Video',#'/mnt/nfs/train51/train_sharp',#"/mnt/nfs/YouHQ-Train",
        #dataroot_lq="/mnt/nfs/train51/train_sharp_BICUBIC",
        num_frames= 17,
        # image_size= (448, 448),
        image_size = (1920, 1080),
        # image_size = (1080, 1920),
        # crop_type="center",
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
# load = "/home/whl/workspace/Open-Sora/CogVideoX_sr/012-CogVideo-2b/epoch1-global_step50000"

# load = "/home/whl/workspace/Open-Sora/CogVideoX_sr_lora_controlnet/002-CogVideo-2b/epoch0-global_step7500"
load = "/home/whl/workspace/Open-Sora/CogVideoX_sr_lora_controlnext_6/000-CogVideo-2b/epoch1-global_step60000"
scheduler = dict(
    type="rflow",
    use_timestep_transform=True,
    num_sampling_steps=5,
    cfg_scale=7.0,
)

aes = 6.5
flow = None
