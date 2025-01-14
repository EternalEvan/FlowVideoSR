from ast import Num
import torch
import numpy as np
from diffusers.utils import export_to_video,load_image
import imageio
from torchvision import transforms
from typing import Union
import pdb
from tqdm.auto import tqdm
import os
import sys
import cv2
import colossalai
import torch.distributed as dist
from colossalai.cluster import DistCoordinator

sys.path.insert(0, os.path.expanduser("/home/whl/anaconda3/envs/opensora/lib/python3.9/site-packages/"))

from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.datasets import save_sample
from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.models.text_encoder.t5 import text_preprocessing
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.inference_utils import (
    add_watermark,
    append_generated,
    append_score_to_prompts,
    apply_mask_strategy,
    collect_references_batch,
    dframe_to_frame,
    extract_json_from_prompts,
    extract_prompts_loop,
    get_save_path_name,
    load_prompts,
    merge_prompt,
    prepare_multi_resolution_info,
    refine_prompts_by_openai,
    split_prompt,
)
from opensora.utils.misc import all_exists, create_logger, is_distributed, is_main_process, to_torch_dtype





saved_latents=[]




def encode_video(pipe, video_path, dtype, device):
    """
    Loads a pre-trained AutoencoderKLCogVideoX model and encodes the video frames.

    Parameters:
    - pipe: CogVideoXPipeline.
    - video_path (str): The path to the video file.
    - dtype (torch.dtype): The data type for computation.
    - device (str): The device to use for computation (e.g., "cuda" or "cpu").

    Returns:
    - torch.Tensor: The encoded video frames.
    """
    video_reader = imageio.get_reader(video_path, "ffmpeg")

    frames = [transforms.ToTensor()(frame) for frame in video_reader]
    video_reader.close()

    frames_tensor = torch.stack(frames).to(device).permute(1, 0, 2, 3).unsqueeze(0).to(dtype)
    #frames_tensor = torch.stack(frames).to(device).unsqueeze(0).to(dtype)
    # frames_tensor = frames_tensor[:, :, :97, :256, :256]
    print(f'frames_tensor.shape = {frames_tensor.shape}')
    with torch.no_grad():
        encoded_frames = pipe.vae.encode(frames_tensor)[0].sample()
        #print(f'encoded_frames = {encoded_frames}')
    return encoded_frames

def encode_multi_image(vae, video_path, dtype, device):
    import os
    from PIL import Image
    import torch
    from torchvision import transforms

    cap = cv2.VideoCapture(video_path)
                
    frames = []
    frame_count = 0

    while frame_count < 17:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        frame_count += 1
    # print(f'frame_count = {frame_count}')
    cap.release()

    frames_tensor = torch.from_numpy(np.array(frames).transpose(3,0,1,2)).to(device).unsqueeze(0).to(dtype)/255.
    with torch.no_grad():
        encoded_frames = vae.encode(frames_tensor)
    
    return encoded_frames

    


def decode_video(vae, latent, dtype, device,num_frames = 17):

    #encoded_frames = torch.load(encoded_tensor_path, weights_only=True).to(device).to(dtype)
    encoded_frames=latent
    with torch.no_grad():
        decoded_frames = vae.decode(encoded_frames,num_frames = num_frames)

    return decoded_frames

def save_video(tensor, output_path,fps):

    frames = tensor[0].squeeze(0).permute(1, 2, 3, 0).to(torch.float32).cpu().numpy()
    frames = np.clip(frames, 0, 1) * 255
    frames = frames.astype(np.uint8)

    writer = imageio.get_writer(output_path, fps=fps)

    for frame in frames:
        writer.append_data(frame)
    writer.close()
    
def generate_video_with_latent(pipe,prompt,output_path,num_inference_steps,latent,guidance_scale,num_frames,fps):

    video = pipe(
            prompt=prompt,
            num_videos_per_prompt=1,  # Number of videos to generate per prompt
            num_inference_steps=num_inference_steps,  # Number of inference steps
            # num_frames=49,  # Number of frames to generate，changed to 49 for diffusers version `0.31.0` and after.
            num_frames=num_frames,
            use_dynamic_cfg=False,  ## This id used for DPM Sechduler, for DDIM scheduler, it should be False
            guidance_scale=guidance_scale,  # Guidance scale for classifier-free guidance, can set to 7 for DPM scheduler
            #generator=torch.Generator().manual_seed(2024),  # Set the seed for reproducibility
            latents=latent
        ).frames[0]
    export_to_video(video, output_path, fps=fps)

@torch.no_grad()
def transfer_image_to_latent(pipe,image_path,video_save_path,dtype=torch.bfloat16,device="cuda",fps=4,num_frames = 17):
    video_encoded=encode_multi_image(pipe,image_path,dtype,device) 
    video_encode_decode=decode_video(pipe,video_encoded,dtype,device,num_frames=num_frames)
    save_video(video_encode_decode,video_save_path,fps)
    
    # latent=video_encoded.permute(0,2,1,3,4)
    # latent=pipe.vae.config.scaling_factor * latent
    
    return video_encoded

@torch.no_grad()
def transfer_video_to_latent(pipe,video_path,video_save_path,dtype=torch.bfloat16,device="cuda",fps=4,num_frame = 17):
    video_encoded=encode_video(pipe,video_path,dtype,device) 
    video_encode_decode=decode_video(pipe,video_encoded,dtype,device,num_frame=num_frame)
    save_video(video_encode_decode,video_save_path,fps)
    
    latent=video_encoded.permute(0,2,1,3,4)
    latent=pipe.vae.config.scaling_factor * latent
    
    return latent

@torch.no_grad()
def transfer_latent_to_video(pipe,latent,video_save_path,dtype=torch.bfloat16,device="cuda",fps=4):
    
    latent=latent.permute(0,2,1,3,4)
    latent=1 / pipe.vae.config.scaling_factor * latent
    
    video_encode_decode=decode_video(pipe,latent,dtype,device)
    save_video(video_encode_decode,video_save_path,fps)

    
@torch.no_grad()
def main():   
    
    #image_path = "/home/whl/workspace/Open-Sora/samples/samples/sample_0000.mp4"
    # image_path = "/home/whl/workspace/video_collect/cat_1280_720_6fps_17frame.mp4"
    # decode_path = "/home/whl/workspace/Open-Sora/sample_msl/video_decode/cat.mp4"
    # z_all_path = "/home/whl/workspace/Open-Sora/sample_msl/z_all/cat.pt"
    # v_all_path = "/home/whl/workspace/Open-Sora/sample_msl/v_all/cat.pt"
    # edit_path = "/home/whl/workspace/Open-Sora/sample_msl/video_edit/cat.mp4"
    
    image_path = "/home/whl/workspace/video_collect/cat_branch_1280_720_6fps_17frame.mp4"
    decode_path = "/home/whl/workspace/Open-Sora/sample_msl/video_decode/cat_branch.mp4"
    z_all_path = "/home/whl/workspace/Open-Sora/sample_msl/z_all/cat_branch.pt"
    v_all_path = "/home/whl/workspace/Open-Sora/sample_msl/v_all/cat_branch.pt"
    edit_path = "/home/whl/workspace/Open-Sora/sample_msl/video_edit/cat_branch.mp4"
    
        
    device="cuda"
    dtype=torch.float16
    prompt = ["a cat is walking on the grass"]   
    #prompt = ["a beautiful waterfall"]
    num_frames=17
    fps=6
    
    
    cfg = parse_configs(training=False)

    
    reference_path = cfg.get("reference_path", [""] * len(prompt))
    mask_strategy = cfg.get("mask_strategy", [""] * len(prompt))
    
    multi_resolution = cfg.get("multi_resolution", None)
    verbose = cfg.get("verbose", 1)
    align = cfg.get("align", None)
    

    
    #load opensora pipe 
    
    # == init distributed env ==
    if is_distributed():
        colossalai.launch_from_torch({})
        coordinator = DistCoordinator()
        enable_sequence_parallelism = coordinator.world_size > 1
        if enable_sequence_parallelism:
            set_sequence_parallel_group(dist.group.WORLD)
    else:
        coordinator = None
        enable_sequence_parallelism = False
        
    # == build text-encoder and vae ==
    
        

    # == build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)
    text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()
    
    
    image_size = cfg.get("image_size", (720,1280))
    input_size = (num_frames, *image_size)
    latent_size = vae.get_latent_size(input_size)
    
    model = (
        build_module(
            cfg.model,
            MODELS,
            input_size=latent_size,
            in_channels=vae.out_channels,
            caption_channels=text_encoder.output_dim,
            model_max_length=text_encoder.model_max_length,
            enable_sequence_parallelism=enable_sequence_parallelism,
        )
        .to(device, dtype)
        .eval()
    )
    
    text_encoder.y_embedder = model.y_embedder  # HACK: for classifier-free guidance
    
    batch_prompts, refs, ms = extract_json_from_prompts(prompt, reference_path, mask_strategy)
    refs = collect_references_batch(refs, vae, image_size)
    
    model_args = prepare_multi_resolution_info(
            multi_resolution, len(batch_prompts), image_size, num_frames, fps, device, dtype
        )   
     
    
    latent=transfer_image_to_latent(vae,image_path,decode_path,dtype = dtype,num_frames = num_frames)  

    # pdb.set_trace()
    #text encode
    
    # n = len(prompt)
    # model_args = text_encoder.encode(prompt)
    
    # y_null = text_encoder.null(n)
    
    # model_args["y"] = torch.cat([model_args["y"], y_null], 0)
    
    model_args = prepare_multi_resolution_info(
            multi_resolution, len(batch_prompts), image_size, num_frames, fps, device, dtype
        )
    masks = torch.ones(latent.shape[2], dtype=torch.float, device=latent.device)
    
    #inversion
    #batch_prompts_loop = ["a cat is walking on the grass aesthetic score: 6.5."]
    batch_prompts_loop = ["a cat is playing with the branch aesthetic score: 6.5."]
    z_all= scheduler.inv_sample_with_flow_guide(
                model,
                text_encoder,
                z=latent,
                prompts=batch_prompts_loop,
                device=device,
                additional_args=model_args,
                progress=True,
                mask=masks,
            )
    # z_all, v_all = scheduler.inv_sample_with_flow_guide(
    #                 model,
    #                 text_encoder,
    #                 z=latent,
    #                 prompts=batch_prompts_loop,
    #                 device=device,
    #                 additional_args=model_args,
    #                 progress=True,
    #                 mask=masks,
    #             )
    
    # torch.save(z_all,z_all_path)
    # torch.save(v_all,v_all_path)
    
    
    #recon 
    # z_all = torch.load(z_all_path)
    # v_all = torch.load(v_all_path)
    
    #batch_prompts_edit = ["a dog is walking on the grass aesthetic score: 6.5."]
    batch_prompts_edit = ["a cat is playing with the branch aesthetic score: 6.5."]      
    samples = scheduler.sample_with_flow_guide(
                    model,
                    text_encoder,
                    z=z_all[-1].to(latent),
                    z_src = z_all[0].to(latent),
                    prompts=batch_prompts_edit,
                    device=device,
                    additional_args=model_args,
                    progress=True,
                    mask=masks,
                )
    # samples = scheduler.sample_with_flow_guide(
    #                 model,
    #                 text_encoder,
    #                 z=z_all[-1].to(latent),
    #                 prompts=batch_prompts_edit,
    #                 device=device,
    #                 additional_args=model_args,
    #                 progress=True,
    #                 mask=masks,
    #                 v_all=v_all,
    #                 ddim_scale=1,
    #                 end_time=30
    #             )
    samples = vae.decode(samples.to(dtype), num_frames=num_frames)
    #edit_path = "/home/whl/workspace/Open-Sora/sample_msl/video_edit/cat_dog_ddimscale1_end30.mp4"    
    save_video(samples,edit_path,fps = fps)

    

if __name__=='__main__':
    main()