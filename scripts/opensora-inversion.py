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


sys.path.insert(0, os.path.expanduser('/home/whl/workspace/'))
from diffusers import CogVideoXPipeline, CogVideoXDDIMScheduler 




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
    # save_video(video_encode_decode,video_save_path,fps)
    
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
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    exp=["rabbit","bird_forest","bird_hand","cat_flower","child_bike","dog_walking","tiger"]
    
    exp_name=exp[3]    
    
    image_path = "/home/whl/workspace/Open-Sora/samples/samples/sample_0000.mp4"
    
    # decode_path = "/home/whl/workspace/output_edit_pre/video_decode/"+exp_name+".mp4"
    # output_path = "/home/whl/workspace/output_edit_pre/video_ddim_recon/"+exp_name+".mp4"
    # all_latent_path = "/home/whl/workspace/output_edit_pre/all_latent_ddim/"+exp_name+".pt"
    
    decode_path = "./output_edit/video_decode/"+exp_name+"_longtext"+".mp4"
    output_path = "/home/whl/workspace/output_edit_pre/video_ddim_recon/"+exp_name+"_longtext"+".mp4"
    all_latent_path = "/home/whl/workspace/output_edit_pre/all_latent_ddim/"+exp_name+"_longtext"+".pt"
    
    device="cuda"
    dtype=torch.float16
    
    prompt_dict = {
                   "rabbit" : "A rabbit is jumping on the grass",
                   "bird_forest" : "A parrot is flying in the forest",
                   #"bird_hand" : "An owl sits on a gloved hand, then spreads its wings to take off in flight",
                   "bird_hand" : "A sequence capturing an owl perched on a gloved hand, preparing to take flight in a serene outdoor setting. The background is softly blurred, highlighting the owl's motion as it gradually unfolds its wings. Warm lighting from the setting or rising sun illuminates the scene, casting a gentle glow on the owl's feathers. The video showcases the elegance and strength of the owl, with its wings spreading wider frame by frame, transitioning from a calm stance to taking flight. The setting is tranquil, emphasizing the natural grace and beauty of the bird in motion.",
                   #"cat_flower" : "A cat is sitting outdoors under a blossoming tree",
                   #"cat_flower" : "A cat is turning its head under a blossoming tree",
                   #"cat_flower" : "A playful cat wearing a harness is sitting on the grass beneath a blossoming cherry tree, surrounded by vibrant pink petals. The cat turns its head curiously, scanning the environment with bright, alert eyes, as if noticing something in the distance. Its movements are natural and fluid, capturing the gentle swaying of the tree branches behind it. The sunlight filters through the blossoms, casting soft shadows on the cat's fur, while a slight breeze causes the petals to flutter around. The background features a park setting with faint outlines of structures blurred, adding depth to the scene.",
                   "cat_flower" : "A cat, dressed in a black harness with subtle patterns, sits comfortably on the grass under the blossoming cherry tree. It starts by looking to the left. Gradually, the cat turns its head to the right, pausing halfway as if something catches its eye, and then continues the motion, its gaze scanning the surroundings with curiosity. At one point, the cat briefly opens its mouth as if to meow or catch a scent in the air. Its movements are natural and fluid, capturing the gentle swaying of the tree branches behind it. The background features a park setting with faint outlines of structures blurred, adding depth to the scene.",
                   #"child_bike" : "A child is riding a bike on the road",
                   "child_bike" : "A young child wearing a teal shirt, black shorts, and a white safety helmet rides a small balance bike along a curved sidewalk in a suburban park. The child grips the handlebars confidently, focusing ahead while learning to maintain balance. The grassy surroundings add a touch of greenery, creating a relaxed outdoor setting. The video captures the slight forward movement, displaying determination and a sense of adventure. The background includes gently sloping lawns and a paved path, emphasizing the everyday charm of a neighborhood stroll while learning to ride.",
                   "dog_walking" : "A man is walking a dog on a leash in a suburban neighborhood",
                   #"tiger" : "A tiger is walking in the forest",
                   "tiger" : "A majestic tiger prowls through a lush forest, its powerful body moving gracefully across the grassy terrain. The sunlight filters through the canopy, casting dappled shadows on the tiger's vibrant orange and black-striped coat. Each frame captures the fluidity of its motion as it steps cautiously, its keen eyes scanning the surroundings. The tiger's posture is both alert and relaxed, embodying a perfect blend of strength and elegance. In the background, tall trees and scattered branches add to the natural ambiance, highlighting the wild beauty of the setting. The scene conveys a sense of quiet power and the timeless allure of the jungle.",
                   "cat_and_duck" : "A cat and a duck are playing on the stone",
                   "cat_grass" : "A cat is walking on the grass"     
    }    
    prompt = ["a beautiful waterfall"]
    num_inference_steps=50
    guidance_scale=6
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

    
    # #inference
    # video = pipe(
    #     prompt=prompt,
    #     num_videos_per_prompt=1,  # Number of videos to generate per prompt
    #     num_inference_steps=num_inference_steps,  # Number of inference steps
    #     # num_frames=49,  # Number of frames to generate，changed to 49 for diffusers version `0.31.0` and after.
    #     num_frames=num_frames,
    #     use_dynamic_cfg=False,  ## This id used for DPM Sechduler, for DDIM scheduler, it should be False
    #     guidance_scale=guidance_scale,  # Guidance scale for classifier-free guidance, can set to 7 for DPM scheduler
    #     generator=torch.Generator().manual_seed(2024),  # Set the seed for reproducibility
    # ).frames[0]
    # export_to_video(video, "/home/whl/workspace/output_edit_pre/video_ddim_recon/test_new_28step.mp4", fps=fps)
    # pdb.set_trace()
    
    # pdb.set_trace()
     
    
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
    #ddim inversion
    video_clips = []
    
    batch_prompts_loop = ['a beautiful waterfall aesthetic score: 6.5.']
    # z_all = scheduler.inv_sample_with_flow_guide(
    #                 model,
    #                 text_encoder,
    #                 z=latent,
    #                 prompts=batch_prompts_loop,
    #                 device=device,
    #                 additional_args=model_args,
    #                 progress=True,
    #                 mask=masks,
    #             )
    
    # torch.save(z_all,"/home/whl/workspace/Open-Sora/output_edit/waterfall_z_all_guide_uncond.pth")
    z_all = torch.load("/home/whl/workspace/Open-Sora/output_edit/waterfall_z_all_guide.pth")
    batch_prompts_edit = ['a beautiful waterfall with blue water aesthetic score: 6.5.']
    # 
    samples = scheduler.sample_with_flow_guide(
                    model,
                    text_encoder,
                    z=z_all[-1].to(latent),z_src = z_all[0].to(latent),
                    prompts=batch_prompts_edit,
                    device=device,
                    additional_args=model_args,
                    progress=True,
                    mask=masks,
                )
    samples = vae.decode(samples.to(dtype), num_frames=num_frames)
    video_clips.append(samples)
    
    #intermidate_latents=invert(pipe,latent, prompt_embeds,negative_prompt_embeds,guidance_scale,num_inference_steps,device=device)
    # torch.save(all_latent,all_latent_path)
    #prompt = "A cat, dressed in a black harness with subtle patterns, walks gracefully across the grass beneath a blossoming cherry tree. It starts by looking to the left, its paws moving softly and deliberately. As it continues forward, the cat turns its head to the right, pausing momentarily as if something catches its attention. Its gaze sweeps the surroundings with curiosity, and for a brief moment, it opens its mouth as if to meow or catch a scent on the breeze. The gentle swaying of the cherry blossoms above complements the cat’s fluid, natural movement. In the background, a blurred park scene adds depth and tranquility to the setting."
    #ddim recon
    save_video(samples,'./samples/samples/recon-guide-blue.mp4',fps = fps)
    #generate_video_with_latent(pipe,prompt,output_path,num_inference_steps,intermidate_latents[-1].unsqueeze(0),guidance_scale,num_frames,fps)
    # pdb.set_trace()
    

if __name__=='__main__':
    main()