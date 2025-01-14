import os
import time
from pprint import pformat

import colossalai
import torch
import torch.distributed as dist
from colossalai.cluster import DistCoordinator
from colossalai.booster import Booster

from mmengine.runner import set_random_seed
from tqdm import tqdm

from opensora.acceleration.parallel_states import set_sequence_parallel_group
from opensora.datasets import save_sample
from opensora.datasets.aspect import get_image_size, get_num_frames
from opensora.models.text_encoder.t5 import text_preprocessing
from opensora.datasets.dataloader import prepare_dataloader
from opensora.registry import DATASETS, MODELS, SCHEDULERS, build_module
from opensora.acceleration.parallel_states import get_data_parallel_group

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
from opensora.utils.ckpt_utils import load
from opensora.utils.misc import all_exists, create_logger, is_distributed, is_main_process, to_torch_dtype
from diffusers import CogVideoXPipeline, CogVideoXDDIMScheduler, CogVideoXDPMScheduler
import torch.nn as nn
import numpy as np
import imageio

def save_video(tensor, output_path):
    """Saves the video frames to a video file."""
    print(f'tensor.shape = {tensor.shape}')
    frames = tensor.to(torch.float32).permute(1, 2, 3, 0).cpu().numpy()
    print(f'Min value: {frames.min()}, Max value: {frames.max()}')
    frames = (frames + 1.0) / 2.0
    frames = np.clip(frames, 0, 1) * 255
    frames = frames.astype(np.uint8)
    # print(f'Min value: {frames.min()}, Max value: {frames.max()}')
    writer = imageio.get_writer(output_path+".mp4", fps=24, format='mp4')
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    return output_path




def main():
    torch.set_grad_enabled(False)
    # ======================================================
    # configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs(training=False)

    # == device and dtype ==
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dtype = cfg.get("dtype", "fp32")
    assert cfg_dtype in ["fp16", "bf16", "fp32"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

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
    set_random_seed(seed=cfg.get("seed", 1024))


    # == init ColossalAI booster ==
    booster = Booster()


    # == init logger ==
    logger = create_logger()
    logger.info("Inference configuration:\n %s", pformat(cfg.to_dict()))
    verbose = cfg.get("verbose", 1)
    progress_wrap = tqdm if verbose == 1 else (lambda x: x)

    # ======================================================
    # build model & load weights
    # ======================================================
    logger.info("Building models...")
    # # == build text-encoder and vae ==
    # text_encoder = build_module(cfg.text_encoder, MODELS, device=device)
    # vae = build_module(cfg.vae, MODELS).to(device, dtype).eval()

    # # == prepare video size ==
    image_size = cfg.get("image_size", None)
    # if image_size is None:
    #     resolution = cfg.get("resolution", None)
    #     aspect_ratio = cfg.get("aspect_ratio", None)
    #     assert (
    #         resolution is not None and aspect_ratio is not None
    #     ), "resolution and aspect_ratio must be provided if image_size is not provided"
    #     image_size = get_image_size(resolution, aspect_ratio)
    num_frames = get_num_frames(cfg.num_frames)

    # # == build diffusion model ==
    # input_size = (num_frames, *image_size)
    # latent_size = vae.get_latent_size(input_size)
    # model = (
    #     build_module(
    #         cfg.model,
    #         MODELS,
    #         input_size=latent_size,
    #         in_channels=vae.out_channels,
    #         caption_channels=text_encoder.output_dim,
    #         model_max_length=text_encoder.model_max_length,
    #         enable_sequence_parallelism=enable_sequence_parallelism,
    #     )
    #     .to(device, dtype)
    #     .eval()
    # )
    # text_encoder.y_embedder = model.y_embedder  # HACK: for classifier-free guidance

    # == build CogVideoX model ==
    pipe = CogVideoXPipeline.from_pretrained(cfg.model_path, torch_dtype=dtype).to(device)

    text_encoder = pipe.text_encoder
    vae = pipe.vae
    model = pipe.transformer
    tokenizer=pipe.tokenizer

    old_weights = model.patch_embed.proj.weight.clone().detach()
    old_bias = model.patch_embed.proj.bias.clone().detach()
    new_weights = torch.zeros_like(old_weights)
    modified_weights = torch.cat([old_weights, new_weights], dim=1)

    model.patch_embed.proj = nn.Conv2d(32, 1920, kernel_size=(2, 2), stride=2, bias=True)
    model.patch_embed.proj.weight.data = modified_weights
    model.patch_embed.proj.bias.data = old_bias

    if cfg.get("load", None) is not None:
        logger.info("Loading checkpoint")
        model, _, _, _, _ = booster.boost(model)
        ret = load(
            booster,
            cfg.load,
            model=model,
            ema=None,
            optimizer=None,
            lr_scheduler=None,
            sampler=None,
        )
    model.to(dtype)



    # == build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)

    logger.info("Building dataset...")
    # == build dataset ==
    dataset = build_module(cfg.dataset, DATASETS)
    logger.info("Dataset contains %s samples.", len(dataset))  # 1021

    # == build dataloader ==
    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.get("batch_size", 2),
        num_workers=cfg.get("num_workers", 4),
        seed=cfg.get("seed", 1024),
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        process_group=get_data_parallel_group(),
        prefetch_factor=cfg.get("prefetch_factor", None),
    )
    dataloader, sampler = prepare_dataloader(
        bucket_config=cfg.get("bucket_config", None),
        num_bucket_build_workers=cfg.get("num_bucket_build_workers", 1),
        **dataloader_args,
    )
    num_steps_per_epoch = len(dataloader)


    # ======================================================
    # inference
    # ======================================================
    # == prepare arguments ==
    fps = cfg.fps
    save_fps = cfg.get("save_fps", fps // cfg.get("frame_interval", 1))
    multi_resolution = cfg.get("multi_resolution", None)
    batch_size = cfg.get("batch_size", 1)
    num_sample = cfg.get("num_sample", 10)
    loop = cfg.get("loop", 1)
    condition_frame_length = cfg.get("condition_frame_length", 5)
    condition_frame_edit = cfg.get("condition_frame_edit", 0.0)
    align = cfg.get("align", None)

    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)
    save_dir1 = cfg.save_dir1
    os.makedirs(save_dir1, exist_ok=True)
    save_dir2 = cfg.save_dir2
    os.makedirs(save_dir2, exist_ok=True)
    sample_name = cfg.get("sample_name", None)
    prompt_as_path = cfg.get("prompt_as_path", False)

    # == resume ==
    # ckptio = GeneralCheckpointIO()
    # ckptio.load_model(model,cfg.load_path,strict=True)

    # model_sharding(ema)
    B=cfg.get("batch_size", 2)
    y = ""

    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=y,
        negative_prompt=None,
        do_classifier_free_guidance=False,
        num_videos_per_prompt=1,
        device=device,
        dtype=dtype
    )
    model_args={}
    model_args['encoder_hidden_states'] = prompt_embeds.repeat(B, 1, 1)
    del text_encoder

    # gc.collect()
    torch.cuda.empty_cache()
    # model.eval()
    # vae.eval()
    start_idx=0
    # == Iter over all samples ==
    for idx, batch in progress_wrap(enumerate(iterable = iter(dataloader), start=0), total=num_steps_per_epoch):
        # == prepare batch prompts ==
        lq = batch.pop("L").to(device, dtype)  # [C, T, H, W]
        
        hq = batch.pop("H").to(device, dtype)
        
        
        print(f'lq.shape = {lq.shape}')
        print(f'hq.shape = {hq.shape}')
        if lq.dim() == 4:
            lq = lq.unsqueeze(0)
        if hq.dim() == 4:
            hq = hq.unsqueeze(0)

        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                model_args[k] = v.to(device, dtype)
     
        lq = vae.encode(lq)[0].sample()  # [B, C, T, H/P, W/P]


        # == Iter over number of sampling for one prompt ==
        # for k in range(num_sample):
            # == prepare save paths ==
        save_path = get_save_path_name(
                save_dir,
                sample_name=str(idx),
                sample_idx= start_idx + 1,
                prompt=None,
                prompt_as_path=False,
                num_sample=num_sample,
                k=0,
            )
        save_path1 = get_save_path_name(
                save_dir1,
                sample_name=str(idx),
                sample_idx= start_idx + 1,
                prompt=None,
                prompt_as_path=False,
                num_sample=num_sample,
                k=0,
            )
        save_path2 = get_save_path_name(
                save_dir2,
                sample_name=str(idx),
                sample_idx= start_idx + 1,
                prompt=None,
                prompt_as_path=False,
                num_sample=num_sample,
                k=0,
            )

        # == Iter over loop generation ==
        # video_clips = []
        model_args["c"] = lq.to(device, dtype) # 1, 4, 29, 90, 160
        # print("model_args[c]",model_args["c"].shape)
        # == sampling ==
  
        noise = torch.randn_like(model_args["c"]).to(dtype)
 


        samples = scheduler.sample_CogVideoX(
            model,
            text_encoder=None,
            z=torch.cat([noise, model_args["c"]], dim=1),#model_args["c"],
            prompts=[""],
            device=device,
            model_kwargs=model_args,
            additional_args=model_args,
            progress=verbose >= 2,
            mask=None,
        )
        

        samples = vae.decode(samples.to(dtype)).sample
        
        lq = vae.decode(lq.to(dtype)).sample
      

        # video_clips.append(samples)
        # == save samples ==
        if is_main_process():

            # save_path = save_sample(
            #     samples[0],
            #     fps=save_fps,
            #     save_path=save_path,
            #     force_video=True,
            #     verbose=verbose >= 2,
            # )
            # save_path = save_sample(
            #     lq[0],
            #     fps=save_fps,
            #     save_path=save_path1,
            #     force_video=True,
            #     verbose=verbose >= 2,
            # )
            # save_path = save_sample(
            #     hq[0],
            #     fps=save_fps,
            #     save_path=save_path2,
            #     force_video=True,
            #     verbose=verbose >= 2,
            # )

            save_path = save_video(samples[0], save_path)
            save_path = save_video(lq[0], save_path1)
            save_path = save_video(hq[0], save_path2)

            if save_path.endswith(".mp4") and cfg.get("watermark", False):
                time.sleep(1)  # prevent loading previous generated video
                # add_watermark(save_path)
        start_idx += 1
    logger.info("Inference finished.")
    logger.info("Saved %s samples to %s", start_idx, save_dir)


if __name__ == "__main__":
    main()
