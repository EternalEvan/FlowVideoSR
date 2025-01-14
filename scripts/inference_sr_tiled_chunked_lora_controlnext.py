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
from opensora.models.my_diffusers import CogVideoXControlNextPipeline, CogVideoXControlNextModel, CogVideoXControlNextPipeline
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import imageio

from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from itertools import chain


class controlnet_wrapper(nn.Module):
    def __init__(self, tranformer, controlnet):
        super().__init__()
        self.controlnet = controlnet
        self.tranformer = tranformer


    # def forward(self, hidden_states, encoder_hidden_states, timestep, block_controlnet_hidden_states=None, control_video=None, return_dict=False, **kwargs):
    def forward(self, hidden_states, timestep, **kwargs):
        assert kwargs["c"] is not None
        controlnet_out = self.controlnet(
            hidden_states=hidden_states,
            encoder_hidden_states=kwargs["encoder_hidden_states"],
            timestep=timestep,
            controlnet_cond=kwargs["c"],
            conditioning_scale=1.0,
            return_dict=False,
        )
        output = self.tranformer(
            hidden_states=hidden_states,
            encoder_hidden_states=kwargs["encoder_hidden_states"],
            timestep=timestep,
            mid_block_additional_residual=controlnet_out,
            # return_dict=return_dict,
            return_dict=False,
        )
        return output


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
    # pipe = CogVideoXPipeline.from_pretrained(cfg.model_path, torch_dtype=dtype).to(device)

    # text_encoder = pipe.text_encoder
    # vae = pipe.vae
    # model = pipe.transformer
    # tokenizer=pipe.tokenizer

    # old_weights = model.patch_embed.proj.weight.clone().detach()
    # old_bias = model.patch_embed.proj.bias.clone().detach()
    # new_weights = torch.zeros_like(old_weights)
    # modified_weights = torch.cat([old_weights, new_weights], dim=1)

    # model.patch_embed.proj = nn.Conv2d(32, 1920, kernel_size=(2, 2), stride=2, bias=True)
    # model.patch_embed.proj.weight.data = modified_weights
    # model.patch_embed.proj.bias.data = old_bias




    pipe = CogVideoXControlNextPipeline.from_pretrained(cfg.model_path, torch_dtype=dtype).to(device)
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    transformer = pipe.transformer
    tokenizer = pipe.tokenizer

    controlnet = CogVideoXControlNextModel.from_transformer(transformer).to(device)
    
    transformer_lora_config = LoraConfig(
        r=cfg.get("lora_rank", 128),
        lora_alpha=cfg.get("lora_alpha", 128),
        init_lora_weights=True,
        target_modules=cfg.get("target_modules", ["to_k", "to_v", "to_q", "to_out.0"]),
    )
    transformer.add_adapter(transformer_lora_config)

    model = controlnet_wrapper(transformer, controlnet)

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

    # 设定切割块的大小
    # patch_size = 448
    patch_size = 512
    stride = patch_size  
    chunk_size = cfg.get("chunk_size", 9)
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


        #### 分块
        B, C, T, H, W = lq.shape

        num_chunks = (T + chunk_size - 1) // chunk_size
        num_overlap = num_chunks * chunk_size - T
        print(f'num_overlap = {num_overlap}')
        processed_chunks = []
        for chunk_idx in range(num_chunks):
            if chunk_idx != num_chunks - 1:
                start_frame = chunk_idx * chunk_size
                end_frame = (chunk_idx + 1) * chunk_size
            else:
                start_frame = T - chunk_size
                end_frame = T
            lq_chunk = lq[:, :, start_frame:end_frame, :, :]

            pad_h = (patch_size - H % patch_size) % patch_size
            pad_w = (patch_size - W % patch_size) % patch_size

            lq_chunk = F.pad(lq_chunk.squeeze(0), (0, pad_w, 0, pad_h), mode='reflect').unsqueeze(0)


            H_padded, W_padded = lq_chunk.shape[-2:]

       
            # 分块操作
            lq_patches = F.unfold(lq_chunk.permute(0,2,1,3,4).view(B*chunk_size, C, H_padded, W_padded), kernel_size=patch_size, stride=stride)
            N_patches = lq_patches.shape[-1]

            # 恢复维度到 [B, T, C, patch_size, patch_size, N_patches]，保证时间维度在第三个维度
            lq_patches = lq_patches.view(B, chunk_size, C, patch_size, patch_size, N_patches).permute(0, 1, 2, 5, 3, 4) #[B, T, C, N_patches, patch_size, patch_size]
            
            print(f'lq_patches.shape = {lq_patches.shape}')
            processed_patches = []
            for i in range(lq_patches.shape[3]):
                lq_patch = lq_patches[:, :, :,  i, :, :]
                lq_patch = lq_patch.permute(0, 2, 1, 3, 4).contiguous()
                model_args["c"] = vae.encode(lq_patch)[0].sample()
                model_args["c"] = model_args["c"].permute(0, 2, 1, 3, 4).contiguous()
                noise = torch.randn_like(model_args["c"]).to(dtype)
                noise = noise.permute(0, 2, 1, 3, 4).contiguous()
                print(f'noise.shape = {noise.shape}')
                samples = scheduler.sample_CogVideoX_ControlNet(#_mid_value(
                    model,
                    text_encoder=None,
                    # z=torch.cat([noise, model_args["c"]], dim=1),#model_args["c"],
                    z=noise,
                    prompts=[""],
                    device=device,
                    model_kwargs=model_args,
                    additional_args=model_args,
                    progress=verbose >= 2,
                    mask=None,
                )
                samples = vae.decode(samples.to(dtype)).sample
                print(f'samples.shape = {samples.shape}')
                processed_patches.append(samples)


            ## 重新拼接
            processed_patches = torch.stack(processed_patches, dim=2)  # [B, C, N_patches, T, patch_size, patch_size]
            print(f'processed_patches.shape = {processed_patches.shape}')
            processed_patches = processed_patches.permute(0, 3, 1, 4, 5, 2).contiguous()  # [B, T, C, patch_size, patch_size, N_patches]
            print(f'processed_patches.shape = {processed_patches.shape}')
            T_patch = processed_patches.shape[1]
            processed_patches = processed_patches.view(B * T_patch, C * patch_size * patch_size, N_patches)   

            reconstructed = F.fold(processed_patches, output_size=(H_padded, W_padded), kernel_size=patch_size, stride=stride)
            reconstructed = reconstructed.view(B, T_patch, C, H_padded, W_padded)
            print(f'reconstructed.shape = {reconstructed.shape}')
            reconstructed = reconstructed[:, :, :, :H, :W]
            reconstructed = reconstructed.permute(0, 2, 1, 3, 4)

            processed_chunks.append(reconstructed)

        if num_overlap > 0:
            processed_chunks[-2] = processed_chunks[-2][:, :, :(chunk_size - num_overlap), :, :]
        final_results = torch.cat(processed_chunks, dim=2)
        print(f'{final_results.shape}, {hq.shape}')
        
       
        # == save samples ==
        if is_main_process():
            lq = lq[..., :H, :W]
            # hq = hq[..., :H, :W]


            save_path = save_video(final_results[0], save_path)
            save_path = save_video(lq[0], save_path1)
            # save_path = save_video(hq[0], save_path2)

            if save_path.endswith(".mp4") and cfg.get("watermark", False):
                time.sleep(1)  # prevent loading previous generated video
                # add_watermark(save_path)
        start_idx += 1
    logger.info("Inference finished.")
    logger.info("Saved %s samples to %s", start_idx, save_dir)


if __name__ == "__main__":
    main()
