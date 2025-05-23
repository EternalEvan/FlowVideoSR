import os
from contextlib import nullcontext
from copy import deepcopy
from datetime import timedelta
from pprint import pformat
from PIL import Image
import torch
import torch.distributed as dist
import wandb
from colossalai.booster import Booster
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device, set_seed
from tqdm import tqdm
import torch.nn.functional as F
from opensora.acceleration.checkpoint import set_grad_checkpoint
from opensora.acceleration.parallel_states import get_data_parallel_group
from opensora.datasets.dataloader import prepare_dataloader
from opensora.registry import DATASETS, MODELS, SCHEDULERS, build_module
from opensora.utils.ckpt_utils import load, model_gathering, model_sharding, record_model_param_shape, save
from opensora.utils.config_utils import define_experiment_workspace, parse_configs, save_training_config
from opensora.utils.lr_scheduler import LinearWarmupLR
from opensora.utils.misc import (
    Timer,
    all_reduce_mean,
    create_logger,
    create_tensorboard_writer,
    format_numel_str,
    get_model_numel,
    requires_grad,
    to_torch_dtype,
)
from opensora.utils.train_utils import MaskGenerator, create_colossalai_plugin, update_ema
import gc,sys
from diffusers.utils import export_to_video,load_image
import torch.nn as nn

from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
from itertools import chain
sys.path.append('/home/whl/workspace/Pyramid-Flow')
from pyramid_dit import PyramidDiTForVideoGeneration
from pyramid_dit import PyramidDiffusionMMDiT_Controlnet
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
os.environ['RANK'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['WORLD_SIZE'] = '1'
class controlnet_wrapper(nn.Module):
    def __init__(self, tranformer, controlnet):
        super().__init__()
        self.controlnet = controlnet
        self.tranformer = tranformer


    # def forward(self, hidden_states, encoder_hidden_states, timestep, block_controlnet_hidden_states=None, control_video=None, return_dict=False, **kwargs):
    def forward(self, hidden_states, timestep, **kwargs):
        assert kwargs["c"] is not None
        controlnet_out = self.controlnet(
            # encoder_hidden_states=hidden_states,
            encoder_hidden_states=kwargs["encoder_hidden_states"],
            timestep=timestep,
            controlnet_cond=kwargs["c"],
            conditioning_scale=1.0,
            return_dict=False,
        )[0]
        output = self.tranformer(
            hidden_states=hidden_states,
            encoder_hidden_states=kwargs["encoder_hidden_states"],
            timestep=timestep,
            block_controlnet_hidden_states=controlnet_out,
            # return_dict=return_dict,
            return_dict=False,
        )
        return output

def main():
    # ======================================================
    # 1. configs & runtime variables
    # == parse configs ==
    cfg = parse_configs(training=True)
    record_time = cfg.get("record_time", False)

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    cfg_dtype = cfg.get("dtype", "bf16")
    assert cfg_dtype in ["fp16", "bf16"], f"Unknown mixed precision {cfg_dtype}"
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))

    # == colossalai init distributed training ==
    dist.init_process_group(backend="nccl", timeout=timedelta(hours=24))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    set_seed(cfg.get("seed", 1024))
    coordinator = DistCoordinator()
    device = get_current_device()

    # == init exp_dir ==
    exp_name, exp_dir = define_experiment_workspace(cfg)
    coordinator.block_all()
    if coordinator.is_master():
        os.makedirs(exp_dir, exist_ok=True)
        save_training_config(cfg.to_dict(), exp_dir)
    coordinator.block_all()

    # == init logger, tensorboard & wandb ==
    logger = create_logger(exp_dir)
    logger.info("Experiment directory created at %s", exp_dir)
    logger.info("Training configuration:\n %s", pformat(cfg.to_dict()))
    if coordinator.is_master():
        tb_writer = create_tensorboard_writer(exp_dir)
        if cfg.get("wandb", False):
            wandb.init(project="Open-Sora", name=exp_name, config=cfg.to_dict(), dir="./outputs/wandb")

    # == init ColossalAI booster ==
    plugin = create_colossalai_plugin(
        plugin=cfg.get("plugin", "zero2"),
        dtype=cfg_dtype,
        grad_clip=cfg.get("grad_clip", 0),
        sp_size=cfg.get("sp_size", 1),
        reduce_bucket_size_in_m=cfg.get("reduce_bucket_size_in_m", 20),
    )
    booster = Booster(plugin=plugin)
    torch.set_num_threads(1)

    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    logger.info("Building dataset...")
    # == build dataset ==
    dataset = build_module(cfg.dataset, DATASETS)
    logger.info("Dataset contains %s samples.", len(dataset))  

    # == build dataloader ==
    dataloader_args = dict(
        dataset=dataset,
        batch_size=cfg.get("batch_size", 4),
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
    # 3. build model
    # ======================================================
    logger.info("Building models...")
    # # == build text-encoder and vae ==
    pipe = PyramidDiTForVideoGeneration(
        '/home/whl/workspace/Open-Sora/pyramid-flow-sd3', dtype,
        model_variant='diffusion_transformer_768p',     # 'diffusion_transformer_384p'
    )
    
    # == build Pyramidal Flow Matching model ==
    
    # pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    text_encoder = pipe.text_encoder.to(device,dtype=dtype)
    requires_grad(text_encoder, False)
    vae = pipe.vae.to(device,dtype=dtype)
    requires_grad(vae, False)
    transformer = pipe.dit.to(device,dtype=dtype)
    # tokenizer = pipe.tokenizer
    # print("Attributes of pipe:", dir(transformer))

    controlnet = PyramidDiffusionMMDiT_Controlnet.from_transformer(transformer)
    controlnet.train()
 
    ####################
    # set lora
    transformer_lora_config = LoraConfig(
        r=cfg.get("lora_rank", 128),
        lora_alpha=cfg.get("lora_alpha", 128),
        init_lora_weights=True,
        target_modules=cfg.get("target_modules", ["to_k", "to_v", "to_q", "to_out.0"]),
    )
    transformer.add_adapter(transformer_lora_config)   
    ###################

    transformer_numel, transformer_numel_trainable = get_model_numel(transformer)
    logger.info(
        "[transformer] Trainable model params: %s, Total model params: %s",
        format_numel_str(transformer_numel_trainable),
        format_numel_str(transformer_numel),
    )

    controlnet_numel, controlnet_numel_trainable = get_model_numel(controlnet)
    logger.info(
        "[controlnet] Trainable model params: %s, Total model params: %s",
        format_numel_str(controlnet_numel_trainable),
        format_numel_str(controlnet_numel),
    )

      # create model_wrapper
    model = controlnet_wrapper(transformer, controlnet)
    pipe.dit = model
    
    # == setup loss function, build scheduler ==
    scheduler = build_module(cfg.scheduler, SCHEDULERS)
    
    # == setup optimizer ==
    from itertools import chain
    optimizer = HybridAdam(
        filter(lambda p: p.requires_grad, chain(transformer.parameters(), controlnet.parameters())),
        adamw_mode=True,
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("adam_eps", 1e-8),
    )

    warmup_steps = cfg.get("warmup_steps", None)

    if warmup_steps is None:
        lr_scheduler = None
    else:
        lr_scheduler = LinearWarmupLR(optimizer, warmup_steps=cfg.get("warmup_steps"))
  
    # == additional preparation ==
    if cfg.get("grad_checkpoint", False):
        set_grad_checkpoint(model)
    if cfg.get("mask_ratios", None) is not None:
        mask_generator = MaskGenerator(cfg.mask_ratios)

    # =======================================================
    # 4. distributed training preparation with colossalai
    # =======================================================
    logger.info("Preparing for distributed training...")
    torch.set_default_dtype(dtype)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=dataloader,
    )
    torch.set_default_dtype(torch.float)
    logger.info("Boosting model for distributed training")

    # == global variables ==
    cfg_epochs = cfg.get("epochs", 1000)
    start_epoch = start_step = log_step = acc_step = 0
    running_loss = 0.0
    logger.info("Training for %s epochs with %s steps per epoch", cfg_epochs, num_steps_per_epoch)

    # == resume ==
    if cfg.get("load", None) is not None:
        logger.info("Loading checkpoint")
        ret = load(
            booster,
            cfg.load,
            model=model,
            ema=None,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            sampler=None if cfg.get("start_from_scratch", False) else sampler,
        )
        if not cfg.get("start_from_scratch", False):
            start_epoch, start_step = ret
        logger.info("Loaded checkpoint %s at epoch %s step %s", cfg.load, start_epoch, start_step)
    
    # model_sharding(ema)
    B=cfg.get("batch_size", 2)
    prompt = "HD video"
    negative_prompt="cartoon style, worst quality, low quality, blurry, absolute black, absolute white, low res, extra limbs, extra digits, misplaced objects, mutated anatomy, monochrome, horror",
   
    prompt_embeds, prompt_attention_mask, pooled_prompt_embeds  = text_encoder(prompt,device)
    negative_prompt_embeds, negative_prompt_attention_mask, negative_pooled_prompt_embeds = text_encoder(negative_prompt, device)
    # do_classifier_free_guidance = True
    # if do_classifier_free_guidance:
    #     prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    #     pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)
    #     prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
   
    model_args={}
    model_args['encoder_hidden_states'] = prompt_embeds.repeat(B, 1, 1)
    model_args['prompt_embeds']= prompt_embeds.repeat(B, 1, 1) 
    model_args['pooled_prompt_embeds'] = pooled_prompt_embeds.repeat(B, 1, 1) 
    model_args['prompt_attention_mask']= prompt_attention_mask.repeat(B, 1, 1) 

    del text_encoder
    gc.collect()
    torch.cuda.empty_cache()
    # =======================================================
    # 5. training loop
    # =======================================================
    dist.barrier()
    timers = {}
    timer_keys = [
        "move_data",
        "encode",
        "mask",
        "diffusion",
        "backward",
        "update_ema",
        "reduce_loss",
    ]
    for key in timer_keys:
        if record_time:
            timers[key] = Timer(key, coordinator=coordinator)
        else:
            timers[key] = nullcontext()
    for epoch in range(start_epoch, cfg_epochs):
        # == set dataloader to new epoch ==
        sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)
        logger.info("Beginning epoch %s...", epoch)
        # == training loop in an epoch ==
        with tqdm(
            enumerate(dataloader_iter, start=start_step),
            desc=f"Epoch {epoch}",
            disable=not coordinator.is_master(),
            initial=start_step,
            total=num_steps_per_epoch,
        ) as pbar:
            for step, batch in pbar:
                torch.cuda.empty_cache()
                timer_list = []
                with timers["move_data"] as move_data_t:
                    # x = batch.pop("video").to(device, dtype)  # [B, C, T, H, W]    [8, 3, 51, 360, 640]
                    
                    lq = batch.pop("L").to(device, dtype)  # [B, C, T, H, W]
                    hq = batch.pop("H").to(device, dtype)  # [B, C, T, H, W]  ([B, 100, 3, 720, 1280])

                    if lq.dim() == 4:
                        lq = lq.unsqueeze(0)
                    if hq.dim() == 4:
                        hq = hq.unsqueeze(0)
                  
    
                if record_time:
                    timer_list.append(move_data_t)
                # == visual and text encoding ==
                with timers["encode"] as encode_t:
                    with torch.no_grad():
                        # Prepare visual inputs
                        if cfg.get("load_video_features", False):
                            lq = lq.to(device, dtype)
                            hq = hq.to(device, dtype)
                        else:
                            lq = vae.encode(lq)[0].sample()  # [B, C, T, H/P, W/P]
                            hq = vae.encode(hq)[0].sample()
                        # Prepare text inputs
                       
                        # else:
                        #     model_args = text_encoder.encode(y)
                        #     model_args['y']= model_args['y'].repeat(B, 1, 1, 1) 
                        #     model_args['mask']= model_args['mask'].repeat(B, 1, 1, 1) 
                if record_time:
                    timer_list.append(encode_t)
                # == mask ==
                with timers["mask"] as mask_t:
                    mask = None
                    if cfg.get("mask_ratios", None) is not None:
                        mask = mask_generator.get_masks(hq)
                    model_args["x_mask"] = None # mask
                if record_time:
                    timer_list.append(mask_t)

                # == video meta info ==
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        model_args[k] = v.to(device, dtype)
                        
                model_args["c"] = lq.to(device, dtype).permute(0, 2, 1, 3, 4).contiguous()
                model_args["hq"] = hq.to(device, dtype)
                # == diffusion loss computation ==
                with timers["diffusion"] as loss_t:
                    # loss_dict = scheduler.training_losses_CogVideoX(model,x_start=lq, model_kwargs=model_args, mask=None)
                    loss_dict = scheduler.training_losses_PFM_ControlNet(pipe,x_start=lq, model_kwargs=model_args)

                if record_time:
                    timer_list.append(loss_t)

                # == backward & update ==
                with timers["backward"] as backward_t:
                    loss = loss_dict["loss"].mean()
                    # loss_1 = loss_dict["loss_1"].mean().item()
                    # loss_2 = loss_dict["loss_2"].mean().item()
                    # loss_3 = loss_dict["loss_3"].mean().item()
                    # print(f'loss_1 = {loss_1}, loss_2 = {loss_2}, loss_3 = {loss_3}')
                    booster.backward(loss=loss, optimizer=optimizer)
                    torch.cuda.empty_cache()
                    gc.collect()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    # update learning rate
                    if lr_scheduler is not None:
                        lr_scheduler.step()
                if record_time:
                    timer_list.append(backward_t)

                # == update EMA ==
                # with timers["update_ema"] as ema_t:
                #     update_ema(ema, model.module, optimizer=optimizer, decay=cfg.get("ema_decay", 0.9999))
                # if record_time:
                #     timer_list.append(ema_t)

                # == update log info ==
                with timers["reduce_loss"] as reduce_loss_t:
                    all_reduce_mean(loss)
                    running_loss += loss.item()
                    global_step = epoch * num_steps_per_epoch + step
                    log_step += 1
                    acc_step += 1
                if record_time:
                    timer_list.append(reduce_loss_t)

                # == logging ==
                if coordinator.is_master() and (global_step + 1) % cfg.get("log_every", 1) == 0:
                    avg_loss = running_loss / log_step
                    # progress bar
                    pbar.set_postfix({"loss": avg_loss, "step": step, "global_step": global_step})
                    # pbar.set_postfix({"loss": avg_loss, "loss_1":loss_1, "loss_2":loss_2, "loss_3":loss_3, "step": step, "global_step": global_step})
                    # print(f'loss_1 = {loss_1}, loss_2 = {loss_2}, loss_3 = {loss_3}')

                    # tensorboard
                    tb_writer.add_scalar("loss", loss.item(), global_step)
                    # wandb
                    if cfg.get("wandb", False):
                        wandb.log(
                            {
                                "iter": global_step,
                                "acc_step": acc_step,
                                "epoch": epoch,
                                "loss": loss.item(),
                                "avg_loss": avg_loss,
                                "lr": optimizer.param_groups[0]["lr"],
                                "debug/move_data_time": move_data_t.elapsed_time,
                                "debug/encode_time": encode_t.elapsed_time,
                                "debug/mask_time": mask_t.elapsed_time,
                                "debug/diffusion_time": loss_t.elapsed_time,
                                "debug/backward_time": backward_t.elapsed_time,
                                # "debug/update_ema_time": ema_t.elapsed_time,
                                "debug/reduce_loss_time": reduce_loss_t.elapsed_time,
                            },
                            step=global_step,
                        )

                    running_loss = 0.0
                    log_step = 0

                # == checkpoint saving ==
                ckpt_every = cfg.get("ckpt_every", 0)
                if ckpt_every > 0 and (global_step + 1) % ckpt_every == 0:
                    # model_gathering(ema, ema_shape_dict)
                    save_dir = save(
                        booster,
                        exp_dir,
                        model=model,
                        ema=None,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        sampler=sampler,
                        epoch=epoch,
                        step=step + 1,
                        global_step=global_step + 1,
                        batch_size=cfg.get("batch_size", None),
                    )
                    # if dist.get_rank() == 0:
                    #     model_sharding(ema)
                    logger.info(
                        "Saved checkpoint at epoch %s, step %s, global_step %s to %s",
                        epoch,
                        step + 1,
                        global_step + 1,
                        save_dir,
                    )
                if record_time:
                    log_str = f"Rank {dist.get_rank()} | Epoch {epoch} | Step {step} | "
                    for timer in timer_list:
                        log_str += f"{timer.name}: {timer.elapsed_time:.3f}s | "
                    print(log_str)

        sampler.reset()
        start_step = 0


if __name__ == "__main__":
    main()