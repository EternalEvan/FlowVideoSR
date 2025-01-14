# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np
import torch
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
from einops import rearrange
# from diffusion.model.builder import MODELS
from opensora.registry import MODELS
# from diffusion.model.utils import auto_grad_checkpoint, to_2tuple
from opensora.acceleration.checkpoint import auto_grad_checkpoint
#from diffusion.model.nets.PixArt_blocks import t2i_modulate, CaptionEmbedder, AttentionKVCompress, MultiHeadCrossAttention, T2IFinalLayer, TimestepEmbedder, SizeEmbedder
# from diffusion.model.nets.PixArt import PixArt
from .pixart_sigma import PixArtMSBlock
from .pixart import PixArt
from opensora.models.layers.blocks import (t2i_modulate,CaptionEmbedder,
                                           KVCompressAttention,MultiHeadCrossAttention,
                                           T2IFinalLayer,TimestepEmbedder,SizeEmbedder,
                                           get_2d_sincos_pos_embed,to_2tuple,PatchEmbed3D,
                                           TimestepEmbedder,PositionEmbedding2D)


from diffusers import ConsistencyDecoderVAE, PixArtAlphaPipeline, Transformer2DModel, DDPMScheduler
from opensora.models.stdit.stdit3 import STDiT3Block
from opensora.utils.ckpt_utils import load_checkpoint


class TemporalSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, drop_path=0., qkv_bias=True,
                 sampling=None, sr_ratio=1, qk_norm=False, **block_kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = KVCompressAttention(
            hidden_size, num_heads=num_heads, qkv_bias=True, sampling=sampling, sr_ratio=sr_ratio,
            qk_norm=qk_norm, **block_kwargs
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(3, hidden_size) / hidden_size ** 0.5)

    def forward(self, x, t, HW=None, **kwargs):
        x = x.permute(1, 0, 2)
        B, N, C = x.shape  #  1024, 6 ,1152
        ## neglecting t
        # shift_mtsa, scale_mtsa, gate_mtsa = (self.scale_shift_table[None] + t.reshape(B, 3, -1)).chunk(3, dim=1)
        shift_mtsa, scale_mtsa, gate_mtsa = (self.scale_shift_table[None]).chunk(3, dim=1)

        # x = x + self.drop_path(gate_mtsa * self.attn(t2i_modulate(self.norm1(x), shift_mtsa, scale_mtsa), HW=HW))
        x_copy = x.clone()
        normalized_x = self.norm1(x_copy)
        modulated_attention = t2i_modulate(normalized_x, shift_mtsa, scale_mtsa)
        attn_output = self.attn(modulated_attention, HW=HW)
        weighted_attn_output = gate_mtsa * attn_output
        drop_path_output = self.drop_path(weighted_attn_output)
        x = x + drop_path_output


        x = x.permute(1, 0, 2)
        return x




#############################################################################
#                                 Core PixArt Model                                #
#################################################################################
@MODELS.register_module()
class PixArtMS_srvideo(PixArt):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
            self,
            input_size=(1, 32, 32), #32,
            patch_size=(1, 2, 2), #2,
            in_channels=4,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            learn_sigma=True,
            pred_sigma=True,
            drop_path: float = 0.,
            caption_channels=4096,
            pe_interpolation=1.,
            config=None,
            model_max_length=300,
            micro_condition=False,
            qk_norm=True,
            kv_compress_config=None,
            enable_flash_attn=False,
            enable_layernorm_kernel=False,
            enable_sequence_parallelism=False,
            **kwargs,
    ):
        input_size=(1, 32, 32)
        super().__init__(
            input_size=input_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            class_dropout_prob=class_dropout_prob,
            learn_sigma=learn_sigma,
            pred_sigma=pred_sigma,
            drop_path=drop_path,
            pe_interpolation=pe_interpolation,
            config=config,
            model_max_length=model_max_length,
            qk_norm=qk_norm,
            kv_compress_config=kv_compress_config,
            **kwargs,
        )
        self.h = self.w = 0
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.t_block = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.enable_flash_attn=enable_flash_attn,
        self.enable_layernorm_kernel=enable_layernorm_kernel,
        self.enable_sequence_parallelism=enable_sequence_parallelism,
               
        # embedding
        self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)
        self.y_embedder = CaptionEmbedder(in_channels=caption_channels, hidden_size=hidden_size, uncond_prob=class_dropout_prob, act_layer=approx_gelu, token_num=model_max_length)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.fps_embedder = SizeEmbedder(hidden_size)
        self.micro_conditioning = micro_condition
        if self.micro_conditioning:
            self.csize_embedder = SizeEmbedder(hidden_size//3)  # c_size embed
            self.ar_embedder = SizeEmbedder(hidden_size//3)     # aspect ratio embed
        
        # spatial blocks
        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        
        if kv_compress_config is None:
            kv_compress_config = {'sampling': None, 'scale_factor': 1, 'kv_compress_layer': []}
        
        self.blocks = nn.ModuleList([
            STDiT3Block(
                hidden_size, num_heads, mlp_ratio=mlp_ratio, drop_path=drop_path[i],   
            )
            for i in range(depth)
        ])
        self.tfas = nn.ModuleList([
            TemporalSelfAttention(
                hidden_size, num_heads=num_heads, qkv_bias=True, sampling=kv_compress_config['sampling'],
                sr_ratio=int(kv_compress_config['scale_factor']) if i in kv_compress_config['kv_compress_layer'] else 1,
                qk_norm=qk_norm,
            )
            for i in range((depth + 1) // 2)
        ])
        
        self.pos_embed = PositionEmbedding2D(hidden_size)

        self.final_layer = T2IFinalLayer(hidden_size,  np.prod(self.patch_size), self.out_channels)
        
        self.initialize()

    def get_dynamic_size(self, x):
        _, _, T, H, W = x.size()
        if T % self.patch_size[0] != 0:
            T += self.patch_size[0] - T % self.patch_size[0]
        if H % self.patch_size[1] != 0:
            H += self.patch_size[1] - H % self.patch_size[1]
        if W % self.patch_size[2] != 0:
            W += self.patch_size[2] - W % self.patch_size[2]
        T = T // self.patch_size[0]
        H = H // self.patch_size[1]
        W = W // self.patch_size[2]
        return (T, H, W)
  
    def encode_text(self, y, mask=None):
        y = self.y_embedder(y, self.training)  # [B, 1, N_token, C]
        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, self.hidden_size)
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, self.hidden_size)
        return y, y_lens
    
    def forward(self, x, timestep, y, mask=None, x_mask=None, fps=None, height=None, width=None, **kwargs):
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """

        B = x.shape[0]
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)
        
        # === get pos embed ===
        _, _, Tx, Hx, Wx = x.size()
        T, H, W = self.get_dynamic_size(x)
        S = H * W
        base_size = round(S**0.5)
        resolution_sq = (height[0].item() * width[0].item()) ** 0.5
        self.input_sq_size=512 ##  ?
        scale = resolution_sq / self.input_sq_size
        pos_emb = self.pos_embed(x, H, W, scale=scale, base_size=base_size)

        # === get timestep embed ===
        t = self.t_embedder(timestep,dtype=x.dtype)  # (N, D)
        fps = self.fps_embedder(fps.unsqueeze(1), B)
        t = t + fps
        t_mlp = self.t_block(t)
        t0 = t0_mlp = None
        if x_mask is not None:
            t0_timestep = torch.zeros_like(timestep)
            t0 = self.t_embedder(t0_timestep, dtype=x.dtype)
            t0 = t0 + fps
            t0_mlp = self.t_block(t0)
        
        # === get y embed ===
        y, y_lens = self.encode_text(y, mask) #  if not self.config.skip_y_embedder:
        
        # === get x embed ===
        x = self.x_embedder(x) # [B, N, C]
        x = rearrange(x, "B (T S) C -> B T S C", T=T, S=S)
        x = x + pos_emb  # (N, T, D), where T = H * W / patch_size ** 2
        
        x = rearrange(x, "B T S C -> B (T S) C", T=T, S=S)
        
        # === blocks ===
        for block_idx in range(len(self.blocks) // 2):
            x = auto_grad_checkpoint(self.blocks[block_idx], x, y, t_mlp, y_lens, x_mask, t0_mlp, T, S)  # (N, T, D) #support grad checkpoint

        for block_idx in range(len(self.blocks) // 2, len(self.blocks)):
            x = auto_grad_checkpoint(self.blocks[block_idx], x, y, t_mlp, y_lens, x_mask, t0_mlp, T, S)  # (N, T, D) #support grad checkpoint
            x = auto_grad_checkpoint(self.tfas[block_idx - (len(self.blocks) // 2)], x, t0, (self.h, self.w), **kwargs)

        # === final layer ===
        x = self.final_layer(x, t, x_mask, t0, T, S)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x, T, H, W, Tx, Hx, Wx)  # (N, out_channels, H, W)
        x = x.to(torch.float32)
        return x

    def forward_with_dpmsolver(self, x, timestep, y, data_info, **kwargs):
        """
        dpm solver donnot need variance prediction
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        model_out = self.forward(x, timestep, y, data_info=data_info, **kwargs)
        return model_out.chunk(2, dim=1)[0]

    def forward_with_cfg(self, x, timestep, y, cfg_scale, data_info, mask=None, **kwargs):
        """
        Forward pass of PixArt, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, timestep, y, mask, data_info=data_info, **kwargs)
        model_out = model_out['x'] if isinstance(model_out, dict) else model_out
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def unpatchify(self, x, N_t, N_h, N_w, R_t, R_h, R_w):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """

        # N_t, N_h, N_w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        # unpad
        x = x[:, :, :R_t, :R_h, :R_w]
        return x

    def initialize(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)
        if self.micro_conditioning:
            nn.init.normal_(self.csize_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.csize_embedder.mlp[2].weight, std=0.02)
            nn.init.normal_(self.ar_embedder.mlp[0].weight, std=0.02)
            nn.init.normal_(self.ar_embedder.mlp[2].weight, std=0.02)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out tfas:
        for tfa in self.tfas:
            nn.init.constant_(tfa.attn.proj.weight, 0)
            nn.init.constant_(tfa.attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


#################################################################################
#                                   PixArt Configs                                  #
#################################################################################
@MODELS.register_module("PixArtMS_srvideo_XL_2")
def PixArtMS_srvideo_XL_2(from_pretrained=None,**kwargs):
    force_huggingface = kwargs.pop("force_huggingface", False)
    if force_huggingface or from_pretrained is not None and not os.path.exists(from_pretrained):
        model = PixArtMS_srvideo.from_pretrained(from_pretrained, **kwargs)
    else:
        model = PixArtMS_srvideo(depth=28, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **kwargs)
        if from_pretrained is not None:
            load_checkpoint(model, from_pretrained)
    return model
    
    





class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.silu = nn.SiLU() #inplace=True
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        x_ = self.conv1(x)
        residual = x_
        out = self.bn1(x_)
        out = self.silu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual  # 恒等映射
        out = self.silu(out)
        return out






















#################################################################################
#################################################################################
from diffusers.utils import BaseOutput
from diffusers.configuration_utils import register_to_config
from diffusers.models.embeddings import PatchEmbed as PatchEmbed_diffusers
from diffusers.models.embeddings import PixArtAlphaTextProjection
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.normalization import AdaLayerNormSingle
from typing import Any, Dict, Optional

class TransformerSRVideo2DModelOutput(BaseOutput):
    sample: torch.FloatTensor


class TransformerSRVideo2DModel(Transformer2DModel):
    @register_to_config
    def __init__(self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        caption_channels: int = None,
        interpolation_scale: float = None,
    ):
        super().__init__(num_attention_heads,
        attention_head_dim,
        in_channels,
        out_channels,
        num_layers,
        dropout,
        norm_num_groups,
        cross_attention_dim,
        attention_bias,
        sample_size,
        num_vector_embeds,
        patch_size,
        activation_fn,
        num_embeds_ada_norm,
        use_linear_projection,
        only_cross_attention,
        double_self_attention,
        upcast_attention,
        norm_type,
        norm_elementwise_affine,
        norm_eps,
        attention_type,
        caption_channels,
        interpolation_scale)
        self.hidden_size = self.inner_dim

  

    def _init_continuous_input(self, norm_type):
        self.norm = torch.nn.GroupNorm(
            num_groups=self.config.norm_num_groups, num_channels=self.in_channels, eps=1e-6, affine=True
        )
        if self.use_linear_projection:
            self.proj_in = torch.nn.Linear(self.in_channels, self.inner_dim)
        else:
            self.proj_in = torch.nn.Conv2d(self.in_channels, self.inner_dim, kernel_size=1, stride=1, padding=0)

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    cross_attention_dim=self.config.cross_attention_dim,
                    activation_fn=self.config.activation_fn,
                    num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                    attention_bias=self.config.attention_bias,
                    only_cross_attention=self.config.only_cross_attention,
                    double_self_attention=self.config.double_self_attention,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    attention_type=self.config.attention_type,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        self.tfas = nn.ModuleList([
            TemporalSelfAttention(
                self.inner_dim, num_heads=self.config.num_attention_heads, qkv_bias=True, sampling=None,
                sr_ratio=1,
                qk_norm=False,
            )
            for _ in range((self.config.num_layers + 1) // 2)
            # for _ in range(4)

        ])
        # Zero-out tfas:
        for tfa in self.tfas:
            nn.init.constant_(tfa.attn.proj.weight, 0)
            nn.init.constant_(tfa.attn.proj.bias, 0)

        # # latent refiner
        # self.refine_ratio = nn.Parameter(torch.tensor(0.5))
        # self.refiner = ResidualBlock(in_channels=8, out_channels=4)
        # nn.init.constant_(self.refiner.conv1.weight, 0)
        # nn.init.constant_(self.refiner.conv1.bias, 0)
        # nn.init.constant_(self.refiner.conv2.weight, 0)
        # nn.init.constant_(self.refiner.conv2.bias, 0)


        if self.use_linear_projection:
            self.proj_out = torch.nn.Linear(self.inner_dim, self.out_channels)
        else:
            self.proj_out = torch.nn.Conv2d(self.inner_dim, self.out_channels, kernel_size=1, stride=1, padding=0)

    def _init_vectorized_inputs(self, norm_type):
        assert self.config.sample_size is not None, "Transformer2DModel over discrete input must provide sample_size"
        assert (
            self.config.num_vector_embeds is not None
        ), "Transformer2DModel over discrete input must provide num_embed"

        self.height = self.config.sample_size
        self.width = self.config.sample_size
        self.num_latent_pixels = self.height * self.width

        self.latent_image_embedding = ImagePositionalEmbeddings(
            num_embed=self.config.num_vector_embeds, embed_dim=self.inner_dim, height=self.height, width=self.width
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    cross_attention_dim=self.config.cross_attention_dim,
                    activation_fn=self.config.activation_fn,
                    num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                    attention_bias=self.config.attention_bias,
                    only_cross_attention=self.config.only_cross_attention,
                    double_self_attention=self.config.double_self_attention,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    attention_type=self.config.attention_type,
                )
                for _ in range(self.config.num_layers)
            ]
        )
        self.tfas = nn.ModuleList([
            TemporalSelfAttention(
                self.inner_dim, num_heads=self.config.num_attention_heads, qkv_bias=True, sampling=None,
                sr_ratio=1,
                qk_norm=False,
            )
            for _ in range((self.config.num_layers + 1) // 2)
            # for _ in range(4)

        ])
        # Zero-out tfas:
        for tfa in self.tfas:
            nn.init.constant_(tfa.attn.proj.weight, 0)
            nn.init.constant_(tfa.attn.proj.bias, 0)

        # # latent refiner
        # self.refine_ratio = nn.Parameter(torch.tensor(0.5))
        # self.refiner = ResidualBlock(in_channels=8, out_channels=4)
        # nn.init.constant_(self.refiner.conv1.weight, 0)
        # nn.init.constant_(self.refiner.conv1.bias, 0)
        # nn.init.constant_(self.refiner.conv2.weight, 0)
        # nn.init.constant_(self.refiner.conv2.bias, 0)

        self.norm_out = nn.LayerNorm(self.inner_dim)
        self.out = nn.Linear(self.inner_dim, self.config.num_vector_embeds - 1)

    def _init_patched_inputs(self, norm_type):
        assert self.config.sample_size is not None, "Transformer2DModel over patched input must provide sample_size"

        self.height = self.config.sample_size
        self.width = self.config.sample_size

        self.patch_size = self.config.patch_size
        interpolation_scale = (
            self.config.interpolation_scale
            if self.config.interpolation_scale is not None
            else max(self.config.sample_size // 64, 1)
        )
        self.pos_embed = PatchEmbed_diffusers(
            height=self.config.sample_size,
            width=self.config.sample_size,
            patch_size=self.config.patch_size,
            in_channels=self.in_channels,
            embed_dim=self.inner_dim,
            interpolation_scale=interpolation_scale,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    self.inner_dim,
                    self.config.num_attention_heads,
                    self.config.attention_head_dim,
                    dropout=self.config.dropout,
                    cross_attention_dim=self.config.cross_attention_dim,
                    activation_fn=self.config.activation_fn,
                    num_embeds_ada_norm=self.config.num_embeds_ada_norm,
                    attention_bias=self.config.attention_bias,
                    only_cross_attention=self.config.only_cross_attention,
                    double_self_attention=self.config.double_self_attention,
                    upcast_attention=self.config.upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=self.config.norm_elementwise_affine,
                    norm_eps=self.config.norm_eps,
                    attention_type=self.config.attention_type,
                )
                for _ in range(self.config.num_layers)

            ]
        )
        self.tfas = nn.ModuleList([
            TemporalSelfAttention(
                self.inner_dim, num_heads=self.config.num_attention_heads, qkv_bias=True, sampling=None,
                sr_ratio=1,
                qk_norm=False,
            )
            for _ in range((self.config.num_layers + 1) // 2)
            # for _ in range(4)

        ])
        # Zero-out tfas:
        for tfa in self.tfas:
            nn.init.constant_(tfa.attn.proj.weight, 0)
            nn.init.constant_(tfa.attn.proj.bias, 0)
        
        # # latent refiner
        # self.refine_ratio = nn.Parameter(torch.tensor(0.5))
        # self.refiner = ResidualBlock(in_channels=8, out_channels=4)
        # nn.init.constant_(self.refiner.conv1.weight, 0)
        # nn.init.constant_(self.refiner.conv1.bias, 0)
        # nn.init.constant_(self.refiner.conv2.weight, 0)
        # nn.init.constant_(self.refiner.conv2.bias, 0)


        if self.config.norm_type != "ada_norm_single":
            self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
            self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
            self.proj_out_2 = nn.Linear(
                self.inner_dim, self.config.patch_size * self.config.patch_size * self.out_channels
            )
        elif self.config.norm_type == "ada_norm_single":
            self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
            self.scale_shift_table = nn.Parameter(torch.randn(2, self.inner_dim) / self.inner_dim**0.5)
            self.proj_out = nn.Linear(
                self.inner_dim, self.config.patch_size * self.config.patch_size * self.out_channels
            )

        # PixArt-Alpha blocks.
        self.adaln_single = None
        self.use_additional_conditions = False
        if self.config.norm_type == "ada_norm_single":
            self.use_additional_conditions = self.config.sample_size == 128
            # TODO(Sayak, PVP) clean this, for now we use sample size to determine whether to use
            # additional conditions until we find better name
            self.adaln_single = AdaLayerNormSingle(
                self.inner_dim, use_additional_conditions=self.use_additional_conditions
            )

        self.caption_projection = None
        if self.caption_channels is not None:
            self.caption_projection = PixArtAlphaTextProjection(
                in_features=self.caption_channels, hidden_size=self.inner_dim
            )


    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        The [`Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """

        input_hidden_states = hidden_states.detach().clone()



        if cross_attention_kwargs is not None:
            if cross_attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` to `cross_attention_kwargs` is deprecated. `scale` will be ignored.")
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        if self.is_input_continuous:
            batch_size, _, height, width = hidden_states.shape
            residual = hidden_states
            hidden_states, inner_dim = self._operate_on_continuous_inputs(hidden_states)
        elif self.is_input_vectorized:
            hidden_states = self.latent_image_embedding(hidden_states)
        elif self.is_input_patches:
            height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
            # print(f'hidden_states.dtype = {hidden_states.dtype}')
            # print(f'encoder_hidden_states.dtype = {encoder_hidden_states.dtype}')
            hidden_states, encoder_hidden_states, timestep, embedded_timestep = self._operate_on_patched_inputs(
                hidden_states, encoder_hidden_states, timestep, added_cond_kwargs
            )

        # 2. Blocks
        for idx, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                    **ckpt_kwargs,
                )
                print(f'timestep = {timestep.shape}')
                if idx >= self.config.num_layers // 2:
                    hidden_states = torhc.utils.checkpoint.checkpoint(
                        create_custom_forward(self.tfas[idx - self.config.num_layers // 2]),
                # if idx >= self.config.num_layers - 4:
                #     hidden_states = torhc.utils.checkpoint.checkpoint(
                #         create_custom_forward(self.tfas[idx - (self.config.num_layers-4)]),
                        hidden_states,
                        timestep
                    )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                )
                if idx >= self.config.num_layers // 2:
                    hidden_states = self.tfas[idx - self.config.num_layers // 2](
                # if idx >= self.config.num_layers -4:
                #     hidden_states = self.tfas[idx - (self.config.num_layers -4)](
                        hidden_states,
                        timestep
                    )






        # 3. Output
        if self.is_input_continuous:
            output = self._get_output_for_continuous_inputs(
                hidden_states=hidden_states,
                residual=residual,
                batch_size=batch_size,
                height=height,
                width=width,
                inner_dim=inner_dim,
            )
        elif self.is_input_vectorized:
            output = self._get_output_for_vectorized_inputs(hidden_states)
        elif self.is_input_patches:
            output = self._get_output_for_patched_inputs(
                hidden_states=hidden_states,
                timestep=timestep,
                class_labels=class_labels,
                embedded_timestep=embedded_timestep,
                height=height,
                width=width,
            )






        if not return_dict:
            return (output,)

        return TransformerSRVideo2DModelOutput(sample=output)