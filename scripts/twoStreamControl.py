import torch
import torch as th
import torch.nn as nn

from functools import partial
from typing import Iterable

from sgm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
    checkpoint
)

from einops import rearrange

from ...modules.attention import BasicTransformerBlock
from ...modules.diffusionmodules.openaimodel import (
    UNetModel,
    Timestep,
    TimestepEmbedSequential,
    ResBlock as ResBlock_orig,
    Downsample,
    Upsample,
    AttentionBlock,
    TimestepBlock
)

from ...util import default, exists

from annotator.midas import MidasDetector
from torchvision import transforms as tt
import numpy as np


class PseudoModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return 0


class TwoStreamControlNet(nn.Module):
    def __init__(
            self,
            in_channels=4,
            model_channels=320,
            out_channels=4,
            hint_channels=3,
            num_res_blocks=2,
            attention_resolutions=(4, 2),
            dropout=0,
            channel_mult=(1, 2, 4),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=64,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            adm_in_channels=2816,
            use_spatial_transformer=True,  # custom transformer support
            transformer_depth=(1,2,10),  # custom transformer support
            context_dim=2048,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=False,
            spatial_transformer_attn_type="softmax-xformers",
            use_linear_in_transformer=True,
            num_classes='sequential',
            infusion2control='cat',         # how to infuse intermediate information into the control net? {'add', 'cat', None}
            infusion2base='add',            # how to infuse intermediate information into the base net? {'add', 'cat'}
            guiding='encoder',              # use just encoder for control or the whole encoder + decoder net? {'encoder', 'encoder_double', 'full'}
            two_stream_mode='cross',        # mode for the two stream infusion. {'cross', 'sequential'}
            control_model_ratio=0.05,        # ratio of the control model size compared to the base model. [0, 1]
            base_model=None,
            learn_embedding=False,
           # control_mode='canny',
            prune_until=None,
            fixed=True,
    ):
        assert infusion2control in ('cat', 'add', None), f'infusion2control needs to be cat, add or None, but not {infusion2control}'
        assert infusion2base == 'add', f'infusion2base only defined for add, but not {infusion2base}'
        assert guiding in ('encoder', 'encoder_double', 'full'), f'guiding has to be encoder, encoder_double or full, but not {guiding}'
        assert two_stream_mode in ('cross', 'sequential'), f'two_stream_mode has to be either cross or sequential, but not {two_stream_mode}'

        super().__init__()

      #  self.control_mode = control_mode
        self.learn_embedding = learn_embedding
        self.infusion2control = infusion2control
        self.infusion2base = infusion2base
        self.in_ch_factor = 1 if infusion2control == 'add' else 2
        self.guiding = guiding
        self.two_stream_mode = two_stream_mode
        self.control_model_ratio = control_model_ratio
        self.out_channels = out_channels
        self.dims = 2
        self.model_channels = model_channels
        self.no_control = False
        self.control_scale = 1.0
        self.prune_until = prune_until
        self.fixed = fixed

        # if control_mode == 'midas':
        #     self.hint_model = MidasDetector()
        # else:
        self.hint_model = None

        ################# start control model variations #################
        if base_model is None:
            base_model = UNetModel(
                adm_in_channels=adm_in_channels, num_classes=num_classes, use_checkpoint=use_checkpoint,
                in_channels=in_channels, out_channels=out_channels, model_channels=model_channels,
                attention_resolutions=attention_resolutions, num_res_blocks=num_res_blocks,
                channel_mult=channel_mult, num_head_channels=num_head_channels, use_spatial_transformer=use_spatial_transformer,
                use_linear_in_transformer=use_linear_in_transformer, transformer_depth=transformer_depth,
                context_dim=context_dim, spatial_transformer_attn_type=spatial_transformer_attn_type,
                legacy=legacy, dropout=dropout,
                conv_resample=conv_resample, dims=dims, use_fp16=use_fp16, num_heads=num_heads,
                num_heads_upsample=num_heads_upsample, use_scale_shift_norm=use_scale_shift_norm,
                resblock_updown=resblock_updown, use_new_attention_order=use_new_attention_order,
                n_embed=n_embed,
                # disable_self_attentions=disable_self_attentions,
                # num_attention_blocks=num_attention_blocks,
                # disable_middle_self_attn=disable_middle_self_attn,
            )  # initialise control model from base model
        self.control_model = ControlledXLUNetModelFixed(
            adm_in_channels=adm_in_channels, num_classes=num_classes, use_checkpoint=use_checkpoint,
            in_channels=in_channels, out_channels=out_channels, model_channels=model_channels,
            attention_resolutions=attention_resolutions, num_res_blocks=num_res_blocks,
            channel_mult=channel_mult, num_head_channels=num_head_channels, use_spatial_transformer=use_spatial_transformer,
            use_linear_in_transformer=use_linear_in_transformer, transformer_depth=transformer_depth,
            context_dim=context_dim, spatial_transformer_attn_type=spatial_transformer_attn_type,
            legacy=legacy, dropout=dropout,
            conv_resample=conv_resample, dims=dims, use_fp16=use_fp16, num_heads=num_heads,
            num_heads_upsample=num_heads_upsample, use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown, use_new_attention_order=use_new_attention_order,
            n_embed=n_embed, fixed=fixed,
            # disable_self_attentions=disable_self_attentions,
            # num_attention_blocks=num_attention_blocks,
            # disable_middle_self_attn=disable_middle_self_attn,
            infusion2control=infusion2control,
            guiding=guiding, two_stream_mode=two_stream_mode, control_model_ratio=control_model_ratio,
        )  # initialise pretrained model

        self.diffusion_model = base_model

        ################# end control model variations #################

        self.enc_zero_convs_out = nn.ModuleList([])
        self.enc_zero_convs_in = nn.ModuleList([])

        self.middle_block_out = nn.ModuleList([])
        self.middle_block_in = nn.ModuleList([])

        self.dec_zero_convs_out = nn.ModuleList([])
        self.dec_zero_convs_in = nn.ModuleList([])

        ch_inout_ctr = {'enc': [], 'mid': [], 'dec': []}
        ch_inout_base = {'enc': [], 'mid': [], 'dec': []}

        ################# Gather Channel Sizes #################
        for module in self.control_model.input_blocks:
            if isinstance(module[0], nn.Conv2d):
                ch_inout_ctr['enc'].append((module[0].in_channels, module[0].out_channels))
            elif isinstance(module[0], (ResBlock, ResBlock_orig)):
                ch_inout_ctr['enc'].append((module[0].channels, module[0].out_channels))
            elif isinstance(module[0], Downsample):
                ch_inout_ctr['enc'].append((module[0].channels, module[-1].out_channels))

        for module in base_model.input_blocks:
            if isinstance(module[0], nn.Conv2d):
                ch_inout_base['enc'].append((module[0].in_channels, module[0].out_channels))
            elif isinstance(module[0], (ResBlock, ResBlock_orig)):
                ch_inout_base['enc'].append((module[0].channels, module[0].out_channels))
            elif isinstance(module[0], Downsample):
                ch_inout_base['enc'].append((module[0].channels, module[-1].out_channels))

        ch_inout_ctr['mid'].append((self.control_model.middle_block[0].channels, self.control_model.middle_block[-1].out_channels))
        ch_inout_base['mid'].append((base_model.middle_block[0].channels, base_model.middle_block[-1].out_channels))

        if guiding not in ('encoder', 'encoder_double'):
            for module in self.control_model.output_blocks:
                if isinstance(module[0], nn.Conv2d):
                    ch_inout_ctr['dec'].append((module[0].in_channels, module[0].out_channels))
                elif isinstance(module[0], (ResBlock, ResBlock_orig)):
                    ch_inout_ctr['dec'].append((module[0].channels, module[0].out_channels))
                elif isinstance(module[-1], Upsample):
                    ch_inout_ctr['dec'].append((module[0].channels, module[-1].out_channels))

        for module in base_model.output_blocks:
            if isinstance(module[0], nn.Conv2d):
                ch_inout_base['dec'].append((module[0].in_channels, module[0].out_channels))
            elif isinstance(module[0], (ResBlock, ResBlock_orig)):
                ch_inout_base['dec'].append((module[0].channels, module[0].out_channels))
            elif isinstance(module[-1], Upsample):
                ch_inout_base['dec'].append((module[0].channels, module[-1].out_channels))

        self.ch_inout_ctr = ch_inout_ctr
        self.ch_inout_base = ch_inout_base

        ################# Build zero convolutions #################
        if two_stream_mode == 'cross':
            ################# cross infusion #################
            # infusion2control
            # add
            if infusion2control == 'add':
                for i in range(len(ch_inout_base['enc'])):
                    self.enc_zero_convs_in.append(self.make_zero_conv(
                        in_channels=ch_inout_base['enc'][i][1], out_channels=ch_inout_ctr['enc'][i][1])
                    )

                if guiding == 'full':
                    self.middle_block_in = self.make_zero_conv(ch_inout_base['mid'][-1][1], ch_inout_ctr['mid'][-1][1])
                    for i in range(len(ch_inout_base['dec']) - 1):
                        self.dec_zero_convs_in.append(self.make_zero_conv(
                            in_channels=ch_inout_base['dec'][i][1], out_channels=ch_inout_ctr['dec'][i][1])
                        )

                # cat - processing full concatenation (all output layers are concatenated without "slimming")
            if infusion2control == 'cat':
                for ch_io_base in ch_inout_base['enc']:
                    self.enc_zero_convs_in.append(self.make_zero_conv(
                        in_channels=ch_io_base[1], out_channels=ch_io_base[1])
                    )

                if guiding == 'full':
                    self.middle_block_in = self.make_zero_conv(ch_inout_base['mid'][-1][1], ch_inout_base['mid'][-1][1])
                    for ch_io_base in ch_inout_base['dec']:
                        self.dec_zero_convs_in.append(self.make_zero_conv(
                            in_channels=ch_io_base[1], out_channels=ch_io_base[1])
                        )

                # None - no changes

            # infusion2base - consider all three guidings
                # add
            if infusion2base == 'add':
                self.middle_block_out = self.make_zero_conv(ch_inout_ctr['mid'][-1][1], ch_inout_base['mid'][-1][1])

                if guiding in ('encoder', 'encoder_double'):
                    self.dec_zero_convs_out.append(
                        self.make_zero_conv(ch_inout_ctr['enc'][-1][1], ch_inout_base['mid'][-1][1])
                    )
                    for i in range(1, len(ch_inout_ctr['enc'])):
                        self.dec_zero_convs_out.append(
                            self.make_zero_conv(ch_inout_ctr['enc'][-(i + 1)][1], ch_inout_base['dec'][i - 1][1])
                        )
                if guiding in ('encoder_double', 'full'):
                    for i in range(len(ch_inout_ctr['enc'])):
                        self.enc_zero_convs_out.append(self.make_zero_conv(
                            in_channels=ch_inout_ctr['enc'][i][1], out_channels=ch_inout_base['enc'][i][1])
                        )

                if guiding == 'full':
                    for i in range(len(ch_inout_ctr['dec'])):
                        self.dec_zero_convs_out.append(self.make_zero_conv(
                            in_channels=ch_inout_ctr['dec'][i][1], out_channels=ch_inout_base['dec'][i][1])
                        )

        self.input_hint_block = TimestepEmbedSequential(
            conv_nd(dims, hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 16, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 32, 32, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            conv_nd(dims, 96, 96, 3, padding=1),
            nn.SiLU(),
            conv_nd(dims, 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(conv_nd(dims, 256, int(model_channels * control_model_ratio), 3, padding=1))
        )

        if self.prune_until is not None:
            self.prune_last(prune_until)
        elif not self.learn_embedding:
            if not self.learn_embedding:
                self.control_model.time_embed = PseudoModule()
        self.control_model.label_emb = PseudoModule()

        scale_list = [1.] * len(self.enc_zero_convs_out) + [1.] + [1.] * len(self.dec_zero_convs_out)
        self.register_buffer('scale_list', torch.tensor(scale_list))

    def prune_last(self, end_layer):
        for n in range(len(self.control_model.input_blocks)):
            if n >= end_layer:
                self.control_model.input_blocks[n] = PseudoModule()

        self.control_model.middle_block = PseudoModule()
        if not self.learn_embedding:
            self.control_model.time_embed = PseudoModule()
        self.control_model.label_emb = PseudoModule()

        self.enc_zero_convs_in = nn.ModuleList(self.enc_zero_convs_in[:end_layer] + [PseudoModule()] * len(self.enc_zero_convs_in[end_layer:]))
        self.enc_zero_convs_out = nn.ModuleList(self.enc_zero_convs_out[:end_layer] + [PseudoModule()] * len(self.enc_zero_convs_out[end_layer:]))
        self.middle_block_in = self.middle_block_out = PseudoModule()
        self.dec_zero_convs_in = nn.ModuleList([PseudoModule()] * len(self.dec_zero_convs_in))
        self.dec_zero_convs_out = nn.ModuleList([PseudoModule()] * len(self.dec_zero_convs_out))

    def make_zero_conv(self, in_channels, out_channels=None):
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        return TimestepEmbedSequential(
            zero_module(conv_nd(self.dims, in_channels, out_channels, 1, padding=0))
        )

    def infuse(self, stream, infusion, mlp, variant, emb, scale=1.0):
        if variant == 'add':
            stream = stream + mlp(infusion, emb) * scale
        elif variant == 'cat':
            stream = torch.cat([stream, mlp(infusion, emb) * scale], dim=1)

        return stream

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: dict, hint: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        x = torch.cat((x, c.get("concat", torch.Tensor([]).type_as(x))), dim=1)
        # in case of classifier free guidance:
        if x.size(0) // 2 == hint.size(0):
            hint = torch.cat([hint, hint], dim=0)
        return self.forward_(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            y=c.get("vector", None),
            hint=hint,
            base_model=self.diffusion_model,
            compute_hint=False #if self.control_mode == 'midas' else False,
            **kwargs,
        )

    def forward_(
        self, x, hint, timesteps, context,
        base_model=None, y=None, precomputed_hint=False,
        no_control=False, compute_hint=False, **kwargs
    ):

        if base_model is None:
            base_model = self.diffusion_model

        if no_control or self.no_control:
            return base_model(x=x, timesteps=timesteps, context=context, y=y, **kwargs)

        if compute_hint:
            hints = []
            for inp in hint:
                hint_processed, _ = self.hint_model(np.array(inp.cpu().permute(1, 2, 0)))
                hints.append(tt.ToTensor()(hint_processed[..., None].repeat(3, 2)))
            hint_processed = torch.stack(hints).to(x.device)
            hint = hint_processed.to(memory_format=torch.contiguous_format).float()

        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        if self.learn_embedding:
            emb = self.control_model.time_embed(t_emb) * self.control_scale ** 0.3 + base_model.time_embed(t_emb) * (1 - self.control_scale ** 0.3)
        else:
            emb = base_model.time_embed(t_emb)

        if y is not None:
            assert y.shape[0] == x.shape[0]
            emb = emb + base_model.label_emb(y)

        if precomputed_hint:
            guided_hint = hint
        else:
            guided_hint = self.input_hint_block(hint, emb, context)

        h_ctr = h_base = x
        hs_base = []
        hs_ctr = []
        it_enc_convs_in = iter(self.enc_zero_convs_in)
        it_enc_convs_out = iter(self.enc_zero_convs_out)
        it_dec_convs_in = iter(self.dec_zero_convs_in)
        it_dec_convs_out = iter(self.dec_zero_convs_out)
        scales = iter(self.scale_list)

        ###################### Cross Control        ######################
        if self.two_stream_mode == 'cross':
            #  input blocks (encoder)
            for module_base, module_ctr in zip(base_model.input_blocks, self.control_model.input_blocks):
                h_base = module_base(h_base, emb, context)
                if "PseudoModule" not in str(type(module_ctr)):
                    h_ctr = module_ctr(h_ctr, emb, context)
                if guided_hint is not None:
                    h_ctr = h_ctr + guided_hint
                    guided_hint = None

                if self.guiding in ('encoder_double', 'full'):
                    h_base = self.infuse(h_base, h_ctr, next(it_enc_convs_out), self.infusion2base, emb, scale=next(scales))

                hs_base.append(h_base)
                hs_ctr.append(h_ctr)

                if "PseudoModule" not in str(type(module_ctr)):
                    h_ctr = self.infuse(h_ctr, h_base, next(it_enc_convs_in), self.infusion2control, emb)

            # mid blocks (bottleneck)
            h_base = base_model.middle_block(h_base, emb, context)
            h_ctr = self.control_model.middle_block(h_ctr, emb, context)

            h_base = self.infuse(h_base, h_ctr, self.middle_block_out, self.infusion2base, emb, scale=next(scales))

            if self.guiding == 'full':
                h_ctr = self.infuse(h_ctr, h_base, self.middle_block_in, self.infusion2control, emb)

            # output blocks (decoder)
            for module_base, module_ctr in zip(
                    base_model.output_blocks,
                    self.control_model.output_blocks if hasattr(
                        self.control_model, 'output_blocks') else [None] * len(base_model.output_blocks)
            ):

                if self.guiding != 'full':
                    h_base = self.infuse(h_base, hs_ctr.pop(), next(it_dec_convs_out), self.infusion2base, emb, scale=next(scales))

                h_base = th.cat([h_base, hs_base.pop()], dim=1)
                h_base = module_base(h_base, emb, context)

                ##### Quick and dirty way of fixing "full" with not applying corrections to the last layer #####
                if self.guiding == 'full':
                    h_ctr = th.cat([h_ctr, hs_ctr.pop()], dim=1)
                    h_ctr = module_ctr(h_ctr, emb, context)
                    if module_base != base_model.output_blocks[-1]:
                        h_base = self.infuse(h_base, h_ctr, next(it_dec_convs_out), self.infusion2base, emb, scale=next(scales))
                        h_ctr = self.infuse(h_ctr, h_base, next(it_dec_convs_in), self.infusion2control, emb)

        h_base = h_base.type(x.dtype)
        return base_model.out(h_base)
