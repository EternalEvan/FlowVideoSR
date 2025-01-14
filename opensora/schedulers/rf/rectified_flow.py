import torch
from torch.distributions import LogisticNormal
import tqdm
from ..iddpm.gaussian_diffusion import _extract_into_tensor, mean_flat

# some code are inspired by https://github.com/magic-research/piecewise-rectified-flow/blob/main/scripts/train_perflow.py
# and https://github.com/magic-research/piecewise-rectified-flow/blob/main/src/scheduler_perflow.py


def timestep_transform(
    t,
    model_kwargs,
    base_resolution=512 * 512,
    base_num_frames=1,
    scale=1.0,
    num_timesteps=1,
):
    # Force fp16 input to fp32 to avoid nan output
    for key in ["height", "width", "num_frames"]:
        if model_kwargs[key].dtype == torch.float16:
            model_kwargs[key] = model_kwargs[key].float()

    t = t / num_timesteps
    resolution = model_kwargs["height"] * model_kwargs["width"]
    ratio_space = (resolution / base_resolution).sqrt()
    # NOTE: currently, we do not take fps into account
    # NOTE: temporal_reduction is hardcoded, this should be equal to the temporal reduction factor of the vae
    # if model_kwargs["num_frames"][0].item() == 1:
    #     num_frames = torch.ones_like(model_kwargs["num_frames"])
    # else:
    num_frames = model_kwargs["num_frames"] // 17 * 5
    ratio_time = (num_frames / base_num_frames).sqrt()

    ratio = ratio_space * ratio_time * scale
    new_t = ratio * t / (1 + (ratio - 1) * t)

    new_t = new_t * num_timesteps
    return new_t


class RFlowScheduler:
    def __init__(
        self,
        num_timesteps=1000,
        num_sampling_steps=10,
        use_discrete_timesteps=False,
        sample_method="uniform",
        loc=0.0,
        scale=1.0,
        use_timestep_transform=False,
        transform_scale=1.0,
    ):
        self.num_timesteps = num_timesteps
        self.num_sampling_steps = num_sampling_steps
        self.use_discrete_timesteps = use_discrete_timesteps

        # sample method
        assert sample_method in ["uniform", "logit-normal"]
        assert (
            sample_method == "uniform" or not use_discrete_timesteps
        ), "Only uniform sampling is supported for discrete timesteps"
        self.sample_method = sample_method
        if sample_method == "logit-normal":
            self.distribution = LogisticNormal(torch.tensor([loc]), torch.tensor([scale]))
            self.sample_t = lambda x: self.distribution.sample((x.shape[0],))[:, 0].to(x.device)

        # timestep transform
        self.use_timestep_transform = use_timestep_transform
        self.transform_scale = transform_scale

    def training_losses(self, model, x_start, model_kwargs=None, noise=None, mask=None, weights=None, t=None):
        """
        Compute training losses for a single timestep.
        Arguments format copied from opensora/schedulers/iddpm/gaussian_diffusion.py/training_losses
        Note: t is int tensor and should be rescaled from [0, num_timesteps-1] to [1,0]
        """
        if t is None:
            if self.use_discrete_timesteps:
                t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device)
            elif self.sample_method == "uniform":
                t = torch.rand((x_start.shape[0],), device=x_start.device) * self.num_timesteps
            elif self.sample_method == "logit-normal":
                t = self.sample_t(x_start) * self.num_timesteps

            if self.use_timestep_transform:
                t = timestep_transform(t, model_kwargs, scale=self.transform_scale, num_timesteps=self.num_timesteps)

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        x_t = self.add_noise(x_start, noise, t)
        if mask is not None:
            t0 = torch.zeros_like(t)
            x_t0 = self.add_noise(x_start, noise, t0)
            x_t = torch.where(mask[:, None, :, None, None], x_t, x_t0)

        terms = {}
        model_output = model(x_t, t, **model_kwargs)
        velocity_pred = model_output.chunk(2, dim=1)[0]
        if weights is None:
            loss = mean_flat((velocity_pred - (x_start - noise)).pow(2), mask=mask)
        else:
            weight = _extract_into_tensor(weights, t, x_start.shape)
            loss = mean_flat(weight * (velocity_pred - (x_start - noise)).pow(2), mask=mask)
        terms["loss"] = loss

        return terms

    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        compatible with diffusers add_noise()
        """
        timepoints = timesteps.float() / self.num_timesteps
        timepoints = 1 - timepoints  # [1,1/1000]

        # timepoint  (bsz) noise: (bsz, 4, frame, w ,h)
        # expand timepoint to noise shape
        timepoints = timepoints.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1)
        timepoints = timepoints.repeat(1, noise.shape[1], noise.shape[2], noise.shape[3], noise.shape[4])

        return timepoints * original_samples + (1 - timepoints) * noise


    def training_losses_CogVideoX(self, model, x_start, model_kwargs=None, noise=None, mask=None, weights=None, t=None):
        """
        Compute training losses for a single timestep.
        Arguments format copied from opensora/schedulers/iddpm/gaussian_diffusion.py/training_losses
        Note: t is int tensor and should be rescaled from [0, num_timesteps-1] to [1,0]
        """
        if t is None:
            if self.use_discrete_timesteps:
                t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device)
            elif self.sample_method == "uniform":
                t = torch.rand((x_start.shape[0],), device=x_start.device) * self.num_timesteps
            elif self.sample_method == "logit-normal":
                t = self.sample_t(x_start) * self.num_timesteps

            if self.use_timestep_transform:
                t = timestep_transform(t, model_kwargs, scale=self.transform_scale, num_timesteps=self.num_timesteps)

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        x_t = self.add_noise(model_kwargs["hq"], noise, t)
        if mask is not None:
            t0 = torch.zeros_like(t)
            x_t0 = self.add_noise(model_kwargs["hq"], noise, t0)
            x_t = torch.where(mask[:, None, :, None, None], x_t, x_t0)

        terms = {}
        # print(f'x_t.shape = {x_t.shape}')
        # print(f'encoder_states.shape = {model_kwargs["encoder_hidden_states"].shape}')
        # print(f'c.shape = {model_kwargs["c"].shape}')
        x_concate = torch.cat([x_t, model_kwargs["c"]], dim=1).permute(0,2,1,3,4).contiguous()
        # print(f'x_concate.shape = {x_concate.shape}')
        model_output = model(x_concate, model_kwargs['encoder_hidden_states'], timestep=t, return_dict=False)
        # print(f'model_output[0].shape = {model_output[0].shape}')
        velocity_pred = model_output[0] #.chunk(2, dim=1)[0]
        velocity_pred = velocity_pred.permute(0, 2, 1, 3, 4)
        if weights is None:
            # print(f'velocity_pred.shape = {velocity_pred.shape}')
            # print(f'noise.shape = {noise.shape}')
            loss = mean_flat((velocity_pred - (model_kwargs["hq"] - noise)).pow(2), mask=mask)

            # v_pred_1, v_pred_2, v_pred_3 = velocity_pred.chunk(3, dim=2)
            # hq_1, hq_2, hq_3 = model_kwargs["hq"].chunk(3, dim=2)
            # n_1, n_2, n_3 = noise.chunk(3, dim=2)
            # loss_1 = mean_flat((v_pred_1 - (hq_1 - n_1)).pow(2), mask=mask)
            # loss_2 = mean_flat((v_pred_2 - (hq_2 - n_2)).pow(2), mask=mask)
            # loss_3 = mean_flat((v_pred_3 - (hq_3 - n_3)).pow(2), mask=mask)

        else:
            weight = _extract_into_tensor(weights, t, x_start.shape)
            loss = mean_flat(weight * (velocity_pred - (model_kwargs["hq"] - noise)).pow(2), mask=mask)
        terms["loss"] = loss
        # terms["loss_1"] = loss_1
        # terms["loss_2"] = loss_2 
        # terms["loss_3"] = loss_3
        

        return terms


   

    def training_losses_CogVideoX_ControlNet(self, model, x_start, model_kwargs=None, noise=None, mask=None, weights=None, t=None):
        """
        Compute training losses for a single timestep.
        Arguments format copied from opensora/schedulers/iddpm/gaussian_diffusion.py/training_losses
        Note: t is int tensor and should be rescaled from [0, num_timesteps-1] to [1,0]
        """
        if t is None:
            if self.use_discrete_timesteps:
                t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device)
            elif self.sample_method == "uniform":
                t = torch.rand((x_start.shape[0],), device=x_start.device) * self.num_timesteps
            elif self.sample_method == "logit-normal":
                t = self.sample_t(x_start) * self.num_timesteps

            if self.use_timestep_transform:
                t = timestep_transform(t, model_kwargs, scale=self.transform_scale, num_timesteps=self.num_timesteps)

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        x_t = self.add_noise(model_kwargs["hq"], noise, t)

        if mask is not None:
            t0 = torch.zeros_like(t)
            x_t0 = self.add_noise(model_kwargs["hq"], noise, t0)
            x_t = torch.where(mask[:, None, :, None, None], x_t, x_t0)

        terms = {}

        x_t = x_t.permute(0,2,1,3,4).contiguous()

        model_output = model(x_t, t, **model_kwargs)
        velocity_pred = model_output[0].permute(0, 2, 1, 3, 4)

        if weights is None:
            loss = mean_flat((velocity_pred - (model_kwargs["hq"] - noise)).pow(2), mask=mask)
        else:
            weight = _extract_into_tensor(weights, t, x_start.shape)
            loss = mean_flat(weight * (velocity_pred - (model_kwargs["hq"] - noise)).pow(2), mask=mask)
        terms["loss"] = loss

        return terms


    def training_losses_PFM_ControlNet(self, pipe, x_start, model_kwargs=None, noise=None, weights=None, t=None):
        # if t is None:
        #     if self.use_discrete_timesteps:
        #         t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device)
        #     elif self.sample_method == "uniform":
        #         t = torch.rand((x_start.shape[0],), device=x_start.device) * self.num_timesteps
        #     elif self.sample_method == "logit-normal":
        #         t = self.sample_t(x_start) * self.num_timesteps

        #     if self.use_timestep_transform:
        #         t = timestep_transform(t, model_kwargs, scale=self.transform_scale, num_timesteps=self.num_timesteps)

        if model_kwargs is None:
            model_kwargs = {}
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape

        x_t = self.add_noise(model_kwargs["hq"], noise, t)
        x_t = x_t.permute(0,2,1,3,4).contiguous() #  b c t h w 
        
        generated_latents_list = []    # The generated results
        last_generated_latents = None
        for unit_index in tqdm(range(num_units)):
            gc.collect()
            torch.cuda.empty_cache()
            
            # if callback:
            #     callback(unit_index, num_units)
            
            # if use_linear_guidance:
            #     self._guidance_scale = guidance_scale_list[unit_index]
            #     self._video_guidance_scale = guidance_scale_list[unit_index]

            if unit_index == 0:
                past_condition_latents = [[] for _ in range(len(stages))]
                intermed_latents = pipe.generate_one_unit(
                    x_t[:,:,:1],
                    past_condition_latents,
                    prompt_embeds,
                    prompt_attention_mask,
                    pooled_prompt_embeds,
                    num_inference_steps,
                    height,
                    width,
                    1,
                    device,
                    dtype,
                    generator,
                    is_first_frame=True,
                )
            else:
                # prepare the condition latents
                past_condition_latents = []
                clean_latents_list = pipe.get_pyramid_latent(torch.cat(generated_latents_list, dim=2), len(stages) - 1)
                
                for i_s in range(len(stages)):
                    last_cond_latent = clean_latents_list[i_s][:,:,-(pipe.frame_per_unit):]

                    stage_input = [torch.cat([last_cond_latent] * 2) if pipe.do_classifier_free_guidance else last_cond_latent]
            
                    # pad the past clean latents
                    cur_unit_num = unit_index
                    cur_stage = i_s
                    cur_unit_ptx = 1

                    while cur_unit_ptx < cur_unit_num:
                        cur_stage = max(cur_stage - 1, 0)
                        if cur_stage == 0:
                            break
                        cur_unit_ptx += 1
                        cond_latents = clean_latents_list[cur_stage][:, :, -(cur_unit_ptx * pipe.frame_per_unit) : -((cur_unit_ptx - 1) * self.frame_per_unit)]
                        stage_input.append(torch.cat([cond_latents] * 2) if pipe.do_classifier_free_guidance else cond_latents)

                    if cur_stage == 0 and cur_unit_ptx < cur_unit_num:
                        cond_latents = clean_latents_list[0][:, :, :-(cur_unit_ptx * pipe.frame_per_unit)]
                        stage_input.append(torch.cat([cond_latents] * 2) if pipe.do_classifier_free_guidance else cond_latents)
                
                    stage_input = list(reversed(stage_input))
                    past_condition_latents.append(stage_input)

                intermed_latents = pipe.generate_one_unit(
                    x_t[:,:, 1 + (unit_index - 1) * pipe.frame_per_unit:1 + unit_index * pipe.frame_per_unit],
                    past_condition_latents,
                    prompt_embeds,
                    prompt_attention_mask,
                    pooled_prompt_embeds,
                    video_num_inference_steps,
                    height,
                    width,
                    pipe.frame_per_unit,
                    device,
                    dtype,
                    generator,
                    is_first_frame=False,
                )

            generated_latents_list.append(intermed_latents[-1])
            last_generated_latents = intermed_latents

        generated_latents = torch.cat(generated_latents_list, dim=2)
        
        model_output = generated_latents
        velocity_pred = model_output[0].permute(0, 2, 1, 3, 4)

        if weights is None:
            loss = mean_flat((velocity_pred - (model_kwargs["hq"] - noise)).pow(2), mask=mask)
        else:
            weight = _extract_into_tensor(weights, t, x_start.shape)
            loss = mean_flat(weight * (velocity_pred - (model_kwargs["hq"] - noise)).pow(2), mask=mask)
        terms = {}
        terms["loss"] = loss

        return terms