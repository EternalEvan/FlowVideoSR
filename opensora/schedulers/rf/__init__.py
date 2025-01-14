from os import times
import torch
from tqdm import tqdm
import pdb
from opensora.registry import SCHEDULERS
from diffusers.utils.torch_utils import randn_tensor
from .rectified_flow import RFlowScheduler, timestep_transform


@SCHEDULERS.register_module("rflow")
class RFLOW:
    def __init__(
        self,
        num_sampling_steps=10,
        num_timesteps=1000,
        cfg_scale=4.0,
        use_discrete_timesteps=False,
        use_timestep_transform=False,
        **kwargs,
    ):
        self.num_sampling_steps = num_sampling_steps
        self.num_timesteps = num_timesteps
        self.cfg_scale = cfg_scale
        self.use_discrete_timesteps = use_discrete_timesteps
        self.use_timestep_transform = use_timestep_transform

        self.scheduler = RFlowScheduler(
            num_timesteps=num_timesteps,
            num_sampling_steps=num_sampling_steps,
            use_discrete_timesteps=use_discrete_timesteps,
            use_timestep_transform=use_timestep_transform,
            **kwargs,
        )

    def sample(
        self,
        model,
        text_encoder,
        z,
        prompts,
        device,
        additional_args=None,
        mask=None,
        guidance_scale=None,
        progress=True,
    ):
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        n = len(prompts)
        # text encoding
        model_args = text_encoder.encode(prompts)
        y_null = text_encoder.null(n)
        model_args["y"] = torch.cat([model_args["y"], y_null], 0)
        if additional_args is not None:
            model_args.update(additional_args)

        # prepare timesteps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        
        pdb.set_trace()
        
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [timestep_transform(t, additional_args, num_timesteps=self.num_timesteps) for t in timesteps]

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)

        progress_wrap = tqdm if progress else (lambda x: x)
        print(timesteps)
        print(mask)
        for i, t in progress_wrap(enumerate(timesteps)):
            # mask for adding noise
            if mask is not None:
                mask_t = mask * self.num_timesteps
                x0 = z.clone()
                x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), t)

                mask_t_upper = mask_t >= t.unsqueeze(1)
                model_args["x_mask"] = mask_t_upper.repeat(2, 1)
                mask_add_noise = mask_t_upper & ~noise_added

                z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                noise_added = mask_t_upper

            # classifier-free guidance
            z_in = torch.cat([z, z], 0)
            t = torch.cat([t, t], 0)
            pred = model(z_in, t, **model_args).chunk(2, dim=1)[0]
            pred_cond, pred_uncond = pred.chunk(2, dim=0)
            v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            # update z
            dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
            
            dt = dt / self.num_timesteps
            z = z + v_pred * dt[:, None, None, None, None]

            if mask is not None:
                z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)

        return z
    
    def sample_load_v(
        self,
        model,
        text_encoder,
        z,
        prompts,
        device,
        additional_args=None,
        mask=None,
        guidance_scale=None,
        progress=True,
        v_all=None,
        ddim_scale=0,
        end_time=0
    ):
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        n = len(prompts)
        # text encoding
        model_args = text_encoder.encode(prompts)
        y_null = text_encoder.null(n)
        model_args["y"] = torch.cat([model_args["y"], y_null], 0)
        if additional_args is not None:
            model_args.update(additional_args)

        # prepare timesteps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        
        
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [timestep_transform(t, additional_args, num_timesteps=self.num_timesteps) for t in timesteps]

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)

        progress_wrap = tqdm if progress else (lambda x: x)
        
        for i, t in progress_wrap(enumerate(timesteps)):
            # mask for adding noise
            if mask is not None:
                mask_t = mask * self.num_timesteps
                x0 = z.clone()
                x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), t)

                mask_t_upper = mask_t >= t.unsqueeze(1)
                model_args["x_mask"] = mask_t_upper.repeat(2, 1)
                mask_add_noise = mask_t_upper & ~noise_added

                z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                noise_added = mask_t_upper

            # classifier-free guidance
            z_in = torch.cat([z, z], 0)
            t = torch.cat([t, t], 0)
            pred = model(z_in, t, **model_args).chunk(2, dim=1)[0]
            pred_cond, pred_uncond = pred.chunk(2, dim=0)
            v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            # update z
            dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
            
            dt = dt / self.num_timesteps
            
            #my code
            if i<end_time:
                v_cond = v_all[len(timesteps)-i-1]
                z = z + v_pred * dt[:, None, None, None, None] * (1-ddim_scale) + v_cond * dt[:, None, None, None, None] * ddim_scale
            else:
                z = z + v_pred * dt[:, None, None, None, None]

            if mask is not None:
                z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)

        return z
    
    def inv_sample(
        self,
        model,
        text_encoder,
        z,
        prompts,
        device,
        additional_args=None,
        mask=None,
        guidance_scale=None,
        progress=True,
    ):
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = 1.0

        n = len(prompts)
        # text encoding
        model_args = text_encoder.encode(prompts)
        y_null = text_encoder.null(n)
        model_args["y"] = torch.cat([model_args["y"], y_null], 0)
        if additional_args is not None:
            model_args.update(additional_args)
       
        # prepare timesteps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        # pdb.set_trace()
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [timestep_transform(t, additional_args, num_timesteps=self.num_timesteps) for t in timesteps]

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)

        progress_wrap = tqdm if progress else (lambda x: x)
        # pdb.set_trace()
        timesteps.append(torch.tensor([0]).to(timesteps[0]))
        timesteps = timesteps[::-1]
        print(timesteps)
        z_all = []
        z_all.append(z)
        for i in progress_wrap(range(self.num_sampling_steps)):
            t = timesteps[i]
            # # mask for adding noise
            # if mask is not None:
            #     mask_t = mask * self.num_timesteps
            #     x0 = z.clone()
            #     x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), t)

            #     mask_t_upper = mask_t >= t.unsqueeze(1)
            #     model_args["x_mask"] = mask_t_upper.repeat(2, 1)
            #     mask_add_noise = mask_t_upper & ~noise_added

            #     z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
            #     noise_added = mask_t_upper

            # classifier-free guidance
            z_in = torch.cat([z, z], 0)
            t = torch.cat([t, t], 0)
            pred = model(z_in, t, **model_args).chunk(2, dim=1)[0]
            pred_cond, pred_uncond = pred.chunk(2, dim=0)
            v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            # update z
            dt = timesteps[i + 1] - timesteps[i] if i < len(timesteps) - 1 else timesteps[i]
            print(timesteps[i])
            print(timesteps[i+1])
            
            dt = dt / self.num_timesteps
            print(torch.var(z))
            z = z - v_pred * dt[:, None, None, None, None]
            
            
            z_all.append(z)


        return z_all
    
    def inv_sample_store_v(
        self,
        model,
        text_encoder,
        z,
        prompts,
        device,
        additional_args=None,
        mask=None,
        guidance_scale=None,
        progress=True,
    ):
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = 1.0
        
        v_all = []
        n = len(prompts)
        # text encoding
        model_args = text_encoder.encode(prompts)
        y_null = text_encoder.null(n)
        model_args["y"] = torch.cat([model_args["y"], y_null], 0)
        if additional_args is not None:
            model_args.update(additional_args)
       
        # prepare timesteps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        # pdb.set_trace()
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [timestep_transform(t, additional_args, num_timesteps=self.num_timesteps) for t in timesteps]

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)

        progress_wrap = tqdm if progress else (lambda x: x)
        # pdb.set_trace()
        timesteps.append(torch.tensor([0]).to(timesteps[0]))
        timesteps = timesteps[::-1]
        z_all = []
        z_all.append(z)
        for i in progress_wrap(range(self.num_sampling_steps)):
            t = timesteps[i]
            # # mask for adding noise
            # if mask is not None:
            #     mask_t = mask * self.num_timesteps
            #     x0 = z.clone()
            #     x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), t)

            #     mask_t_upper = mask_t >= t.unsqueeze(1)
            #     model_args["x_mask"] = mask_t_upper.repeat(2, 1)
            #     mask_add_noise = mask_t_upper & ~noise_added

            #     z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
            #     noise_added = mask_t_upper

            # classifier-free guidance
            z_in = torch.cat([z, z], 0)
            t = torch.cat([t, t], 0)
            pred = model(z_in, t, **model_args).chunk(2, dim=1)[0]
            pred_cond, pred_uncond = pred.chunk(2, dim=0)
            v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            # update z
            dt = timesteps[i + 1] - timesteps[i] if i < len(timesteps) - 1 else timesteps[i]            
            dt = dt / self.num_timesteps
            z = z - v_pred * dt[:, None, None, None, None]
            
            z_all.append(z)
            v_all.append(v_pred)
        
        return z_all, v_all


    def inv_sample_with_flow_guide(
        self,
        model,
        text_encoder,
        z, 
        prompts,
        device,
        additional_args=None,
        mask=None,
        guidance_scale=None,
        progress=True,
    ):
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = 1.0

        n = len(prompts)
        # text encoding
        model_args = text_encoder.encode(prompts)
        y_null = text_encoder.null(n)
        model_args["y"] = torch.cat([model_args["y"], y_null], 0)
        if additional_args is not None:
            model_args.update(additional_args)
       
        # prepare timesteps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        # pdb.set_trace()
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [timestep_transform(t, additional_args, num_timesteps=self.num_timesteps) for t in timesteps]

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)

        progress_wrap = tqdm if progress else (lambda x: x)
        # pdb.set_trace()
        timesteps.append(torch.tensor([0]).to(timesteps[0]))
        timesteps = timesteps[::-1]
        z_set = randn_tensor(z.shape)
        z_set = z_set.to(z)
        z_all = []
        z_all.append(z)
        for i in progress_wrap(range(self.num_sampling_steps)):
            t = timesteps[i]
            # # mask for adding noise
            # if mask is not None:
            #     mask_t = mask * self.num_timesteps
            #     x0 = z.clone()
            #     x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), t)

            #     mask_t_upper = mask_t >= t.unsqueeze(1)
            #     model_args["x_mask"] = mask_t_upper.repeat(2, 1)
            #     mask_add_noise = mask_t_upper & ~noise_added

            #     z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
            #     noise_added = mask_t_upper

            # classifier-free guidance
            z_in = torch.cat([z, z], 0)
            t = torch.cat([t, t], 0)
            
            v_cond = (z_set-z)/(1 - t[0]/self.num_timesteps)
                
      
                
            pred = model(z_in, t, **model_args).chunk(2, dim=1)[0]
            pred_cond, pred_uncond = pred.chunk(2, dim=0)
            v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            v_pred = pred_uncond
            # update z
            dt = timesteps[i + 1] - timesteps[i] 
 
            
            dt = dt / self.num_timesteps
            
            z = z - v_pred * dt[:, None, None, None, None]*0.5 + v_cond * dt[:, None, None, None, None]*0.5
            print(torch.var(z))
            
            z_all.append(z)


        return z_all
    

    def sample_with_flow_guide(
        self,
        model,
        text_encoder,
        z,z_src,
        prompts,
        device,
        additional_args=None,
        mask=None,
        guidance_scale=None,
        progress=True,
    ):
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale

        n = len(prompts)
        # text encoding
        model_args = text_encoder.encode(prompts)
        y_null = text_encoder.null(n)
        model_args["y"] = torch.cat([model_args["y"], y_null], 0)
        if additional_args is not None:
            model_args.update(additional_args)

        # prepare timesteps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        

        
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [timestep_transform(t, additional_args, num_timesteps=self.num_timesteps) for t in timesteps]

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)

        progress_wrap = tqdm if progress else (lambda x: x)
        print(timesteps)
  
        for i, t in progress_wrap(enumerate(timesteps)):
            # mask for adding noise
            if mask is not None:
                mask_t = mask * self.num_timesteps
                x0 = z.clone()
                x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), t)

                mask_t_upper = mask_t >= t.unsqueeze(1)
                model_args["x_mask"] = mask_t_upper.repeat(2, 1)
                mask_add_noise = mask_t_upper & ~noise_added

                z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
                noise_added = mask_t_upper

            
            # pdb.set_trace()
            # classifier-free guidance
            z_in = torch.cat([z, z], 0)
            t = torch.cat([t, t], 0)
            pred = model(z_in, t, **model_args).chunk(2, dim=1)[0]
            pred_cond, pred_uncond = pred.chunk(2, dim=0)
            v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            # update z
            dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
            
            dt = dt / self.num_timesteps
            
            #if i >=0 and i<30:
            if i >=0 and i<50:  
                v_cond = (z_src-z)/(t[0]/self.num_timesteps)
                
                z = z + v_pred * dt[:, None, None, None, None]*0.5+ v_cond * dt[:, None, None, None, None]*0.5
            else: 
                
                z = z + v_pred * dt[:, None, None, None, None]
            if mask is not None:
                z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)

        return z
    
      
    def sample_CogVideoX(
        self,
        model,
        text_encoder,
        z,
        prompts,
        device,
        model_kwargs=None,
        additional_args=None,
        mask=None,
        guidance_scale=None,
        progress=True,
    ):
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale



        # prepare timesteps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [timestep_transform(t, additional_args, num_timesteps=self.num_timesteps) for t in timesteps]

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)

        noise, lq = z.chunk(2, dim=1)


        progress_wrap = tqdm if progress else (lambda x: x)
        for i, t in progress_wrap(enumerate(timesteps)):
            # mask for adding noise
            # if mask is not None:
            #     mask_t = mask * self.num_timesteps
            #     x0 = z.clone()
            #     x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), t)

            #     mask_t_upper = mask_t >= t.unsqueeze(1)
            #     model_args["x_mask"] = mask_t_upper.repeat(2, 1)
            #     mask_add_noise = mask_t_upper & ~noise_added

            #     z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
            #     noise_added = mask_t_upper

            # classifier-free guidance
            z_in = torch.cat([noise, lq], dim=1).permute(0, 2, 1, 3, 4).contiguous().to(torch.bfloat16)
            t = t #torch.cat([t, t], 0)
            # pred = model(z_in, t, **model_args).chunk(2, dim=1)[0]
            # pred_cond, pred_uncond = pred.chunk(2, dim=0)
            # v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            v_pred = model(z_in, model_kwargs['encoder_hidden_states'], timestep=t, return_dict=False)[0]
            v_pred = v_pred.permute(0, 2, 1, 3, 4)

            # update z
            dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
            dt = dt / self.num_timesteps
            # z = z + v_pred * dt[:, None, None, None, None]
            noise = noise + v_pred * dt[:, None, None, None, None]

            # if mask is not None:
            #     z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)

        return noise

    def sample_CogVideoX_ControlNet(
        self,
        model,
        text_encoder,
        z,
        prompts,
        device,
        model_kwargs=None,
        additional_args=None,
        mask=None,
        guidance_scale=None,
        progress=True,
    ):
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale


        # prepare timesteps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [timestep_transform(t, additional_args, num_timesteps=self.num_timesteps) for t in timesteps]

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)

        progress_wrap = tqdm if progress else (lambda x: x)
        for i, t in progress_wrap(enumerate(timesteps)):
            # mask for adding noise
            # if mask is not None:
            #     mask_t = mask * self.num_timesteps
            #     x0 = z.clone()
            #     x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), t)

            #     mask_t_upper = mask_t >= t.unsqueeze(1)
            #     model_args["x_mask"] = mask_t_upper.repeat(2, 1)
            #     mask_add_noise = mask_t_upper & ~noise_added

            #     z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
            #     noise_added = mask_t_upper

            # classifier-free guidance
            # z_in = torch.cat([z, z], 0)
            z_in = z.permute(0, 2, 1, 3, 4).contiguous().to(torch.bfloat16)
            # t = torch.cat([t, t], 0)
            t = t
            # pred = model(z_in, t, **model_args).chunk(2, dim=1)[0]
            v_pred = model(z_in, t, **model_kwargs)[0].permute(0, 2, 1, 3, 4)
            # pred_cond, pred_uncond = pred.chunk(2, dim=0)
            # v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            # update z
            dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
            dt = dt / self.num_timesteps
            z = z + v_pred * dt[:, None, None, None, None]

            # if mask is not None:
            #     z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)

        return z
    
    def sample_CogVideoX_ControlNet_mid_value(
        self,
        model,
        text_encoder,
        z,
        prompts,
        device,
        model_kwargs=None,
        additional_args=None,
        mask=None,
        guidance_scale=None,
        progress=True,
        midpoint_step = 3,
    ):
        # if no specific guidance scale is provided, use the default scale when initializing the scheduler
        if guidance_scale is None:
            guidance_scale = self.cfg_scale


        # prepare timesteps
        timesteps = [(1.0 - i / self.num_sampling_steps) * self.num_timesteps for i in range(self.num_sampling_steps)]
        if self.use_discrete_timesteps:
            timesteps = [int(round(t)) for t in timesteps]
        timesteps = [torch.tensor([t] * z.shape[0], device=device) for t in timesteps]
        if self.use_timestep_transform:
            timesteps = [timestep_transform(t, additional_args, num_timesteps=self.num_timesteps) for t in timesteps]

        if mask is not None:
            noise_added = torch.zeros_like(mask, dtype=torch.bool)
            noise_added = noise_added | (mask == 1)

        progress_wrap = tqdm if progress else (lambda x: x)
        for i, t in progress_wrap(enumerate(timesteps)):
            # mask for adding noise
            # if mask is not None:
            #     mask_t = mask * self.num_timesteps
            #     x0 = z.clone()
            #     x_noise = self.scheduler.add_noise(x0, torch.randn_like(x0), t)

            #     mask_t_upper = mask_t >= t.unsqueeze(1)
            #     model_args["x_mask"] = mask_t_upper.repeat(2, 1)
            #     mask_add_noise = mask_t_upper & ~noise_added

            #     z = torch.where(mask_add_noise[:, None, :, None, None], x_noise, x0)
            #     noise_added = mask_t_upper

            # classifier-free guidance
            # z_in = torch.cat([z, z], 0)
            z_in = z.permute(0, 2, 1, 3, 4).contiguous().to(torch.bfloat16)
            # t = torch.cat([t, t], 0)
            t = t
            # pred = model(z_in, t, **model_args).chunk(2, dim=1)[0]
            v_pred = model(z_in, t, **model_kwargs)[0].permute(0, 2, 1, 3, 4)
            # pred_cond, pred_uncond = pred.chunk(2, dim=0)
            # v_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            print(timesteps)
     
            # update z
            if i == midpoint_step:
                dt = timesteps[i] - timesteps[-1]
                dt = dt / self.num_timesteps
                z = z + v_pred * dt[:, None, None, None, None]
            
                return z
            else:
            
                dt = timesteps[i] - timesteps[i + 1] if i < len(timesteps) - 1 else timesteps[i]
                dt = dt / self.num_timesteps
                z = z + v_pred * dt[:, None, None, None, None]

            # if mask is not None:
            #     z = torch.where(mask_t_upper[:, None, :, None, None], z, x0)

        return z


    def training_losses(self, model, x_start, model_kwargs=None, noise=None, mask=None, weights=None, t=None):
        return self.scheduler.training_losses(model, x_start, model_kwargs, noise, mask, weights, t)

    def training_losses_CogVideoX(self, model, x_start, model_kwargs=None, noise=None, mask=None, weights=None, t=None):
        return self.scheduler.training_losses_CogVideoX(model, x_start, model_kwargs, noise, mask, weights, t)

    def training_losses_CogVideoX_ControlNet(self, model, x_start, model_kwargs=None, noise=None, mask=None, weights=None, t=None):
        return self.scheduler.training_losses_CogVideoX_ControlNet(model, x_start, model_kwargs, noise, mask, weights, t)
    
    def training_losses_PFM_ControlNet(self, model, x_start, model_kwargs=None, noise=None, mask=None, weights=None, t=None):
        return self.scheduler.training_losses_PFM_ControlNet(model, x_start, model_kwargs, noise, mask, weights, t)