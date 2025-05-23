# src/uncond_ts_diff/sampler/observation_guidance.py
import numpy as np
import torch
import torch.nn.functional as F
from gluonts.torch.util import lagged_sequence_values

from typing import Optional, Dict, Tuple, List 

from uncond_ts_diff.predictor import PyTorchPredictorWGrads 
from uncond_ts_diff.utils import extract
from uncond_ts_diff.model import TSDiff 

PREDICTION_INPUT_NAMES = [
    "past_target",
    "past_observed_values",
    "feat_static_cat",        
    "feat_static_real",       
    "past_feat_dynamic_real", 
    "future_feat_dynamic_real",
]

class Guidance(torch.nn.Module):
    _missing_scenarios = ["none", "RM", "BM-B", "BM-E"]

    def __init__(
        self,
        model: TSDiff, 
        prediction_length: int,
        scale: float = 1.0, 
        num_samples: int = 1,
        guidance: str = "quantile", 
        missing_scenario: str = "none",
        missing_values: int = 0,
    ):
        super().__init__()
        assert missing_scenario in self._missing_scenarios
        self.model = model 
        self.prediction_length = prediction_length
        self.context_length = model.context_length 
        self.scale = scale
        self.num_samples = num_samples
        self.guidance = guidance
        self.missing_scenario = missing_scenario
        self.missing_values = missing_values

    def quantile_loss(self, y_prediction: torch.Tensor, y_target: torch.Tensor) -> torch.Tensor:
        assert y_target.shape == y_prediction.shape, \
            f"Target shape {y_target.shape} must match prediction shape {y_prediction.shape}"
        device = y_prediction.device
        
        if y_prediction.shape[-1] > 1:
            y_prediction_target_channel = y_prediction[..., 0].unsqueeze(-1)
            y_target_target_channel = y_target[..., 0].unsqueeze(-1)
        else:
            y_prediction_target_channel = y_prediction
            y_target_target_channel = y_target

        batch_size_x_num_samples = y_target_target_channel.shape[0]
        batch_size = batch_size_x_num_samples // self.num_samples
        
        q_levels = (torch.arange(1, self.num_samples + 1, device=device, dtype=torch.float32)) / (self.num_samples + 1)
        q_tensor = q_levels.repeat_interleave(batch_size).view(-1, 1, 1)
        
        e = y_target_target_channel - y_prediction_target_channel
        loss = torch.max(q_tensor * e, (q_tensor - 1) * e)
        return loss

    def energy_func(self, 
                    y_candidate: torch.Tensor, 
                    t: torch.Tensor,           
                    ground_truth_for_loss: torch.Tensor, # Renamed from observation_ground_truth
                    observation_mask_for_loss: torch.Tensor, 
                    features_for_denoise: Optional[torch.Tensor]
                   ) -> torch.Tensor:
        predicted_x0 = self.model.fast_denoise(y_candidate, t, features_for_denoise)
        predicted_x0_target_channel = predicted_x0[..., 0].unsqueeze(-1)

        if self.guidance == "MSE":
            loss_values = F.mse_loss(predicted_x0_target_channel, ground_truth_for_loss, reduction="none")
        elif self.guidance == "quantile":
            loss_values = self.quantile_loss(predicted_x0_target_channel, ground_truth_for_loss)
        else:
            raise ValueError(f"Unknown guidance type: {self.guidance}!")
        
        return (loss_values * observation_mask_for_loss).sum()

    def score_func(self, 
                   y_candidate: torch.Tensor, 
                   t: torch.Tensor, 
                   ground_truth_for_loss: torch.Tensor, # Renamed
                   observation_mask_for_loss: torch.Tensor, 
                   features_for_denoise: Optional[torch.Tensor]
                  ) -> torch.Tensor:
        with torch.enable_grad():
            y_candidate_for_grad = y_candidate.detach().clone().requires_grad_(True)
            energy = self.energy_func(
                y_candidate_for_grad, t, ground_truth_for_loss, 
                observation_mask_for_loss, features_for_denoise
            )
            score = -torch.autograd.grad(outputs=energy, inputs=y_candidate_for_grad)[0]
        return score

    def scale_func(self, y: torch.Tensor, t: torch.Tensor, base_scale: float) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement scale_func for guidance step size.")

    # --- 修改 guide 方法签名以匹配 forward 中的调用 ---
    def guide(self, 
              initial_sample_state: torch.Tensor, # This is the x_t candidate, repeated for samples
              observation_mask_for_loss: torch.Tensor, 
              ground_truth_for_loss: torch.Tensor, 
              features_for_backbone: Optional[torch.Tensor], 
              guidance_strength_knob: float 
             ) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the main guidance loop.")
    # --- 结束修改 ---

    def forward(
        self,
        past_target: torch.Tensor,
        past_observed_values: torch.Tensor,
        feat_static_cat: Optional[torch.Tensor] = None,
        feat_static_real: Optional[torch.Tensor] = None,
        past_feat_dynamic_real: Optional[torch.Tensor] = None,
        future_feat_dynamic_real: Optional[torch.Tensor] = None,
    ):
        device = self.model.device 

        data_for_extraction = {
            "past_target": past_target.to(device),
            "past_observed_values": past_observed_values.to(device),
            "future_target": torch.zeros(past_target.shape[0], self.prediction_length, device=device),
        }
        if feat_static_cat is not None: data_for_extraction["feat_static_cat"] = feat_static_cat.to(device)
        if feat_static_real is not None: data_for_extraction["feat_static_real"] = feat_static_real.to(device)
        if past_feat_dynamic_real is not None: data_for_extraction["past_feat_dynamic_real"] = past_feat_dynamic_real.to(device)
        if future_feat_dynamic_real is not None: data_for_extraction["future_feat_dynamic_real"] = future_feat_dynamic_real.to(device)

        x_for_backbone_scaled, scale_params_for_unscale, features_for_backbone_conditioning = \
            self.model._extract_features(data_for_extraction)
        
        x_for_backbone_scaled = x_for_backbone_scaled.to(device)
        if features_for_backbone_conditioning is not None:
            features_for_backbone_conditioning = features_for_backbone_conditioning.to(device)

        batch_size, total_seq_len, _ = x_for_backbone_scaled.shape
        observation_ground_truth_target_scaled = x_for_backbone_scaled[..., 0].unsqueeze(-1) 
        
        context_mask_2d = past_observed_values[:, -self.context_length :] 
        if context_mask_2d.ndim == 3 and context_mask_2d.shape[-1] == 1: context_mask_2d = context_mask_2d.squeeze(-1)
        future_mask_part_2d = torch.zeros((batch_size, self.prediction_length), device=device, dtype=context_mask_2d.dtype)
        observation_mask_target_channel_only = torch.cat([context_mask_2d, future_mask_part_2d], dim=1).unsqueeze(-1)

        # Prepare inputs for the guide method
        initial_sample_state_repeated = x_for_backbone_scaled.repeat_interleave(self.num_samples, dim=0)
        observation_mask_for_loss_repeated = observation_mask_target_channel_only.repeat_interleave(self.num_samples, dim=0)
        ground_truth_for_loss_repeated = observation_ground_truth_target_scaled.repeat_interleave(self.num_samples, dim=0)
        
        features_for_guidance_repeated = None
        if features_for_backbone_conditioning is not None:
            features_for_guidance_repeated = features_for_backbone_conditioning.repeat_interleave(self.num_samples, dim=0)
        
        # Call the guide method (implemented by DDPMGuidance or DDIMGuidance)
        pred_scaled_all_channels = self.guide(
            initial_sample_state=initial_sample_state_repeated, 
            observation_mask_for_loss=observation_mask_for_loss_repeated, 
            ground_truth_for_loss=ground_truth_for_loss_repeated, 
            features_for_backbone=features_for_guidance_repeated,
            guidance_strength_knob=self.scale 
        )
        
        pred_target_channel_scaled = pred_scaled_all_channels[..., 0] 
        pred_target_channel_scaled = pred_target_channel_scaled.reshape(batch_size, self.num_samples, total_seq_len)
        pred_unscaled = pred_target_channel_scaled * scale_params_for_unscale
        return pred_unscaled[..., self.context_length :]

    def get_predictor(self, input_transform, batch_size=40, device=None):
        return PyTorchPredictorWGrads(
            prediction_length=self.prediction_length,
            input_names=PREDICTION_INPUT_NAMES, 
            prediction_net=self,
            batch_size=batch_size,
            input_transform=input_transform,
            device=device,
        )

class DDPMGuidance(Guidance):
    def __init__(
        self, model: TSDiff, prediction_length: int, scale: float = 1.0,
        num_samples: int = 1, guidance: str = "quantile",
        missing_scenario: str = "none", missing_values: int = 0
    ):
        super().__init__(
            model, prediction_length, scale, num_samples, guidance,
            missing_scenario, missing_values
        )

    def scale_func(self, y_candidate: torch.Tensor, t: torch.Tensor, base_scale_knob: float) -> torch.Tensor:
        posterior_variance_dev = self.model.posterior_variance.to(y_candidate.device)
        return extract(posterior_variance_dev, t, y_candidate.shape) * base_scale_knob

    # --- 修改 guide 方法签名以匹配 Guidance.forward 中的调用 ---
    @torch.no_grad()
    def guide(self, 
              initial_sample_state: torch.Tensor, # Shape (B*S, L, C_input_to_backbone)
              observation_mask_for_loss: torch.Tensor, 
              ground_truth_for_loss: torch.Tensor, 
              features_for_backbone: Optional[torch.Tensor], 
              guidance_strength_knob: float 
             ) -> torch.Tensor:
        
        device = self.model.device # Use model's device
        batch_size_x_num_samples = initial_sample_state.shape[0]
        
        # For DDPM, sampling starts from pure noise matching the shape of initial_sample_state
        current_sample_state = torch.randn_like(initial_sample_state, device=device) 

        for i in reversed(range(0, self.model.timesteps)):
            t_tensor = torch.full((batch_size_x_num_samples,), i, device=device, dtype=torch.long)
            
            current_sample_state = self.model.p_sample(
                current_sample_state, t_tensor, i, features_for_backbone
            )
            
            if guidance_strength_knob > 0:
                score_val = self.score_func(
                    current_sample_state.detach().clone(), 
                    t_tensor, 
                    ground_truth_for_loss, 
                    observation_mask_for_loss, 
                    features_for_backbone 
                )
                guidance_step_size = self.scale_func(current_sample_state, t_tensor, guidance_strength_knob)
                current_sample_state = current_sample_state + guidance_step_size * score_val
        return current_sample_state
    # --- 结束修改 ---


class DDIMGuidance(Guidance):
    _skip_types = ["uniform", "quadratic"]
    def __init__(
        self, model: TSDiff, prediction_length: int, eta: float = 0.0,
        skip_factor: int = 1, skip_type: str = "uniform", scale: float = 1.0, 
        num_samples: int = 1, guidance: str = "quantile",
        missing_scenario: str = "none", missing_values: int = 0
    ):
        super().__init__(
            model, prediction_length, scale, num_samples, guidance,
            missing_scenario, missing_values
        )
        assert skip_type in self._skip_types
        self.eta = eta
        self.skip_factor = skip_factor
        self.skip_type = skip_type

    def scale_func(self, y_candidate: torch.Tensor, t: torch.Tensor, base_scale_knob: float) -> torch.Tensor:
        sqrt_one_minus_alphas_cumprod_dev = self.model.sqrt_one_minus_alphas_cumprod.to(y_candidate.device)
        # The base_scale_knob here is the overall guidance strength 's'
        # The scale_func for DDIM noise guidance is often just sqrt(1-alpha_bar_t)
        return extract(sqrt_one_minus_alphas_cumprod_dev, t, y_candidate.shape) # base_scale_knob is applied later

    def _get_timesteps(self) -> torch.Tensor: 
        if self.skip_type == "uniform":
            num_inference_steps = max(1, self.model.timesteps // self.skip_factor)
            timesteps_np = np.linspace(0, self.model.timesteps - 1, num_inference_steps, dtype=np.int64)
        elif self.skip_type == "quadratic":
            num_inference_steps = max(1, self.model.timesteps // self.skip_factor)
            timesteps_np = ((np.linspace(0, np.sqrt(self.model.timesteps -1 + 1e-6), num_inference_steps)) ** 2).astype(np.int64)
        else: 
            timesteps_np = np.arange(self.model.timesteps).astype(np.int64)
        
        timesteps_np = np.unique(np.concatenate(([0], timesteps_np))) 
        return torch.from_numpy(timesteps_np).to(torch.long).flip(dims=[0]) 

    # --- 修改 guide 方法签名 ---
    @torch.no_grad()
    def guide(self, 
              initial_sample_state: torch.Tensor, # Shape (B*S, L, C_input_to_backbone)
              observation_mask_for_loss: torch.Tensor, 
              ground_truth_for_loss: torch.Tensor, 
              features_for_backbone: Optional[torch.Tensor], 
              guidance_strength_knob: float # This is self.scale from __init__
             ) -> torch.Tensor:
        
        device = self.model.device
        batch_size_x_num_samples = initial_sample_state.shape[0]
        seq_len = initial_sample_state.shape[1]
        num_channels_backbone = initial_sample_state.shape[2]

        current_sample_state = torch.randn((batch_size_x_num_samples, seq_len, num_channels_backbone), device=device)
        
        timesteps_ddim = self._get_timesteps().to(device)

        for i in range(len(timesteps_ddim)):
            t_val = timesteps_ddim[i]
            t_current_tensor = torch.full((batch_size_x_num_samples,), t_val.item(), device=device, dtype=torch.long)
            
            t_prev_val = timesteps_ddim[i + 1] if i < len(timesteps_ddim) - 1 else -1 
            t_prev_tensor = torch.full((batch_size_x_num_samples,), t_prev_val.item(), device=device, dtype=torch.long)

            predicted_noise_unconditional = self.model.backbone(current_sample_state, t_current_tensor, features_for_backbone)
            
            noise_pred_final = predicted_noise_unconditional
            if guidance_strength_knob > 0:
                with torch.enable_grad():
                    seq_for_score = current_sample_state.detach().clone().requires_grad_(True)
                    score_val = self.score_func( # score_val is -dE/dx_t = d(log P(obs|denoised(x_t)))/dx_t
                        seq_for_score, t_current_tensor, 
                        ground_truth_for_loss, 
                        observation_mask_for_loss, 
                        features_for_backbone
                    )
                
                # Adjustment factor for noise: sqrt(1-alpha_bar_t)
                # The self.scale_func now returns this factor * base_scale_knob.
                # If we want self.scale to be the 's' in eps_hat = eps_theta - s * factor * score,
                # then scale_func should just return the 'factor'.
                # Let's adjust scale_func for DDIM to return just sqrt(1-alpha_bar_t)
                # and apply guidance_strength_knob (self.scale) here.
                
                # Re-calling scale_func for DDIM which should ideally just return sqrt(1-alpha_bar_t)
                # For DDIM, scale_func was: extract(self.model.sqrt_one_minus_alphas_cumprod.to(y.device), t, y.shape) * base_scale_knob
                # Let's assume base_scale_knob in scale_func is 1 for this factor.
                adjustment_factor = self.scale_func(current_sample_state, t_current_tensor, 1.0) # Gets sqrt(1-alpha_bar_t)
                
                noise_pred_final = predicted_noise_unconditional - guidance_strength_knob * adjustment_factor * score_val
            
            current_sample_state = self.model.p_sample_genddim(
                x=current_sample_state,
                t=t_current_tensor,
                t_index=t_val.item(),
                t_prev=t_prev_tensor,
                eta=self.eta,
                features=features_for_backbone,
                noise_pred_override=noise_pred_final 
            )
        return current_sample_state
    # --- 结束修改 ---