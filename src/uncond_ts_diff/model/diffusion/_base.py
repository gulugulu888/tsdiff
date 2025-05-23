# src/uncond_ts_diff/model/diffusion/_base.py
from typing import Optional, Dict, Tuple, List # <--- 确保 List 在这里被导入

import logging 
import numpy as np
import pandas as pd 
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

from gluonts.time_feature import time_features_from_frequency_str, TimeFeature 
from gluonts.torch.modules.feature import FeatureEmbedder
from gluonts.torch.modules.scaler import MeanScaler, NOPScaler
import tqdm

from uncond_ts_diff.utils import extract 

logger = logging.getLogger(__name__)

PREDICTION_INPUT_NAMES = [
    "past_target",
    "past_observed_values",
    "feat_static_cat",
    "feat_static_real",
    # --- 修改以下两行 ---
    "past_feat_dynamic_real", # 原来是 "past_time_feat"
    "future_feat_dynamic_real", # 原来是 "future_time_feat"
    # --- 结束修改 ---
]

class TSDiffBase(pl.LightningModule):
    def __init__(
        self,
        backbone_parameters: dict,
        timesteps: int,
        diffusion_scheduler, 
        context_length: int,
        prediction_length: int,
        num_feat_dynamic_real: int = 0, 
        num_feat_static_cat: int = 0,   
        num_feat_static_real: int = 0,  
        cardinalities: Optional[List[int]] = None, 
        freq: Optional[str] = None,
        normalization: str = "none",
        use_features: bool = False, # Parameter received
        use_lags: bool = True,      # Parameter received
        lr: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters( # Call this early if you rely on self.hparams immediately
            "timesteps", 
            "context_length", 
            "prediction_length",
            "num_feat_dynamic_real", 
            "num_feat_static_cat",
            "num_feat_static_real",
            "freq",
            "normalization",
            "use_features", # Saved to hparams
            "use_lags",   # Saved to hparams
            "lr"
        )

        self.timesteps = timesteps
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.lr = lr
        self.freq = freq 
        
        # --- 添加这两行 ---
        self.use_features = use_features
        self.use_lags = use_lags
        # --- 结束添加 ---

        # Diffusion schedule parameters
        self.betas = diffusion_scheduler(timesteps)
        # ... (rest of the __init__ method remains the same as the previous complete version) ...
        self.sqrt_one_minus_beta = torch.sqrt(1.0 - self.betas) 
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = (
            self.betas
            * (1.0 - self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        
        self.logs = {} 

        self.normalization = normalization
        if normalization == "mean":
            self.scaler = MeanScaler(dim=1, keepdim=True) 
        else: 
            self.scaler = NOPScaler(dim=1, keepdim=True)

        _cardinalities = cardinalities if cardinalities is not None else []
        _embedding_dims = [min(50, (cat + 1) // 2) for cat in _cardinalities] if _cardinalities else []
        self.embedder = FeatureEmbedder(
            cardinalities=_cardinalities,
            embedding_dims=_embedding_dims,
        )

        self.time_features: List[TimeFeature] = [] 
        if self.freq is not None:
            supported_base_freq_names = ['Y', 'A', 'Q', 'M', 'W', 'D', 'B', 'H', 'T', 'min', 'S']
            try:
                parsed_offset = pd.tseries.frequencies.to_offset(self.freq)
                if parsed_offset is not None and parsed_offset.name in supported_base_freq_names:
                    self.time_features = time_features_from_frequency_str(self.freq)
                    logger.info(f"Automatically derived {len(self.time_features)} time features for frequency '{self.freq}'.")
                else:
                    logger.warning(f"Frequency '{self.freq}' (parsed base name: '{parsed_offset.name if parsed_offset else 'N/A'}') "
                                   f"is not in the list of directly supported base frequencies for automatic time feature generation "
                                   f"by `time_features_from_frequency_str`. No calendar time features will be automatically added. "
                                   f"Model will rely on lags and learned dynamics.")
                    self.time_features = []
            except RuntimeError as e: 
                 logger.warning(f"RuntimeError generating time features for frequency '{self.freq}': {e}. "
                                f"No calendar time features will be automatically added.")
                 self.time_features = []
            except Exception as e_freq: 
                 logger.warning(f"Could not parse frequency string '{self.freq}' or generate time features: {e_freq}. "
                                f"No calendar time features will be added.")
                 self.time_features = []
        
        # These are the counts of *external* features passed in the data dictionary
        self.num_feat_dynamic_real_external = num_feat_dynamic_real 
        self.num_feat_static_cat_external = num_feat_static_cat     
        self.num_feat_static_real_external = num_feat_static_real   
        
        # Total number of dynamic real features including generated time features
        self.num_feat_dynamic_real = num_feat_dynamic_real + len(self.time_features)
        # Total number of static cat features (usually just from embedder if external static_cat is 0)
        self.num_feat_static_cat = self.embedder.num_features if _cardinalities else 0 
        # Total number of static real features
        self.num_feat_static_real = num_feat_static_real
        
        self.best_crps = np.inf

    def _extract_features(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError("Subclasses must implement _extract_features.")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode="min", 
            factor=0.5, 
            patience=10, 
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss_epoch", 
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def log(self, name, value, **kwargs):
        super().log(name, value, **kwargs)
        if isinstance(value, torch.Tensor):
            value_item = value.detach().cpu().item()
        else:
            value_item = value
        
        if hasattr(self, 'trainer') and self.trainer and self.trainer.global_rank == 0: 
            if name not in self.logs:
                self.logs[name] = []
            self.logs[name].append(value_item)

    def get_logs_as_dataframe(self) -> pd.DataFrame: 
        max_len = 0
        if self.logs:
            max_len = max((len(v) for v in self.logs.values() if isinstance(v, list)), default=0)
        
        logs_for_df = {}
        for k, v in self.logs.items():
            if isinstance(v, list) and len(v) == max_len:
                logs_for_df[k] = v
            elif isinstance(v, list) and len(v) < max_len : 
                logs_for_df[k] = v + [np.nan] * (max_len - len(v))

        if 'epoch' not in logs_for_df or len(logs_for_df.get('epoch',[])) != max_len:
            if max_len > 0 : 
                 logs_for_df['epoch'] = list(range(max_len)) 
            elif not logs_for_df : 
                 return pd.DataFrame()
        try:
            return pd.DataFrame.from_dict(logs_for_df)
        except ValueError as e:
            print(f"Error creating DataFrame from logs (likely due to unequal lengths): {e}")
            print(f"Current log dict keys and lengths: { {k: len(v) if isinstance(v,list) else 1 for k,v in logs_for_df.items()} }")
            return pd.DataFrame() 

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start, device=self.device)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod.to(self.device), t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod.to(self.device), t, x_start.shape)
        return (
            sqrt_alphas_cumprod_t * x_start
            + sqrt_one_minus_alphas_cumprod_t * noise
        )

    def p_losses(
        self,
        x_start, 
        t,       
        conditioning_features=None, 
        noise=None,
        loss_type="l2", 
        reduction="mean",
    ):
        if noise is None:
            noise = torch.randn_like(x_start, device=self.device)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise) 
        predicted_noise = self.backbone(x_noisy, t, conditioning_features)
        if loss_type == "l1":
            loss = F.l1_loss(noise, predicted_noise, reduction=reduction)
        elif loss_type == "l2":
            loss = F.mse_loss(noise, predicted_noise, reduction=reduction)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise, reduction=reduction)
        else:
            raise NotImplementedError()
        return loss, x_noisy, predicted_noise

    @torch.no_grad()
    def p_sample(self, x, t, t_index, features=None): 
        betas_t = extract(self.betas.to(self.device), t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod.to(self.device), t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas.to(self.device), t, x.shape)
        predicted_noise = self.backbone(x, t, features)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        if t_index == 0: 
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance.to(self.device), t, x.shape)
            noise_for_sampling = torch.randn_like(x, device=self.device)
            return model_mean + torch.sqrt(posterior_variance_t) * noise_for_sampling

    @torch.no_grad()
    def p_sample_ddim(self, x, t, features=None, noise_pred_override=None): 
        if noise_pred_override is None:
            noise_pred = self.backbone(x, t, features)
        else:
            noise_pred = noise_pred_override
        alphas_cumprod_t = extract(self.alphas_cumprod.to(self.device), t, x.shape)
        alphas_cumprod_prev_t = extract(self.alphas_cumprod_prev.to(self.device), t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = (1.0 - alphas_cumprod_t).sqrt()
        pred_x0 = (x - sqrt_one_minus_alphas_cumprod_t * noise_pred) / alphas_cumprod_t.sqrt()
        dir_xt = (1.0 - alphas_cumprod_prev_t).sqrt() * noise_pred 
        x_prev = alphas_cumprod_prev_t.sqrt() * pred_x0 + dir_xt
        return x_prev

    @torch.no_grad()
    def p_sample_genddim(
        self,
        x: torch.Tensor, 
        t: torch.Tensor, 
        t_index: int,    
        t_prev: Optional[torch.Tensor] = None, 
        eta: float = 0.0, 
        features=None,
        noise_pred_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noise_pred_override is None:
            noise_pred = self.backbone(x, t, features) 
        else:
            noise_pred = noise_pred_override

        alphas_cumprod_t = extract(self.alphas_cumprod.to(self.device), t, x.shape) 
        if t_prev is None: 
            alphas_cumprod_t_prev = extract(self.alphas_cumprod_prev.to(self.device), t, x.shape) 
        else: 
            alphas_cumprod_t_prev = extract(self.alphas_cumprod.to(self.device), t_prev, x.shape) if t_index > 0 else torch.ones_like(alphas_cumprod_t)
        
        pred_x0 = (x - (1 - alphas_cumprod_t).sqrt() * noise_pred) / alphas_cumprod_t.sqrt()
        variance = 0.0 
        if t_index > 0 and eta > 0:
            term1_safe = (1 - alphas_cumprod_t_prev)
            term2_safe = (1 - alphas_cumprod_t)
            term3_safe = (1 - alphas_cumprod_t / (alphas_cumprod_t_prev + 1e-8) ) 
            variance_numerator = term1_safe * term3_safe
            variance_denominator = term2_safe + 1e-8 
            variance_term = torch.clamp(variance_numerator / variance_denominator, min=0.0)
            sigma_t = eta * torch.sqrt(variance_term)
            variance = sigma_t**2
        
        term_under_sqrt = torch.clamp((1 - alphas_cumprod_t_prev - variance), min=0.0)
        direction_xt = term_under_sqrt.sqrt() * noise_pred
        x_prev_mean = alphas_cumprod_t_prev.sqrt() * pred_x0 + direction_xt
        
        if t_index == 0 or eta == 0.0: 
            return x_prev_mean
        else:
            return x_prev_mean + torch.sqrt(variance) * torch.randn_like(x, device=self.device)

    @torch.no_grad()
    def sample(self, noise_input_tensor, features=None, use_ddim=False, ddim_skip_factor=1, ddim_eta=0.0):
        current_samples = noise_input_tensor.to(self.device)
        if use_ddim:
            num_inference_steps = self.timesteps // ddim_skip_factor
            timesteps_ddim = torch.from_numpy(np.linspace(0, self.timesteps -1 , num_inference_steps, dtype=np.int64)[::-1].copy()).to(self.device)
            for i, t_val in enumerate(tqdm(timesteps_ddim, desc="DDIM Sampling")):
                t_current_tensor = torch.full((current_samples.shape[0],), t_val, device=self.device, dtype=torch.long)
                t_prev_val = timesteps_ddim[i + 1] if i < len(timesteps_ddim) - 1 else -1 
                t_prev_tensor = torch.full((current_samples.shape[0],), t_prev_val, device=self.device, dtype=torch.long)
                current_samples = self.p_sample_genddim(current_samples, t_current_tensor, t_index=t_val.item(), t_prev=t_prev_tensor, eta=ddim_eta, features=features)
        else: 
            for i in tqdm(reversed(range(0, self.timesteps)), desc="DDPM Sampling", total=self.timesteps):
                t = torch.full((current_samples.shape[0],), i, device=self.device, dtype=torch.long)
                current_samples = self.p_sample(current_samples, t, i, features=features)
        return current_samples 

    def fast_denoise(self, xt, t, features=None, noise_pred_override=None):
        if noise_pred_override is None:
            noise_pred = self.backbone(xt, t, features)
        else:
            noise_pred = noise_pred_override
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod.to(self.device), t, xt.shape)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod.to(self.device), t, xt.shape)
        pred_x0 = (xt - sqrt_one_minus_alphas_cumprod_t * noise_pred) / (sqrt_alphas_cumprod_t + 1e-8) 
        return pred_x0

    def forward(self, x_noisy_input, t_diffusion_step, conditioning_feats=None):
        return self.backbone(x_noisy_input, t_diffusion_step, conditioning_feats)

    def training_step(self, batch_data_dict: dict, batch_idx: int):
        x_start_scaled, _, conditioning_features = self._extract_features(batch_data_dict)
        x_start_scaled = x_start_scaled.to(self.device)
        if conditioning_features is not None:
            conditioning_features = conditioning_features.to(self.device)
        t = torch.randint(0, self.timesteps, (x_start_scaled.shape[0],), device=self.device).long()
        loss, _, _ = self.p_losses(x_start_scaled, t, conditioning_features, loss_type="l2") 
        self.log("train_loss_step", loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss 

    def training_epoch_end(self, training_step_outputs):
        if training_step_outputs : # Ensure not empty
            avg_epoch_loss = torch.stack([x['loss'] for x in training_step_outputs]).mean()
            self.log("train_loss_epoch", avg_epoch_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch_data_dict: dict, batch_idx: int):
        x_start_scaled, _, conditioning_features = self._extract_features(batch_data_dict)
        x_start_scaled = x_start_scaled.to(self.device)
        if conditioning_features is not None:
            conditioning_features = conditioning_features.to(self.device)
        t = torch.randint(0, self.timesteps, (x_start_scaled.shape[0],), device=self.device).long()
        loss, _, _ = self.p_losses(x_start_scaled, t, conditioning_features, loss_type="l2")
        self.log("valid_loss_step", loss, on_step=True, on_epoch=True, logger=True) # Log on epoch as well for aggregation
        return loss 

    def validation_epoch_end(self, validation_step_outputs):
        if validation_step_outputs: # Ensure not empty
            avg_epoch_loss = torch.stack(validation_step_outputs).mean() 
            self.log("valid_loss_epoch", avg_epoch_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)