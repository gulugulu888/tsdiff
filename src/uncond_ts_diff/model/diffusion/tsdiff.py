# src/uncond_ts_diff/model/diffusion/tsdiff.py
import copy
import torch
from gluonts.torch.util import lagged_sequence_values

from uncond_ts_diff.arch import BackboneModel
from uncond_ts_diff.model.diffusion._base import TSDiffBase
from uncond_ts_diff.utils import get_lags_for_freq # Ensure this import is correct
from typing import Optional, Dict, Tuple, List 
import logging 
logger = logging.getLogger(__name__)
class TSDiff(TSDiffBase):
    def __init__(
        self,
        backbone_parameters,
        timesteps,
        diffusion_scheduler,
        context_length,
        prediction_length,
        num_feat_dynamic_real: int = 0,
        num_feat_static_cat: int = 0,
        num_feat_static_real: int = 0,
        cardinalities=None,
        freq=None, # freq will be passed from config
        normalization="none",
        use_features=False,
        use_lags=True,
        init_skip=True,
        lr=1e-3,
        # Add num_input_variates for multi-variate case later
        # num_input_variates: int = 1, # For Phase 2
    ):
        super().__init__(
            backbone_parameters=backbone_parameters, # Pass directly
            timesteps=timesteps,
            diffusion_scheduler=diffusion_scheduler,
            context_length=context_length,
            prediction_length=prediction_length,
            num_feat_dynamic_real=num_feat_dynamic_real,
            num_feat_static_cat=num_feat_static_cat,
            num_feat_static_real=num_feat_static_real,
            cardinalities=cardinalities,
            freq=freq, # Pass freq to super
            normalization=normalization,
            use_features=use_features,
            use_lags=use_lags, # This will be used by TSDiffBase if needed
            lr=lr,
        )

        self.freq = freq # Store freq from config
        # self.num_input_variates = num_input_variates # For Phase 2

        backbone_params_copy = backbone_parameters.copy()

        if use_lags:
            if not self.freq:
                raise ValueError("Frequency `freq` must be provided when use_lags is True.")
            # Pass context_length to get_lags_for_freq
            self.lags_seq = get_lags_for_freq(self.freq, context_length_for_lags=self.context_length)
            if not self.lags_seq: # Ensure lags_seq is not empty
                print(f"Warning: get_lags_for_freq returned empty for freq='{self.freq}'. Defaulting to simple lags or no lags.")
                # Decide a fallback, e.g. no lags or a default small lag
                # For now, let's assume it returns at least [1]
            
            # Assuming single variate for now (Phase 1)
            # input_dim is 1 (target) + number of lags
            backbone_params_copy["input_dim"] = 1 + len(self.lags_seq)
            backbone_params_copy["output_dim"] = 1 + len(self.lags_seq) # Output should match input structure if predicting noise on all
        else:
            self.lags_seq = [] # Or [0] depending on how lagged_sequence_values handles empty list
            # Assuming single variate
            backbone_params_copy["input_dim"] = 1
            backbone_params_copy["output_dim"] = 1
        
        self.input_dim_to_backbone = backbone_params_copy["input_dim"]

        # Calculate num_features for BackboneModel based on what TSDiffBase prepares
        # TSDiffBase._extract_features will create 'features' tensor.
        # For single variate, no external features:
        _num_additional_features_for_backbone = 0
        if self.use_features: # Now self.use_features is correctly an instance attribute
             _num_additional_features_for_backbone = (
                self.num_feat_static_real +       # This is total static real (external + log_scale)
                self.num_feat_static_cat +        # This is total embedded static cat (from embedder.num_features)
                self.num_feat_dynamic_real        # This is total dynamic real (external + time_features)
                                                  # The log_scale is now part of self.num_feat_static_real if added in _extract_features
            )
             # More precise calculation based on TSDiffBase's attributes:
             _num_additional_features_for_backbone = self.hparams.num_feat_static_real + \
                                                     self.embedder.num_features + \
                                                     self.hparams.num_feat_dynamic_real + \
                                                     len(self.time_features) + \
                                                     1 # For log_scale feature added in _extract_features

        backbone_params_copy["num_features"] = _num_additional_features_for_backbone

        self.backbone = BackboneModel(
            **backbone_params_copy,
            init_skip=init_skip,
        )
        self.ema_rate = [] 
        self.ema_state_dicts = [
            copy.deepcopy(self.backbone.state_dict())
            for _ in range(len(self.ema_rate))
        ]


    def _extract_features(self, data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        past_target = data["past_target"] 
        past_observed = data["past_observed_values"]
        future_target_gt = data["future_target"]

        if past_target.ndim == 3 and past_target.shape[-1] == 1: past_target = past_target.squeeze(-1)
        if past_observed.ndim == 3 and past_observed.shape[-1] == 1: past_observed = past_observed.squeeze(-1)
        if future_target_gt.ndim == 3 and future_target_gt.shape[-1] == 1: future_target_gt = future_target_gt.squeeze(-1)

        prior_for_lags = past_target[:, : -self.context_length]    
        context_for_input = past_target[:, -self.context_length :] 
        context_observed_for_scaling = past_observed[:, -self.context_length :]
        
        scaled_context, scale_val = self.scaler(context_for_input, context_observed_for_scaling)
        
        scale_params_for_ts = scale_val.unsqueeze(-1) if scale_val.ndim == 1 else scale_val
        if scale_params_for_ts.ndim == 2 : scale_params_for_ts = scale_params_for_ts.unsqueeze(-1) # Ensure (B,1,1)

        scaled_prior_for_lags = prior_for_lags / scale_val 
        scaled_future_gt = future_target_gt / scale_val       
        x_target_unlagged = torch.cat([scaled_context, scaled_future_gt], dim=1).unsqueeze(-1)

        x_for_backbone: torch.Tensor
        if self.use_lags and self.lags_seq:
            lags = lagged_sequence_values(
                self.lags_seq, scaled_prior_for_lags, 
                torch.cat([scaled_context, scaled_future_gt], dim=1), dim=1,
            ) 
            x_for_backbone = torch.cat([x_target_unlagged, lags], dim=-1)
        else:
            x_for_backbone = x_target_unlagged

        final_features_for_backbone: Optional[torch.Tensor] = None
        if self.use_features: # This flag controls if TSDiff uses additional features for its backbone
            features_list_for_concat = []
            
            # 1. Static features (categorical embeddings + real static + log_scale)
            static_feats_to_cat_for_expansion = []
            if data.get("feat_static_cat") is not None and self.hparams.num_feat_static_cat > 0: # Use hparams for original count
                static_feats_to_cat_for_expansion.append(self.embedder(data["feat_static_cat"]))
            if data.get("feat_static_real") is not None and self.hparams.num_feat_static_real > 0:
                static_feats_to_cat_for_expansion.append(data["feat_static_real"])
            
            log_scale_feature = torch.log(scale_val + 1e-8) # scale_val is (B,1)
            static_feats_to_cat_for_expansion.append(log_scale_feature)

            if static_feats_to_cat_for_expansion:
                combined_static_feat = torch.cat(static_feats_to_cat_for_expansion, dim=1) 
                expanded_static_feat = combined_static_feat.unsqueeze(1).expand(
                    -1, self.context_length + self.prediction_length, -1 
                )
                features_list_for_concat.append(expanded_static_feat)

            # 2. Dynamic features (Time features, Age, and any external dynamic real)
            # These are now expected to be in "past_feat_dynamic_real" and "future_feat_dynamic_real"
            # after InstanceSplitter processes the output of VstackFeatures.
            dynamic_feats_parts = []
            if data.get("past_feat_dynamic_real") is not None: # <--- MODIFIED KEY
                dynamic_feats_parts.append(data["past_feat_dynamic_real"])
            if data.get("future_feat_dynamic_real") is not None: # <--- MODIFIED KEY
                dynamic_feats_parts.append(data["future_feat_dynamic_real"])
            
            if dynamic_feats_parts:
                combined_dynamic_feat = torch.cat(dynamic_feats_parts, dim=1)
                features_list_for_concat.append(combined_dynamic_feat)
            
            if features_list_for_concat:
                final_features_for_backbone = torch.cat(features_list_for_concat, dim=-1)
                # Check expected num_features for backbone
                expected_num_feats = self.backbone.num_features # Accessing it from backbone directly
                if final_features_for_backbone.shape[-1] != expected_num_feats:
                    logger.warning(f"_extract_features: Shape of final_features_for_backbone ({final_features_for_backbone.shape}) "
                                   f"last dim does not match BackboneModel's expected num_features ({expected_num_feats}). This might cause errors.")
        
        return x_for_backbone, scale_params_for_ts, final_features_for_backbone

    # ... (on_train_batch_end and sample_n might need slight adjustment if input_dim_to_backbone changes meaning)
    @torch.no_grad()
    def sample_n(
        self,
        num_samples: int = 1,
        return_lags: bool = False, # This flag might be less relevant if output_dim is just num_variates
    ):
        device = next(self.backbone.parameters()).device
        seq_len = self.context_length + self.prediction_length

        # input_dim_to_backbone is 1 (target) + num_lags for single variate with lags
        # or just 1 if no lags.
        samples_shape = (num_samples, seq_len, self.input_dim_to_backbone)
        samples = torch.randn(samples_shape, device=device)

        # Assuming features are not needed for unconditional sampling, or would be dummy features
        # If your model was trained with time/static features, you might need to provide them here
        # For pure unconditional generation, features=None is typical.
        dummy_features_for_sampling = None 
        # If self.use_features was true during training, you might need to create placeholder features
        # that match the expected dimension for num_features in BackboneModel.
        # This part can be complex if features are essential to the backbone's structure.
        # For now, assuming unconditional generation doesn't strictly require them or backbone handles None.

        for i in reversed(range(0, self.timesteps)):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            samples = self.p_sample(
                samples, 
                t, 
                i,
                features=dummy_features_for_sampling # Pass None or dummy features
            )
        
        # The backbone outputs shape (B, L, output_dim_of_backbone)
        # For single variate with lags, output_dim_of_backbone = 1 + num_lags
        # We are interested in the first channel, which is the actual time series value.
        if self.use_lags and self.lags_seq and not return_lags:
            # Return only the first channel (the actual series, not the concatenated lags)
            return samples[..., 0].cpu().numpy() 
        else:
            # Return all channels (e.g., if no lags, or if user wants lags too)
            return samples.cpu().numpy()

    def on_train_batch_end(self, outputs, batch, batch_idx): # Keep as is
        for rate, state_dict in zip(self.ema_rate, self.ema_state_dicts):
            update_ema(state_dict, self.backbone.state_dict(), rate=rate)

# Helper function (can be outside class or as a static method if preferred)
def update_ema(target_state_dict, source_state_dict, rate=0.99): # Keep as is
    with torch.no_grad():
        for key, value in source_state_dict.items():
            ema_value = target_state_dict[key]
            ema_value.copy_(
                rate * ema_value + (1.0 - rate) * value.cpu(),
                non_blocking=True,
            )