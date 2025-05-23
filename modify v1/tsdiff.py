# src/uncond_ts_diff/model/diffusion/tsdiff.py
import copy
import torch
from gluonts.torch.util import lagged_sequence_values

from uncond_ts_diff.arch import BackboneModel
from uncond_ts_diff.model.diffusion._base import TSDiffBase
from uncond_ts_diff.utils import get_lags_for_freq # Ensure this import is correct

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
        if self.use_features: # This flag in TSDiff controls if time/static feats are used
             _num_additional_features_for_backbone = (
                self.num_feat_static_real + \
                self.embedder.num_features + # Sum of embedding_dims from FeatureEmbedder
                self.num_feat_dynamic_real + \
                1 # log_scale feature added in _extract_features
            )
        
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

    # ... (rest of TSDiff class, _extract_features, sample_n, etc. remain largely the same for Phase 1)
    # _extract_features will need changes for multivariate in Phase 2
    def _extract_features(self, data):
        # This is for single-variate case. Will need significant changes for multi-variate.
        prior = data["past_target"][:, : -self.context_length]
        context = data["past_target"][:, -self.context_length :]
        context_observed = data["past_observed_values"][:, -self.context_length :]
        
        # Scaler expects (batch, time) or (batch, time, 1)
        # Ensure context is (batch, time) for scaler
        if context.ndim == 3 and context.shape[-1] == 1:
            context_squeezed = context.squeeze(-1)
            context_observed_squeezed = context_observed.squeeze(-1)
        else: # Assuming it's already (batch, time)
            context_squeezed = context
            context_observed_squeezed = context_observed

        if self.normalization == "zscore": # This was not in TSDiffBase, but in some of your examples
            # zscore scaling typically needs stats (mean, std) computed over training set
            # For now, let's assume MeanScaler or NOPScaler as in TSDiffBase
            # If you have precomputed stats in data["stats"]:
            # scaled_context_squeezed, scale = self.scaler(context_squeezed, context_observed_squeezed, data["stats"])
            raise NotImplementedError("Z-score scaling with precomputed stats needs careful implementation here.")
        else: # MeanScaler or NOPScaler
            scaled_context_squeezed, scale = self.scaler(context_squeezed, context_observed_squeezed)
        
        # Reshape scale and scaled_context back if needed
        scaled_context = scaled_context_squeezed.unsqueeze(-1) # (batch, context_len, 1)
        # Scale should be (batch, 1, 1) for broadcasting with (batch, time, 1)
        if scale.ndim == 2 and scale.shape[-1] == 1: # (batch, 1) from scaler
            scale_params_for_ts = scale.unsqueeze(-1) # (batch, 1, 1)
        elif scale.ndim == 1: # (batch,) if keepdim=False was used in scaler
            scale_params_for_ts = scale.unsqueeze(-1).unsqueeze(-1) # (batch, 1, 1)
        else: # Assume it's already (batch, 1, 1) or compatible
            scale_params_for_ts = scale


        # Prepare features for the backbone
        features_for_backbone = []
        if self.use_features:
            static_feats_list = []
            if data["feat_static_cat"] is not None and self.num_feat_static_cat > 0:
                static_feats_list.append(self.embedder(data["feat_static_cat"]))
            if data["feat_static_real"] is not None and self.num_feat_static_real > 0:
                static_feats_list.append(data["feat_static_real"])
            
            # Add log_scale as a static real feature
            static_feats_list.append(scale_params_for_ts.log().squeeze(-1)) # Squeeze last dim if it was (B,1,1) -> (B,1)

            if static_feats_list:
                static_feat_combined = torch.cat(static_feats_list, dim=1)
                # Expand to match time dimension: (B, num_static_features) -> (B, T, num_static_features)
                expanded_static_feat = static_feat_combined.unsqueeze(1).expand(
                    -1, self.context_length + self.prediction_length, -1
                )
                features_for_backbone.append(expanded_static_feat)

            time_feats_list = []
            if data["past_time_feat"] is not None: # (B, T_past, num_time_features)
                time_feats_list.append(data["past_time_feat"][:, -self.context_length:])
            if data["future_time_feat"] is not None: # (B, T_future, num_time_features)
                time_feats_list.append(data["future_time_feat"])
            
            if time_feats_list:
                time_features_combined = torch.cat(time_feats_list, dim=1) # (B, T_total, num_time_features)
                features_for_backbone.append(time_features_combined)
        
        # Combine all features if any
        final_features_for_backbone = torch.cat(features_for_backbone, dim=-1) if features_for_backbone else None


        # Prepare x (target + lags)
        # Ensure future_target is (batch, pred_len, 1) for single variate
        future_target_squeezed = data["future_target"]
        if future_target_squeezed.ndim == 3 and future_target_squeezed.shape[-1] == 1:
            future_target_squeezed = future_target_squeezed.squeeze(-1)

        scaled_prior = prior.squeeze(-1) / scale.squeeze(-1) if prior.ndim ==3 else prior / scale.squeeze(-1) # (B, T_prior)
        scaled_future = future_target_squeezed / scale.squeeze(-1) # (B, T_pred)

        # x_unlagged is the time series itself (scaled context + scaled future)
        x_unlagged = torch.cat([scaled_context, scaled_future.unsqueeze(-1)], dim=1) # (B, T_total, 1)

        if self.use_lags and self.lags_seq:
            # lagged_sequence_values expects target for lags to be 2D (batch, time)
            # scaled_prior is (B, T_prior), scaled_context_squeezed is (B, T_context), scaled_future is (B, T_pred)
            lags = lagged_sequence_values(
                self.lags_seq,
                scaled_prior, # Prior for initial lags
                torch.cat([scaled_context_squeezed, scaled_future], dim=1), # Full series for lag calculation
                dim=1, # Time dimension
            ) # lags shape: (B, T_total, num_lags)
            
            # Concatenate target with its lags: (B, T_total, 1 + num_lags)
            x_for_backbone = torch.cat([x_unlagged, lags], dim=-1)
        else:
            x_for_backbone = x_unlagged # (B, T_total, 1)
        
        # scale_params_for_ts should be (B,1,1) to correctly unscale (B,T,1) or (B,num_samples,T)
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