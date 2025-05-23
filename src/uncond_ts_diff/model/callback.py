# src/uncond_ts_diff/model/callback.py
import copy 
import math
from pathlib import Path
import logging 
from typing import List, Dict, Any, Optional # <--- 确保这些被导入

import numpy as np
import torch
import pytorch_lightning as pl 
from pytorch_lightning import Callback 

from gluonts.dataset.field_names import FieldName
from gluonts.evaluation import make_evaluation_predictions, Evaluator
# InstanceSplitter 自身不需要在这里导入，因为它被 create_splitter 内部使用
# from gluonts.transform import TestSplitSampler, InstanceSplitter 
from gluonts.dataset.common import ListDataset 
from gluonts.transform import Chain # <--- 确保 Chain 被导入

# Assuming these are correctly located relative to this file or in PYTHONPATH
from uncond_ts_diff.sampler import DDPMGuidance, DDIMGuidance
from uncond_ts_diff.utils import create_splitter 

logger = logging.getLogger(__name__)


class GradNormCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_before_optimizer_step(
        self,
        trainer: "pl.Trainer", 
        pl_module: "pl.LightningModule",
        optimizer, 
        opt_idx: int,
    ) -> None:
        if hasattr(pl_module, 'log') and callable(pl_module.log):
            pl_module.log(
                "grad_norm", self.grad_norm(pl_module.parameters()), prog_bar=True, logger=True
            )
        else:
            current_grad_norm = self.grad_norm(pl_module.parameters())
            if hasattr(trainer, 'logger') and hasattr(trainer.logger, 'log_metrics'):
                 trainer.logger.log_metrics({"grad_norm_manual": current_grad_norm.item()}, step=trainer.global_step)

    def grad_norm(self, parameters):
        parameters = [p for p in parameters if p.grad is not None]
        if not parameters: 
            return torch.tensor(0.0)
        device = parameters[0].grad.device
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), p=2).to(device) for p in parameters] 
            ),
            p=2, 
        )
        return total_norm


class PredictiveScoreCallback(Callback): 
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        model: pl.LightningModule, # Type hint for PL module
        transformation: Chain, # Type hint for GluonTS Chain
        train_dataloader, 
        train_batch_size: int,
        test_dataset: Any, # Can be any GluonTS dataset
        eval_every: int =10,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.model_lightning_module = model 
        self.transformation = transformation
        self.train_dataloader = train_dataloader
        self.train_batch_size = train_batch_size
        self.test_dataset_gluonts = test_dataset 
        self.eval_every = eval_every
        self.n_pred_samples = 10000 

    def _generate_real_samples(
        self, data_loader, num_samples: int, n_timesteps: int,
        batch_size: int, cache_path: Path,
    ):
        if cache_path.exists():
            try:
                real_samples = np.load(cache_path)
                if len(real_samples) >= num_samples:
                    return real_samples[:num_samples]
            except Exception as e:
                logger.warning(f"Could not load cached real samples from {cache_path}: {e}")
        real_samples_list = []
        num_collected = 0
        iters_since_last_batch = 0 # To prevent infinite loop if dataloader is exhausted early
        max_iters_no_batch = 5 # Arbitrary limit

        while num_collected < num_samples:
            batch_found_in_epoch = False
            for batch in data_loader: 
                batch_found_in_epoch = True
                iters_since_last_batch = 0
                past_target_np = batch["past_target"].cpu().numpy()
                future_target_np = batch["future_target"].cpu().numpy()
                
                # Determine axis for concatenation and slicing based on typical shapes
                # Common shape from GluonTS: (batch_size, sequence_length, num_features_or_1)
                # Or (batch_size, sequence_length) if already squeezed
                ts_concat_axis = -2 # Assumes time is the second to last dimension
                if past_target_np.ndim == 2: # (batch_size, sequence_length)
                    ts_concat_axis = -1

                ts = np.concatenate([past_target_np, future_target_np], axis=ts_concat_axis) 
                
                if ts.ndim == 3 and ts.shape[-1] == 1: 
                    ts = ts[..., 0] 
                
                current_batch_samples = ts[:, -n_timesteps:]
                real_samples_list.append(current_batch_samples)
                num_collected += current_batch_samples.shape[0]
                if num_collected >= num_samples: break
            
            if not batch_found_in_epoch: # Dataloader might be exhausted
                iters_since_last_batch +=1
                if iters_since_last_batch > max_iters_no_batch:
                    logger.warning("PredictiveScoreCallback: Train Dataloader exhausted before collecting enough real samples.")
                    break
            if num_collected >= num_samples: break # Break outer loop too
        
        if not real_samples_list:
            logger.error("PredictiveScoreCallback: No real samples could be generated.")
            return np.array([])
        real_samples_arr = np.concatenate(real_samples_list, axis=0)[:num_samples]
        try: np.save(cache_path, real_samples_arr)
        except Exception as e: logger.warning(f"Could not save real samples to cache {cache_path}: {e}")
        return real_samples_arr

    def _generate_synth_samples(
        self, model_pl_module: pl.LightningModule, num_samples: int, batch_size: int = 1000
    ):
        synth_samples_list = []
        num_generated = 0
        while num_generated < num_samples:
            current_batch_size = min(batch_size, num_samples - num_generated)
            # Assuming sample_n is a method of the pl_module (TSDiff)
            if not hasattr(model_pl_module, 'sample_n'):
                logger.error("PredictiveScoreCallback: Model does not have a 'sample_n' method.")
                return np.array([])
            samples = model_pl_module.sample_n(num_samples=current_batch_size) 
            synth_samples_list.append(samples)
            num_generated += samples.shape[0]
        
        if not synth_samples_list:
            logger.error("PredictiveScoreCallback: No synthetic samples could be generated.")
            return np.array([])
        return np.concatenate(synth_samples_list, axis=0)[:num_samples]

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if (pl_module.current_epoch + 1) % self.eval_every == 0:
            logger.info(f"PredictiveScoreCallback: Running at epoch {pl_module.current_epoch + 1}")
            eval_model = pl_module 
            original_training_state = eval_model.training
            eval_model.eval()

            synth_samples_scaled = self._generate_synth_samples(eval_model, self.n_pred_samples)
            
            if synth_samples_scaled.size == 0:
                logger.error("PredictiveScoreCallback: Synthetic sample generation failed. Skipping LPS calculation.")
                if original_training_state: eval_model.train()
                return

            from uncond_ts_diff.metrics import linear_pred_score 
            
            logger.info("PredictiveScoreCallback: Calculating LPS for synthetic data...")
            try:
                # Ensure model has 'normalization' attribute or get it from hparams
                scaling_type = getattr(eval_model, 'normalization', eval_model.hparams.get('normalization', 'none'))
                synth_metrics, _, _ = linear_pred_score(
                    samples=synth_samples_scaled, 
                    context_length=self.context_length,
                    prediction_length=self.prediction_length,
                    test_dataset=self.test_dataset_gluonts, 
                    scaling_type=scaling_type 
                )
                pl_module.log("LPS_synth_ND", synth_metrics.get("ND", float('nan')), on_epoch=True, logger=True)
                pl_module.log("LPS_synth_NRMSE", synth_metrics.get("NRMSE", float('nan')), on_epoch=True, logger=True)
            except Exception as e_lps:
                logger.error(f"PredictiveScoreCallback: Error during LPS calculation: {e_lps}")
            
            if original_training_state: eval_model.train()


class EvaluateCallback(Callback):
    def __init__(
        self,
        context_length: int,
        prediction_length: int,
        sampler: str, 
        sampler_kwargs: dict,
        num_samples: int,
        model: pl.LightningModule, 
        transformation: Chain,     
        val_dataset: List[Dict[str, Any]], 
        test_dataset: Any, 
        eval_every: int = 50,
    ):
        super().__init__()
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.sampler_name = sampler
        self.num_samples = num_samples
        self.sampler_kwargs = sampler_kwargs
        self.model_lightning_module = model 
        self.transformation = transformation 
        self.val_data_list = val_dataset 
        self.test_dataset_gluonts = test_dataset 
        self.eval_every = eval_every
        self.log_metrics = {"CRPS", "ND", "NRMSE", "MSE"} 

        if self.sampler_name == "ddpm":
            self.GuidanceClass = DDPMGuidance
        elif self.sampler_name == "ddim":
            self.GuidanceClass = DDIMGuidance
        else:
            raise ValueError(f"Unknown sampler type: {self.sampler_name}")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if (pl_module.current_epoch + 1) % self.eval_every == 0:
            logger.info(f"EvaluateCallback: Running evaluation at epoch {pl_module.current_epoch + 1}")
            device = pl_module.device 

            original_training_state = pl_module.training # Save current mode
            pl_module.eval() # Set to eval mode

            original_state_dict = copy.deepcopy(pl_module.backbone.state_dict())
            
            ema_states_to_eval = [("main_model", pl_module.backbone.state_dict())] 
            if hasattr(pl_module, 'ema_rate') and hasattr(pl_module, 'ema_state_dicts') and \
               pl_module.ema_rate and pl_module.ema_state_dicts:
                logger.info("EvaluateCallback: Evaluating EMA models.")
                ema_states_to_eval.extend(
                    [(f"ema_{rate}", state_dict) for rate, state_dict in zip(pl_module.ema_rate, pl_module.ema_state_dicts)]
                )
            
            for model_label, state_dict_to_eval in ema_states_to_eval:
                logger.info(f"EvaluateCallback: Evaluating with {model_label} weights...")
                pl_module.backbone.load_state_dict(state_dict_to_eval, strict=True)
                
                guidance_sampler_instance = self.GuidanceClass(
                    model=pl_module, 
                    prediction_length=self.prediction_length,
                    num_samples=self.num_samples,
                    **self.sampler_kwargs,
                )

                instance_splitter_for_eval = create_splitter( 
                    past_length=self.context_length + max(pl_module.lags_seq if hasattr(pl_module, 'lags_seq') and pl_module.lags_seq else [0]),
                    future_length=self.prediction_length,
                    mode="test", 
                )
                
                predictor_for_eval = guidance_sampler_instance.get_predictor(
                    input_transform=instance_splitter_for_eval, 
                    batch_size=max(1, 1024 // self.num_samples), 
                    device=device,
                )
                evaluator = Evaluator(quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]) 

                if self.val_data_list:
                    logger.info(f"  Evaluating on val_dataset ({len(self.val_data_list)} series)...")
                    # self.val_data_list is List[Dict], self.transformation is Chain
                    # Create a temporary ListDataset to apply transformations
                    temp_val_list_dataset = ListDataset(self.val_data_list, freq=pl_module.hparams.get('freq', None))
                    transformed_valdata_for_eval = self.transformation.apply(temp_val_list_dataset, is_train=False)
                    
                    forecast_it_val, ts_it_val = make_evaluation_predictions(
                        dataset=transformed_valdata_for_eval, 
                        predictor=predictor_for_eval,
                        num_samples=self.num_samples,
                    )
                    forecasts_val = list(forecast_it_val)
                    tss_val = list(ts_it_val)

                    metrics_val, _ = evaluator(iter(tss_val), iter(forecasts_val))
                    metrics_val["CRPS"] = metrics_val["mean_wQuantileLoss"] 
                    
                    for metric_name in self.log_metrics:
                        if metric_name in metrics_val:
                            pl_module.log(f"val_{metric_name}_{model_label}", metrics_val[metric_name], on_epoch=True, prog_bar=True, logger=True)
                    
                    if hasattr(pl_module, 'best_crps') and metrics_val["CRPS"] < pl_module.best_crps and model_label == "main_model":
                        pl_module.best_crps = metrics_val["CRPS"]
                        logger.info(f"  New best val_CRPS for main model: {pl_module.best_crps:.4f}")
                else:
                    logger.info("  val_data_list is empty, skipping validation set evaluation.")

                if self.test_dataset_gluonts:
                    try:
                        # Check if test_dataset_gluonts is non-empty before proceeding
                        # Iterating to check length can be slow for large file datasets.
                        # A better check might be specific to the dataset type or a sample.
                        # For now, let's assume if it's provided, we try to use it.
                        # if len(list(self.test_dataset_gluonts)) == 0: 
                        #      logger.info("  test_dataset_gluonts is empty, skipping test set evaluation.")
                        # else:
                        logger.info(f"  Evaluating on test_dataset...")
                        transformed_testdata_for_eval = self.transformation.apply(self.test_dataset_gluonts, is_train=False)
                        forecast_it_test, ts_it_test = make_evaluation_predictions(
                            dataset=transformed_testdata_for_eval, predictor=predictor_for_eval, num_samples=self.num_samples,
                        )
                        forecasts_test = list(forecast_it_test)
                        tss_test = list(ts_it_test)
                        metrics_test, _ = evaluator(iter(tss_test), iter(forecasts_test))
                        metrics_test["CRPS"] = metrics_test["mean_wQuantileLoss"]
                        for metric_name in self.log_metrics:
                            if metric_name in metrics_test:
                                 pl_module.log(f"test_{metric_name}_{model_label}", metrics_test[metric_name], on_epoch=True, logger=True)
                    except Exception as e_test_eval:
                        logger.error(f"  Error during test set evaluation in EvaluateCallback: {e_test_eval}")
                else:
                    logger.info("  test_dataset_gluonts not provided, skipping test set evaluation.")

            pl_module.backbone.load_state_dict(original_state_dict, strict=True)
            logger.info("EvaluateCallback: Restored original model weights.")
            if original_training_state: pl_module.train() # Restore original training mode