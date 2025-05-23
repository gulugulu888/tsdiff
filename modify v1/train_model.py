# bin/train_model.py
import logging
import argparse
from pathlib import Path
import sys 
import yaml # For loading config
import torch
from tqdm.auto import tqdm # For progress bars

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar 
# from pytorch_lightning.profilers import PyTorchProfiler # Optional for profiling

from gluonts.dataset.loader import TrainDataLoader, ValidationDataLoader # ValidationDataLoader might be needed if use_validation_set=True and EvaluateCallback is used
from gluonts.dataset.split import OffsetSplitter # Used if setup is 'missing_values' or for custom val split
from gluonts.itertools import Cached
from gluonts.torch.batchify import batchify # For collating batches
from gluonts.evaluation import make_evaluation_predictions, Evaluator # For final evaluation
from gluonts.dataset.field_names import FieldName # For MaskInput if used

# TSDiff specific imports
import uncond_ts_diff.configs as diffusion_configs # For diffusion_small_config etc.
from uncond_ts_diff.dataset import get_gts_dataset # Your modified version
from uncond_ts_diff.model.callback import EvaluateCallback # If using validation callback
from uncond_ts_diff.model import TSDiff # Your modified version
from uncond_ts_diff.sampler import DDPMGuidance, DDIMGuidance # For final evaluation
from uncond_ts_diff.utils import (
    create_transforms, 
    create_splitter, 
    add_config_to_argparser, # Utility for overriding config with CLI args
    filter_metrics, # For final evaluation
    MaskInput, # If handling missing values
)

# Setup global logger for the script
logger = logging.getLogger(__name__) # Use __name__ for module-level logger

# --- Model Creation ---
def create_model(config: dict) -> TSDiff:
    """Instantiates the TSDiff model based on the configuration."""
    logger.info("Creating TSDiff model instance...")
    # For single variate (Phase 1), num_input_variates is implicitly 1
    # For Phase 2 (multi-variate), you would add num_input_variates to config and pass it here.
    # num_input_variates = config.get("num_input_variates", 1)

    model = TSDiff(
        backbone_parameters=getattr(diffusion_configs, config["diffusion_config"])["backbone_parameters"],
        timesteps=getattr(diffusion_configs, config["diffusion_config"])["timesteps"],
        diffusion_scheduler=getattr(diffusion_configs, config["diffusion_config"])["diffusion_scheduler"],
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        num_feat_dynamic_real=config.get("num_feat_dynamic_real", 0), # Get from config or default
        num_feat_static_cat=config.get("num_feat_static_cat", 0),
        num_feat_static_real=config.get("num_feat_static_real", 0),
        cardinalities=config.get("cardinalities", None), # TSDiffBase handles None
        freq=config["freq"],
        normalization=config["normalization"],
        use_features=config["use_features"],
        use_lags=config["use_lags"],
        init_skip=config["init_skip"],
        lr=config["lr"],
        # num_input_variates=num_input_variates, # For Phase 2
    )
    logger.info("TSDiff model created.")
    return model

# --- Evaluation (typically for end-of-training or separate script) ---
# This is a simplified version of the evaluate_guidance from the original train_model.py
# It's often better to have evaluation in a separate script (like guidance_experiment.py)
def run_final_evaluation(
    config: dict, 
    model: TSDiff, 
    test_dataset_gluonts, # This should be dataset.test from TrainDatasets
    transformation # The same transformation used for training/validation data
):
    logger.info(f"Running final evaluation with {config.get('num_samples', 100)} samples.") # Use num_samples from config if present
    
    # Determine sampler based on config
    guidance_sampler_name = config.get("sampler", "ddpm") # Default to ddpm if not specified
    sampler_params = config.get("sampler_params", {"guidance": "quantile", "scale": 4.0}) # Default params
    
    GuidanceClass = DDPMGuidance if guidance_sampler_name == "ddpm" else DDIMGuidance
    
    # For standard forecasting, no missing data scenario is applied here unless specified
    # The 'missing_data_configs' in train_tsdiff.yaml is for evaluating robustness to missing data,
    # not typically for the primary forecasting task's final eval unless that's the goal.
    # Here, we assume a standard forecasting setup for the final eval.
    
    sampler_instance = GuidanceClass(
        model=model,
        prediction_length=config["prediction_length"],
        num_samples=config.get("num_samples", 100), # Number of samples for evaluation
        **sampler_params # Pass sampler-specific params like scale, eta, skip_factor
    )

    # Test splitter for evaluation instances
    test_splitter = create_splitter(
        past_length=config["context_length"] + max(model.lags_seq if model.lags_seq else [0]),
        future_length=config["prediction_length"],
        mode="test", # Ensures it takes data from the end of series
    )
    
    # Apply transformations to the test dataset
    # Note: test_dataset_gluonts is dataset.test, which is already a GluonTS dataset object
    transformed_testdata = transformation.apply(test_dataset_gluonts, is_train=False) 
    # No MaskInput here unless specifically evaluating missing data imputation on test set

    predictor = sampler_instance.get_predictor(
        input_transform=test_splitter, # Apply splitter after general transforms
        batch_size=config.get("eval_batch_size", max(1, 1280 // config.get('num_samples',100))), # Adjust batch size for eval
        device=config["device"],
    )

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=transformed_testdata, # Pass the transformed test data
        predictor=predictor,
        num_samples=config.get('num_samples', 100),
    )
    
    logger.info("Generating forecasts for evaluation...")
    forecasts = list(tqdm(forecast_it, total=len(transformed_testdata)))
    tss = list(ts_it) # Ground truth series
    
    evaluator = Evaluator() # Default quantiles [0.1, ..., 0.9]
    metrics, _ = evaluator(iter(tss), iter(forecasts)) # Pass iterators
    
    # Filter and log desired metrics
    metrics_to_log = filter_metrics(metrics, select={"ND", "NRMSE", "mean_wQuantileLoss", "MSE"}) # Added MSE
    logger.info(f"Final Evaluation Metrics: {metrics_to_log}")
    return metrics_to_log


# --- Main Training Function ---
def main(config: dict, log_dir: str, cli_args: argparse.Namespace): # Pass cli_args for config path
    dataset_path = config["dataset"]
    freq = config["freq"]
    prediction_length = config["prediction_length"]
    context_length = config["context_length"]

    logger.info(f"Attempting to load dataset from: {dataset_path}")
    logger.info(f"Expected frequency from config: {freq}")
    logger.info(f"Expected prediction length from config: {prediction_length}")

    dataset_gluonts = get_gts_dataset(dataset_path, config_freq=freq, config_prediction_length=prediction_length)
    
    if dataset_gluonts is None:
        logger.error(f"Failed to load dataset '{dataset_path}'. Exiting.")
        sys.exit(1)

    logger.info(f"Dataset loaded. Metadata freq: {dataset_gluonts.metadata.freq}, Metadata pred_len: {dataset_gluonts.metadata.prediction_length}")

    if dataset_gluonts.metadata.freq != freq:
        logger.error(
            f"FATAL: Loaded dataset's metadata frequency ('{dataset_gluonts.metadata.freq}') "
            f"does not match training configuration frequency ('{freq}').\n"
            f"Ensure data in '{dataset_path}' (check its metadata.json) was prepared with freq '{freq}', "
            f"OR update training config YAML ('{cli_args.config}') to set 'freq: \"{dataset_gluonts.metadata.freq}\"'.")
        sys.exit(1)
    
    if dataset_gluonts.metadata.prediction_length != prediction_length:
        logger.warning(
            f"Loaded dataset metadata prediction length ({dataset_gluonts.metadata.prediction_length}) "
            f"differs from config prediction length ({prediction_length}). "
            f"Model will use prediction length from config: {prediction_length}."
        )
    
    model = create_model(config)

    # Data for training loop
    if config["setup"] == "forecasting":
        training_data_for_loader = dataset_gluonts.train
    elif config["setup"] == "missing_values": # Not our primary use case now
        total_len_for_offset = context_length + prediction_length
        missing_values_splitter = OffsetSplitter(offset=-total_len_for_offset)
        training_data_for_loader, _ = missing_values_splitter.split(dataset_gluonts.train)
    else:
        logger.error(f"Unknown setup type: {config['setup']}"); sys.exit(1)

    # Transformations
    # num_feat_dynamic_real etc. are 0 for single-variate if not using external features.
    # model.time_features is initialized in TSDiffBase based on freq.
    transformation = create_transforms(
        num_feat_dynamic_real=config.get("num_feat_dynamic_real", 0),
        num_feat_static_cat=config.get("num_feat_static_cat", 0),
        num_feat_static_real=config.get("num_feat_static_real", 0),
        time_features=model.time_features, 
        prediction_length=prediction_length,
    )

    training_splitter = create_splitter(
        past_length=context_length + max(model.lags_seq if model.lags_seq else [0]),
        future_length=prediction_length,
        mode="train",
    )
    
    # Apply general transformations first, then the instance splitter for training
    transformed_training_data = transformation.apply(training_data_for_loader, is_train=True)

    train_dataloader = TrainDataLoader(
        Cached(transformed_training_data), # Cache transformed data for efficiency
        batch_size=config["batch_size"],
        stack_fn=batchify, 
        transform=training_splitter, # InstanceSplitter applied per batch fetch
        num_batches_per_epoch=config["num_batches_per_epoch"],
    )

    # Callbacks
    callbacks = []
    # Validation data and callback (optional but recommended)
    val_dataloader = None
    if config.get("use_validation_set", False): # Check if validation is enabled in config
        logger.info("Setting up validation using EvaluateCallback.")
        # EvaluateCallback needs a validation dataset. We can split from train or use dataset.test
        # For simplicity, let's assume EvaluateCallback can handle dataset.test or a split.
        # The original EvaluateCallback in TSDiff might expect a specific val_dataset format.
        # If dataset_gluonts.test is substantial, it can be used.
        # Or, split from training_data_for_loader if test set is for final holdout.
        
        # This is a simplified setup for validation. The original EvaluateCallback might need more.
        # It's often better to use PyTorch Lightning's built-in validation loop if possible.
        # For now, let's assume EvaluateCallback is set up as in the original repo if used.
        # If using PL's val loop, you'd define `val_dataloader` similarly to `train_dataloader`
        # but with mode="val" in create_splitter and using a validation split of data.
        
        # Example for PL validation loop (if EvaluateCallback is not used or adapted)
        # _, val_data_split = OffsetSplitter(offset=-prediction_length * 5).split(dataset_gluonts.train) # Take last 5*PL windows for val
        # transformed_val_data = transformation.apply(val_data_split, is_train=False) # is_train=False for val
        # val_splitter = create_splitter(
        #    past_length=context_length + max(model.lags_seq if model.lags_seq else [0]),
        #    future_length=prediction_length,
        #    mode="val", # ValidationSplitSampler
        # )
        # val_dataloader = ValidationDataLoader(
        #    Cached(transformed_val_data),
        #    batch_size=config["batch_size"], # Can use a larger batch size for validation
        #    stack_fn=batchify,
        #    transform=val_splitter,
        # )
        # logger.info(f"Validation dataloader configured with {len(val_data_split) if val_data_split else 0} series.")
        
        # If using the TSDiff's EvaluateCallback:
        if config.get("use_evaluate_callback", True): # Add a flag to control this
            num_rolling_evals = 5 # Example: number of forecast windows from the end of each val series
            train_val_splitter = OffsetSplitter(offset=-config["prediction_length"] * num_rolling_evals)
            _, val_gen = train_val_splitter.split(dataset_gluonts.train) # Split from train
            
            # ConcatDataset was a custom class in TSDiff utils, ensure it's available or adapt
            # For simplicity, let's assume val_gen can be directly used if EvaluateCallback supports it
            # Or convert val_gen to a ListDataset
            val_dataset_for_callback = list(val_gen.generate_instances(config["prediction_length"], num_rolling_evals))


            callbacks.append(EvaluateCallback(
                context_length=config["context_length"],
                prediction_length=config["prediction_length"],
                sampler=config["sampler"], # Sampler type for validation forecasts
                sampler_kwargs=config["sampler_params"],
                num_samples=config.get("num_samples_val_callback", 16), # num_samples for val callback
                model=model, # Pass the model instance
                transformation=transformation, # Pass general transformations
                # test_dataset=dataset_gluonts.test, # Pass actual test set for callback to also eval on test
                val_dataset=val_dataset_for_callback, # Pass validation data
                eval_every=config["eval_every"],
            ))
            logger.info("EvaluateCallback added for validation.")
        else: # Use PyTorch Lightning's own validation loop
            # Prepare val_dataloader as shown above (commented out section)
            # And ensure your TSDiff model has `validation_step` and `validation_epoch_end` methods.
            # The TSDiffBase already has these.
            logger.info("Using PyTorch Lightning's default validation loop (ensure val_dataloader is set up if needed).")


    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        save_top_k=config.get("save_top_k_checkpoints", 3),
        monitor="train_loss", # Or "valid_loss" if PL validation loop is robustly configured
        mode="min",
        filename=f"{Path(dataset_path).name}"+"-{epoch:03d}-{train_loss:.3f}", # Dynamic filename
        save_last=True,
        save_weights_only=True, # Save only weights to reduce size
        dirpath=Path(log_dir) / "checkpoints" # Save checkpoints in a subfolder
    )
    callbacks.append(checkpoint_callback)
    callbacks.append(RichProgressBar()) # For a nice progress bar

    # PyTorch Lightning Trainer
    trainer = pl.Trainer(
        accelerator="gpu" if config["device"].startswith("cuda") else "cpu",
        devices=[int(config["device"].split(":")[-1])] if config["device"].startswith("cuda") else "auto",
        max_epochs=config["max_epochs"],
        enable_progress_bar=True, # RichProgressBar handles this
        num_sanity_val_steps=0, # Disable sanity check for faster start
        callbacks=callbacks,
        default_root_dir=log_dir, # For logs and checkpoints
        gradient_clip_val=config.get("gradient_clip_val", None),
        check_val_every_n_epoch=config.get("eval_every", 1) if val_dataloader or any(isinstance(cb, EvaluateCallback) for cb in callbacks) else 1e6, # Validate only if val setup
        # profiler="pytorch" if config.get("profile", False) else None, # Optional profiler
    )
    
    logger.info(f"Starting training. Logging to {trainer.logger.log_dir}")
    trainer.fit(
        model, 
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader # Pass val_dataloader if using PL's val loop
    )
    logger.info("Training completed.")

    # Load best model for final evaluation (if enabled)
    best_model_path_from_callback = checkpoint_callback.best_model_path
    best_ckpt_for_final_eval = Path(trainer.logger.log_dir) / "checkpoints" / "best_checkpoint.ckpt" # Consistent name

    if best_model_path_from_callback and Path(best_model_path_from_callback).exists():
        logger.info(f"Best checkpoint path from callback: {best_model_path_from_callback}")
        # Create a 'best_checkpoint.ckpt' symlink or copy for easier access
        # Note: ModelCheckpoint might already save a 'last.ckpt'. 'best_model_path' is more reliable.
        # For simplicity, we'll just load from best_model_path_from_callback
        # Or, if you want a fixed name:
        if not best_ckpt_for_final_eval.exists() or \
           Path(best_model_path_from_callback).stat().st_mtime > best_ckpt_for_final_eval.stat().st_mtime :
            logger.info(f"Saving best model state_dict to {best_ckpt_for_final_eval}")
            torch.save(
                torch.load(best_model_path_from_callback)["state_dict"], # Save only state_dict
                best_ckpt_for_final_eval,
            )
    elif checkpoint_callback.last_model_path and Path(checkpoint_callback.last_model_path).exists():
        logger.warning(f"No 'best' model checkpoint found, using 'last' model checkpoint: {checkpoint_callback.last_model_path}")
        best_model_path_from_callback = checkpoint_callback.last_model_path
        torch.save(
            torch.load(best_model_path_from_callback)["state_dict"],
            best_ckpt_for_final_eval,
        )
    else:
        logger.error("No best or last model checkpoint found. Cannot perform final evaluation.")
        best_ckpt_for_final_eval = None


    if best_ckpt_for_final_eval and best_ckpt_for_final_eval.exists() and config.get("do_final_eval", True):
        logger.info(f"Loading best model from {best_ckpt_for_final_eval} for final evaluation.")
        # Re-create model and load state dict (safer than reusing trainer.model)
        final_model = create_model(config)
        final_model.load_state_dict(torch.load(best_ckpt_for_final_eval))
        final_model.to(config["device"])
        final_model.eval()

        final_metrics = run_final_evaluation(config, final_model, dataset_gluonts.test, transformation)
        metrics_save_path = Path(trainer.logger.log_dir) / "final_evaluation_metrics.json"
        with open(metrics_save_path, "w") as fp:
            json.dump({"config_used": config, "metrics": final_metrics}, fp, indent=4)
        logger.info(f"Final evaluation metrics saved to {metrics_save_path}")
    else:
        logger.info("Final evaluation not performed or no checkpoint found.")

    logger.info(f"All results, logs, and checkpoints saved in: {trainer.logger.log_dir}")


# --- Script Entry Point ---
if __name__ == "__main__":
    # Setup basic logging configuration for the script
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout) # Ensure logs go to console
        ]
    )

    parser = argparse.ArgumentParser(description="Train TSDiff model.")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--out_dir", type=str, default="./lightning_logs_tsdiff", help="Base directory for logs and checkpoints."
    )
    # Add other CLI arguments that can override config values if needed
    # Example: parser.add_argument("--lr", type=float, help="Override learning rate.")
    
    args = parser.parse_args()

    # Load YAML configuration
    try:
        with open(args.config, "r") as fp:
            config_from_yaml = yaml.safe_load(fp)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration file {args.config}: {e}")
        sys.exit(1)

    # --- Override config with CLI arguments if provided (example) ---
    # if args.lr is not None:
    #     config_from_yaml["lr"] = args.lr
    # You can use add_config_to_argparser utility here if you have many such overrides

    # Set up output directory for this specific run
    # Using a timestamp or versioning for unique log_dir is handled by PyTorch Lightning's logger
    # default_root_dir in Trainer will create version_X subdirectories.
    
    # Call main training function
    main(config=config_from_yaml, log_dir=args.out_dir, cli_args=args)