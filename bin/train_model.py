# bin/train_model.py
import logging
import argparse
from pathlib import Path
import sys 
import yaml 
import torch
from tqdm.auto import tqdm 
import numpy as np # Ensure numpy is imported
import json      # Ensure json is imported

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar 

from gluonts.dataset.loader import TrainDataLoader, ValidationDataLoader 
from gluonts.dataset.split import OffsetSplitter 
from gluonts.itertools import Cached
from gluonts.torch.batchify import batchify 
from gluonts.evaluation import make_evaluation_predictions, Evaluator 
from gluonts.dataset.field_names import FieldName 
from gluonts.transform import Chain, AsNumpyArray 

import uncond_ts_diff.configs as diffusion_configs 
from uncond_ts_diff.dataset import get_gts_dataset 
from uncond_ts_diff.model.callback import EvaluateCallback 
from uncond_ts_diff.model import TSDiff 
from uncond_ts_diff.sampler import DDPMGuidance, DDIMGuidance 
from uncond_ts_diff.utils import (
    create_transforms, 
    create_splitter, 
    add_config_to_argparser, 
    filter_metrics, 
    MaskInput, 
)

logger = logging.getLogger(__name__)

def create_model(config: dict) -> TSDiff:
    logger.info("Creating TSDiff model instance...")
    try:
        diffusion_model_config = getattr(diffusion_configs, config["diffusion_config"])
    except AttributeError:
        logger.error(f"Diffusion config '{config['diffusion_config']}' not found in uncond_ts_diff.configs.")
        sys.exit(1)
    model = TSDiff(
        backbone_parameters=diffusion_model_config["backbone_parameters"],
        timesteps=diffusion_model_config["timesteps"],
        diffusion_scheduler=diffusion_model_config["diffusion_scheduler"],
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        num_feat_dynamic_real=config.get("num_feat_dynamic_real", 0),
        num_feat_static_cat=config.get("num_feat_static_cat", 0),
        num_feat_static_real=config.get("num_feat_static_real", 0),
        cardinalities=config.get("cardinalities", None),
        freq=config["freq"],
        normalization=config["normalization"],
        use_features=config.get("use_features", False),
        use_lags=config.get("use_lags", True),
        init_skip=config.get("init_skip", True),
        lr=config["lr"],
    )
    logger.info("TSDiff model created.")
    return model

def run_final_evaluation(
    config: dict, 
    model_to_eval: TSDiff, 
    test_dataset_gluonts, 
    transformation_pipeline
):
    logger.info(f"Running final evaluation with {config.get('num_samples_final_eval', 100)} samples.")
    guidance_sampler_name = config.get("sampler", "ddpm")
    sampler_params_config = config.get("sampler_params", {"guidance": "quantile", "scale": 4.0})
    GuidanceClass = DDPMGuidance if guidance_sampler_name == "ddpm" else DDIMGuidance
    
    sampler_instance = GuidanceClass(
        model=model_to_eval,
        prediction_length=config["prediction_length"],
        num_samples=config.get("num_samples_final_eval", 100),
        **sampler_params_config
    )
    test_splitter = create_splitter(
        past_length=config["context_length"] + max(model_to_eval.lags_seq if model_to_eval.lags_seq else [0]),
        future_length=config["prediction_length"],
        mode="test",
    )
    transformed_testdata = transformation_pipeline.apply(test_dataset_gluonts, is_train=False) 
    predictor = sampler_instance.get_predictor(
        input_transform=test_splitter,
        batch_size=config.get("eval_batch_size", max(1, 1280 // config.get('num_samples_final_eval',100))),
        device=config["device"],
    )
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=transformed_testdata, 
        predictor=predictor,
        num_samples=config.get('num_samples_final_eval', 100),
    )
    logger.info("Generating forecasts for final evaluation...")
    forecasts = list(tqdm(forecast_it, total=len(transformed_testdata) if hasattr(transformed_testdata, "__len__") else None))
    tss = list(ts_it) 
    evaluator = Evaluator() 
    metrics, _ = evaluator(iter(tss), iter(forecasts)) 
    metrics_to_log = filter_metrics(metrics, select={"ND", "NRMSE", "mean_wQuantileLoss", "MSE"})
    logger.info(f"Final Evaluation Metrics: {metrics_to_log}")
    return metrics_to_log

def main(config: dict, log_dir: Path, cli_args: argparse.Namespace):
    dataset_path = config["dataset"]
    freq = config["freq"]
    prediction_length = config["prediction_length"]
    context_length = config["context_length"]

    logger.info(f"Attempting to load dataset from: {dataset_path}")
    logger.info(f"Expected frequency from config: {freq}")
    logger.info(f"Expected prediction length from config: {prediction_length}")

    dataset_gluonts = get_gts_dataset(dataset_path, config_freq=freq, config_prediction_length=prediction_length)
    if dataset_gluonts is None: logger.error(f"Failed to load dataset '{dataset_path}'. Exiting."); sys.exit(1)
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
            f"differs from config prediction length ({prediction_length}). Model will use config: {prediction_length}."
        )
    
    model = create_model(config)

    base_transformation_for_split = Chain([AsNumpyArray(field=FieldName.TARGET, expected_ndim=1, dtype=np.float32)])
    training_data_source_for_loader = None
    data_for_val_split = None

    if config["setup"] == "forecasting":
        training_data_source_for_loader = dataset_gluonts.train
        data_for_val_split = base_transformation_for_split.apply(dataset_gluonts.train, is_train=True)
    elif config["setup"] == "missing_values":
        temp_transformed_train = base_transformation_for_split.apply(dataset_gluonts.train, is_train=True)
        total_len_for_offset = context_length + prediction_length
        missing_values_splitter = OffsetSplitter(offset=-total_len_for_offset)
        training_data_source_for_loader, data_for_val_split = missing_values_splitter.split(temp_transformed_train)
    else:
        logger.error(f"Unknown setup type: {config['setup']}"); sys.exit(1)

    full_transformation_pipeline = create_transforms(
        num_feat_dynamic_real=config.get("num_feat_dynamic_real", 0),
        num_feat_static_cat=config.get("num_feat_static_cat", 0),
        num_feat_static_real=config.get("num_feat_static_real", 0),
        time_features=model.time_features, # 从模型实例获取，TSDiffBase中已根据freq处理
        prediction_length=prediction_length,
        freq_str=config["freq"] # <--- 添加这一行，传递配置文件中的频率字符串
    )
    training_instance_splitter = create_splitter(
        past_length=context_length + max(model.lags_seq if model.lags_seq else [0]),
        future_length=prediction_length, mode="train",
    )
    transformed_training_data_for_loader = full_transformation_pipeline.apply(training_data_source_for_loader, is_train=True)
    train_dataloader = TrainDataLoader(
        Cached(transformed_training_data_for_loader), batch_size=config["batch_size"],
        stack_fn=batchify, transform=training_instance_splitter, 
        num_batches_per_epoch=config["num_batches_per_epoch"],
        shuffle_buffer_length=config.get("shuffle_buffer_length", 1000)
    )

    callbacks_list = []
    pytorch_val_dataloader = None 

    if config.get("use_validation_set", False):
        if config.get("use_evaluate_callback", True): 
            logger.info("Setting up validation using TSDiff's EvaluateCallback.")
            num_train_series = 0
            try: num_train_series = len(list(dataset_gluonts.train))
            except Exception as e_len: logger.warning(f"Could not determine num_train_series for validation split: {e_len}.")
            
            num_val_series_for_callback = config.get("num_val_series_for_callback", max(1, int(num_train_series * 0.1)) if num_train_series > 10 else (1 if num_train_series > 0 else 0) )
            val_series_list_for_callback = []

            if num_train_series > 0:
                all_transformed_training_entries = list(data_for_val_split) # data_for_val_split has target as np.array
                actual_num_val_series = min(num_val_series_for_callback, len(all_transformed_training_entries))
                if actual_num_val_series > 0 :
                    val_series_list_for_callback = all_transformed_training_entries[-actual_num_val_series:]
                    logger.info(f"Using last {len(val_series_list_for_callback)} series from training data for EvaluateCallback's val_dataset.")
                else: logger.warning("Not enough training series for EvaluateCallback validation split.")
            else: logger.warning("Training data source empty, cannot create val_dataset for EvaluateCallback.")

            if val_series_list_for_callback:
                callbacks_list.append(EvaluateCallback(
                    context_length=context_length, prediction_length=prediction_length,
                    sampler=config.get("sampler_val_callback", config.get("sampler","ddpm")), 
                    sampler_kwargs=config.get("sampler_params_val_callback", config.get("sampler_params", {})),
                    num_samples=config.get("num_samples_val_callback", 16), model=model, 
                    transformation=full_transformation_pipeline, 
                    val_dataset=val_series_list_for_callback, 
                    test_dataset=dataset_gluonts.test, 
                    eval_every=config["eval_every"],
                ))
                logger.info(f"EvaluateCallback added with {len(val_series_list_for_callback)} validation series.")
            else: logger.warning("EvaluateCallback not added as no validation series were prepared.")
        else: 
            logger.info("TSDiff's EvaluateCallback is disabled. If using PyTorch Lightning's validation loop, ensure val_dataloader is configured.")
            # Placeholder for PL val_dataloader setup if needed
            # _, val_ds_for_pl_loop_source = OffsetSplitter(...).split(data_for_val_split)
            # ... setup pytorch_val_dataloader ...
            pass

    # --- Corrected checkpoint_monitor logic ---
    checkpoint_monitor = "train_loss_epoch" 
    monitor_mode = "min" 
    
    if pytorch_val_dataloader is not None and not config.get("use_evaluate_callback", True): 
        checkpoint_monitor = "valid_loss_epoch" 
        logger.info(f"ModelCheckpoint will monitor PyTorch Lightning's: {checkpoint_monitor}")
    elif any(isinstance(cb, EvaluateCallback) for cb in callbacks_list) and \
         config.get("use_evaluate_callback", True) and \
         config.get("monitor_evaluate_callback_metric_name"):
        checkpoint_monitor = config["monitor_evaluate_callback_metric_name"]
        monitor_mode = config.get("monitor_evaluate_callback_metric_mode", "min") 
        logger.info(f"ModelCheckpoint configured to monitor EvaluateCallback metric: {checkpoint_monitor} (mode: {monitor_mode})")
    else:
        logger.info(f"ModelCheckpoint will monitor default: {checkpoint_monitor} (mode: {monitor_mode})")
    # --- End corrected logic ---

    run_specific_checkpoint_dir = Path(log_dir) / cli_args.config.split('/')[-1].replace('.yaml','').replace('.yml','').replace('.json','') / "checkpoints"
    run_specific_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=config.get("save_top_k_checkpoints", 3),
        monitor=checkpoint_monitor, 
        mode=monitor_mode,
        filename=f"{Path(dataset_path).stem}"+"-{epoch:03d}-{"+checkpoint_monitor+":.3f}",
        save_last=True, save_weights_only=True, dirpath=run_specific_checkpoint_dir 
    )
    callbacks_list.append(checkpoint_callback)
    
    if not any(isinstance(cb, RichProgressBar) for cb in callbacks_list):
        callbacks_list.append(RichProgressBar()) 
    else: 
        for cb_idx, cb_instance in enumerate(callbacks_list):
            if isinstance(cb_instance, RichProgressBar): callbacks_list.pop(cb_idx); break
        callbacks_list.append(RichProgressBar())

    trainer = pl.Trainer(
        accelerator="gpu" if config["device"].startswith("cuda") else "cpu",
        devices=[int(config["device"].split(":")[-1])] if config["device"].startswith("cuda") and ":" in config["device"] else "auto",
        max_epochs=config["max_epochs"], enable_progress_bar=True, 
        num_sanity_val_steps=0, callbacks=callbacks_list, default_root_dir=log_dir, 
        gradient_clip_val=config.get("gradient_clip_val", None),
        check_val_every_n_epoch=config.get("eval_every", 1) if pytorch_val_dataloader or any(isinstance(cb, EvaluateCallback) for cb in callbacks_list) else int(1e6),
    )
    
    logger.info(f"Starting training. Logging to {trainer.logger.log_dir}")
    try:
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=pytorch_val_dataloader)
        logger.info("Training completed.")
    except Exception as e_fit:
        logger.error(f"Error during trainer.fit: {e_fit}"); import traceback; logger.error(traceback.format_exc()); sys.exit(1)

    best_model_path_from_callback = checkpoint_callback.best_model_path
    final_eval_metrics_result = "Final eval not performed or no valid checkpoint found."
    path_to_load_for_final_eval = None

    if best_model_path_from_callback and Path(best_model_path_from_callback).exists():
        path_to_load_for_final_eval = best_model_path_from_callback
        logger.info(f"Best checkpoint path from callback: {path_to_load_for_final_eval}")
    elif checkpoint_callback.last_model_path and Path(checkpoint_callback.last_model_path).exists():
        path_to_load_for_final_eval = checkpoint_callback.last_model_path
        logger.warning(f"No 'best' model checkpoint found by monitor '{checkpoint_monitor}'. Using 'last' model: {path_to_load_for_final_eval}")
    else:
        logger.error("No best or last model checkpoint found by ModelCheckpoint callback.")

    if path_to_load_for_final_eval and config.get("do_final_eval", True):
        logger.info(f"Loading model from {path_to_load_for_final_eval} for final evaluation.")
        eval_model = create_model(config) 
        try:
            ckpt = torch.load(path_to_load_for_final_eval, map_location=config["device"])
            eval_model.load_state_dict(ckpt['state_dict'])
            eval_model.to(config["device"]); eval_model.eval()
            final_eval_metrics_result = run_final_evaluation(config, eval_model, dataset_gluonts.test, full_transformation_pipeline)
        except Exception as e_eval_load:
            logger.error(f"Error loading/evaluating from {path_to_load_for_final_eval}: {e_eval_load}")
            final_eval_metrics_result = f"Error during final eval: {e_eval_load}"
    else:
        logger.info("Final evaluation skipped.")

    results_summary_path = Path(trainer.logger.log_dir) / "training_run_summary.json"
    summary_data = {
        "config_used_path": cli_args.config, "config_content": config,
        "best_checkpoint_path_from_callback": str(best_model_path_from_callback) if best_model_path_from_callback else None,
        "final_evaluation_metrics": final_eval_metrics_result, "log_directory": str(trainer.logger.log_dir)
    }
    try:
        with open(results_summary_path, "w", encoding="utf-8") as fp:
            class CustomEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, Path): return str(obj)
                    if isinstance(obj, (np.integer)): return int(obj)
                    if isinstance(obj, (np.floating)): return float(obj)
                    if isinstance(obj, (np.bool_)): return bool(obj)
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    if callable(obj): return f"<function {obj.__name__}>"
                    try: return super(CustomEncoder, self).default(obj)
                    except TypeError: return f"<object of type {type(obj).__name__} not serializable>"
            json.dump(summary_data, fp, indent=4, cls=CustomEncoder)
        logger.info(f"Training run summary saved to {results_summary_path}")
    except Exception as e_json:
        logger.error(f"Could not serialize run summary to JSON: {e_json}.")
        with open(Path(trainer.logger.log_dir) / "training_run_summary.txt", 'w', encoding='utf-8') as f:
             f.write(str(summary_data))
    logger.info(f"All results, logs, and checkpoints saved in: {trainer.logger.log_dir}")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)]
    )
    parser = argparse.ArgumentParser(description="Train TSDiff model.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to the YAML configuration file.")
    parser.add_argument("--out_dir", type=str, default="./lightning_runs_tsdiff", help="Base directory for logs and checkpoints.")
    args = parser.parse_args()
    try:
        with open(args.config, "r", encoding="utf-8") as fp: config_from_yaml = yaml.safe_load(fp)
    except FileNotFoundError: logger.error(f"Config file not found: {args.config}"); sys.exit(1)
    except yaml.YAMLError as e: logger.error(f"Error parsing YAML config {args.config}: {e}"); sys.exit(1)
    except Exception as e: logger.error(f"Unexpected error reading config {args.config}: {e}"); sys.exit(1)
    
    if "device" not in config_from_yaml:
        config_from_yaml["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device not specified in config, defaulted to: {config_from_yaml['device']}")
    
    main(config=config_from_yaml, log_dir=Path(args.out_dir), cli_args=args)