# custom_scripts/predict_and_evaluate_fdr.py
import argparse
import json
import logging
from pathlib import Path
import sys

# --- 添加以下导入 ---
from typing import Optional, Dict, Tuple, List, Any # 确保所有类型提示都被导入
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml 
from tqdm.auto import tqdm

from gluonts.dataset.common import ListDataset
from gluonts.dataset.jsonl import JsonLinesFile
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.torch.model.predictor import PyTorchPredictor 
from gluonts.dataset.field_names import FieldName # <--- 导入 FieldName
# --- 结束添加 ---


# Ensure src directory is in Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# TSDiff specific imports
import uncond_ts_diff.configs as diffusion_configs
from uncond_ts_diff.model import TSDiff 
from uncond_ts_diff.sampler import DDPMGuidance, DDIMGuidance 
from uncond_ts_diff.utils import create_transforms, create_splitter, filter_metrics

logger = logging.getLogger(__name__)

def load_config_for_prediction(config_path: Path) -> dict:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if "device" not in config: 
            config["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
        return config
    except Exception as e:
        logger.error(f"Error loading prediction config from {config_path}: {e}")
        raise

def load_trained_model(config: dict, ckpt_path: Path) -> TSDiff:
    logger.info(f"Loading trained model from checkpoint: {ckpt_path}")
    try:
        diffusion_model_config = getattr(diffusion_configs, config["diffusion_config"])
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
            lr=config.get("lr", 1e-3), 
        )
        
        checkpoint = torch.load(ckpt_path, map_location=config["device"])
        if 'state_dict' in checkpoint:
            state_dict = {k.replace("model.", "", 1) if k.startswith("model.") else k: v 
                          for k, v in checkpoint['state_dict'].items()}
            model.load_state_dict(state_dict)
        else: 
            model.load_state_dict(checkpoint)
            
        model.to(config["device"])
        model.eval() 
        logger.info("Trained model loaded successfully and set to evaluation mode.")
        return model
    except Exception as e:
        logger.error(f"Error loading trained model from {ckpt_path}: {e}")
        raise

def load_jsonl_test_dataset(dataset_path: Path, freq: str, prediction_length: int) -> ListDataset:
    logger.info(f"Loading test dataset from: {dataset_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Test data file not found: {dataset_path}")
    
    data_entries = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            entry[FieldName.START] = pd.Period(entry[FieldName.START], freq=freq)
            entry[FieldName.TARGET] = np.array(entry[FieldName.TARGET], dtype=np.float32)
            data_entries.append(entry)
    
    if not data_entries:
        logger.warning(f"No data entries loaded from {dataset_path}.")
    
    return ListDataset(data_entries, freq=freq)


def generate_forecasts_and_evaluate(
    model: TSDiff,
    test_gluonts_dataset: ListDataset, 
    config: dict,
    num_samples_forecast: int = 100,
) -> Tuple[List[Any], Dict[str, float], List[pd.DataFrame], pd.DataFrame]: # item_metrics is DataFrame
    
    logger.info(f"Generating forecasts with {num_samples_forecast} samples per series...")

    transformation_pipeline = create_transforms(
        num_feat_dynamic_real=config.get("num_feat_dynamic_real", 0),
        num_feat_static_cat=config.get("num_feat_static_cat", 0),
        num_feat_static_real=config.get("num_feat_static_real", 0),
        time_features=model.time_features, 
        prediction_length=config["prediction_length"],
        freq_str=config["freq"]
    )
    
    transformed_testdata = transformation_pipeline.apply(test_gluonts_dataset, is_train=False)

    guidance_sampler_name = config.get("sampler", "ddpm") 
    sampler_params_config = config.get("sampler_params", {"guidance": "quantile", "scale": 4.0})
    GuidanceClass = DDPMGuidance if guidance_sampler_name == "ddpm" else DDIMGuidance
    
    sampler_instance = GuidanceClass(
        model=model, 
        prediction_length=config["prediction_length"],
        num_samples=num_samples_forecast, 
        **sampler_params_config
    )

    prediction_instance_splitter = create_splitter(
        past_length=config["context_length"] + max(model.lags_seq if model.lags_seq else [0]),
        future_length=config["prediction_length"],
        mode="test", 
    )

    # Note: PyTorchPredictorWGrads was from your sampler code, if not using it, use PyTorchPredictor
    # For this script, standard PyTorchPredictor should be fine as we don't need grads during inference here.
    predictor = sampler_instance.get_predictor( # This returns PyTorchPredictorWGrads
        input_transform=prediction_instance_splitter, 
        batch_size=config.get("eval_batch_size", max(1, 256 // num_samples_forecast)), 
        device=config["device"],
    )

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=transformed_testdata, 
        predictor=predictor,
        num_samples=num_samples_forecast, 
    )
    
    logger.info("Collecting forecasts and ground truth series...")
    forecast_objects = list(tqdm(forecast_it, total=len(transformed_testdata) if hasattr(transformed_testdata, "__len__") else None))
    ground_truth_series_dfs = list(ts_it) 

    logger.info("Calculating evaluation metrics...")
    evaluator = Evaluator(quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]) 
    agg_metrics, item_metrics_df = evaluator(iter(ground_truth_series_dfs), iter(forecast_objects)) # item_metrics is a DataFrame
    
    metrics_to_log = filter_metrics(agg_metrics, select={"ND", "NRMSE", "MSE", "abs_error", "mean_wQuantileLoss"})
    metrics_to_log["CRPS"] = agg_metrics.get("mean_wQuantileLoss", float('nan')) 
    
    print("\n--- Aggregate Evaluation Metrics ---")
    for name, value in metrics_to_log.items():
        print(f"  {name}: {value:.4f}")
    
    return forecast_objects, metrics_to_log, ground_truth_series_dfs, item_metrics_df


def plot_sample_forecasts(
    ground_truth_dfs: List[pd.DataFrame],
    forecast_objects: List[Any], 
    config: dict,
    output_dir: Path,
    num_series_to_plot: int = 5,
    num_samples_to_plot_in_forecast: int = 20 
):
    logger.info(f"Generating plots for the first {num_series_to_plot} series...")
    output_dir.mkdir(parents=True, exist_ok=True)

    num_to_plot = min(num_series_to_plot, len(ground_truth_dfs), len(forecast_objects))

    for i in range(num_to_plot):
        ts_df = ground_truth_dfs[i] 
        forecast = forecast_objects[i] 

        plt.figure(figsize=(15, 7)) # Wider plot
        
        if isinstance(ts_df, pd.DataFrame):
            ts_values = ts_df.iloc[:, 0] if not ts_df.empty else pd.Series([])
        else: 
            ts_values = ts_df if isinstance(ts_df, pd.Series) else pd.Series([])
        
        if ts_values.empty:
            logger.warning(f"Ground truth series {i} is empty. Skipping plot.")
            plt.close()
            continue

        # Plot entire ground truth available (context + future)
        plt.plot(ts_values.index.to_timestamp(), ts_values.values, label="Ground Truth", color="blue", linewidth=1)

        # Plot forecast samples
        if hasattr(forecast, 'samples') and forecast.samples is not None and forecast.samples.shape[0] > 0:
            # Ensure forecast_samples is 2D (num_prediction_samples, prediction_length)
            forecast_samples_raw = forecast.samples
            if forecast_samples_raw.ndim == 3 and forecast_samples_raw.shape[-1] == 1: # (num_pred_samples, pred_len, 1)
                forecast_samples_for_plot = forecast_samples_raw[:, :, 0]
            elif forecast_samples_raw.ndim == 2: # (num_pred_samples, pred_len)
                forecast_samples_for_plot = forecast_samples_raw
            else:
                logger.warning(f"Unexpected forecast.samples shape for series {i}: {forecast_samples_raw.shape}. Skipping sample paths.")
                forecast_samples_for_plot = np.array([])


            forecast_start_time = forecast.start_date 
            # Ensure freq is valid for period_range
            try:
                current_freq = forecast.freq if hasattr(forecast, 'freq') and forecast.freq else config["freq"]
                pd.Period(forecast_start_time, freq=current_freq) # Test if freq is valid with start_date
                forecast_index = pd.period_range(
                    start=forecast_start_time, 
                    periods=config["prediction_length"], 
                    freq=current_freq
                )
            except ValueError as e_freq:
                logger.warning(f"Invalid frequency '{config['freq']}' for pd.period_range with start_date '{forecast_start_time}' for series {i}. Error: {e_freq}. Skipping forecast plot.")
                plt.close()
                continue
            
            if forecast_samples_for_plot.shape[1] != config["prediction_length"]:
                 logger.warning(f"Forecast samples length ({forecast_samples_for_plot.shape[1]}) does not match prediction length ({config['prediction_length']}) for series {i}. Skipping sample paths.")
            else:
                for j in range(min(num_samples_to_plot_in_forecast, forecast_samples_for_plot.shape[0])):
                    plt.plot(forecast_index.to_timestamp(), forecast_samples_for_plot[j], color="lightcoral", alpha=0.2, linewidth=0.5)
            
                median_forecast = np.median(forecast_samples_for_plot, axis=0)
                plt.plot(forecast_index.to_timestamp(), median_forecast, color="red", linewidth=1.5, label=f"Median Forecast")
        else:
            logger.warning(f"No samples found in forecast object for series {i}.")

        
        plt.title(f"Forecast vs Ground Truth - Item: {forecast.item_id if hasattr(forecast, 'item_id') else i+1}", fontsize=14)
        plt.xlabel("Time", fontsize=10)
        plt.ylabel("Value", fontsize=10)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout() # Adjust layout
        
        plot_save_path = output_dir / f"forecast_series_{forecast.item_id if hasattr(forecast, 'item_id') else i+1}.png"
        try:
            plt.savefig(plot_save_path)
        except Exception as e_save:
            logger.error(f"Error saving plot {plot_save_path}: {e_save}")
        plt.close()
        logger.info(f"Saved plot to {plot_save_path}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    parser = argparse.ArgumentParser(description="Predict and evaluate using a trained TSDiff model.")
    parser.add_argument("-c", "--config_file", type=str, required=True, help="Path to the YAML training configuration file used for the model.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the trained model checkpoint (.ckpt or .pth).")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test data JSON Lines file (e.g., test.jsonl).")
    parser.add_argument("--output_dir", type=str, default="./fdr_predictions_eval", help="Directory to save prediction results and plots.")
    parser.add_argument("--num_forecast_samples", type=int, default=100, help="Number of sample paths to generate for each forecast.")
    parser.add_argument("--num_plot_series", type=int, default=5, help="Number of time series to plot forecasts for.")

    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config = load_config_for_prediction(Path(args.config_file))
    logger.info(f"Configuration loaded from {args.config_file}")

    model = load_trained_model(config, Path(args.ckpt_path))

    test_dataset = load_jsonl_test_dataset(Path(args.test_data_path), config["freq"], config["prediction_length"])
    if not test_dataset or not hasattr(test_dataset, 'list_data') or not test_dataset.list_data : 
        logger.error("Test dataset is empty or could not be loaded. Exiting.")
        return

    forecast_objects, agg_metrics, ground_truth_dfs, item_metrics_df = generate_forecasts_and_evaluate(
        model, test_dataset, config, num_samples_forecast=args.num_forecast_samples
    )

    results_summary = {
        "config_file_used": args.config_file,
        "checkpoint_path_used": args.ckpt_path,
        "test_data_path_used": args.test_data_path,
        "aggregate_metrics": agg_metrics,
    }
    summary_save_path = output_path / "prediction_summary_metrics.json"
    with open(summary_save_path, 'w', encoding='utf-8') as f:
        # Custom encoder for Path and numpy types
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, Path): return str(obj)
                if isinstance(obj, (np.integer)): return int(obj)
                if isinstance(obj, (np.floating)): return float(obj)
                if isinstance(obj, (np.bool_)): return bool(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                return super(CustomEncoder, self).default(obj)
        json.dump(results_summary, f, indent=4, cls=CustomEncoder)
    logger.info(f"Prediction summary saved to {summary_save_path}")

    # Save item_metrics DataFrame to CSV
    item_metrics_save_path = output_path / "item_metrics.csv"
    try:
        item_metrics_df.to_csv(item_metrics_save_path, index=False)
        logger.info(f"Item-wise metrics saved to {item_metrics_save_path}")
    except Exception as e_csv:
        logger.error(f"Could not save item_metrics to CSV: {e_csv}")


    if args.num_plot_series > 0 and ground_truth_dfs and forecast_objects:
        plot_sample_forecasts(
            ground_truth_dfs, 
            forecast_objects, 
            config, 
            output_path / "plots", 
            num_series_to_plot=args.num_plot_series
        )
    logger.info(f"Prediction and evaluation complete. Results in {output_path}")

if __name__ == "__main__":
    main()