# custom_scripts/predict_and_evaluate_fdr.py
import argparse
import json
import logging
from pathlib import Path
import sys
import warnings

from typing import Optional, Dict, Tuple, List, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from tqdm.auto import tqdm
import seaborn as sns

from gluonts.dataset.common import ListDataset
from gluonts.dataset.jsonl import JsonLinesFile
from gluonts.evaluation import Evaluator, make_evaluation_predictions
from gluonts.torch.model.predictor import PyTorchPredictor
from gluonts.dataset.field_names import FieldName
from gluonts.model.forecast import QuantileForecast

# Ensure src directory is in Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# TSDiff specific imports
import uncond_ts_diff.configs as diffusion_configs
from uncond_ts_diff.model import TSDiff
from uncond_ts_diff.sampler import DDPMGuidance, DDIMGuidance
from uncond_ts_diff.utils import create_transforms, create_splitter, filter_metrics

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def load_config_for_prediction(config_path: Path) -> dict:
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if "device" not in config:
            config["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
        logger.info(f"Configuration loaded successfully from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading prediction config from {config_path}: {e}")
        raise

def load_trained_model(config: dict, ckpt_path: Path) -> TSDiff:
    logger.info(f"Loading trained model from checkpoint: {ckpt_path}")
    try:
        # Ensure diffusion_config specified in the config exists in diffusion_configs
        if not hasattr(diffusion_configs, config["diffusion_config"]):
            logger.error(f"Diffusion config '{config['diffusion_config']}' not found in uncond_ts_diff.configs.")
            raise ValueError(f"Diffusion config '{config['diffusion_config']}' not found.")

        diffusion_model_config = getattr(diffusion_configs, config["diffusion_config"])

        # Prepare backbone parameters by ensuring all expected keys are present
        # Default values can be added here if they are not in diffusion_model_config["backbone_parameters"]
        # but are expected by TSDiff or BackboneModel
        backbone_params = diffusion_model_config.get("backbone_parameters", {}).copy()
        
        # TSDiff expects these from its direct config, not necessarily through backbone_parameters dict from diffusion_config
        # So, they are passed directly in TSDiff instantiation.

        model = TSDiff(
            backbone_parameters=backbone_params, # This should be the sub-dict for the backbone
            timesteps=diffusion_model_config["timesteps"],
            diffusion_scheduler=diffusion_model_config["diffusion_scheduler"],
            context_length=config["context_length"],
            prediction_length=config["prediction_length"],
            num_feat_dynamic_real=config.get("num_feat_dynamic_real", 0),
            num_feat_static_cat=config.get("num_feat_static_cat", 0),
            num_feat_static_real=config.get("num_feat_static_real", 0),
            cardinalities=config.get("cardinalities", []), # Ensure it's a list
            freq=config["freq"],
            normalization=config.get("normalization", "none"),
            use_features=config.get("use_features", False),
            use_lags=config.get("use_lags", True),
            init_skip=config.get("init_skip", True), # init_skip is part of TSDiff now
            lr=config.get("lr", 1e-3),
        )

        checkpoint = torch.load(ckpt_path, map_location=config["device"])
        if 'state_dict' in checkpoint:
            # Remove "model." prefix if present (common in PyTorch Lightning checkpoints)
            state_dict = {k.replace("model.", "", 1): v for k, v in checkpoint['state_dict'].items()}
            # Remove "backbone." prefix specifically for backbone weights if model is nested
            state_dict = {k.replace("backbone.", "", 1) if "backbone." in k else k: v for k,v in state_dict.items()}

        elif 'ema_state_dicts' in checkpoint and checkpoint['ema_state_dicts']:
             # Prioritize EMA weights if available and configured
            logger.info("Using EMA weights from checkpoint for evaluation.")
            # Assuming the first EMA rate's state_dict is what we want, or the one for a specific rate
            # This part might need adjustment based on how EMA states are stored/chosen
            ema_state_to_load = checkpoint['ema_state_dicts'][0] # Or choose based on a specific EMA rate
            state_dict = {k.replace("model.", "", 1).replace("backbone.","",1): v for k, v in ema_state_to_load.items()}
        else:
            # Raw state_dict
            state_dict = {k.replace("model.", "", 1).replace("backbone.","",1): v for k, v in checkpoint.items()}


        # Load into the model's backbone
        # If TSDiff directly contains the S4 model as self.backbone as per its __init__
        try:
            model.backbone.load_state_dict(state_dict, strict=False) # Use strict=False initially for debugging
            logger.info("Model backbone state dict loaded.")
        except RuntimeError as e:
            logger.error(f"RuntimeError loading state_dict into backbone: {e}")
            logger.info("Attempting to load into the main model...")
            try:
                model.load_state_dict(state_dict, strict=False) # Fallback for different checkpoint structures
                logger.info("Main model state_dict loaded.")
            except RuntimeError as e_main:
                logger.error(f"RuntimeError loading state_dict into main model: {e_main}")
                raise e_main


        model.to(config["device"])
        model.eval()
        logger.info("Trained model loaded successfully and set to evaluation mode.")
        return model
    except Exception as e:
        logger.error(f"Error loading trained model from {ckpt_path}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
def load_jsonl_test_dataset(dataset_path: Path, freq: str, prediction_length: int) -> ListDataset:
    logger.info(f"Loading test dataset from: {dataset_path}")
    if not dataset_path.exists():
        logger.error(f"Test data file not found: {dataset_path}")
        raise FileNotFoundError(f"Test data file not found: {dataset_path}")

    data_entries = []
    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            try:
                entry = json.loads(line)
                
                # 保持start字段为字符串格式，让GluonTS自己处理
                # 不在这里转换为Period对象
                if not isinstance(entry[FieldName.START], str):
                    entry[FieldName.START] = str(entry[FieldName.START])
                
                # 确保TARGET是numpy数组
                entry[FieldName.TARGET] = np.array(entry[FieldName.TARGET], dtype=np.float32)

                # 处理其他特征字段
                if FieldName.FEAT_DYNAMIC_REAL in entry:
                    entry[FieldName.FEAT_DYNAMIC_REAL] = np.array(entry[FieldName.FEAT_DYNAMIC_REAL], dtype=np.float32)
                if FieldName.FEAT_STATIC_CAT in entry:
                    entry[FieldName.FEAT_STATIC_CAT] = np.array(entry[FieldName.FEAT_STATIC_CAT], dtype=int)
                if FieldName.FEAT_STATIC_REAL in entry:
                    entry[FieldName.FEAT_STATIC_REAL] = np.array(entry[FieldName.FEAT_STATIC_REAL], dtype=np.float32)

                data_entries.append(entry)
            except Exception as e:
                logger.error(f"Error parsing line {line_idx+1} in {dataset_path}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue 

    logger.info(f"Successfully loaded {len(data_entries)} entries from {dataset_path}")
    
    if not data_entries:
        logger.error(f"No data entries successfully loaded from {dataset_path}")
        return None

    # 创建ListDataset时使用freq参数
    try:
        dataset = ListDataset(data_entries, freq=freq)
        logger.info(f"Created ListDataset with {len(data_entries)} entries and frequency '{freq}'")
        return dataset
    except Exception as e:
        logger.error(f"Error creating ListDataset: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def generate_forecasts_and_evaluate(
    model: TSDiff,
    test_gluonts_dataset: ListDataset,
    config: dict,
    num_samples_forecast: int = 100,
) -> Tuple[List[Any], Dict[str, float], List[pd.DataFrame], pd.DataFrame, List[Dict[str,Any]]]:

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
        past_length=config["context_length"] + max(model.lags_seq if hasattr(model, "lags_seq") and model.lags_seq else [0]),
        future_length=config["prediction_length"],
        mode="test",
    )

    predictor = sampler_instance.get_predictor(
        input_transform=prediction_instance_splitter,
        batch_size=config.get("eval_batch_size", max(1, 128 // num_samples_forecast)), # Reduced batch size
        device=config["device"],
    )

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=transformed_testdata,
        predictor=predictor,
        num_samples=num_samples_forecast,
    )

    logger.info("Collecting forecasts and ground truth series...")
    forecast_objects = list(tqdm(forecast_it, total=len(test_gluonts_dataset)))
    ground_truth_series_dfs = list(ts_it)


    logger.info("Calculating evaluation metrics...")
    # Define a comprehensive set of quantiles
    eval_quantiles = [0.01, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 0.99]
    evaluator = Evaluator(quantiles=eval_quantiles) # MSTL can be slow, consider disabling if not needed or data is not seasonal

    agg_metrics, item_metrics_df = evaluator(iter(ground_truth_series_dfs), iter(forecast_objects))

    # Add common metrics and ensure they are present
    metrics_to_log = {"CRPS": agg_metrics.get("mean_wQuantileLoss", float('nan'))}
    common_metrics_keys = ["ND", "NRMSE", "MSE", "abs_error", "MAPE", "sMAPE", "MASE"] # MASE might be tricky without full history
    for key in common_metrics_keys:
        metrics_to_log[key] = agg_metrics.get(key, float('nan'))
    
    # Store per-item quantile losses if available from item_metrics_df
    per_item_quantile_losses = []
    for q_idx, q_val in enumerate(eval_quantiles):
        quantile_loss_col_name = f"Coverage[{q_val:.2f}]" # This is coverage, not quantile loss directly
                                                         # True quantile loss per item is harder to get directly from evaluator output
        mql_col_name = f"QuantileLoss[{q_val:.2f}]" # GluonTS >=0.10 uses this format
        if mql_col_name in item_metrics_df.columns:
             item_metrics_df[f'wQuantileLoss@{q_val:.2f}'] = item_metrics_df[mql_col_name]


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
    num_sample_paths_to_plot: int = 20,
    plot_quantiles: Optional[List[float]] = None
):
    logger.info(f"Generating forecast plots for the first {num_series_to_plot} series...")
    output_dir.mkdir(parents=True, exist_ok=True)
    num_to_plot = min(num_series_to_plot, len(ground_truth_dfs), len(forecast_objects))

    if plot_quantiles is None:
        plot_quantiles = [0.1, 0.5, 0.9] # Default quantiles to plot for intervals
    # Ensure 0.5 is present for median line if other quantiles are specified
    if 0.5 not in plot_quantiles:
        plot_quantiles.append(0.5)
        plot_quantiles.sort()


    for i in range(num_to_plot):
        ts_df = ground_truth_dfs[i]
        forecast = forecast_objects[i]
        item_id = forecast.item_id if hasattr(forecast, 'item_id') else f"series_{i+1}"

        plt.figure(figsize=(18, 8))
        ax = plt.gca()

        # Plot ground truth
        # The ground truth from ts_it includes both context and prediction window
        full_context_plus_future_gt = ts_df.iloc[:,0] # Assuming first column is target
        full_context_plus_future_gt.plot(ax=ax, label="Ground Truth (Observed)", color="blue", linewidth=1.5)

        # Determine forecast start date and index
        forecast_start_dt = pd.Timestamp(forecast.start_date.to_timestamp(how='S'))
        pred_len = config["prediction_length"]
        forecast_index = pd.date_range(start=forecast_start_dt, periods=pred_len, freq=config["freq"])

        # Plot sample paths (optional and limited)
        if num_sample_paths_to_plot > 0 and hasattr(forecast, 'samples') and forecast.samples.shape[0] > 0:
            num_paths = min(num_sample_paths_to_plot, forecast.samples.shape[0])
            for j in range(num_paths):
                sample_path = pd.Series(forecast.samples[j], index=forecast_index)
                sample_path.plot(ax=ax, color="lightcoral", alpha=0.2, linewidth=0.7, label="_nolegend_")

        # Plot prediction intervals and median from quantiles
        if isinstance(forecast, QuantileForecast):
            # Try to get specific quantiles for prediction interval
            # For a 80% PI, we need 0.1 and 0.9. For 90% PI, 0.05 and 0.95.
            # Example: Plot 80% PI (0.1 and 0.9) and 50% PI (0.25 and 0.75)
            pi_levels = [(0.1, 0.9), (0.25, 0.75)] # 80% and 50% PI
            pi_colors = ['peachpuff', 'lightsalmon']

            for pi_idx, (lower_q, upper_q) in enumerate(pi_levels):
                if str(lower_q) in forecast.forecast_keys and str(upper_q) in forecast.forecast_keys:
                    lower_bound = pd.Series(forecast.quantile(str(lower_q)), index=forecast_index)
                    upper_bound = pd.Series(forecast.quantile(str(upper_q)), index=forecast_index)
                    ax.fill_between(
                        forecast_index, lower_bound, upper_bound,
                        color=pi_colors[pi_idx % len(pi_colors)], alpha=0.3,
                        label=f'{(upper_q-lower_q)*100:.0f}% PI'
                    )
            # Plot median
            if '0.5' in forecast.forecast_keys:
                 median_forecast = pd.Series(forecast.quantile('0.5'), index=forecast_index)
                 median_forecast.plot(ax=ax, color="red", linewidth=1.5, label=f"Median Forecast ({forecast.forecast_keys[0].split(':')[0]})")

        elif hasattr(forecast, 'mean'): # If it's a MeanForecast or similar
            mean_forecast = pd.Series(forecast.mean, index=forecast_index)
            mean_forecast.plot(ax=ax, color="red", linewidth=1.5, label="Mean Forecast")


        # Highlight prediction window start
        plt.axvline(forecast_start_dt, color='gray', linestyle='--', linewidth=1, label='Forecast Start')

        plt.title(f"Forecast vs. Ground Truth - Item: {item_id}", fontsize=16)
        plt.xlabel("Time", fontsize=12)
        plt.ylabel("Value", fontsize=12)
        plt.legend(fontsize=10, loc='upper left')
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()

        plot_save_path = output_dir / f"forecast_{item_id}.png"
        try:
            plt.savefig(plot_save_path, dpi=150)
        except Exception as e_save:
            logger.error(f"Error saving plot {plot_save_path}: {e_save}")
        plt.close()
        logger.info(f"Saved forecast plot to {plot_save_path}")

def plot_error_distribution(
    ground_truth_dfs: List[pd.DataFrame],
    forecast_objects: List[Any],
    config: dict,
    output_dir: Path,
    num_series_to_analyze: int = 5
):
    logger.info(f"Generating error distribution plots for the first {num_series_to_analyze} series...")
    output_dir.mkdir(parents=True, exist_ok=True)
    num_to_analyze = min(num_series_to_analyze, len(ground_truth_dfs), len(forecast_objects))
    all_residuals = []

    for i in range(num_to_analyze):
        ts_df = ground_truth_dfs[i]
        forecast = forecast_objects[i]
        item_id = forecast.item_id if hasattr(forecast, 'item_id') else f"series_{i+1}"

        # 获取预测长度
        pred_len = config["prediction_length"]
        
        # 直接获取真实值的最后 pred_len 个点，而不是尝试用 forecast_index 索引
        # ground truth 已经包含了完整的序列（context + prediction）
        actual_values_prediction_window = ts_df.iloc[-pred_len:, 0].values
        
        # 获取预测值
        if isinstance(forecast, QuantileForecast) and '0.5' in forecast.forecast_keys:
            median_forecast_vals = forecast.quantile('0.5')
        elif hasattr(forecast, 'mean'):
            median_forecast_vals = forecast.mean
        else: # Fallback to median of samples if available
            if hasattr(forecast, 'samples') and forecast.samples.shape[0] > 0:
                median_forecast_vals = np.median(forecast.samples, axis=0)
            else:
                logger.warning(f"Cannot determine point forecast for error calculation for series {item_id}. Skipping error plot.")
                continue
        
        if len(actual_values_prediction_window) != len(median_forecast_vals):
            logger.warning(f"Length mismatch for series {item_id}. Actuals: {len(actual_values_prediction_window)}, Forecast: {len(median_forecast_vals)}. Skipping.")
            continue

        residuals = actual_values_prediction_window - median_forecast_vals
        all_residuals.extend(residuals)

        plt.figure(figsize=(12, 6))
        sns.histplot(residuals, kde=True, bins=20)
        plt.title(f"Error (Residual) Distribution - Item: {item_id}", fontsize=16)
        plt.xlabel("Error (Actual - Forecasted Median)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        plot_save_path = output_dir / f"error_dist_{item_id}.png"
        plt.savefig(plot_save_path, dpi=150)
        plt.close()
        logger.info(f"Saved error distribution plot to {plot_save_path}")

    if all_residuals:
        plt.figure(figsize=(12, 6))
        sns.histplot(all_residuals, kde=True, bins=30, color='skyblue')
        plt.title(f"Aggregated Error Distribution (First {num_to_analyze} Series)", fontsize=16)
        plt.xlabel("Error (Actual - Forecasted Median)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.grid(True, linestyle=':', alpha=0.7)
        plt.tight_layout()
        agg_plot_save_path = output_dir / "error_dist_aggregated.png"
        plt.savefig(agg_plot_save_path, dpi=150)
        plt.close()
        logger.info(f"Saved aggregated error distribution plot to {agg_plot_save_path}")
        
def plot_item_wise_metrics_distribution(
    item_metrics_df: pd.DataFrame,
    output_dir: Path
):
    logger.info("Generating item-wise metric distribution plots...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_to_plot = ['ND', 'NRMSE', 'MSE', 'abs_error', 'CRPS', 'MAPE', 'sMAPE']
    # Filter for metrics actually present in the DataFrame
    available_metrics = [m for m in metrics_to_plot if m in item_metrics_df.columns]

    if not available_metrics:
        logger.warning("No suitable item-wise metrics found in DataFrame to plot distributions.")
        return

    num_metrics = len(available_metrics)
    # Determine subplot layout (e.g., 2 columns)
    ncols = 2
    nrows = (num_metrics + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8 * ncols, 6 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for i, metric_name in enumerate(available_metrics):
        ax = axes_flat[i]
        if item_metrics_df[metric_name].isnull().all():
             ax.text(0.5, 0.5, f'{metric_name}\n(All NaN values)', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        else:
            sns.histplot(item_metrics_df[metric_name].dropna(), ax=ax, kde=True, bins=20)
            ax.set_title(f"Distribution of {metric_name}", fontsize=14)
            ax.set_xlabel(metric_name, fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.5)

    # Hide any unused subplots
    for j in range(i + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.tight_layout()
    plot_save_path = output_dir / "item_metrics_distributions.png"
    plt.savefig(plot_save_path, dpi=150)
    plt.close()
    logger.info(f"Saved item-wise metrics distribution plot to {plot_save_path}")

def plot_aggregate_metrics_summary(
    agg_metrics: Dict[str, float],
    output_dir: Path
):
    logger.info("Generating aggregate metrics summary plot...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Filter out NaN values for plotting
    metrics_to_plot = {k: v for k, v in agg_metrics.items() if pd.notna(v)}
    if not metrics_to_plot:
        logger.warning("No non-NaN aggregate metrics to plot.")
        return

    metric_names = list(metrics_to_plot.keys())
    metric_values = list(metrics_to_plot.values())

    plt.figure(figsize=(max(10, len(metric_names) * 1.2), 6))
    bars = sns.barplot(x=metric_names, y=metric_values, palette="viridis")
    plt.title("Aggregate Evaluation Metrics Summary", fontsize=16)
    plt.ylabel("Metric Value", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(axis='y', linestyle=':', alpha=0.7)

    # Add text labels on bars
    for bar in bars.patches:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * max(metric_values, default=0), # Adjust offset
                 f'{yval:.3f}', ha='center', va='bottom', fontsize=9)


    plt.tight_layout()
    plot_save_path = output_dir / "aggregate_metrics_summary.png"
    plt.savefig(plot_save_path, dpi=150)
    plt.close()
    logger.info(f"Saved aggregate metrics summary plot to {plot_save_path}")

def generate_text_report(
    config: dict,
    agg_metrics: Dict[str, float],
    item_metrics_df: pd.DataFrame,
    output_dir: Path,
    num_series_total: int
):
    logger.info("Generating text report...")
    report_path = output_dir / "evaluation_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("TSDiff Model Evaluation Report\n")
        f.write("="*60 + "\n\n")

        f.write("--- Configuration ---\n")
        f.write(f"Training Config File: {config.get('original_config_file_path', 'N/A')}\n")
        f.write(f"Checkpoint Path: {config.get('original_ckpt_path', 'N/A')}\n")
        f.write(f"Test Data Path: {config.get('original_test_data_path', 'N/A')}\n")
        f.write(f"Model Frequency: {config.get('freq', 'N/A')}\n")
        f.write(f"Context Length: {config.get('context_length', 'N/A')}\n")
        f.write(f"Prediction Length: {config.get('prediction_length', 'N/A')}\n")
        f.write(f"Number of Forecast Samples: {config.get('num_forecast_samples_run', 'N/A')}\n")
        f.write(f"Device Used: {config.get('device', 'N/A')}\n")
        f.write("\n")

        f.write("--- Data Summary ---\n")
        f.write(f"Total time series in test set: {num_series_total}\n")
        f.write("\n")

        f.write("--- Aggregate Performance Metrics ---\n")
        for name, value in agg_metrics.items():
            f.write(f"  {name}: {value:.4f}\n")
        f.write("\n")

        f.write("--- Item-wise Metrics Summary ---\n")
        metrics_to_describe = ['ND', 'NRMSE', 'MSE', 'CRPS', 'sMAPE']
        available_metrics = [m for m in metrics_to_describe if m in item_metrics_df.columns and item_metrics_df[m].notna().any()]
        
        if available_metrics:
            description_df = item_metrics_df[available_metrics].describe().transpose()
            f.write(description_df.to_string(float_format="%.4f"))
            f.write("\n\n")
        else:
            f.write("No item-wise metrics available for summary statistics.\n\n")

        f.write("--- Notes ---\n")
        if pd.isna(agg_metrics.get('MASE')):
            f.write("- MASE could not be calculated (requires full historical series for scaling, or specific setup).\n")
        f.write("- CRPS is reported as mean_wQuantileLoss from the GluonTS evaluator.\n")
        f.write("- Plots for individual forecasts, error distributions, and metric distributions are saved separately.\n")
        f.write("="*60 + "\n")
        f.write("End of Report\n")
        f.write("="*60 + "\n")
    logger.info(f"Text report saved to {report_path}")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler("predict_evaluate_fdr.log", mode='w')] # Log to file as well
    )

    parser = argparse.ArgumentParser(description="Predict and evaluate using a trained TSDiff model for FDR data.")
    parser.add_argument("-c", "--config_file", type=str, required=True, help="Path to the YAML training configuration file used for the model.")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the trained model checkpoint (.ckpt or .pth).")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test data JSON Lines file (e.g., test.jsonl).")
    parser.add_argument("--output_dir", type=str, default="./fdr_predictions_eval_detailed", help="Directory to save prediction results, plots, and report.")
    parser.add_argument("--num_forecast_samples", type=int, default=100, help="Number of sample paths to generate for each forecast.")
    parser.add_argument("--num_plot_series", type=int, default=5, help="Number of time series to plot forecasts for.")
    parser.add_argument("--num_error_analysis_series", type=int, default=5, help="Number of series for detailed error distribution plots.")


    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    plots_output_path = output_path / "plots"
    plots_output_path.mkdir(parents=True, exist_ok=True)


    config = load_config_for_prediction(Path(args.config_file))
    # Store original paths in config for reporting
    config['original_config_file_path'] = args.config_file
    config['original_ckpt_path'] = args.ckpt_path
    config['original_test_data_path'] = args.test_data_path
    config['num_forecast_samples_run'] = args.num_forecast_samples


    model = load_trained_model(config, Path(args.ckpt_path))

    test_dataset = load_jsonl_test_dataset(Path(args.test_data_path), config["freq"], config["prediction_length"])
    # 改进的空数据集检查
    if test_dataset is None:
        logger.error("Test dataset is None. Exiting.")
        return
    
    # 检查数据集是否真的有数据
    try:
        test_dataset_list = list(test_dataset)
        num_total_series = len(test_dataset_list)
        logger.info(f"Test dataset contains {num_total_series} time series")
        if num_total_series == 0:
            logger.error("Test dataset is empty. Exiting.")
            return
    except Exception as e:
        logger.error(f"Error accessing test dataset: {e}")
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
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, Path): return str(obj)
                if isinstance(obj, (np.integer, np.int_)): return int(obj)
                if isinstance(obj, (np.floating, np.float_)): return float(obj)
                if isinstance(obj, (np.bool_)): return bool(obj)
                if isinstance(obj, np.ndarray): return obj.tolist()
                return super(CustomEncoder, self).default(obj)
        json.dump(results_summary, f, indent=4, cls=CustomEncoder)
    logger.info(f"Prediction summary (JSON) saved to {summary_save_path}")

    item_metrics_save_path = output_path / "item_metrics_detailed.csv"
    try:
        item_metrics_df.to_csv(item_metrics_save_path, index=True, index_label="item_index") # Save index too
        logger.info(f"Item-wise metrics (CSV) saved to {item_metrics_save_path}")
    except Exception as e_csv:
        logger.error(f"Could not save item_metrics to CSV: {e_csv}")


    if args.num_plot_series > 0 and ground_truth_dfs and forecast_objects:
        plot_sample_forecasts(
            ground_truth_dfs,
            forecast_objects,
            config,
            plots_output_path,
            num_series_to_plot=args.num_plot_series,
            num_sample_paths_to_plot=20 # Limit sample paths in plot
        )

    if args.num_error_analysis_series > 0 and ground_truth_dfs and forecast_objects:
         plot_error_distribution(
            ground_truth_dfs,
            forecast_objects,
            config,
            plots_output_path,
            num_series_to_analyze=args.num_error_analysis_series
        )

    if not item_metrics_df.empty:
        plot_item_wise_metrics_distribution(item_metrics_df, plots_output_path)

    if agg_metrics:
        plot_aggregate_metrics_summary(agg_metrics, plots_output_path)
        
    generate_text_report(config, agg_metrics, item_metrics_df, output_path, num_total_series)

    logger.info(f"Prediction and evaluation complete. Results in {output_path.resolve()}")

if __name__ == "__main__":
    main()