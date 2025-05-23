# custom_scripts/interactive_fdr_setup.py
import argparse
from pathlib import Path
import json
import sys
from typing import Any, Dict, List 
import numpy as np 
import warnings

# Suppress specific Pandas warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pandas.core.resample")

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

try:
    import yaml 
except ImportError:
    print("Warning: PyYAML is not installed. JSON config file is preferred for data_preprocessing_interactive_config if YAML is not available.")
    yaml = None

try:
    from src.fdr_project.data_loader import (
        get_mat_file_paths,
        discover_variables_from_sample_file,
        analyze_variable_across_files,
        get_user_decision_for_variable,
        process_mat_file_with_decisions,
        save_gluonts_jsonl,
        ask_yes_no  # <--- 确保 ask_yes_no 被导入
    )
except ImportError as e:
    print(f"Error importing from src.fdr_project.data_loader: {e}")
    print("Please ensure that 'src/fdr_project/data_loader.py' exists and is correctly structured with all necessary functions, including 'ask_yes_no'.")
    sys.exit(1)


def main_interactive_setup():
    parser = argparse.ArgumentParser(description="Interactively set up FDR data processing for a target variable.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/fdr/data_preprocessing_interactive_config.yaml",
        help="Path to the data preprocessing YAML or JSON config file (relative to project root)."
    )
    args = parser.parse_args()

    config_file_path = project_root / args.config
    if not config_file_path.exists():
        print(f"Error: Configuration file not found at {config_file_path}")
        sys.exit(1)
    
    config_data = {}
    try:
        with open(config_file_path, 'r', encoding='utf-8') as f:
            if config_file_path.suffix.lower() in [".yaml", ".yml"]:
                if yaml is None:
                    print("Error: PyYAML not installed, cannot load .yaml config. Please use .json or install PyYAML (`pip install pyyaml`).")
                    sys.exit(1)
                config_data = yaml.safe_load(f)
            elif config_file_path.suffix.lower() == ".json":
                config_data = json.load(f)
            else:
                print(f"Error: Unsupported config file format: {config_file_path.suffix}. Use .yaml or .json.")
                sys.exit(1)
    except Exception as e:
        print(f"Error loading or parsing config file {config_file_path}: {e}")
        sys.exit(1)

    fdr_mat_dir_relative = config_data.get("fdr_mat_source_path")
    output_base_dir_relative = config_data.get("output_base_directory")
    sample_files_count = config_data.get("sample_files_for_analysis", 3)
    min_len_factor = config_data.get("min_len_factor_for_filtering", 1.5)
    test_split_ratio = config_data.get("test_split_ratio", 0.2)

    if not fdr_mat_dir_relative or not output_base_dir_relative:
        print("Error: 'fdr_mat_source_path' and 'output_base_directory' must be defined in the config file.")
        sys.exit(1)

    fdr_data_root = project_root / fdr_mat_dir_relative
    output_base_dir = project_root / output_base_dir_relative
    output_base_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- FDR Data Interactive Setup ---")
    print(f"Using MAT file source: {fdr_data_root}")
    print(f"Processed data will be saved under: {output_base_dir}")

    mat_files = get_mat_file_paths(fdr_data_root)
    if not mat_files:
        print(f"No MAT files found in the specified source directory: {fdr_data_root}. Exiting.")
        return

    available_variables = discover_variables_from_sample_file(mat_files[0])
    if not available_variables:
        print("No suitable time series variables found in the sample MAT file. Exiting.")
        return

    print("\nAvailable time series variables found in sample file:")
    for i, var_name in enumerate(available_variables):
        print(f"  {i+1}. {var_name}")

    target_variable = ""
    while True:
        try:
            choice_idx_str = input("Enter the number of the variable you want to process: ").strip()
            if not choice_idx_str: continue
            choice_idx = int(choice_idx_str) - 1
            if 0 <= choice_idx < len(available_variables):
                target_variable = available_variables[choice_idx]
                break
            else: print("  Invalid choice. Please enter a number from the list.")
        except ValueError: print("  Invalid input. Please enter a number.")
    print(f"You selected variable: '{target_variable}'")

    var_analysis_results = analyze_variable_across_files(mat_files, target_variable, sample_files_count)
    
    if var_analysis_results.get("num_files_effectively_analyzed", 0) == 0 :
        print(f"Variable '{target_variable}' was not found or analyzable in any sample files. Cannot proceed.")
        return
    # rates_found_hz check is implicitly handled by get_user_decision_for_variable

    if target_variable == "CAS":
        print(f"\n  --- Specific Note for Variable: '{target_variable}' ---")
        cas_dtype_list = var_analysis_results.get("dtypes_found_in_samples", [])
        is_uint8 = any('uint8' in dt.lower() for dt in cas_dtype_list)
        all_samples_zero = var_analysis_results.get("all_zero_samples_count", 0) == var_analysis_results.get("num_files_effectively_analyzed", 0) and var_analysis_results.get("num_files_effectively_analyzed",0) > 0

        if is_uint8:
            print(f"    WARNING: '{target_variable}' is stored as uint8 in some sample files.")
            if all_samples_zero:
                 print(f"    CRITICAL WARNING: '{target_variable}' (uint8) was ALL ZEROS in all {var_analysis_results.get('num_files_effectively_analyzed',0)} sample file(s) checked where it was uint8. "
                       f"It is highly unlikely to be useful for prediction if this pattern holds for all uint8 CAS files.")
                 if not ask_yes_no(f"    '{target_variable}' (when uint8) appears non-informative in samples. Do you REALLY want to proceed with processing this variable (uint8 instances might be skipped)?"):
                    print(f"Skipping '{target_variable}' due to data quality concerns regarding uint8 instances.")
                    return
            else: # uint8 but not all zeros in sample
                print(f"             The processing script will attempt to convert it to float32.")
                print(f"             It will also skip individual files if this variable's data within that file (when uint8) is mostly zeros.")
                if not ask_yes_no(f"    Knowing '{target_variable}' is sometimes uint8 and might be filtered if zero-heavy, do you still want to proceed?"):
                    print(f"Skipping '{target_variable}' based on user decision due to uint8 type concerns.")
                    return
    
    user_decisions = get_user_decision_for_variable(var_analysis_results)

    if not user_decisions.get("process_this_variable", True):
        print(f"Skipping processing for variable '{target_variable}' based on user decision.")
        return
    
    print("\n--- Model Configuration (for filtering sequences) ---")
    model_context_length = 0
    while True:
        try:
            context_len_str = input(f"Enter the intended context_length for TSDiff model training with '{target_variable}': ").strip()
            if not context_len_str: continue
            model_context_length = int(context_len_str)
            if model_context_length <= 0: raise ValueError("Context length must be positive.")
            break
        except ValueError as e: print(f"  Invalid input: {e}")
    
    model_prediction_length = 0
    while True:
        try:
            pred_len_str = input(f"Enter the intended prediction_length for TSDiff model training with '{target_variable}': ").strip()
            if not pred_len_str: continue
            model_prediction_length = int(pred_len_str)
            if model_prediction_length <= 0: raise ValueError("Prediction length must be positive.")
            break
        except ValueError as e: print(f"  Invalid input: {e}")

    min_required_length_for_series = int((model_context_length + model_prediction_length) * min_len_factor)
    print(f"Sequences for '{target_variable}' shorter than {min_required_length_for_series} samples (after any processing) will be skipped.")

    print(f"\n--- Processing all MAT files for '{target_variable}' using your decisions ---")
    all_final_series: List[Dict[str, Any]] = []
    
    for mat_file in mat_files:
        series_data = process_mat_file_with_decisions(
            mat_file,
            target_variable,
            user_decisions, 
            min_required_length_for_series
        )
        if series_data:
            all_final_series.append(series_data)
    
    if not all_final_series:
        print(f"No series for '{target_variable}' met the criteria after processing all files. No data saved.")
        return

    freq_suffix = str(user_decisions['target_hz']) + 'hz' if user_decisions.get('resample_enabled', False) else 'origfreq'
    variable_specific_output_dir = output_base_dir / f"{target_variable.replace('.', '_')}_{freq_suffix}"
    variable_specific_output_dir.mkdir(parents=True, exist_ok=True)

    num_total = len(all_final_series)
    num_test = int(num_total * test_split_ratio)
    num_train = num_total - num_test

    train_data = all_final_series[:num_train]
    test_data = all_final_series[num_train:]

    final_dataset_frequency_for_gluonts = user_decisions["final_pandas_freq"]

    print(f"\nTotal '{target_variable}' series successfully processed and filtered: {num_total}")
    print(f"  Training series: {len(train_data)}")
    print(f"  Test series: {len(test_data)}")
    print(f"  Final dataset frequency for GluonTS config: {final_dataset_frequency_for_gluonts}")

    save_gluonts_jsonl(train_data, variable_specific_output_dir / "train.jsonl")
    save_gluonts_jsonl(test_data, variable_specific_output_dir / "test.jsonl")

    run_configuration_summary = {
        "input_script_config_file": str(config_file_path.relative_to(project_root) if hasattr(config_file_path, "is_relative_to") and config_file_path.is_relative_to(project_root) else str(config_file_path.resolve())),
        "data_source": {
            "fdr_mat_directory": str(fdr_data_root.relative_to(project_root) if hasattr(fdr_data_root, "is_relative_to") and fdr_data_root.is_relative_to(project_root) else str(fdr_data_root.resolve())),
            "target_variable_selected": target_variable,
            "num_mat_files_scanned": len(mat_files)
        },
        "variable_analysis_and_user_decisions": {
            "initial_sample_analysis": var_analysis_results,
            "user_processing_choices": user_decisions
        },
        "output_dataset_info": {
            "output_directory": str(variable_specific_output_dir.relative_to(project_root) if hasattr(variable_specific_output_dir, "is_relative_to") and variable_specific_output_dir.is_relative_to(project_root) else str(variable_specific_output_dir.resolve())),
            "gluonts_metadata_freq": final_dataset_frequency_for_gluonts,
            "num_train_series_saved": len(train_data),
            "num_test_series_saved": len(test_data)
        },
        "model_length_params_used_for_filtering": {
             "context_length": model_context_length,
             "prediction_length": model_prediction_length,
             "min_len_factor_applied": min_len_factor,
             "min_required_samples_per_series": min_required_length_for_series
        }
    }
    config_save_path = variable_specific_output_dir / "data_processing_run_summary.json"
    try:
        with open(config_save_path, 'w', encoding='utf-8') as f:
            class CustomEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, Path): return str(obj)
                    if isinstance(obj, (np.integer)): return int(obj)
                    if isinstance(obj, (np.floating)): return float(obj)
                    if isinstance(obj, (np.bool_)): return bool(obj)
                    if isinstance(obj, np.ndarray): return obj.tolist()
                    return super(CustomEncoder, self).default(obj)
            json.dump(run_configuration_summary, f, indent=4, cls=CustomEncoder)
        print(f"Run summary saved to: {config_save_path}")
    except TypeError as te:
        print(f"Could not serialize run summary to JSON: {te}. Saving as text.")
        with open(variable_specific_output_dir / "data_processing_run_summary.txt", 'w', encoding='utf-8') as f:
             f.write(str(run_configuration_summary))

    gluonts_metadata = {
        "freq": final_dataset_frequency_for_gluonts,
        "prediction_length": model_prediction_length
    }
    with open(variable_specific_output_dir / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(gluonts_metadata, f, indent=4)
    print(f"GluonTS metadata.json saved in: {variable_specific_output_dir}")

    print(f"\nInteractive data setup complete for '{target_variable}'.")
    print(f"You can now use the directory '{str(variable_specific_output_dir.resolve())}' as the 'dataset' path in your TSDiff training config.") # Use resolved path
    print(f"Ensure the 'freq' in TSDiff config matches: {final_dataset_frequency_for_gluonts}")
    print(f"Ensure 'prediction_length' and 'context_length' in TSDiff config match: {model_prediction_length} and {model_context_length}")

if __name__ == "__main__":
    main_interactive_setup()