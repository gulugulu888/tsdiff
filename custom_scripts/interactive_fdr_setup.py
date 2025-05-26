# custom_scripts/interactive_fdr_setup.py
import argparse
from pathlib import Path
import json
import sys
from typing import Any, Dict, List, Optional, Tuple, Set # Added Set
import numpy as np 
import pandas as pd 
import warnings

import scipy

# Suppress specific Pandas warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pandas.core.resample")

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

try:
    import yaml 
except ImportError:
    print("Warning: PyYAML is not installed. JSON config file is preferred if YAML is not available.")
    yaml = None

try:
    from src.fdr_project.data_loader import (
        get_mat_file_paths,
        discover_variables_from_sample_file,
        analyze_variable_across_files,
        get_user_decision_for_variable,
        process_mat_file_with_decisions,
        save_gluonts_jsonl,
        ask_yes_no,
        parse_fdr_timestamp_from_mat, 
        get_series_from_mat,          
        determine_inflight_window_from_flight_indicator 
    )
except ImportError as e:
    print(f"Error importing from src.fdr_project.data_loader: {e}")
    print("Please ensure that 'src/fdr_project/data_loader.py' is correctly structured with all necessary functions.")
    sys.exit(1)


def main_interactive_setup():
    parser = argparse.ArgumentParser(description="Interactively set up FDR data processing for a target variable, using a flight indicator for windowing.")
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
                    print("Error: PyYAML not installed, cannot load .yaml config. Use .json or install PyYAML (`pip install pyyaml`).")
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
    sample_files_count = config_data.get("sample_files_for_analysis", 5)
    min_len_factor = config_data.get("min_len_factor_for_filtering", 1.5)
    test_split_ratio = config_data.get("test_split_ratio", 0.2)

    if not fdr_mat_dir_relative or not output_base_dir_relative:
        print("Error: 'fdr_mat_source_path' and 'output_base_directory' must be defined in the config file.")
        sys.exit(1)

    fdr_data_root = project_root / fdr_mat_dir_relative
    output_base_dir = project_root / output_base_dir_relative
    output_base_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- FDR Data Interactive Setup (with Flight Window Filtering) ---")
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

    # --- Flight Indicator Variable Setup ---
    print("\n--- Flight Phase Indicator Setup ---")
    default_flight_indicator_var = "CAS"
    flight_indicator_var_name = default_flight_indicator_var # Initialize with default

    if default_flight_indicator_var not in available_variables:
        print(f"Warning: Default flight indicator '{default_flight_indicator_var}' not found among discovered variables: {available_variables}")
        print("Available variables:")
        for i, var_name_option in enumerate(available_variables): print(f"  {i+1}. {var_name_option}")
        while True:
            try:
                choice_idx_str = input(f"Enter the number of the variable to use as flight phase indicator (or type 'skip' to process without flight windowing): ").strip()
                if choice_idx_str.lower() == 'skip':
                    flight_indicator_var_name = None; break
                if not choice_idx_str: continue
                choice_idx = int(choice_idx_str) - 1
                if 0 <= choice_idx < len(available_variables):
                    flight_indicator_var_name = available_variables[choice_idx]; break
                else: print("  Invalid choice.")
            except ValueError: print("  Invalid input. Please enter a number or 'skip'.")
    else:
        if not ask_yes_no(f"Use '{default_flight_indicator_var}' as the flight phase indicator? (y/n): "):
            print("Available variables:")
            for i, var_name_option in enumerate(available_variables): print(f"  {i+1}. {var_name_option}")
            while True:
                try:
                    choice_idx_str = input(f"Enter the number of the variable for flight phase (or type 'skip' to process without specific flight windowing): ").strip()
                    if choice_idx_str.lower() == 'skip':
                        flight_indicator_var_name = None; break
                    if not choice_idx_str: continue
                    choice_idx = int(choice_idx_str) - 1
                    if 0 <= choice_idx < len(available_variables):
                        flight_indicator_var_name = available_variables[choice_idx]; break
                    else: print("  Invalid choice.")
                except ValueError: print("  Invalid input. Please enter a number or 'skip'.")
    
    flight_indicator_processing_decisions = None
    flight_indicator_ground_threshold = 0.0

    if flight_indicator_var_name:
        print(f"Using '{flight_indicator_var_name}' for flight phase determination.")
        indicator_analysis = analyze_variable_across_files(mat_files, flight_indicator_var_name, sample_files_count)
        if indicator_analysis.get("num_files_effectively_analyzed", 0) == 0:
            print(f"Critical: Flight indicator '{flight_indicator_var_name}' could not be analyzed. Processing will continue without indicator-based windowing.")
            flight_indicator_var_name = None # Disable indicator if it cannot be analyzed
        else:
            flight_indicator_processing_decisions = get_user_decision_for_variable(
                indicator_analysis, 
                is_flight_indicator_setup=True
            )
            if not flight_indicator_processing_decisions.get("process_this_variable", False):
                print(f"Flight indicator '{flight_indicator_var_name}' was marked not to be processed. Proceeding without indicator-based windowing.")
                flight_indicator_var_name = None
            else:
                while True: # Get threshold for the chosen indicator
                    try:
                        threshold_units = indicator_analysis.get("units", ["unknown units"])[0] if indicator_analysis.get("units") else "unknown units"
                        threshold_str = input(f"Enter the ground threshold for '{flight_indicator_var_name}' (in {threshold_units}, e.g., for CAS in knots, try 30-50) to define 'in-flight': ").strip()
                        flight_indicator_ground_threshold = float(threshold_str)
                        if flight_indicator_ground_threshold < 0: raise ValueError("Threshold must be non-negative.")
                        break
                    except ValueError as e: print(f"  Invalid input: {e}")
    else:
        print("Proceeding without specific flight window filtering based on an indicator variable.")

    # --- Pass 1: Determine In-flight Windows (if indicator is chosen and valid) ---
    flight_windows_map: Dict[Path, Optional[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
    num_files_with_flight_window = 0

    if flight_indicator_var_name and flight_indicator_processing_decisions:
        print(f"\n--- Determining In-Flight Windows using '{flight_indicator_var_name}' (threshold: {flight_indicator_ground_threshold}) ---")
        processed_mat_count_pass1 = 0
        for mat_file in mat_files:
            processed_mat_count_pass1 += 1
            if processed_mat_count_pass1 % 50 == 0 or processed_mat_count_pass1 == len(mat_files):
                 print(f"  Determining flight window for MAT file {processed_mat_count_pass1}/{len(mat_files)}...")
            
            mat_data_struct = scipy.io.loadmat(mat_file, simplify_cells=True)
            mat_start_ts = parse_fdr_timestamp_from_mat(mat_data_struct, mat_file.name)
            
            forced_hz_for_indicator = flight_indicator_processing_decisions.get("determined_original_hz")

            indicator_series_tuple = get_series_from_mat(
                mat_data_struct, 
                flight_indicator_var_name, 
                mat_start_ts, 
                mat_file,
                forced_original_hz=forced_hz_for_indicator
            )
            
            if indicator_series_tuple is None:
                flight_windows_map[mat_file] = None; continue
            
            indicator_series_orig_rate, indicator_actual_orig_hz = indicator_series_tuple
            
            series_for_window_detection = indicator_series_orig_rate
            if flight_indicator_processing_decisions.get("resample_enabled", False):
                target_hz_indicator_window = flight_indicator_processing_decisions["target_hz"]
                target_freq_indicator_window = flight_indicator_processing_decisions["final_pandas_freq"]
                
                if abs(indicator_actual_orig_hz - target_hz_indicator_window) > 1e-6:
                    resampled_series: Optional[pd.Series] = None
                    if indicator_actual_orig_hz > target_hz_indicator_window: 
                        agg_method = flight_indicator_processing_decisions.get("resample_agg_method", "mean")
                        resampled_series = indicator_series_orig_rate.resample(target_freq_indicator_window).agg(agg_method)
                    else: 
                        fill_method = flight_indicator_processing_decisions.get("resample_fill_method", "ffill")
                        if fill_method == "interpolate":
                            interp_method = flight_indicator_processing_decisions.get("interpolate_method", "linear")
                            resampled_series = indicator_series_orig_rate.resample(target_freq_indicator_window).interpolate(method=interp_method)
                        else:
                             resampled_series = indicator_series_orig_rate.resample(target_freq_indicator_window).fillna(method=fill_method)
                    if resampled_series is not None:
                        series_for_window_detection = resampled_series.dropna()
            
            if series_for_window_detection.empty:
                flight_windows_map[mat_file] = None; continue
                
            if series_for_window_detection.isnull().any():
                nan_handle_method_indicator = flight_indicator_processing_decisions.get("nan_handling_method", "interpolate_linear")
                if nan_handle_method_indicator == "interpolate_linear":
                    series_for_window_detection = series_for_window_detection.interpolate(method='linear', limit_direction='both').bfill().ffill()
                elif nan_handle_method_indicator == "ffill_then_bfill":
                    series_for_window_detection = series_for_window_detection.ffill().bfill()
                elif nan_handle_method_indicator == "to_zero": # Should be rare for CAS/indicator
                    series_for_window_detection = series_for_window_detection.fillna(0.0)
                series_for_window_detection = series_for_window_detection.dropna()

            if series_for_window_detection.empty:
                flight_windows_map[mat_file] = None; continue

            flight_window = determine_inflight_window_from_flight_indicator(series_for_window_detection, flight_indicator_ground_threshold)
            flight_windows_map[mat_file] = flight_window
            if flight_window:
                num_files_with_flight_window +=1
        
        print(f"Flight windows determined for {num_files_with_flight_window}/{len(mat_files)} MAT files using '{flight_indicator_var_name}'.")
        if num_files_with_flight_window == 0 and len(mat_files) > 0:
            print(f"Warning: No flight windows could be determined from any MAT file using '{flight_indicator_var_name}'.")
            if not ask_yes_no("Do you want to proceed with processing the target variable on the full data (no flight window filtering)? (y/n)"):
                print("Exiting data setup.")
                return
            flight_indicator_var_name = None # Disable for subsequent steps
            for mat_file in mat_files: flight_windows_map[mat_file] = None # Ensure all are None
    else: # User skipped flight indicator or indicator setup failed and they chose to skip
        print("Proceeding without specific flight window filtering based on an indicator variable.")
        for mat_file in mat_files: flight_windows_map[mat_file] = None


    # --- Select Target Variable for Final Dataset ---
    print("\n--- Target Variable Selection for Final Dataset ---")
    print("Available time series variables:")
    for i, var_name in enumerate(available_variables): print(f"  {i+1}. {var_name}")
    
    target_variable_for_dataset = ""
    while True:
        try:
            choice_idx_str = input("Enter the number of the variable you want to process for the dataset: ").strip()
            if not choice_idx_str: continue
            choice_idx = int(choice_idx_str) - 1
            if 0 <= choice_idx < len(available_variables):
                target_variable_for_dataset = available_variables[choice_idx]; break
            else: print("  Invalid choice.")
        except ValueError: print("  Invalid input. Please enter a number.")
    print(f"You selected variable: '{target_variable_for_dataset}' for the final dataset.")

    target_var_analysis_results = analyze_variable_across_files(mat_files, target_variable_for_dataset, sample_files_count)
    if target_var_analysis_results.get("num_files_effectively_analyzed", 0) == 0:
        print(f"Target variable '{target_variable_for_dataset}' was not found/analyzable in sample files. Cannot proceed.")
        return

    user_decisions_for_target_var = get_user_decision_for_variable(
        target_var_analysis_results, 
        is_flight_indicator_setup=False, 
        target_var_selected_by_user=target_variable_for_dataset
    )
    if not user_decisions_for_target_var.get("process_this_variable", True):
        print(f"Skipping processing for variable '{target_variable_for_dataset}' based on user decision.")
        return

    print("\n--- Model Configuration (for filtering sequences by length) ---")
    model_context_length = 0
    while True:
        try:
            context_len_str = input(f"Enter the intended context_length for TSDiff model training with '{target_variable_for_dataset}': ").strip()
            model_context_length = int(context_len_str)
            if model_context_length <= 0: raise ValueError("Context length must be positive.")
            break
        except ValueError as e: print(f"  Invalid input: {e}")
    
    model_prediction_length = 0
    while True:
        try:
            pred_len_str = input(f"Enter the intended prediction_length for TSDiff model training with '{target_variable_for_dataset}': ").strip()
            model_prediction_length = int(pred_len_str)
            if model_prediction_length <= 0: raise ValueError("Prediction length must be positive.")
            break
        except ValueError as e: print(f"  Invalid input: {e}")

    min_required_length_for_series = int((model_context_length + model_prediction_length) * min_len_factor)
    print(f"Sequences for '{target_variable_for_dataset}' shorter than {min_required_length_for_series} samples (after all processing) will be skipped.")

    # --- Pass 2: Process Target Variable using Flight Windows ---
    processing_context_message = f"using flight windows from '{flight_indicator_var_name}'" if flight_indicator_var_name else "on full data (no indicator-based windowing)"
    print(f"\n--- Processing all MAT files for '{target_variable_for_dataset}' {processing_context_message} ---")
    
    all_final_series: List[Dict[str, Any]] = []
    processed_mat_count_main = 0
    skipped_due_no_flight_window_main = 0

    for mat_file in mat_files:
        processed_mat_count_main += 1
        if processed_mat_count_main % 20 == 0 or processed_mat_count_main == len(mat_files):
             print(f"  Processing MAT file {processed_mat_count_main}/{len(mat_files)} for '{target_variable_for_dataset}'...")
        
        current_flight_window_for_file = flight_windows_map.get(mat_file) # Will be None if flight_indicator_var_name is None
        
        if flight_indicator_var_name and current_flight_window_for_file is None:
            skipped_due_no_flight_window_main +=1
            continue # Skip this file for the target var if indicator was used but no window found
            
        series_data = process_mat_file_with_decisions(
            mat_file_path=mat_file,
            target_variable_name_to_process=target_variable_for_dataset,
            processing_decisions_for_target_var=user_decisions_for_target_var,
            min_length_after_processing=min_required_length_for_series,
            flight_window_abs_timestamps=current_flight_window_for_file 
        )
        if series_data:
            all_final_series.append(series_data)
    
    if flight_indicator_var_name:
        print(f"Skipped {skipped_due_no_flight_window_main} MAT files for '{target_variable_for_dataset}' because no valid flight window was found from '{flight_indicator_var_name}'.")
    
    if not all_final_series:
        print(f"No series for '{target_variable_for_dataset}' met the criteria after processing. No data saved.")
        return

    # --- Output Naming and Saving ---
    target_var_freq_suffix = str(user_decisions_for_target_var['target_hz']) + 'Hz' \
        if user_decisions_for_target_var.get('resample_enabled', False) else 'origrate'
    target_var_freq_suffix = target_var_freq_suffix.replace(".","p") # make it filename friendly
    
    output_dir_name_parts = [target_variable_for_dataset.replace('.', '_'), target_var_freq_suffix]
    if flight_indicator_var_name and flight_indicator_processing_decisions:
        indicator_thresh_info = str(flight_indicator_ground_threshold).replace('.','p') + "units" # Using generic units
        indicator_name_cleaned = flight_indicator_var_name.replace('.','_')
        output_dir_name_parts.append(f"filtBy{indicator_name_cleaned}{indicator_thresh_info}")
    
    variable_specific_output_dir = output_base_dir / "_".join(output_dir_name_parts)
    variable_specific_output_dir.mkdir(parents=True, exist_ok=True)

    num_total = len(all_final_series)
    num_test = int(num_total * test_split_ratio)
    num_train = num_total - num_test
    if num_train == 0 and num_total > 0: # Ensure train is not empty if total is not
        if num_total == 1: num_train = 1; num_test = 0
        else: # if multiple, but num_train became 0, adjust
            num_train = num_total - num_test
            if num_train <=0 and num_total > 0 : # if still 0, put at least one in train
                num_train = 1
                num_test = num_total -1


    if num_train == 0 and num_test == 0 and num_total > 0: # Should not happen if logic above is correct
         print("Warning: Train/Test split resulted in zero samples for both. Putting all in train.")
         num_train = num_total
    elif num_total == 0:
        print("No data to save after train/test split. Exiting")
        return

    train_data = all_final_series[:num_train]
    test_data = all_final_series[num_train:] # This could be empty if num_train took all
    
    final_gluonts_freq_for_target = user_decisions_for_target_var["final_pandas_freq"]

    print(f"\nTotal '{target_variable_for_dataset}' series successfully processed {processing_context_message}: {num_total}")
    print(f"  Training series: {len(train_data)}")
    print(f"  Test series: {len(test_data)}")
    print(f"  Final GluonTS frequency for '{target_variable_for_dataset}': {final_gluonts_freq_for_target}")

    save_gluonts_jsonl(train_data, variable_specific_output_dir / "train.jsonl")
    save_gluonts_jsonl(test_data, variable_specific_output_dir / "test.jsonl")

    # --- Save Run Summary and Metadata ---
    run_summary = {
        "config_file_used": str(config_file_path.resolve()),
        "fdr_mat_source_path": str(fdr_data_root.resolve()),
        "output_base_directory": str(output_base_dir.resolve()),
        "flight_phase_determination": {
            "indicator_variable": flight_indicator_var_name if flight_indicator_var_name else "N/A",
            "indicator_processing_decisions": flight_indicator_processing_decisions if flight_indicator_processing_decisions else "N/A",
            "indicator_ground_threshold": flight_indicator_ground_threshold if flight_indicator_var_name else "N/A",
            "num_mat_files_with_flight_window_found": num_files_with_flight_window if flight_indicator_var_name else "N/A"
        },
        "target_variable_processing": {
            "name": target_variable_for_dataset,
            "analysis_summary": target_var_analysis_results,
            "user_decisions": user_decisions_for_target_var,
        },
        "dataset_output": {
            "final_output_directory": str(variable_specific_output_dir.resolve()),
            "num_train_series": len(train_data),
            "num_test_series": len(test_data),
            "gluonts_frequency": final_gluonts_freq_for_target,
        },
        "filtering_parameters": {
            "model_context_length_for_min_len": model_context_length,
            "model_prediction_length_for_min_len": model_prediction_length,
            "min_len_factor": min_len_factor,
            "min_required_samples_per_series": min_required_length_for_series,
        }
    }
    summary_save_path = variable_specific_output_dir / "data_processing_run_summary.json"
    try:
        with open(summary_save_path, 'w', encoding='utf-8') as f:
            class CustomEncoder(json.JSONEncoder):
                def default(self, o): # Changed 'obj' to 'o' to match superclass if any warning
                    if isinstance(o, Path): return str(o)
                    if isinstance(o, (np.integer, np.int64)): return int(o)
                    if isinstance(o, (np.floating, np.float64)): return float(o)
                    if isinstance(o, (np.bool_)): return bool(o)
                    if isinstance(o, np.ndarray): return o.tolist()
                    if isinstance(o, pd.Timestamp): return o.isoformat()
                    if isinstance(o, pd.Period): return str(o) 
                    if isinstance(o, Set): return list(o) 
                    try:
                        return super().default(o) # Call super().default(o)
                    except TypeError:
                        return f"<object of type {type(o).__name__} not serializable>"
            json.dump(run_summary, f, indent=4, cls=CustomEncoder)
        print(f"Run summary saved to: {summary_save_path}")
    except Exception as e_json: 
        print(f"Could not serialize run summary to JSON: {e_json}. Saving as text.")
        with open(variable_specific_output_dir / "data_processing_run_summary.txt", 'w', encoding='utf-8') as f:
             f.write(str(run_summary))

    gluonts_metadata = {
        "freq": final_gluonts_freq_for_target,
        "prediction_length": model_prediction_length 
    }
    with open(variable_specific_output_dir / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(gluonts_metadata, f, indent=4)
    print(f"GluonTS metadata.json saved in: {variable_specific_output_dir}")

    print(f"\nInteractive data setup complete for '{target_variable_for_dataset}'.")
    print(f"Output directory: {str(variable_specific_output_dir.resolve())}")

if __name__ == "__main__":
    main_interactive_setup()