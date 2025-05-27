# src/fdr_project/data_loader.py
import json
import scipy.io
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import warnings
import calendar

# Suppress specific Pandas warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pandas.core.resample")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in cast")


def get_mat_file_paths(fdr_data_root: Path) -> List[Path]:
    """Scans the directory for .mat files and sorts them."""
    mat_files = sorted(list(fdr_data_root.glob("*.mat")))
    if not mat_files:
        print(f"Warning: No .mat files found in {fdr_data_root}")
    return mat_files

def discover_variables_from_sample_file(sample_mat_file: Path) -> List[str]:
    """
    Loads a sample MAT file and discovers potential time series variables.
    A variable is considered a time series if it's a struct with 'data' and 'Rate' fields.
    """
    print(f"\nDiscovering variables from sample file: {sample_mat_file.name}...")
    try:
        mat_data = scipy.io.loadmat(sample_mat_file, simplify_cells=True)
        potential_variables = []
        for var_name, content in mat_data.items():
            if isinstance(content, dict) and 'data' in content and 'Rate' in content:
                data_field = content['data']
                # Check if data field is a numpy array with content or a scalar number
                if (isinstance(data_field, np.ndarray) and data_field.ndim > 0 and data_field.size > 0) or \
                   isinstance(data_field, (int, float, np.number)) or \
                   (isinstance(data_field, np.ndarray) and data_field.size == 1): # Handle scalar in array
                    potential_variables.append(var_name)
        print(f"Found {len(potential_variables)} potential time series variables: {potential_variables}")
        return sorted(potential_variables)
    except Exception as e:
        print(f"Error discovering variables in {sample_mat_file.name}: {e}")
        return []

def analyze_variable_across_files(
    mat_files: List[Path],
    var_name: str,
    num_files_to_sample: int = 5
) -> Dict[str, Any]:
    """Analyzes a specific variable across a sample of MAT files to check for consistency."""
    print(f"\nAnalyzing variable '{var_name}' across up to {num_files_to_sample} sample files...")
    rates: Set[float] = set()
    lengths: List[int] = []
    descriptions: Set[str] = set()
    units_set: Set[str] = set()
    dtypes_set: Set[str] = set()
    nan_count_total = 0
    inf_count_total = 0
    total_elements_checked_for_nan_inf = 0
    all_zero_samples_count = 0
    files_actually_checked = 0
    non_array_data_field_count = 0

    files_to_check = mat_files[:min(len(mat_files), num_files_to_sample)]

    for mat_file in files_to_check:
        try:
            mat_data = scipy.io.loadmat(mat_file, simplify_cells=True)
            if var_name in mat_data:
                content = mat_data[var_name]
                if isinstance(content, dict) and 'data' in content and 'Rate' in content:
                    files_actually_checked += 1
                    try:
                        rate_val = content['Rate']
                        if isinstance(rate_val, np.ndarray) and rate_val.size == 1: # Handle rate as array
                            rate_val = rate_val.item()
                        rates.add(float(rate_val))
                    except (ValueError, TypeError):
                        print(f"  Warning: Could not parse 'Rate' for '{var_name}' in {mat_file.name} (value: {content.get('Rate')}).")

                    data_array_raw = content['data']
                    if not isinstance(data_array_raw, np.ndarray):
                        if isinstance(data_array_raw, (int, float, np.number)): # Handle scalar data
                            data_array_raw = np.array([data_array_raw])
                        else:
                            non_array_data_field_count += 1
                            # print(f"  Warning: 'data' field for '{var_name}' in {mat_file.name} is not a numpy array or scalar (type: {type(data_array_raw)}). Skipping data analysis for this entry.")
                            continue
                    
                    data_array = data_array_raw.flatten()
                    if data_array.size == 0: 
                        lengths.append(0)
                        continue

                    dtypes_set.add(str(data_array.dtype))
                    lengths.append(len(data_array))

                    if 'Description' in content and isinstance(content['Description'], str): descriptions.add(content['Description'])
                    if 'Units' in content and isinstance(content['Units'], str): units_set.add(content['Units'])

                    try:
                        if np.issubdtype(data_array.dtype, np.number): 
                            data_float = data_array.astype(np.float64) 
                            nan_count_total += np.isnan(data_float).sum()
                            inf_count_total += np.isinf(data_float).sum()
                            total_elements_checked_for_nan_inf += data_float.size
                            if data_float.size > 0 and np.count_nonzero(data_float) == 0 :
                                all_zero_samples_count +=1
                        # else:
                            # print(f"  Data for '{var_name}' in {mat_file.name} is not numeric (dtype: {data_array.dtype}). Cannot check for NaN/Inf.")
                    except (ValueError, TypeError) as e_conv:
                         print(f"  Warning: Could not convert data of '{var_name}' in {mat_file.name} to float for NaN/Inf/zero check: {e_conv}")
        except Exception as e:
            print(f"  Error analyzing '{var_name}' in {mat_file.name}: {e}")
            continue
            
    analysis = {
        "variable_name": var_name,
        "consistent_rate": len(rates) <= 1,
        "rates_found_hz": sorted(list(rates)), 
        "consistent_dtype_in_samples": len(dtypes_set) <= 1,
        "dtypes_found_in_samples": list(dtypes_set),
        "avg_length": np.mean(lengths) if lengths else 0,
        "min_length": np.min(lengths) if lengths else 0,
        "max_length": np.max(lengths) if lengths else 0,
        "descriptions": list(descriptions),
        "units": list(units_set),
        "nan_found_in_samples": nan_count_total > 0,
        "total_nans_in_samples": nan_count_total,
        "inf_found_in_samples": inf_count_total > 0,
        "total_infs_in_samples": inf_count_total,
        "total_elements_for_nan_inf_check": total_elements_checked_for_nan_inf,
        "all_zero_samples_count": all_zero_samples_count,
        "num_files_effectively_analyzed": files_actually_checked,
        "non_array_data_field_count": non_array_data_field_count
    }
    print(f"Analysis for '{var_name}':")
    for key, value in analysis.items():
        if key not in ["descriptions", "units"] or value:
             print(f"  - {key.replace('_', ' ').capitalize()}: {value}")
    return analysis

def ask_yes_no(prompt_message: str) -> bool:
    """Helper to ask a yes/no question."""
    while True:
        response = input(f"{prompt_message} (y/n): ").strip().lower()
        if response == 'y':
            return True
        elif response == 'n':
            return False
        else:
            print("  Invalid input. Please enter 'y' or 'n'.")

def get_user_decision_for_variable(
    var_analysis: Dict[str, Any],
    is_flight_indicator_setup: bool = False, 
    target_var_selected_by_user: Optional[str] = None 
    ) -> Dict[str, Any]:
    """Gets user decisions on how to process the variable based on analysis."""
    decisions = {"process_this_variable": True}
    var_name = var_analysis['variable_name']
    setup_context = "(for flight windowing)" if is_flight_indicator_setup else f"(as final target: {target_var_selected_by_user or var_name})"
    print(f"\n--- Decisions for Variable: {var_name} {setup_context} ---")

    if var_analysis["num_files_effectively_analyzed"] == 0:
        print(f"  Warning: Variable '{var_name}' was not found or was not analyzable in any of the sample files. Skipping.")
        decisions["process_this_variable"] = False
        return decisions

    if not is_flight_indicator_setup: 
        if not var_analysis["consistent_dtype_in_samples"]:
            print(f"  Issue: Inconsistent data types found in samples for '{var_name}': {var_analysis['dtypes_found_in_samples']}.")
            if not ask_yes_no("  Attempt to proceed by casting all to float32? (If 'n', this variable will be skipped)"):
                decisions["process_this_variable"] = False
                return decisions
        elif var_analysis["dtypes_found_in_samples"]:
            dtype_str = var_analysis["dtypes_found_in_samples"][0]
            if dtype_str not in ['float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool']:
                 print(f"  Warning: Data type '{dtype_str}' for '{var_name}' might not be ideal. Will attempt to cast to float32 during processing.")

    original_rate_hz: Optional[float] = None
    rates_found = var_analysis.get("rates_found_hz", [])
    if not rates_found:
        print(f"  Critical Issue: No sampling rate information found for '{var_name}'. Skipping variable.")
        decisions["process_this_variable"] = False
        return decisions

    if len(rates_found) == 1:
        original_rate_hz = rates_found[0]
        print(f"  Original sampling rate for '{var_name}': {original_rate_hz} Hz.")
    else: 
        print(f"  Issue: Inconsistent sampling rates found for '{var_name}': {rates_found}.")
        if not ask_yes_no(f"  Attempt to proceed with '{var_name}'? You'll need to choose one rate to assume as original OR select resampling. (If 'n', skip this variable)"):
            decisions["process_this_variable"] = False
            return decisions
    
    prompt_resample_q = f"  Do you want to apply uniform resampling for '{var_name}'?"
    if is_flight_indicator_setup and original_rate_hz and original_rate_hz != 4.0 and original_rate_hz > 0:
        prompt_resample_q = f"  '{var_name}' (rate {original_rate_hz}Hz) is for flight windowing. For robust window detection, consider resampling (e.g., to 4Hz). Resample '{var_name}' for this purpose?"
    elif not is_flight_indicator_setup and original_rate_hz and original_rate_hz > 20:
         print(f"  Suggestion: Rate {original_rate_hz}Hz for '{var_name}' is high. Uniform resampling is recommended for TSDiff.")


    if ask_yes_no(prompt_resample_q):
        decisions["resample_enabled"] = True
        while True:
            try:
                default_target_hz_suggestion = ""
                if is_flight_indicator_setup and original_rate_hz and original_rate_hz > 4.0 : default_target_hz_suggestion = " (e.g., 4.0 for robust windowing)"
                elif original_rate_hz : default_target_hz_suggestion = f" (original was {original_rate_hz})"
                
                target_hz_str = input(f"    Enter target sampling frequency in Hz for '{var_name}'{default_target_hz_suggestion}: ").strip()
                decisions["target_hz"] = float(target_hz_str)
                if decisions["target_hz"] <= 0: raise ValueError("Frequency must be positive.")
                
                # Determine effective original_hz for comparison if it was inconsistent
                effective_original_hz_for_comparison = original_rate_hz
                if not effective_original_hz_for_comparison and rates_found: # inconsistent and not chosen yet
                    # This case should be rare if logic above forces a choice for original_rate_hz
                    effective_original_hz_for_comparison = rates_found[0] # fallback to first found for comparison message

                if effective_original_hz_for_comparison:
                    if decisions["target_hz"] > effective_original_hz_for_comparison: print(f"    Note: Target Hz ({decisions['target_hz']}) for '{var_name}' is higher than original ({effective_original_hz_for_comparison}). This implies upsampling.")
                    elif decisions["target_hz"] < effective_original_hz_for_comparison: print(f"    Note: Target Hz ({decisions['target_hz']}) for '{var_name}' is lower than original ({effective_original_hz_for_comparison}). This implies downsampling.")
                break
            except ValueError as e: print(f"    Invalid input: {e}")
        
        effective_original_hz_for_methods = original_rate_hz
        if not effective_original_hz_for_methods: # If original_rate_hz is still None (inconsistent rates)
             while True: # Force user to pick one for deciding agg/fill methods
                try:
                    chosen_orig_rate_str = input(f"    Rates for '{var_name}' were {rates_found}. To decide agg/fill methods for resampling, please select ONE rate from this list to consider as 'original': ").strip()
                    effective_original_hz_for_methods = float(chosen_orig_rate_str)
                    if effective_original_hz_for_methods not in rates_found: raise ValueError("Rate not in the detected list.")
                    original_rate_hz = effective_original_hz_for_methods # Update original_rate_hz with user's choice for consistency
                    break
                except ValueError as e: print(f"    Invalid input: {e}")
        
        is_downsampling = effective_original_hz_for_methods and decisions["target_hz"] < effective_original_hz_for_methods
        is_upsampling = effective_original_hz_for_methods and decisions["target_hz"] > effective_original_hz_for_methods

        if is_downsampling:
            agg_methods = ["mean", "median", "first", "last", "sum"]
            default_agg = "mean"
            print(f"    '{var_name}' will be downsampled.")
            while True:
                agg_method_str = input(f"    Choose downsampling aggregation for '{var_name}' ({', '.join(agg_methods)}, default: {default_agg}): ").strip().lower() or default_agg
                if agg_method_str in agg_methods:
                    decisions["resample_agg_method"] = agg_method_str; decisions["resample_fill_method"] = None; break
                else: print(f"    Invalid method. Choose from: {', '.join(agg_methods)}")
        elif is_upsampling:
            upsample_methods = ["ffill", "bfill", "interpolate"]
            default_upsample = "interpolate" if is_flight_indicator_setup else "ffill"
            print(f"    '{var_name}' will be upsampled.")
            while True:
                fill_method_str = input(f"    Choose upsampling fill/interpolation for '{var_name}' ({', '.join(upsample_methods)}, default: {default_upsample}): ").strip().lower() or default_upsample
                if fill_method_str in upsample_methods:
                    decisions["resample_fill_method"] = fill_method_str; decisions["resample_agg_method"] = None
                    if fill_method_str == "interpolate": decisions["interpolate_method"] = "linear" 
                    break
                else: print(f"    Invalid method. Choose from: {', '.join(upsample_methods)}")
        else: 
             print(f"    Target rate for '{var_name}' ({decisions['target_hz']} Hz) matches effective original or implies mixed methods if original was inconsistent. Defaults will be used if necessary.")
             decisions["resample_agg_method"] = "mean" 
             decisions["resample_fill_method"] = "ffill"
        
        decisions["final_pandas_freq"] = hz_to_pandas_freq(decisions["target_hz"])
    else: # No resampling enabled for this variable
        decisions["resample_enabled"] = False
        decisions["resample_agg_method"] = None
        decisions["resample_fill_method"] = None
        if original_rate_hz is None: # Rates were inconsistent, user must choose one now as it's the final rate
            print(f"  Rates for '{var_name}' were inconsistent: {rates_found}.")
            while True:
                try:
                    chosen_rate_str = input(f"    Since not resampling '{var_name}', please choose ONE rate from {rates_found} to be its declared frequency: ").strip()
                    original_rate_hz = float(chosen_rate_str)
                    if original_rate_hz not in rates_found: raise ValueError("Chosen rate not in the detected list.")
                    break
                except ValueError as e: print(f"    Invalid input: {e}")
        decisions["target_hz"] = original_rate_hz # This is the declared final rate (Hz value)
        decisions["final_pandas_freq"] = hz_to_pandas_freq(original_rate_hz) 
        
    print(f"  '{var_name}' will be processed. Final GluonTS frequency string: {decisions['final_pandas_freq']}")

    if not is_flight_indicator_setup: # NaN/Inf handling for the actual target variable
        if var_analysis["nan_found_in_samples"] or var_analysis["inf_found_in_samples"]:
            print(f"  Issue: NaNs or Infs found in sample data for '{var_name}'.")
            nan_handling_options = ["skip_file", "interpolate_linear", "ffill_then_bfill", "to_zero", "keep_as_is"]
            default_nan = "interpolate_linear"
            while True:
                nan_choice = input(f"    How to handle files of '{var_name}' with NaNs/Infs? ({', '.join(nan_handling_options)}, default: {default_nan}): ").strip().lower() or default_nan
                if nan_choice in nan_handling_options: decisions["nan_handling_method"] = nan_choice; break
                else: print(f"    Invalid choice. Options are: {', '.join(nan_handling_options)}")
        else:
            decisions["nan_handling_method"] = "keep_as_is"
    else: # For flight indicator variable during its setup for windowing, be robust
        decisions["nan_handling_method"] = "interpolate_linear" 
        if var_analysis["nan_found_in_samples"] or var_analysis["inf_found_in_samples"]:
            print(f"  Note: NaNs/Infs found in '{var_name}' (for windowing). Will use default: '{decisions['nan_handling_method']}'.")

    decisions["determined_original_hz"] = original_rate_hz # Store the rate (Hz) used as the basis before any resampling for this var
    return decisions

def parse_fdr_timestamp_from_mat(mat_data: Dict[str, Any], filename: str) -> pd.Timestamp:
    """
    Returns a fixed default start timestamp (e.g., 1970-01-01) for all series.
    The actual date/time from MAT files are ignored due to known inaccuracies.
    """
    return pd.Timestamp("1970-01-01 00:00:00")

def hz_to_pandas_freq(hz: float) -> str:
    """Converts a frequency in Hz to a Pandas offset alias string."""
    if hz <= 0: raise ValueError("Frequency in Hz must be positive.")
    
    if abs(hz - 1.0) < 1e-9: return "S"      
    if abs(hz - (1/60.0)) < 1e-9: return "T" 
    if abs(hz - (1/3600.0)) < 1e-9: return "H" 
    
    period_seconds = 1.0 / hz
    
    if abs(period_seconds - round(period_seconds)) < 1e-9 and period_seconds >= 1:
        return f"{int(round(period_seconds))}S"
        
    period_ms = period_seconds * 1000.0
    if abs(period_ms - round(period_ms)) < 1e-9 and period_ms >= 1:
        return f"{int(round(period_ms))}L"
        
    period_us = period_seconds * 1_000_000.0
    if abs(period_us - round(period_us)) < 1e-9 and period_us >= 1:
        return f"{int(round(period_us))}U"
        
    period_ns = period_seconds * 1_000_000_000.0
    approx_ns = max(1, int(round(period_ns))) 
    return f"{approx_ns}N"

def get_series_from_mat(
    mat_data_struct: Dict[str, Any], 
    var_name_to_extract: str,
    mat_file_overall_start_ts: pd.Timestamp, 
    file_path_for_debug: Path,
    forced_original_hz: Optional[float] = None 
) -> Optional[Tuple[pd.Series, float]]:
    """
    Extracts a variable from .mat data structure and returns it as a Pandas Series
    with a DatetimeIndex based on its original sampling rate.
    Returns the series and its actual original Hz from file, or None if an error occurs.
    `forced_original_hz` is used to construct the initial DatetimeIndex if the variable's
    rate was inconsistent and the user chose a specific one to assume.
    """
    if var_name_to_extract not in mat_data_struct or not isinstance(mat_data_struct[var_name_to_extract], dict):
        return None

    var_content = mat_data_struct[var_name_to_extract]
    if 'data' not in var_content or 'Rate' not in var_content:
        return None

    actual_original_hz_from_file: float
    try:
        rate_field = var_content['Rate']
        if isinstance(rate_field, np.ndarray) and rate_field.size == 1:
            actual_original_hz_from_file = float(rate_field.item())
        else:
            actual_original_hz_from_file = float(rate_field)
        if actual_original_hz_from_file <= 0: return None
    except (ValueError, TypeError): return None
    
    hz_for_indexing = forced_original_hz if forced_original_hz is not None else actual_original_hz_from_file

    raw_data = var_content['data']
    if not isinstance(raw_data, np.ndarray):
        if isinstance(raw_data, (int, float, np.number)): 
            raw_data = np.array([raw_data])
        else: return None
            
    ts_values = raw_data.flatten().astype(np.float32) 
    if ts_values.size == 0: return None

    try:
        original_pd_freq_for_idx = hz_to_pandas_freq(hz_for_indexing)
        datetime_idx = pd.date_range(
            start=mat_file_overall_start_ts, periods=len(ts_values), freq=original_pd_freq_for_idx
        )
        # Return the series AND the actual original Hz from the file
        return pd.Series(ts_values, index=datetime_idx), actual_original_hz_from_file
    except ValueError: return None


def determine_inflight_window_from_flight_indicator(
    flight_indicator_series: pd.Series, 
    ground_threshold: float
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Determines the in-flight window (start and end timestamps) from a flight indicator series.
    Input series must have a DatetimeIndex.
    """
    if flight_indicator_series.empty: return None
    
    numeric_values = pd.to_numeric(flight_indicator_series.values, errors='coerce')
    valid_value_mask = ~np.isnan(numeric_values)

    if not valid_value_mask.any(): return None

    # Consider only valid numeric values for threshold comparison
    candidate_inflight_indices = np.where(numeric_values[valid_value_mask] > ground_threshold)[0]
    
    if len(candidate_inflight_indices) == 0: return None
        
    # Map these relative indices back to the original series' index positions using the mask
    actual_series_indices = np.arange(len(flight_indicator_series))[valid_value_mask]

    flight_start_actual_idx = actual_series_indices[candidate_inflight_indices[0]]
    flight_end_actual_idx = actual_series_indices[candidate_inflight_indices[-1]]
    
    if flight_end_actual_idx <= flight_start_actual_idx : return None

    flight_start_ts = flight_indicator_series.index[flight_start_actual_idx]
    flight_end_ts = flight_indicator_series.index[flight_end_actual_idx]
    
    return flight_start_ts, flight_end_ts

def process_mat_file_with_decisions(
    mat_file_path: Path,
    target_variable_name_to_process: str,
    processing_decisions_for_target_var: Dict[str, Any], 
    min_length_after_processing: int,
    flight_window_abs_timestamps: Optional[Tuple[pd.Timestamp, pd.Timestamp]]
) -> Optional[Dict[str, Any]]:
    try:
        mat_data_struct = scipy.io.loadmat(mat_file_path, simplify_cells=True)
        mat_overall_start_ts = parse_fdr_timestamp_from_mat(mat_data_struct, mat_file_path.name)

        forced_hz_for_target_var_initial_load = processing_decisions_for_target_var.get("determined_original_hz")

        series_and_actual_hz = get_series_from_mat(
            mat_data_struct, 
            target_variable_name_to_process, 
            mat_overall_start_ts, 
            mat_file_path,
            forced_original_hz=forced_hz_for_target_var_initial_load
        )
        
        if series_and_actual_hz is None: return None
        target_var_series_orig_rate, actual_original_hz_target_var_from_file = series_and_actual_hz
        
        current_ts_pd_series = target_var_series_orig_rate

        if flight_window_abs_timestamps:
            start_flight_abs_ts, end_flight_abs_ts = flight_window_abs_timestamps
            if not current_ts_pd_series.index.is_monotonic_increasing:
                current_ts_pd_series = current_ts_pd_series.sort_index()
            
            # Ensure the slice is valid and series has data within the window
            if start_flight_abs_ts > current_ts_pd_series.index[-1] or end_flight_abs_ts < current_ts_pd_series.index[0]:
                return None # Flight window is outside the data range of this variable
            current_ts_pd_series = current_ts_pd_series.loc[start_flight_abs_ts:end_flight_abs_ts]
            if current_ts_pd_series.empty: return None
        # If flight_window_abs_timestamps is None, we process the whole series (no CAS-based trim)

        current_start_ts_timestamp = current_ts_pd_series.index[0]
        final_pd_freq_str_for_target_var = processing_decisions_for_target_var["final_pandas_freq"]

        if processing_decisions_for_target_var["resample_enabled"]:
            target_hz_for_resample = processing_decisions_for_target_var["target_hz"]
            
            # Only resample if effective original rate significantly differs from target rate
            if abs(actual_original_hz_target_var_from_file - target_hz_for_resample) > 1e-6:
                resampled_pd_target_var: Optional[pd.Series] = None
                if actual_original_hz_target_var_from_file > target_hz_for_resample: 
                    agg_method = processing_decisions_for_target_var["resample_agg_method"]
                    resampled_pd_target_var = current_ts_pd_series.resample(final_pd_freq_str_for_target_var).agg(agg_method)
                elif actual_original_hz_target_var_from_file < target_hz_for_resample: 
                    fill_method = processing_decisions_for_target_var["resample_fill_method"]
                    if fill_method == "interpolate":
                        interp_method = processing_decisions_for_target_var.get("interpolate_method", "linear")
                        resampled_pd_target_var = current_ts_pd_series.resample(final_pd_freq_str_for_target_var).interpolate(method=interp_method)
                    else:
                        resampled_pd_target_var = current_ts_pd_series.resample(final_pd_freq_str_for_target_var).fillna(method=fill_method)
                
                if resampled_pd_target_var is not None:
                    current_ts_pd_series = resampled_pd_target_var.dropna()
            else: 
                 current_ts_pd_series = current_ts_pd_series.asfreq(final_pd_freq_str_for_target_var) # Ensure index conforms

            if current_ts_pd_series.empty: return None
            current_start_ts_timestamp = current_ts_pd_series.index[0]
        else: 
             current_ts_pd_series = current_ts_pd_series.asfreq(final_pd_freq_str_for_target_var)
             if current_ts_pd_series.empty : return None 
             current_ts_pd_series = current_ts_pd_series.dropna() 
             if current_ts_pd_series.empty : return None
             current_start_ts_timestamp = current_ts_pd_series.index[0]

        current_start_period = current_start_ts_timestamp.to_period(final_pd_freq_str_for_target_var)
        current_ts_data_float32 = current_ts_pd_series.values.astype(np.float32)

        nan_handling_method = processing_decisions_for_target_var.get("nan_handling_method", "keep_as_is")
        has_nan = np.isnan(current_ts_data_float32).any()
        has_inf = np.isinf(current_ts_data_float32).any()

        if has_nan or has_inf:
            if nan_handling_method == "skip_file": return None
            temp_series_for_nan_handling = pd.Series(current_ts_data_float32) 
            if nan_handling_method == "to_zero":
                temp_series_for_nan_handling = temp_series_for_nan_handling.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            elif nan_handling_method == "interpolate_linear":
                temp_series_for_nan_handling = temp_series_for_nan_handling.replace([np.inf, -np.inf], np.nan).interpolate(method='linear', limit_direction='both').fillna(method='bfill').fillna(method='ffill')
            elif nan_handling_method == "ffill_then_bfill":
                temp_series_for_nan_handling = temp_series_for_nan_handling.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
            current_ts_data_float32 = temp_series_for_nan_handling.to_numpy().astype(np.float32)
            if np.isnan(current_ts_data_float32).any() or np.isinf(current_ts_data_float32).any(): return None

        if len(current_ts_data_float32) < min_length_after_processing:
            return None
        
        return {
            "target": current_ts_data_float32,
            "start": current_start_period, 
            "item_id": f"{mat_file_path.stem}_{target_variable_name_to_process}",
            "final_gluonts_freq": final_pd_freq_str_for_target_var
        }
    except Exception as e:
        print(f"  Major error during processing of {mat_file_path.name} for '{target_variable_name_to_process}': {e}")
        import traceback
        traceback.print_exc()
        return None

def save_gluonts_jsonl(dataset: List[Dict[str, Any]], file_path: Path):
    """Saves the dataset in GluonTS JSON Lines format."""
    if not dataset:
        print(f"No data to save to {file_path}.")
        return
    print(f"Saving {len(dataset)} series to {file_path}...")
    try:
        with file_path.open('w', encoding='utf-8') as f:
            for series_data in dataset:
                start_value_for_json = str(series_data["start"])
                gluonts_entry = {
                    "start": start_value_for_json, 
                    "target": series_data["target"].tolist(), 
                    "item_id": series_data.get("item_id", f"unknown_item_{pd.Timestamp.now().isoformat()}"),
                    "feat_static_cat": [0], 
                }
                f.write(json.dumps(gluonts_entry) + "\n")
        print(f"Successfully saved data to {file_path}")
    except Exception as e:
        print(f"Error saving data to {file_path}: {e}")