# src/fdr_project/data_loader.py
import json
import scipy.io
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import warnings
import calendar

# Suppress specific Pandas warnings that might occur with mixed-frequency resampling
warnings.filterwarnings("ignore", category=UserWarning, module="pandas.core.resample")

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
                if isinstance(content['data'], np.ndarray) and content['data'].ndim > 0 and content['data'].size > 0:
                     potential_variables.append(var_name)
                elif isinstance(content['data'], np.ndarray) and content['data'].ndim == 0 and content['data'].size == 1:
                     potential_variables.append(var_name)
        print(f"Found {len(potential_variables)} potential time series variables.")
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

    files_to_check = mat_files[:min(len(mat_files), num_files_to_sample)]

    for mat_file in files_to_check:
        try:
            mat_data = scipy.io.loadmat(mat_file, simplify_cells=True)
            if var_name in mat_data:
                content = mat_data[var_name]
                if isinstance(content, dict) and 'data' in content and 'Rate' in content:
                    files_actually_checked += 1
                    try:
                        rates.add(float(content['Rate']))
                    except (ValueError, TypeError):
                        print(f"  Warning: Could not parse 'Rate' for '{var_name}' in {mat_file.name}.")

                    data_array_raw = content['data']
                    if not isinstance(data_array_raw, np.ndarray): # Handle non-array data field
                        print(f"  Warning: 'data' field for '{var_name}' in {mat_file.name} is not a numpy array (type: {type(data_array_raw)}). Skipping this entry for analysis.")
                        continue
                    
                    data_array = data_array_raw.flatten()
                    dtypes_set.add(str(data_array.dtype))
                    lengths.append(len(data_array))

                    if 'Description' in content and isinstance(content['Description'], str): descriptions.add(content['Description'])
                    if 'Units' in content and isinstance(content['Units'], str): units_set.add(content['Units'])

                    try:
                        data_float = data_array.astype(np.float64)
                        nan_count_total += np.isnan(data_float).sum()
                        inf_count_total += np.isinf(data_float).sum()
                        total_elements_checked_for_nan_inf += data_float.size
                        if data_float.size > 0 and np.count_nonzero(data_float) == 0 :
                            all_zero_samples_count +=1
                    except (ValueError, TypeError):
                         print(f"  Warning: Could not convert data of '{var_name}' in {mat_file.name} to float for NaN/Inf/zero check.")
        except Exception as e:
            print(f"  Could not analyze '{var_name}' in {mat_file.name}: {e}")
            continue
            
    analysis = {
        "variable_name": var_name,
        "consistent_rate": len(rates) <= 1,
        "rates_found_hz": list(rates),
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
        "num_files_effectively_analyzed": files_actually_checked
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

def get_user_decision_for_variable(var_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Gets user decisions on how to process the variable based on analysis."""
    decisions = {"process_this_variable": True}
    var_name = var_analysis['variable_name']
    print(f"\n--- Decisions for Variable: {var_name} ---")

    if var_analysis["num_files_effectively_analyzed"] == 0:
        print(f"  Warning: Variable '{var_name}' was not found or was not analyzable in any of the sample files. Skipping.")
        decisions["process_this_variable"] = False
        return decisions

    if not var_analysis["consistent_dtype_in_samples"]:
        print(f"  Issue: Inconsistent data types found in samples: {var_analysis['dtypes_found_in_samples']}.")
        if not ask_yes_no("  Attempt to proceed by casting all to float32? (If 'n', this variable will be skipped)"):
            decisions["process_this_variable"] = False
            return decisions
    elif var_analysis["dtypes_found_in_samples"]:
        dtype_str = var_analysis["dtypes_found_in_samples"][0]
        if dtype_str not in ['float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']:
             print(f"  Warning: Data type '{dtype_str}' might not be ideal. Will attempt to cast to float32 during processing.")

    original_rate_hz: Optional[float] = None
    rates_found = var_analysis.get("rates_found_hz", [])
    if not rates_found:
        print(f"  Critical Issue: No sampling rate information found for '{var_name}'. Skipping variable.")
        decisions["process_this_variable"] = False
        return decisions

    if var_analysis["consistent_rate"]:
        original_rate_hz = rates_found[0]
        print(f"  Consistent original sampling rate: {original_rate_hz} Hz.")
    else:
        print(f"  Issue: Inconsistent sampling rates found: {rates_found}.")
        if not ask_yes_no("  Attempt to proceed? You'll need to choose a rate or opt for resampling. (If 'n', skip variable)"):
            decisions["process_this_variable"] = False
            return decisions

    if original_rate_hz and original_rate_hz > 20:
        print(f"  Suggestion: Rate {original_rate_hz}Hz is high. Uniform resampling is recommended for TSDiff.")
    
    if ask_yes_no("  Do you want to apply uniform resampling (downsample or upsample) to a target frequency for this variable?"):
        decisions["resample_enabled"] = True
        while True:
            try:
                target_hz_str = input(f"    Enter target sampling frequency in Hz (e.g., 1 for 1Hz, 0.5 for 1 sample every 2s, 4 for 4Hz): ").strip()
                decisions["target_hz"] = float(target_hz_str)
                if decisions["target_hz"] <= 0: raise ValueError("Frequency must be positive.")
                if original_rate_hz: # Only print comparison if original_rate_hz is known
                    if decisions["target_hz"] > original_rate_hz:
                        print(f"    Note: Target Hz ({decisions['target_hz']}) is higher than original ({original_rate_hz}). This implies upsampling.")
                    elif decisions["target_hz"] < original_rate_hz:
                        print(f"    Note: Target Hz ({decisions['target_hz']}) is lower than original ({original_rate_hz}). This implies downsampling.")
                break
            except ValueError as e: print(f"    Invalid input: {e}")
        
        is_downsampling = original_rate_hz and decisions["target_hz"] < original_rate_hz
        is_upsampling = original_rate_hz and decisions["target_hz"] > original_rate_hz

        if is_downsampling:
            agg_methods = ["mean", "first", "last", "median", "sum"]
            print("    You are downsampling.")
            while True:
                agg_method_str = input(f"    Choose downsampling aggregation method ({', '.join(agg_methods)}): ").strip().lower()
                if agg_method_str in agg_methods:
                    decisions["resample_agg_method"] = agg_method_str; decisions["resample_fill_method"] = None; break
                else: print(f"    Invalid method. Choose from: {', '.join(agg_methods)}")
        elif is_upsampling:
            upsample_methods = ["ffill", "bfill", "interpolate"]
            print("    You are upsampling.")
            while True:
                fill_method_str = input(f"    Choose upsampling fill/interpolation method ({', '.join(upsample_methods)}): ").strip().lower()
                if fill_method_str in upsample_methods:
                    decisions["resample_fill_method"] = fill_method_str; decisions["resample_agg_method"] = None
                    if fill_method_str == "interpolate": decisions["interpolate_method"] = "linear" 
                    break
                else: print(f"    Invalid method. Choose from: {', '.join(upsample_methods)}")
        else:
             print(f"    Resampling to {decisions['target_hz']} Hz. Method will be applied per file based on its original rate relative to target.")
             decisions["resample_agg_method"] = "mean" 
             decisions["resample_fill_method"] = "ffill"
             if not ask_yes_no(f"    Use '{decisions['resample_agg_method']}' for potential downsampling and '{decisions['resample_fill_method']}' for potential upsampling to {decisions['target_hz']}Hz? (y/n)"):
                agg_methods = ["mean", "first", "last", "median", "sum"]; upsample_methods = ["ffill", "bfill", "interpolate"]
                while True:
                    agg_m = input(f"      Enter default downsample aggregation ({', '.join(agg_methods)}): ").strip().lower()
                    if agg_m in agg_methods: decisions["resample_agg_method"] = agg_m; break
                while True:
                    fill_m = input(f"      Enter default upsample fill ({', '.join(upsample_methods)}): ").strip().lower()
                    if fill_m in upsample_methods: decisions["resample_fill_method"] = fill_m; break
        
        decisions["final_pandas_freq"] = hz_to_pandas_freq(decisions["target_hz"])
    else: 
        decisions["resample_enabled"] = False
        decisions["resample_agg_method"] = None
        decisions["resample_fill_method"] = None
        if original_rate_hz is None: 
            print(f"  Rates were inconsistent: {rates_found}.")
            while True:
                try:
                    chosen_rate_str = input(f"    Since you are not resampling, please choose ONE rate from {rates_found} to be the declared frequency for all files of '{var_name}': ").strip()
                    chosen_rate = float(chosen_rate_str)
                    if chosen_rate not in rates_found: raise ValueError("Rate not in the detected list.")
                    original_rate_hz = chosen_rate 
                    print(f"    All files for '{var_name}' will be assumed to have/target this rate: {original_rate_hz} Hz.")
                    break
                except ValueError as e: print(f"    Invalid input: {e}")
        decisions["target_hz"] = original_rate_hz
        decisions["final_pandas_freq"] = hz_to_pandas_freq(original_rate_hz)
        
    print(f"  This variable will be processed. Final GluonTS frequency will be: {decisions['final_pandas_freq']}")

    if var_analysis["nan_found_in_samples"] or var_analysis["inf_found_in_samples"]:
        print(f"  Issue: NaNs or Infs found in sample data for '{var_name}'.")
        nan_handling_options = ["skip_file", "interpolate_linear", "ffill_then_bfill", "to_zero", "keep_as_is"]
        while True:
            nan_choice = input(f"    How to handle files with NaNs/Infs? ({', '.join(nan_handling_options)}): ").strip().lower()
            if nan_choice in nan_handling_options: decisions["nan_handling_method"] = nan_choice; break
            else: print(f"    Invalid choice. Options are: {', '.join(nan_handling_options)}")
    else:
        decisions["nan_handling_method"] = "keep_as_is"
    return decisions

def validate_and_fix_timestamp(year_val, month_val, day_val, hour_val, minute_val, sec_val=0) -> str:
    default_timestamp_str = "1900-01-01 00:00:00"
    valid_year_range = (1970, 2050) 
    original_inputs = f"Y:{year_val} M:{month_val} D:{day_val} H:{hour_val} m:{minute_val} s:{sec_val}"
    corrected_parts_log = []
    try:
        year = int(year_val); month = int(month_val); day = int(day_val)
        hour = int(hour_val); minute = int(minute_val); second = int(sec_val)
        valid = True
        if not (valid_year_range[0] <= year <= valid_year_range[1]):
            corrected_parts_log.append(f"Year({year} not in {valid_year_range})")
            valid = False
        if not (1 <= month <= 12):
            corrected_parts_log.append(f"Month({month} not in 1-12)")
            valid = False
        if valid: # Only check day if month and year were somewhat plausible
            try:
                _, last_day_of_month = calendar.monthrange(year, month)
                if not (1 <= day <= last_day_of_month):
                    corrected_parts_log.append(f"Day({day} not in 1-{last_day_of_month} for Y{year}M{month})")
                    valid = False
            except Exception: valid = False; corrected_parts_log.append(f"Day({day}) with Y{year}M{month} invalid")
        elif not (1 <= day <= 31): corrected_parts_log.append(f"Day({day}) not in 1-31"); valid = False
        if not (0 <= hour <= 23): corrected_parts_log.append(f"Hour({hour}) not in 0-23"); valid = False
        if not (0 <= minute <= 59): corrected_parts_log.append(f"Minute({minute}) not in 0-59"); valid = False
        if not (0 <= second <= 59): corrected_parts_log.append(f"Second({second}) not in 0-59"); valid = False
        if not valid:
            print(f"    [TS Warning] Invalid components found: {'; '.join(corrected_parts_log)}. Original: [{original_inputs}]. Defaulting timestamp.")
            return default_timestamp_str
        return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:02d}"
    except (ValueError, TypeError) as e:
        print(f"    [TS Error] Could not parse timestamp components ({original_inputs}): {e}. Using default.")
        return default_timestamp_str

def decode_bcd_value(decimal_read_from_mat: int, max_expected_decimal_after_decode: int) -> Optional[int]:
    if not isinstance(decimal_read_from_mat, (int, np.integer)): return None
    try:
        if 0 <= decimal_read_from_mat <= 0x99 : 
            tens_digit_from_hex = (decimal_read_from_mat >> 4) & 0xF
            ones_digit_from_hex = decimal_read_from_mat & 0xF
            if 0 <= tens_digit_from_hex <= 9 and 0 <= ones_digit_from_hex <= 9:
                decoded_decimal_val = tens_digit_from_hex * 10 + ones_digit_from_hex
                if 0 <= decoded_decimal_val <= max_expected_decimal_after_decode:
                    return decoded_decimal_val
    except Exception: pass 
    if 0 <= decimal_read_from_mat <= max_expected_decimal_after_decode:
        return int(decimal_read_from_mat)
    return None

def parse_fdr_timestamp_from_mat(mat_data: Dict[str, Any], filename: str) -> pd.Timestamp:
    default_ts_str = "1900-01-01 00:00:00"
    try:
        year_raw = int(mat_data.get('DATE_YEAR', {}).get('data', [99999]).flat[0])
        month_raw = int(mat_data.get('DATE_MONTH', {}).get('data', [99]).flat[0])
        day_raw = int(mat_data.get('DATE_DAY', {}).get('data', [99]).flat[0])
        hour_raw = int(mat_data.get('GMT_HOUR', {}).get('data', [99]).flat[0])
        minute_raw = int(mat_data.get('GMT_MINUTE', {}).get('data', [999]).flat[0])
        second_raw = int(mat_data.get('GMT_SEC', {}).get('data', [99]).flat[0])
        
        # print(f"  Debug Raw Timestamp Values in {filename}: Y_raw={year_raw}, M_raw={month_raw}, D_raw={day_raw}, H_raw={hour_raw}, Min_raw={minute_raw}, S_raw={second_raw}")
        year_parsed = year_raw 
        month_parsed = decode_bcd_value(month_raw, 12)
        day_parsed = decode_bcd_value(day_raw, 31) 
        hour_parsed = decode_bcd_value(hour_raw, 23)
        minute_parsed = decode_bcd_value(minute_raw, 59)
        second_parsed = decode_bcd_value(second_raw, 59)
        # print(f"  Debug Decoded/Attempted Timestamp Values for {filename}: Y_parsed={year_parsed}, M_parsed={month_parsed}, D_parsed={day_parsed}, H_parsed={hour_parsed}, Min_parsed={minute_parsed}, S_parsed={second_parsed}")
        
        timestamp_str = validate_and_fix_timestamp(
            year_parsed,
            month_parsed if month_parsed is not None else 99,
            day_parsed if day_parsed is not None else 99,
            hour_parsed if hour_parsed is not None else 99,
            minute_parsed if minute_parsed is not None else 999,
            second_parsed if second_parsed is not None else 99
        )
        return pd.Timestamp(timestamp_str)
    except Exception as e:
        print(f"  Major error during timestamp parsing in {filename} ({e}). Defaulting.")
        return pd.Timestamp(default_ts_str)

def hz_to_pandas_freq(hz: float) -> str:
    if hz <= 0: raise ValueError("Frequency in Hz must be positive.")
    if abs(hz - 1.0) < 1e-9: return "S"
    if abs(hz - 2.0) < 1e-9: return "500L"
    if abs(hz - 4.0) < 1e-9: return "250L"
    if abs(hz - 8.0) < 1e-9: return "125L"
    if abs(hz - 16.0) < 1e-9: return "62500U"
    if abs(hz - 0.25) < 1e-9: return "4S"
    if abs(hz - 0.5) < 1e-9: return "2S"

    period_seconds = 1.0 / hz
    if abs(period_seconds - round(period_seconds)) < 1e-9 and period_seconds > 0:
        return f"{int(round(period_seconds))}S"
    period_ms = period_seconds * 1000.0
    if abs(period_ms - round(period_ms)) < 1e-9 and period_ms >= 1:
        return f"{int(round(period_ms))}L"
    period_us = period_seconds * 1_000_000.0
    if abs(period_us - round(period_us)) < 1e-9 and period_us >= 1:
        return f"{int(round(period_us))}U"
    period_ns = period_seconds * 1_000_000_000.0
    if abs(period_ns - round(period_ns)) < 1e-9 and period_ns >=1 :
         return f"{int(round(period_ns))}N"
    approx_ns = max(1, int(round(period_ns)))
    # print(f"  Warning: No standard integer Pandas freq for {hz}Hz (period: {period_seconds:.4g}s). Approximating to {approx_ns}N.")
    return f"{approx_ns}N"

def process_mat_file_with_decisions(
    mat_file_path: Path,
    target_variable_name: str,
    processing_decisions: Dict[str, Any],
    min_length_after_processing: int
) -> Optional[Dict[str, Any]]:
    try:
        mat_data = scipy.io.loadmat(mat_file_path, simplify_cells=True)
        if target_variable_name not in mat_data or not isinstance(mat_data[target_variable_name], dict):
            return None
        var_struct = mat_data[target_variable_name]
        if 'data' not in var_struct or 'Rate' not in var_struct:
            return None
            
        raw_ts_data_untyped = var_struct['data'].flatten()
        if raw_ts_data_untyped.size == 0: return None

        original_hz: float
        try:
            original_hz = float(var_struct['Rate'])
            if original_hz <= 0:
                print(f"  Skipping {mat_file_path.name}: Invalid original rate {original_hz}Hz for '{target_variable_name}'.")
                return None
        except (ValueError, TypeError) as e:
            print(f"  Skipping {mat_file_path.name}: Could not parse 'Rate' for '{target_variable_name}': {e}")
            return None
        
        current_ts_data_float32: Optional[np.ndarray] = None
        if target_variable_name == "CAS":
            if raw_ts_data_untyped.dtype == np.uint8:
                meaningful_values_ratio = np.count_nonzero(raw_ts_data_untyped > 1) / raw_ts_data_untyped.size if raw_ts_data_untyped.size > 0 else 0
                if meaningful_values_ratio < 0.05: return None
            try: current_ts_data_float32 = raw_ts_data_untyped.astype(np.float32)
            except ValueError: return None
        else:
            try: current_ts_data_float32 = raw_ts_data_untyped.astype(np.float32)
            except ValueError: return None
        
        if current_ts_data_float32 is None: return None

        start_ts = parse_fdr_timestamp_from_mat(mat_data, mat_file_path.name)
        original_pd_freq = hz_to_pandas_freq(original_hz)
        
        current_ts_for_resample = current_ts_data_float32
        current_start_ts = start_ts
        final_pd_freq_for_series = original_pd_freq

        if processing_decisions["resample_enabled"]:
            target_hz = processing_decisions["target_hz"]
            target_pd_freq_config = processing_decisions["final_pandas_freq"]
            final_pd_freq_for_series = target_pd_freq_config

            if abs(original_hz - target_hz) > 1e-6 : 
                datetime_idx: Optional[pd.DatetimeIndex] = None
                try:
                    datetime_idx = pd.date_range(start=current_start_ts, periods=len(current_ts_for_resample), freq=original_pd_freq)
                except ValueError as e_dr: 
                    print(f"    Warning: Could not create DateRangeIndex for resampling {mat_file_path.name} with freq '{original_pd_freq}' (orig_hz={original_hz}). Error: {e_dr}. Using original data but declaring target freq.")
                    # Fallback: use original data but still declare target frequency
                    # No actual resampling happens, current_ts_data_float32 remains as is
                    # This might lead to frequency inconsistencies if not handled carefully downstream
                    # Or, decide to skip the file:
                    # print(f"    Skipping {mat_file_path.name} due to date_range issue for resampling.")
                    # return None
                
                if datetime_idx is not None: # Proceed if index was created
                    series_pd = pd.Series(current_ts_for_resample, index=datetime_idx)
                    resampled_pd: Optional[pd.Series] = None
                    if original_hz > target_hz: 
                        agg_method = processing_decisions["resample_agg_method"]
                        resampled_pd = series_pd.resample(target_pd_freq_config).agg(agg_method).dropna()
                    else: 
                        fill_method = processing_decisions["resample_fill_method"]
                        if fill_method == "interpolate":
                            interp_method = processing_decisions.get("interpolate_method", "linear")
                            resampled_pd = series_pd.resample(target_pd_freq_config).interpolate(method=interp_method).dropna()
                        else: 
                            resampled_pd = series_pd.resample(target_pd_freq_config).fillna(method=fill_method).dropna()
                    
                    if resampled_pd is not None and not resampled_pd.empty:
                        current_ts_data_float32 = resampled_pd.to_numpy().astype(np.float32)
                        current_start_ts = resampled_pd.index[0]
                    elif resampled_pd is not None and resampled_pd.empty: return None
        else: 
            final_pd_freq_for_series = processing_decisions["final_pandas_freq"]

        nan_handling = processing_decisions.get("nan_handling_method", "keep_as_is")
        has_nan = np.isnan(current_ts_data_float32).any()
        has_inf = np.isinf(current_ts_data_float32).any()

        if has_nan or has_inf:
            if nan_handling == "skip_file": return None
            temp_series = pd.Series(current_ts_data_float32)
            if nan_handling == "to_zero":
                temp_series = temp_series.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            elif nan_handling == "interpolate_linear":
                temp_series = temp_series.replace([np.inf, -np.inf], np.nan).interpolate(method='linear', limit_direction='both').fillna(method='bfill').fillna(method='ffill')
            elif nan_handling == "ffill_then_bfill":
                temp_series = temp_series.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
            current_ts_data_float32 = temp_series.to_numpy().astype(np.float32)
            if np.isnan(current_ts_data_float32).any() or np.isinf(current_ts_data_float32).any(): return None
        
        if len(current_ts_data_float32) < min_length_after_processing: return None

        return {
            "target": current_ts_data_float32,
            "start": current_start_ts,
            "item_id": f"{mat_file_path.stem}_{target_variable_name}",
            "final_gluonts_freq": final_pd_freq_for_series
        }
    except Exception as e:
        print(f"  Major error during decided processing of {mat_file_path.name} for '{target_variable_name}': {e}")
        return None

def save_gluonts_jsonl(dataset: List[Dict[str, Any]], file_path: Path):
    if not dataset:
        print(f"No data to save to {file_path}.")
        return
    print(f"Saving {len(dataset)} series to {file_path}...")
    try:
        with file_path.open('w', encoding='utf-8') as f:
            for series_data in dataset:
                gluonts_entry = {
                    "start": str(series_data["start"]),
                    "target": series_data["target"].tolist(),
                    "item_id": series_data.get("item_id", f"unknown_item_{pd.Timestamp.now().isoformat()}"),
                    "feat_static_cat": [0], 
                }
                f.write(json.dumps(gluonts_entry) + "\n")
        print(f"Successfully saved data to {file_path}")
    except Exception as e:
        print(f"Error saving data to {file_path}: {e}")