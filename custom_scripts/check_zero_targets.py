import json
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

# --- 配置 ---
processed_data_dir = Path("./data/processed_fdr_interactive/CAS_origfreq") # 您的数据路径
test_file = processed_data_dir / "test.jsonl"
prediction_length = 40 # 从您的配置中获取
# --- 结束配置 ---

zero_target_in_prediction_window_count = 0
near_zero_target_in_prediction_window_count = 0
non_zero_sum_count = 0
total_test_series = 0
problematic_series_ids_zero = []
problematic_series_ids_near_zero = []
abs_target_sums = []

if not test_file.exists():
    print(f"Test file not found: {test_file}")

with open(test_file, 'r', encoding='utf-8') as f:
    for line_idx, line in enumerate(f):
        total_test_series += 1
        entry = json.loads(line)
        target = np.array(entry["target"], dtype=np.float32)
        
        item_id = entry.get("item_id", f"series_{line_idx}")

        if len(target) >= prediction_length:
            prediction_window_target = target[-prediction_length:]
            
            abs_target_sum_window = np.sum(np.abs(prediction_window_target))
            abs_target_sums.append(abs_target_sum_window)

            if abs_target_sum_window == 0.0: # 完全为零
                zero_target_in_prediction_window_count += 1
                problematic_series_ids_zero.append(item_id)
            elif abs_target_sum_window < 1e-5: # 非常接近零 (阈值可调)
                near_zero_target_in_prediction_window_count += 1
                problematic_series_ids_near_zero.append(item_id)
            else:
                non_zero_sum_count +=1
        elif len(target) > 0 : 
            abs_target_sum_window = np.sum(np.abs(target))
            abs_target_sums.append(abs_target_sum_window)
            if abs_target_sum_window == 0.0:
                 zero_target_in_prediction_window_count += 1
                 problematic_series_ids_zero.append(item_id)
            elif abs_target_sum_window < 1e-5:
                 near_zero_target_in_prediction_window_count += 1
                 problematic_series_ids_near_zero.append(item_id)
            else:
                non_zero_sum_count +=1
        else: # target 数组为空
            abs_target_sums.append(0.0)
            zero_target_in_prediction_window_count += 1
            problematic_series_ids_zero.append(item_id)


print(f"\n--- Target Value Analysis for Test Set ({test_file}) ---")
print(f"Total test series checked: {total_test_series}")
print(f"Series with EXACTLY zero target sum in prediction window: {zero_target_in_prediction_window_count}")
print(f"Series with NEARLY zero target sum (<1e-5) in prediction window: {near_zero_target_in_prediction_window_count}")
print(f"Series with non-zero target sum in prediction window: {non_zero_sum_count}")

if total_test_series > 0:
    perc_zero = zero_target_in_prediction_window_count / total_test_series
    perc_near_zero = near_zero_target_in_prediction_window_count / total_test_series
    print(f"Percentage of series with zero-sum target in prediction window: {perc_zero:.2%}")
    print(f"Percentage of series with near-zero-sum target in prediction window: {perc_near_zero:.2%}")

if problematic_series_ids_zero:
    print(f"IDs of series with EXACTLY zero target sum (first 5): {problematic_series_ids_zero[:5]}")
if problematic_series_ids_near_zero:
    print(f"IDs of series with NEARLY zero target sum (first 5): {problematic_series_ids_near_zero[:5]}")

if abs_target_sums:
    plt.figure(figsize=(10,6))
    plt.hist(abs_target_sums, bins=50, log=True) # Use log scale for y-axis if sums vary a lot
    plt.title("Distribution of Absolute Target Sums in Prediction Window (Test Set)")
    plt.xlabel("Absolute Sum of Target in Prediction Window")
    plt.ylabel("Frequency (Log Scale)")
    plt.grid(True, alpha=0.5)
    plot_path = processed_data_dir / "test_set_target_sum_distribution.png"
    plt.savefig(plot_path)
    print(f"Saved distribution plot to: {plot_path}")
    plt.close()
