# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
from typing import Type, Dict
from pathlib import Path
from argparse import ArgumentParser, ArgumentTypeError
from functools import partial
import re
from gluonts.time_feature import TimeFeature
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import torch
from torch.utils.data import Dataset
from pandas.tseries.frequencies import to_offset
from typing import List, Optional 
from gluonts.core.component import validated
from gluonts.dataset import DataEntry
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.split import split
from gluonts.dataset.util import period_index
from gluonts.transform import (
    Chain,
    RemoveFields,
    SetField,
    AsNumpyArray,
    AddObservedValuesIndicator,
    AddTimeFeatures,
    AddAgeFeature,
    VstackFeatures,
    MapTransformation,
    ExpectedNumInstanceSampler,
    InstanceSplitter,
    TestSplitSampler,
    ValidationSplitSampler,
)
from gluonts.model.forecast import SampleForecast
from gluonts.transform import Transformation
class ConvertStartToPeriod(Transformation):
    """
    Converts a string 'start_field' to a pandas.Period object with a given frequency.
    Also ensures that if the field is already a Period object, its frequency matches.
    """
    def __init__(self, field: str, freq: str, errors: str = 'raise'):
        self.field = field
        self.freq = freq
        self.errors = errors # 'raise', 'coerce', 'ignore' for pd.Period

    def __call__(self, data_it, is_train):
        for data_entry in data_it:
            if self.field in data_entry:
                start_val = data_entry[self.field]
                try:
                    if isinstance(start_val, str):
                        data_entry[self.field] = pd.Period(start_val, freq=self.freq)
                    elif isinstance(start_val, pd.Period):
                        if start_val.freqstr != self.freq:
                            # print(f"Warning: Correcting frequency of start_field. Was {start_val.freqstr}, target {self.freq}")
                            data_entry[self.field] = start_val.asfreq(self.freq)
                    # If it's already a Period with correct freq, do nothing
                    # Handle other types like pd.Timestamp if necessary, convert to Period
                    elif isinstance(start_val, pd.Timestamp):
                        data_entry[self.field] = start_val.to_period(freq=self.freq)

                except Exception as e:
                    # Log or handle error based on self.errors policy
                    if self.errors == 'raise':
                        raise ValueError(f"Error converting start_field '{start_val}' to Period with freq '{self.freq}': {e}") from e
                    elif self.errors == 'coerce':
                        print(f"Warning: Coercing start_field '{start_val}' to NaT due to conversion error with freq '{self.freq}': {e}")
                        data_entry[self.field] = pd.NaT # Or handle differently
                    # if 'ignore', do nothing, leave as is
            yield data_entry
# --- 结束 ConvertStartToPeriod 类 ---

sns.set(
    style="white",
    font_scale=1.1,
    rc={"figure.dpi": 125, "lines.linewidth": 2.5, "axes.linewidth": 1.5},
)


def filter_metrics(metrics, select={"ND", "NRMSE", "mean_wQuantileLoss"}):
    return {m: metrics[m].item() for m in select}


def extract(a, t, x_shape): # a is the tensor to gather from, t is the index tensor
    batch_size = t.shape[0]
    # --- 修改以下行 ---
    # 原来的代码: out = a.gather(-1, t.cpu())
    # 确保 a 和 t 在同一个设备上。
    # 如果 a 已经在正确的设备上 (例如 self.device)，那么 t 也应该在该设备上。
    # t 在 q_sample 中创建时应该已经在 self.device 上了。
    # 所以，不需要 t.cpu()，除非 a 也在 cpu 上。
    # 假设 a 已经被正确地移动到了目标设备 (如 self.device)
    
    # 检查设备是否匹配，如果不匹配，则将 t 移动到 a 的设备
    if a.device != t.device:
        t_on_a_device = t.to(a.device)
    else:
        t_on_a_device = t
        
    out = a.gather(-1, t_on_a_device) 
    # --- 结束修改 ---
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(a.device) # Ensure output is also on a's device



def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = (
        torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    )
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.1
    return torch.linspace(beta_start, beta_end, timesteps)


def plot_train_stats(df: pd.DataFrame, y_keys=None, skip_first_epoch=True):
    if skip_first_epoch:
        df = df.iloc[1:, :]
    if y_keys is None:
        y_keys = ["train_loss", "valid_loss"]

    fix, ax = plt.subplots(1, 1, figsize=(6.5, 4))
    for y_key in y_keys:
        sns.lineplot(
            ax=ax,
            data=df,
            x="epochs",
            y=y_key,
            label=y_key.replace("_", " ").capitalize(),
        )
    ax.legend()
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    plt.show()


def get_lags_for_freq(freq_str: str, context_length_for_lags: Optional[int] = None) -> List[int]:
    """
    Generates a list of lags appropriate for the given frequency string.
    Lags are filtered to be less than context_length_for_lags if provided.
    """
    # print(f"utils.py: get_lags_for_freq called with: freq_str='{freq_str}', context_length_for_lags={context_length_for_lags}")
    
    offset = to_offset(freq_str)
    if offset is None:
        raise ValueError(f"Could not parse frequency string: '{freq_str}'")
            
    # print(f"utils.py: Parsed offset: name='{offset.name}', n={offset.n}")

    lags_seq = []
    base_unit_multiplier = offset.n 

    if offset.name in ['S', 'L', 'U', 'N']: 
        points_per_second = 1.0 
        if offset.name == 'S': points_per_second = 1.0 / base_unit_multiplier
        elif offset.name == 'L': points_per_second = 1000.0 / base_unit_multiplier
        elif offset.name == 'U': points_per_second = 1_000_000.0 / base_unit_multiplier
        elif offset.name == 'N': points_per_second = 1_000_000_000.0 / base_unit_multiplier
        
        if points_per_second <= 0: points_per_second = 1.0 

        # Example for 4Hz ("250L"): points_per_second = 1000.0 / 250 = 4.0
        # Lags in terms of number of points.
        if freq_str == "250L": # Specific for 4Hz
             lags_seq = [
                1, 2, 3, 4, # up to 1s
                8,          # 2s
                12,         # 3s
                20,         # 5s
                40,         # 10s
                120,        # 30s
                240         # 60s (1 min)
                # Add more if context_length allows, e.g., 480 (2min)
            ]
        else: # Generic for other S, L, U, N frequencies
            second_intervals = [1, 2, 5, 10, 30, 60, 5*60, 10*60, 30*60, 60*60]
            lags_seq = [int(round(s_interval * points_per_second)) for s_interval in second_intervals]
            if points_per_second > 1.1:
                 num_short_lags = min(int(round(points_per_second)), 7) 
                 short_term_point_lags = list(range(1, num_short_lags + 1))
                 lags_seq.extend(short_term_point_lags)

    elif offset.name in ['T', 'min']: 
        minute_intervals = [1, 2, 3, 5, 10, 15, 30, 60, 120, 24*60]
        lags_seq = [int(round(m_interval / base_unit_multiplier)) for m_interval in minute_intervals]

    elif offset.name == 'H': 
        hour_intervals = [1, 2, 3, 6, 12, 24, 2*24, 3*24, 7*24]
        lags_seq = [int(round(h_interval / base_unit_multiplier)) for h_interval in hour_intervals]

    elif offset.name in ['D', 'B']: 
        day_intervals = [1, 2, 3, 5, 7, 14, 21, 30, 60, 90, 180, 365]
        lags_seq = [int(round(d_interval / base_unit_multiplier)) for d_interval in day_intervals]
        if offset.name == 'B': 
            lags_seq.extend([int(round(w * 5 / base_unit_multiplier)) for w in [1,2,3,4,8,12, 26, 52]])
    
    elif offset.name.startswith('W'): 
        week_intervals = [1,2,3,4,8,13,26,52]
        lags_seq = [int(round(w_interval / base_unit_multiplier)) for w_interval in week_intervals]
    
    elif offset.name.startswith('M'): 
        month_intervals = [1,2,3,6,9,12,18,24]
        lags_seq = [int(round(m_interval / base_unit_multiplier)) for m_interval in month_intervals]

    else:
        print(f"Warning: No specific lag sequence defined for frequency name '{offset.name}' (from {freq_str}). Using generic lags.")
        lags_seq = [1, 2, 3, 4, 5, 6, 7]

    final_lags = sorted(list(set(lag for lag in lags_seq if lag > 0)))
    
    if not final_lags: 
        final_lags = [1]
        print(f"Warning: No positive lags generated for freq '{freq_str}'. Defaulting to lags_seq=[1].")

    if context_length_for_lags is not None:
        original_lag_count = len(final_lags)
        final_lags = [lag for lag in final_lags if lag < context_length_for_lags]
        if not final_lags and original_lag_count > 0 : 
             smallest_original_lag = min(set(lag for lag in lags_seq if lag > 0)) if any(lag > 0 for lag in lags_seq) else 1
             final_lags = [smallest_original_lag] 
             if smallest_original_lag >= context_length_for_lags : # If even the smallest is too large
                 final_lags = [1] # Fallback to 1 if context is extremely small
                 print(f"Warning: Smallest generated lag ({smallest_original_lag}) for '{freq_str}' was >= context_length ({context_length_for_lags}). Defaulting to lags_seq=[1].")
             else:
                 print(f"Warning: All generated lags for '{freq_str}' were >= context_length ({context_length_for_lags}). Using smallest lag: {final_lags}.")
        elif not final_lags and original_lag_count == 0: 
             final_lags = [1]
             print(f"Warning: No lags were generated for '{freq_str}' and context_length was applied. Defaulting to lags_seq=[1].")

    # print(f"utils.py: Final lags_seq for '{freq_str}' (context: {context_length_for_lags}): {final_lags}")
    return final_lags

def create_transforms(
    num_feat_dynamic_real: int,    
    num_feat_static_cat: int,      
    num_feat_static_real: int,     
    time_features: List[TimeFeature], 
    prediction_length: int,
    freq_str: str 
):
    # Initialize list of transformations
    transformations = []

    # 1. Handle removal of static fields if they are not configured to be used
    #    If num_feat_static_cat is 0, AsNumpyArray for it will be skipped.
    #    If num_feat_static_real is 0, AsNumpyArray for it will be skipped.
    #    SetField will ensure these fields exist with defaults if needed by PREDICTION_INPUT_NAMES.
    
    # Ensure FEAT_STATIC_CAT exists, with default if not provided from data & config
    if num_feat_static_cat == 0:
        transformations.append(SetField(output_field=FieldName.FEAT_STATIC_CAT, value=[0]))
    transformations.append(
        AsNumpyArray(field=FieldName.FEAT_STATIC_CAT, expected_ndim=1, dtype=int)
    )

    # Ensure FEAT_STATIC_REAL exists, with default if not provided from data & config
    if num_feat_static_real == 0:
        transformations.append(SetField(output_field=FieldName.FEAT_STATIC_REAL, value=[0.0]))
    transformations.append(
        AsNumpyArray(field=FieldName.FEAT_STATIC_REAL, expected_ndim=1, dtype=np.float32)
    )

    # 2. Process TARGET and other time-based features
    transformations.extend([
        AsNumpyArray(field=FieldName.TARGET, expected_ndim=1, dtype=np.float32),
        ConvertStartToPeriod(field=FieldName.START, freq=freq_str), # Ensure START is a Period
        AddObservedValuesIndicator(
            target_field=FieldName.TARGET,
            output_field=FieldName.OBSERVED_VALUES,
        ),
        AddTimeFeatures( 
            start_field=FieldName.START, 
            target_field=FieldName.TARGET, 
            output_field=FieldName.FEAT_TIME, # Creates 'time_feat'
            time_features=time_features, 
            pred_length=prediction_length,
        ),
        AddAgeFeature(
            target_field=FieldName.TARGET,
            output_field=FieldName.FEAT_AGE, # Creates 'age_feat'
            pred_length=prediction_length,
            log_scale=True,
        ),
    ])

    # 3. Stack dynamic features (time, age, and any *external* dynamic real)
    #    into a single FEAT_DYNAMIC_REAL field.
    input_fields_for_vstack = [FieldName.FEAT_TIME, FieldName.FEAT_AGE]
    
    # If external dynamic real features are configured (num_feat_dynamic_real > 0),
    # we assume they are already present in the data_entry under FieldName.FEAT_DYNAMIC_REAL.
    # VstackFeatures will then combine the generated FEAT_TIME, FEAT_AGE, and this existing FEAT_DYNAMIC_REAL.
    # If num_feat_dynamic_real == 0, then only FEAT_TIME and FEAT_AGE are stacked.
    if num_feat_dynamic_real > 0:
        # This implies that the input data should have a FieldName.FEAT_DYNAMIC_REAL field
        # if num_feat_dynamic_real is > 0 in the config.
        # If it doesn't, an error will occur here or earlier.
        # The `interactive_fdr_setup.py` currently doesn't add external dynamic features.
        # So, for your current setup, num_feat_dynamic_real in config should be 0.
        input_fields_for_vstack.append(FieldName.FEAT_DYNAMIC_REAL) # <--- CORRECTED: Add existing FEAT_DYNAMIC_REAL if it's supposed to be there

    # VstackFeatures creates/overwrites the output_field with the stacked inputs.
    # The output field should be FEAT_DYNAMIC_REAL as InstanceSplitter expects this
    # to create past_feat_dynamic_real and future_feat_dynamic_real.
    transformations.append(
        VstackFeatures( 
            output_field=FieldName.FEAT_DYNAMIC_REAL, 
            input_fields=input_fields_for_vstack,
            # delete_input_fields=True # Default is False. If True, FEAT_TIME and FEAT_AGE are removed after stacking.
                                     # This is usually desired to avoid duplicate data.
        )
    )
    
    # Optional: If you want to explicitly remove the original FEAT_TIME and FEAT_AGE after stacking
    # transformations.append(RemoveFields(field_names=[FieldName.FEAT_TIME, FieldName.FEAT_AGE]))
    # However, VstackFeatures with a common output_field might handle this implicitly if input fields are part of it.
    # If output_field is FEAT_DYNAMIC_REAL and FEAT_TIME was an input, the original FEAT_TIME is effectively gone.

    return Chain(transformations)

def create_splitter(past_length: int, future_length: int, mode: str = "train"):
    if mode == "train":
        instance_sampler = ExpectedNumInstanceSampler(
            num_instances=1,
            min_past=past_length,
            min_future=future_length,
        )
    elif mode == "val":
        instance_sampler = ValidationSplitSampler(min_future=future_length)
    elif mode == "test":
        instance_sampler = TestSplitSampler()
    else: # Added else for robustness
        raise ValueError(f"Unknown mode '{mode}' for create_splitter.")

    splitter = InstanceSplitter(
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=instance_sampler,
        past_length=past_length,
        future_length=future_length,
        # --- 修改 time_series_fields ---
        time_series_fields=[
            FieldName.FEAT_DYNAMIC_REAL, # <--- 从 FEAT_TIME 改为 FEAT_DYNAMIC_REAL
            FieldName.OBSERVED_VALUES
        ],
        # --- 结束修改 ---
        # Output field names for dynamic features will now be:
        # past_feat_dynamic_real, future_feat_dynamic_real
    )
    return splitter
def get_next_file_num(
    base_fname: str,
    base_dir: Path,
    file_type: str = "yaml",
    separator: str = "-",
):
    """Gets the next available file number in a directory.
    e.g., if `base_fname="results"` and `base_dir` has
    files ["results-0.yaml", "results-1.yaml"],
    this function returns 2.

    Parameters
    ----------
    base_fname
        Base name of the file.
    base_dir
        Base directory where files are located.

    Returns
    -------
        Next available file number
    """
    if file_type == "":
        # Directory
        items = filter(
            lambda x: x.is_dir() and x.name.startswith(base_fname),
            base_dir.glob("*"),
        )
    else:
        # File
        items = filter(
            lambda x: x.name.startswith(base_fname),
            base_dir.glob(f"*.{file_type}"),
        )
    run_nums = list(
        map(lambda x: int(x.stem.replace(base_fname + separator, "")), items)
    ) + [-1]

    return max(run_nums) + 1


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentTypeError("Boolean value expected.")


def add_config_to_argparser(config: Dict, parser: ArgumentParser):
    for k, v in config.items():
        sanitized_key = re.sub(r"[^\w\-]", "", k).replace("-", "_")
        val_type = type(v)
        if val_type not in {int, float, str, bool}:
            print(f"WARNING: Skipping key {k}!")
            continue
        if val_type == bool:
            parser.add_argument(f"--{sanitized_key}", type=str2bool, default=v)
        else:
            parser.add_argument(f"--{sanitized_key}", type=val_type, default=v)
    return parser


class AddMeanAndStdFeature(MapTransformation):
    @validated()
    def __init__(
        self,
        target_field: str,
        output_field: str,
        dtype: Type = np.float32,
    ) -> None:
        self.target_field = target_field
        self.feature_name = output_field
        self.dtype = dtype

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        data[self.feature_name] = np.array(
            [data[self.target_field].mean(), data[self.target_field].std()]
        )

        return data


class ScaleAndAddMeanFeature(MapTransformation):
    def __init__(
        self, target_field: str, output_field: str, prediction_length: int
    ) -> None:
        """Scale the time series using mean scaler and
        add the scale to `output_field`.

        Parameters
        ----------
        target_field
            Key for target time series
        output_field
            Key for the mean feature
        prediction_length
            prediction length, only the time series before the
            last `prediction_length` timesteps is used for
            scale computation
        """
        self.target_field = target_field
        self.feature_name = output_field
        self.prediction_length = prediction_length

    def map_transform(self, data, is_train: bool):
        scale = np.mean(
            np.abs(data[self.target_field][..., : -self.prediction_length]),
            axis=-1,
            keepdims=True,
        )
        scale = np.maximum(scale, 1e-7)
        scaled_target = data[self.target_field] / scale
        data[self.target_field] = scaled_target
        data[self.feature_name] = scale

        return data


class ScaleAndAddMinMaxFeature(MapTransformation):
    def __init__(
        self, target_field: str, output_field: str, prediction_length: int
    ) -> None:
        """Scale the time series using min-max scaler and
        add the scale to `output_field`.

        Parameters
        ----------
        target_field
            Key for target time series
        output_field
            Key for the min-max feature
        prediction_length
            prediction length, only the time series before the
            last `prediction_length` timesteps is used for
            scale computation
        """
        self.target_field = target_field
        self.feature_name = output_field
        self.prediction_length = prediction_length

    def map_transform(self, data, is_train: bool):
        full_seq = data[self.target_field][..., : -self.prediction_length]
        min_val = np.min(full_seq, axis=-1, keepdims=True)
        max_val = np.max(full_seq, axis=-1, keepdims=True)
        loc = min_val
        scale = np.maximum(max_val - min_val, 1e-7)
        scaled_target = (full_seq - loc) / scale
        data[self.target_field] = scaled_target
        data[self.feature_name] = (loc, scale)

        return data


def descale(data, scale, scaling_type):
    if scaling_type == "mean":
        return data * scale
    elif scaling_type == "min-max":
        loc, scale = scale
        return data * scale + loc
    else:
        raise ValueError(f"Unknown scaling type: {scaling_type}")


def predict_and_descale(predictor, dataset, num_samples, scaling_type):
    """Generates forecasts using the predictor on the test
    dataset and then scales them back to the original space
    using the scale feature from `ScaleAndAddMeanFeature`
    or `ScaleAndAddMinMaxFeature` transformation.

    Parameters
    ----------
    predictor
        GluonTS predictor
    dataset
        Test dataset
    num_samples
        Number of forecast samples
    scaling_type
        Scaling type should be one of {"mean", "min-max"}
        Min-max scaling is used in TimeGAN, defaults to "mean"

    Yields
    ------
        SampleForecast objects

    Raises
    ------
    ValueError
        If the predictor generates Forecast objects other than SampleForecast
    """
    forecasts = predictor.predict(dataset, num_samples=num_samples)
    for input_ts, fcst in zip(dataset, forecasts):
        scale = input_ts["scale"]
        if isinstance(fcst, SampleForecast):
            fcst.samples = descale(
                fcst.samples, scale, scaling_type=scaling_type
            )
        else:
            raise ValueError("Only SampleForecast objects supported!")
        yield fcst


def to_dataframe_and_descale(input_label, scaling_type) -> pd.DataFrame:
    """Glues together "input" and "label" time series and scales
    the back using the scale feature from transformation.

    Parameters
    ----------
    input_label
        Input-Label pair generated from the test template
    scaling_type
        Scaling type should be one of {"mean", "min-max"}
        Min-max scaling is used in TimeGAN, defaults to "mean"

    Returns
    -------
        A DataFrame containing the time series
    """
    start = input_label[0][FieldName.START]
    scale = input_label[0]["scale"]
    targets = [entry[FieldName.TARGET] for entry in input_label]
    full_target = np.concatenate(targets, axis=-1)
    full_target = descale(full_target, scale, scaling_type=scaling_type)
    index = period_index(
        {FieldName.START: start, FieldName.TARGET: full_target}
    )
    return pd.DataFrame(full_target.transpose(), index=index)


def make_evaluation_predictions_with_scaling(
    dataset, predictor, num_samples: int = 100, scaling_type="mean"
):
    """A customized version of `make_evaluation_predictions` utility
    that first scales the test time series, generates the forecast and
    the scales it back to the original space.

    Parameters
    ----------
    dataset
        Test dataset
    predictor
        GluonTS predictor
    num_samples, optional
        Number of test samples, by default 100
    scaling_type, optional
        Scaling type should be one of {"mean", "min-max"}
        Min-max scaling is used in TimeGAN, defaults to "mean"

    Returns
    -------
        A tuple of forecast and time series iterators
    """
    window_length = predictor.prediction_length + predictor.lead_time
    _, test_template = split(dataset, offset=-window_length)
    test_data = test_template.generate_instances(window_length)
    input_test_data = list(test_data.input)

    return (
        predict_and_descale(
            predictor,
            input_test_data,
            num_samples=num_samples,
            scaling_type=scaling_type,
        ),
        map(
            partial(to_dataframe_and_descale, scaling_type=scaling_type),
            test_data,
        ),
    )


class PairDataset(Dataset):
    def __init__(self, x, y) -> None:
        super().__init__()
        assert x.shape[0] == y.shape[0]
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]


class GluonTSNumpyDataset:
    """GluonTS dataset from a numpy array.

    Parameters
    ----------
    data
        Numpy array of samples with shape [N, T].
    start_date, optional
        Dummy start date field, by default pd.Period("2023", "H")
    """

    def __init__(
        self, data: np.ndarray, start_date: pd.Period = pd.Period("2023", "H")
    ):
        self.data = data
        self.start_date = start_date

    def __iter__(self):
        for ts in self.data:
            item = {"target": ts, "start": self.start_date}
            yield item

    def __len__(self):
        return len(self.data)


class MaskInput(MapTransformation):
    @validated()
    def __init__(
        self,
        target_field: str,
        observed_field: str,
        context_length: int,
        missing_scenario: str,
        missing_values: int,
        dtype: Type = np.float32,
    ) -> None:
        # FIXME: Remove hardcoding of fields
        self.target_field = target_field
        self.observed_field = observed_field
        self.context_length = context_length
        self.missing_scenario = missing_scenario
        self.missing_values = missing_values
        self.dtype = dtype

    def map_transform(self, data: DataEntry, is_train: bool) -> DataEntry:
        data = deepcopy(data)
        data["orig_past_target"] = data["past_target"].copy()
        if self.missing_scenario == "BM-E" and self.missing_values > 0:
            data["past_target"][-self.missing_values :] = 0
            data["past_observed_values"][-self.missing_values :] = 0
        elif self.missing_scenario == "BM-B" and self.missing_values > 0:
            data["past_target"][
                -self.context_length : -self.context_length
                + self.missing_values
            ] = 0
            data["past_observed_values"][
                -self.context_length : -self.context_length
                + self.missing_values
            ] = 0
        elif self.missing_scenario == "RM" and self.missing_values > 0:
            weights = torch.ones(self.context_length)
            missing_idxs = -self.context_length + torch.multinomial(
                weights, self.missing_values, replacement=False
            )
            data["past_target"][missing_idxs] = 0
            data["past_observed_values"][missing_idxs] = 0
        return data


class ConcatDataset:
    def __init__(self, test_pairs, axis=-1) -> None:
        self.test_pairs = test_pairs
        self.axis = axis

    def _concat(self, test_pairs):
        for t1, t2 in test_pairs:
            yield {
                "target": np.concatenate(
                    [t1["target"], t2["target"]], axis=self.axis
                ),
                "start": t1["start"],
            }

    def __iter__(self):
        yield from self._concat(self.test_pairs)
