# src/uncond_ts_diff/dataset.py
import os
import tarfile
from pathlib import Path
from urllib import request
import json 

from gluonts.dataset.common import MetaData, TrainDatasets, load_datasets
from gluonts.dataset.repository.datasets import get_dataset, get_download_path
from gluonts.dataset.jsonl import JsonLinesFile 

default_dataset_path: Path = get_download_path() / "datasets"
wiki2k_download_link: str = "https://github.com/awslabs/gluonts/raw/b89f203595183340651411a41eeb0ee60570a4d9/datasets/wiki2000_nips.tar.gz" # noqa: E501

def _load_custom_jsonl_dataset(
    dataset_dir_path: Path,
    config_freq: str, 
    config_prediction_length: int
) -> TrainDatasets:
    train_file = dataset_dir_path / "train.jsonl"
    test_file = dataset_dir_path / "test.jsonl"
    meta_file = dataset_dir_path / "metadata.json"

    if not train_file.exists():
        raise FileNotFoundError(f"Custom dataset error: Train file not found at {train_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Custom dataset error: Test file not found at {test_file}")
    
    actual_dataset_freq_from_meta = config_freq 
    dataset_meta_prediction_length = config_prediction_length

    if meta_file.exists():
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                ds_meta = json.load(f)
            
            meta_freq = ds_meta.get("freq")
            meta_pred_len = ds_meta.get("prediction_length")

            if meta_freq:
                actual_dataset_freq_from_meta = meta_freq
            if meta_pred_len is not None:
                dataset_meta_prediction_length = meta_pred_len

            if actual_dataset_freq_from_meta != config_freq:
                print(f"Warning: Training config frequency ('{config_freq}') differs from dataset's metadata.json frequency ('{actual_dataset_freq_from_meta}'). "
                      f"The TSDiff model and GluonTS Dataset object will be built using the config frequency: '{config_freq}'. " # Clarified
                      f"Ensure data in '{dataset_dir_path}' was actually prepared for this frequency ('{config_freq}') "
                      f"or update training config's 'freq' to '{actual_dataset_freq_from_meta}' if metadata is correct source of truth for data's actual freq.")
                # For consistency, we will use config_freq for the MetaData object passed to TrainDatasets
                # actual_dataset_freq_from_meta is now just for the warning.
        except Exception as e:
            print(f"Warning: Could not correctly parse metadata.json from {meta_file}: {e}. Relying on config values.")
            # actual_dataset_freq_from_meta remains config_freq
            # dataset_meta_prediction_length remains config_prediction_length
            
    else:
        print(f"Warning: metadata.json not found in {dataset_dir_path}. "
              f"Using freq='{config_freq}' and prediction_length={config_prediction_length} from training config for GluonTS Dataset object.")
        # actual_dataset_freq_from_meta remains config_freq
        # dataset_meta_prediction_length remains config_prediction_length

    # The MetaData object for TrainDatasets will use the frequency from the training configuration.
    # This is what the model and its components (like lag calculation) will rely on.
    metadata_for_gluonts = MetaData(freq=config_freq, prediction_length=config_prediction_length)

    # --- 修改此处：移除 JsonLinesFile 初始化时的 freq 参数 ---
    return TrainDatasets(
        metadata=metadata_for_gluonts,
        train=JsonLinesFile(path=train_file), # freq 参数已移除
        test=JsonLinesFile(path=test_file),   # freq 参数已移除
    )



def get_gts_dataset(dataset_name_or_path: str, config_freq: str, config_prediction_length: int) -> TrainDatasets:
    path_obj = Path(dataset_name_or_path)
    
    if path_obj.is_dir() and (path_obj / "metadata.json").exists() and \
       (path_obj / "train.jsonl").exists() and (path_obj / "test.jsonl").exists():
        print(f"Loading custom dataset from directory: {path_obj}")
        return _load_custom_jsonl_dataset(path_obj, config_freq, config_prediction_length)
    
    elif dataset_name_or_path == "wiki2000_nips":
        wiki_dataset_path = default_dataset_path / dataset_name_or_path
        Path(default_dataset_path).mkdir(parents=True, exist_ok=True)
        if not wiki_dataset_path.exists():
            tar_file_path = wiki_dataset_path.parent / f"{dataset_name_or_path}.tar.gz"
            request.urlretrieve(wiki2k_download_link, tar_file_path)
            with tarfile.open(tar_file_path) as tar: tar.extractall(path=wiki_dataset_path.parent)
            os.remove(tar_file_path)
        
        loaded_ds = load_datasets(
            metadata=wiki_dataset_path / "metadata",
            train=wiki_dataset_path / "train",
            test=wiki_dataset_path / "test",
        )
        if loaded_ds.metadata.freq is None: loaded_ds.metadata.freq = config_freq
        if loaded_ds.metadata.prediction_length is None: loaded_ds.metadata.prediction_length = config_prediction_length
        return loaded_ds
        
    else: 
        print(f"Attempting to load dataset '{dataset_name_or_path}' from GluonTS built-in repository.")
        try:
            ds = get_dataset(dataset_name_or_path, path=default_dataset_path)
            if ds.metadata.freq is None: ds.metadata.freq = config_freq
            if ds.metadata.prediction_length is None: ds.metadata.prediction_length = config_prediction_length
            return ds
        except Exception as e:
            raise ValueError(
                f"Dataset '{dataset_name_or_path}' not found as a custom path "
                f"nor in the GluonTS repository. Error: {e}"
            )