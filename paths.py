#!/usr/bin/env python3
"""Path helpers for processed shard datasets and experiment outputs."""

import os

# Root directory for processed shard datasets.
# Override with: export CXR_DATA_ROOT=/path/to/all_processed_data
DATA_ROOT = os.environ.get(
    "CXR_DATA_ROOT",
    os.path.join(os.path.expanduser("~"), "data", "cxr_processed"),
)

AVAILABLE_DATASETS = [
    "mimic_shards",
    "indiana_shards",
    "aug_indiana_extended",
    "indiana_shards_zero_shot",
]

OUTPUTS_DIR = os.environ.get("CXR_OUTPUTS_DIR", "outputs")
SAVED_MODELS_DIR = os.environ.get("CXR_SAVED_MODELS_DIR", "saved_models")
LOGS_DIR = os.environ.get("CXR_LOGS_DIR", "logs")


def get_shard_base_path(dataset_name):
    if dataset_name not in AVAILABLE_DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Available: {AVAILABLE_DATASETS}"
        )
    return os.path.join(DATA_ROOT, dataset_name)


def get_train_shards_dir(dataset_name):
    return os.path.join(get_shard_base_path(dataset_name), "train")


def get_val_shards_dir(dataset_name):
    return os.path.join(get_shard_base_path(dataset_name), "val")


def get_test_shards_dir(dataset_name):
    return os.path.join(get_shard_base_path(dataset_name), "test")


def get_metadata_path(dataset_name):
    return os.path.join(get_shard_base_path(dataset_name), "metadata.pkl")


def ensure_output_dirs():
    for directory in (OUTPUTS_DIR, SAVED_MODELS_DIR, LOGS_DIR):
        os.makedirs(directory, exist_ok=True)


def print_paths(dataset_name):
    print(f"Dataset: {dataset_name}")
    print(f"  DATA_ROOT : {DATA_ROOT}")
    print(f"  shards    : {get_shard_base_path(dataset_name)}")
    print(f"  train     : {get_train_shards_dir(dataset_name)}")
    print(f"  val       : {get_val_shards_dir(dataset_name)}")
    print(f"  test      : {get_test_shards_dir(dataset_name)}")
    print(f"  metadata  : {get_metadata_path(dataset_name)}")
