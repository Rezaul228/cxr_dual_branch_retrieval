#!/usr/bin/env python3
"""Central configuration for dual-branch chest X-ray retrieval."""

import os

# Active dataset key (must match a folder under CXR_DATA_ROOT / paths.DATA_ROOT)
DATASET_MODE = os.environ.get("CXR_DATASET_MODE", "aug_indiana_extended")

DATASET_CONFIGS = {
    "mimic_shards": {
        "vocab_size": 10805,
        "max_token_length": 128,
        "embed_dim": 256,
        "num_heads": 8,
        "num_layers": 2,
        "temperature": 0.07,
        "batch_size": 256,
        "learning_rate": 1e-4,
        "epochs": 60,
        "shard_size": 100,
        "train_samples": None,
        "val_samples": None,
    },
    "indiana_shards": {
        "vocab_size": 2559,
        "max_token_length": 64,
        "embed_dim": 256,
        "num_heads": 8,
        "num_layers": 2,
        "temperature": 0.07,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "epochs": 15,
        "shard_size": 100,
        "train_samples": None,
        "val_samples": None,
    },
    "aug_indiana_extended": {
        "vocab_size": 10870,
        "max_token_length": 128,
        "embed_dim": 256,
        "num_heads": 8,
        "num_layers": 2,
        "temperature": 0.07,
        "batch_size": 128,
        "learning_rate": 1e-4,
        "epochs": 50,
        "shard_size": 100,
        "train_samples": None,
        "val_samples": None,
    },
    "indiana_shards_zero_shot": {
        "vocab_size": 2552,
        "max_token_length": 128,
        "embed_dim": 256,
        "num_heads": 8,
        "num_layers": 2,
        "temperature": 0.07,
        "batch_size": 32,
        "learning_rate": 1e-4,
        "epochs": 20,
        "shard_size": 50,
        "train_samples": None,
        "val_samples": None,
    },
}

DEFAULT_TRAINING_CONFIG = {
    "device": "cuda",
    "num_workers": 0,
    "pin_memory": True,
    "gradient_clip": 1.0,
    "weight_decay": 1e-5,
    "scheduler_patience": 10,
    "scheduler_factor": 0.5,
    "early_stopping_patience": 10,
    "save_best_only": True,
    "monitor_metric": "recall@1",
    # Loss mix used in the reported experiments
    "synergy_weight": 0.64,
    "main_weight": 0.20,
    "ortho_weight": 0.15,
}


def get_current_config():
    if DATASET_MODE not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset mode: {DATASET_MODE}. "
            f"Available: {list(DATASET_CONFIGS.keys())}"
        )
    return DATASET_CONFIGS[DATASET_MODE]


def get_vocab_size():
    return get_current_config()["vocab_size"]


def get_max_token_length():
    return get_current_config()["max_token_length"]


def get_embed_dim():
    return get_current_config()["embed_dim"]


def get_default_batch_size():
    return get_current_config()["batch_size"]


def get_default_learning_rate():
    return get_current_config()["learning_rate"]


def get_default_epochs():
    return get_current_config()["epochs"]


def get_default_weight_decay():
    return DEFAULT_TRAINING_CONFIG["weight_decay"]


def get_default_train_samples():
    return get_current_config().get("train_samples")


def get_default_val_samples():
    return get_current_config().get("val_samples")


def switch_dataset(dataset_name):
    global DATASET_MODE
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_CONFIGS.keys())}"
        )
    DATASET_MODE = dataset_name


def print_current_config():
    cfg = get_current_config()
    print(f"Dataset mode     : {DATASET_MODE}")
    print(f"Vocab size       : {cfg['vocab_size']}")
    print(f"Max token length : {cfg['max_token_length']}")
    print(f"Embed dim        : {cfg['embed_dim']}")
    print(f"Batch size       : {cfg['batch_size']}")
    print(f"Learning rate    : {cfg['learning_rate']}")
    print(f"Epochs           : {cfg['epochs']}")
