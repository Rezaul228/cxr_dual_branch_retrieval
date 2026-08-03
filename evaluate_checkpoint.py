#!/usr/bin/env python3
"""Evaluate a saved checkpoint on the configured test split."""

import argparse
import os

import torch
from torch.utils.data import DataLoader

import config
from data_loader import CXRDataLoader
from evaluate import evaluate_cross_modal_retrieval_streaming
from models import MultimodalFusion


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate dual-branch retrieval checkpoint")
    parser.add_argument("--model-path", required=True, help="Path to model_weights.pth")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-samples", type=int, default=None)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override config.DATASET_MODE",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dataset:
        config.switch_dataset(args.dataset)

    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(args.model_path)

    print(f"Dataset: {config.DATASET_MODE}")
    print(f"Device: {args.device}")
    print(f"Checkpoint: {args.model_path}")

    data_loader = CXRDataLoader(
        batch_size=args.batch_size,
        shard_subfolder=config.DATASET_MODE,
    )
    data_loader.load_data(skip_processing=True)
    test_dataset = data_loader.get_test_data(num_samples=args.test_samples)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    model = MultimodalFusion(vocab_size=config.get_vocab_size()).to(args.device)
    state = torch.load(args.model_path, map_location=args.device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()

    results = evaluate_cross_modal_retrieval_streaming(
        model=model,
        test_dataset=test_loader,
        k_values=[1, 5, 10],
        batch_size=args.batch_size,
        visualize=False,
    )
    print("\nSummary")
    for key in ("avg_mrr", "avg_recall@1", "avg_recall@5", "avg_recall@10"):
        if key in results:
            print(f"  {key}: {results[key]:.4f}")


if __name__ == "__main__":
    main()
