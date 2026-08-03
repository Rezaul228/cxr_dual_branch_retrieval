#!/usr/bin/env python3
"""Isolated orthogonal-loss ablation: full model vs matched no-ortho training run."""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from datetime import datetime

import torch

import config
from ablation_study import (
    compute_orthogonal_loss,
    load_base_model,
    prepare_test_loader,
    training_loss_mix_string,
    write_reviewer_ablation_log,
)
from evaluate import evaluate_cross_modal_retrieval_streaming
from visualize import (
    create_ablation_contribution_analysis,
    create_ablation_performance_comparison,
)


def evaluate_model(model, test_loader, batch_size):
    return evaluate_cross_modal_retrieval_streaming(
        model=model,
        test_dataset=test_loader,
        k_values=[1, 5, 10],
        batch_size=batch_size,
        visualize=False,
        num_vis_examples=0,
    )


def load_prior_ablation_results(ablation_json):
    if not ablation_json:
        return {}
    if not os.path.isfile(ablation_json):
        raise FileNotFoundError(f"Ablation JSON not found: {ablation_json}")
    with open(ablation_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a dict of variant -> metrics in {ablation_json}")
    return data


def attach_ortho_metrics(results, ortho_metrics):
    results = dict(results)
    results.update(
        {
            "orthogonal-loss": ortho_metrics["orthogonal-loss"],
            "img_orthogonal_loss": ortho_metrics["img_orthogonal_loss"],
            "txt_orthogonal_loss": ortho_metrics["txt_orthogonal_loss"],
            "synergy_branch_similarity": ortho_metrics["synergy_branch_similarity"],
            "difference_branch_similarity": ortho_metrics[
                "difference_branch_similarity"
            ],
            "specialization_gap": ortho_metrics["specialization_gap"],
        }
    )
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare a full model trained with orthogonal loss against a matched "
            "checkpoint trained with ortho_weight=0."
        )
    )
    parser.add_argument(
        "--full-model",
        required=True,
        help="Path to full-model weights (trained with orthogonal loss)",
    )
    parser.add_argument(
        "--no-ortho-model",
        required=True,
        help="Path to matched weights trained with ortho_weight=0",
    )
    parser.add_argument(
        "--ablation-json",
        default=None,
        help=(
            "Optional prior architecture-ablation results JSON to merge. "
            "If omitted, only full_model and no_orthogonal_loss entries are written."
        ),
    )
    parser.add_argument(
        "--save-dir",
        default="comprehensive_ablation_results",
        help="Directory for plots, JSON, and reviewer log",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device string, e.g. cuda or cpu (default: auto)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Evaluation batch size (default: 32)",
    )
    parser.add_argument(
        "--dataset",
        default=config.DATASET_MODE,
        help=f"Dataset mode (default: {config.DATASET_MODE})",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=None,
        help="Limit test samples (default: all)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    for path, label in (
        (args.full_model, "full-model"),
        (args.no_ortho_model, "no-ortho-model"),
    ):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{label} checkpoint not found: {path}")

    dataset_mode = args.dataset
    if dataset_mode != config.DATASET_MODE:
        config.switch_dataset(dataset_mode)

    config.print_current_config()
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    test_dataset, test_loader = prepare_test_loader(
        dataset_mode, args.batch_size, test_samples=args.test_samples
    )
    print(f"Test samples: {len(test_dataset)}")

    ablation_results = deepcopy(load_prior_ablation_results(args.ablation_json))

    print("\n=== Full model (WITH orthogonal loss) ===")
    full_model = load_base_model(args.full_model, device)
    full_retrieval = evaluate_model(full_model, test_loader, args.batch_size)
    ortho_full = compute_orthogonal_loss(full_model, test_loader, device=device)
    print(f"orthogonal-loss: {ortho_full['orthogonal-loss']:.6f}")
    ablation_results["full_model"] = attach_ortho_metrics(full_retrieval, ortho_full)

    print("\n=== No-orthogonal-loss model (trained with ortho_weight=0) ===")
    no_ortho_model = load_base_model(args.no_ortho_model, device)
    no_ortho_retrieval = evaluate_model(no_ortho_model, test_loader, args.batch_size)
    ortho_none = compute_orthogonal_loss(no_ortho_model, test_loader, device=device)
    ablation_results["no_orthogonal_loss"] = attach_ortho_metrics(
        no_ortho_retrieval, ortho_none
    )

    print("\nIsolated orthogonal-loss ablation:")
    print(
        f"  full_model:          MRR={ablation_results['full_model']['avg_mrr']:.4f} "
        f"R@1={ablation_results['full_model']['avg_recall@1']:.4f} "
        f"ortho={ablation_results['full_model']['orthogonal-loss']:.4f}"
    )
    print(
        f"  no_orthogonal_loss:  MRR={ablation_results['no_orthogonal_loss']['avg_mrr']:.4f} "
        f"R@1={ablation_results['no_orthogonal_loss']['avg_recall@1']:.4f} "
        f"ortho={ablation_results['no_orthogonal_loss']['orthogonal-loss']:.4f}"
    )

    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    create_ablation_performance_comparison(ablation_results, args.save_dir, timestamp)
    create_ablation_contribution_analysis(ablation_results, args.save_dir, timestamp)

    results_json = os.path.join(
        args.save_dir, f"orthogonal_ablation_results_{timestamp}.json"
    )
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"Results JSON saved to: {results_json}")

    train_cfg = config.DEFAULT_TRAINING_CONFIG
    syn = train_cfg.get("synergy_weight", 0.64)
    main_w = train_cfg.get("main_weight", 0.20)
    ortho_w = train_cfg.get("ortho_weight", 0.15)
    # When ortho_weight=0, redistributed mass goes to main; synergy stays fixed.
    no_ortho_main = main_w + ortho_w

    write_reviewer_ablation_log(
        ablation_results=ablation_results,
        ortho_metrics=ortho_full,
        save_dir=args.save_dir,
        timestamp=timestamp,
        model_path=args.full_model,
        test_samples=len(test_dataset),
        dataset_mode=dataset_mode,
        extra_context={
            "isolated_orthogonal_ablation": "yes",
            "full_model_path": args.full_model,
            "no_orthogonal_loss_model_path": args.no_ortho_model,
            "full_model_train_weights": training_loss_mix_string(),
            "no_ortho_train_weights": (
                f"synergy {syn:.2f} / main {no_ortho_main:.2f} / orthogonal 0.00 "
                "(ortho mass redistributed to main)"
            ),
            "prior_ablation_json": args.ablation_json or "none",
            "results_json": os.path.basename(results_json),
        },
    )

    print(f"\nUpdated artifacts under {args.save_dir}/ with timestamp {timestamp}")


if __name__ == "__main__":
    main()
