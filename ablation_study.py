#!/usr/bin/env python3
"""Essential architecture ablation study for dual-branch CXR retrieval."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
import paths
from data_loader import CXRDataLoader
from evaluate import evaluate_cross_modal_retrieval_streaming
from models import MultimodalFusion
from visualize import (
    create_ablation_contribution_analysis,
    create_ablation_performance_comparison,
)

# Essential ablation variants: (result_key, ablation_type)
ESSENTIAL_VARIANTS = [
    ("full_model", "none"),
    ("no_local_attention", "no_local_attention"),
    ("no_global_attention", "no_global_attention"),
    ("no_gating_mechanism", "no_gating_mechanism"),
    ("no_local_gating", "no_local_gating"),
    ("no_global_gating", "no_global_gating"),
    ("no_global_to_local_feedback", "no_global_to_local_feedback"),
    ("no_synergy_branch", "no_synergy_branch"),
    ("no_difference_branch", "no_difference_branch"),
    ("single_coattention_layer", "single_coattention_layer"),
]

DETAILED_ABLATION_TYPES = {
    "no_local_attention",
    "no_global_attention",
    "no_gating_mechanism",
    "no_local_gating",
    "no_global_gating",
    "no_global_to_local_feedback",
    "single_coattention_layer",
}


def load_tokenizer_from_metadata(dataset_mode=None):
    """Load tokenizer from shard metadata for the active dataset."""
    shard_subfolder = dataset_mode or config.DATASET_MODE
    metadata_path = paths.get_metadata_path(shard_subfolder)

    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    tokenizer = metadata.get("tokenizer")
    if tokenizer is None:
        raise ValueError(f"No tokenizer found in metadata: {metadata_path}")

    if hasattr(tokenizer, "word_index") and not hasattr(tokenizer, "word2idx"):
        tokenizer.word2idx = tokenizer.word_index
        tokenizer.idx2word = tokenizer.index_word

    return tokenizer


def compute_orthogonal_loss(model, data_loader, device=None):
    """
    Compute dual-branch orthogonal loss on a dataloader.

    Matches training definition:
        img_orthogonal_loss = mean(|sum(synergy_img * diff_img, dim=1)|)
        txt_orthogonal_loss = mean(|sum(synergy_txt * diff_txt, dim=1)|)
        orthogonal_loss = (img + txt) / 2
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    img_ortho_vals = []
    txt_ortho_vals = []
    synergy_sims = []
    diff_sims = []
    n_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, dict):
                images = batch["images"]
                texts = batch["captions"]
            else:
                images, texts = batch

            if images.shape[1] != 3:
                images = images.permute(0, 3, 1, 2)

            images = images.to(device)
            texts = texts.to(device)

            _, _, synergy_img, synergy_txt, diff_img, diff_txt = model(
                (images, texts),
                training=False,
                return_branch_embeddings=True,
            )

            img_ortho = torch.abs(torch.sum(synergy_img * diff_img, dim=1))
            txt_ortho = torch.abs(torch.sum(synergy_txt * diff_txt, dim=1))
            img_ortho_vals.append(img_ortho.cpu())
            txt_ortho_vals.append(txt_ortho.cpu())

            synergy_sims.append(F.cosine_similarity(synergy_img, synergy_txt, dim=1).cpu())
            diff_sims.append(F.cosine_similarity(diff_img, diff_txt, dim=1).cpu())
            n_samples += images.size(0)

    img_orthogonal_loss = torch.cat(img_ortho_vals).mean().item()
    txt_orthogonal_loss = torch.cat(txt_ortho_vals).mean().item()
    orthogonal_loss = (img_orthogonal_loss + txt_orthogonal_loss) / 2.0
    synergy_sim = torch.cat(synergy_sims).mean().item()
    diff_sim = torch.cat(diff_sims).mean().item()

    return {
        "orthogonal-loss": orthogonal_loss,
        "orthogonal_loss": orthogonal_loss,
        "img_orthogonal_loss": img_orthogonal_loss,
        "txt_orthogonal_loss": txt_orthogonal_loss,
        "synergy_branch_similarity": synergy_sim,
        "difference_branch_similarity": diff_sim,
        "specialization_gap": synergy_sim - diff_sim,
        "n_samples": n_samples,
    }


def training_loss_mix_string():
    """Format synergy / main / orthogonal weights from DEFAULT_TRAINING_CONFIG."""
    train_cfg = config.DEFAULT_TRAINING_CONFIG
    syn = train_cfg.get("synergy_weight", 0.64)
    main = train_cfg.get("main_weight", 0.20)
    ortho = train_cfg.get("ortho_weight", 0.15)
    return f"synergy {syn:.2f} / main {main:.2f} / orthogonal {ortho:.2f}"


def write_reviewer_ablation_log(
    ablation_results,
    ortho_metrics,
    save_dir,
    timestamp,
    model_path,
    test_samples,
    extra_context=None,
    dataset_mode=None,
):
    """Write a reviewer-facing text log with ablation metrics and orthogonal-loss."""
    os.makedirs(save_dir, exist_ok=True)
    log_path = os.path.join(save_dir, f"ablation_reviewer_report_{timestamp}.txt")

    dataset_mode = dataset_mode or config.DATASET_MODE
    baseline = ablation_results.get("full_model", {})
    baseline_mrr = baseline.get("avg_mrr", 0.0)
    cfg = config.get_current_config()

    lines = []
    lines.append("=" * 78)
    lines.append("ABLATION STUDY — REVIEWER REPORT")
    lines.append("=" * 78)
    lines.append(f'Generated: {datetime.now().isoformat(timespec="seconds")}')
    lines.append(f"Timestamp tag: {timestamp}")
    lines.append("")

    lines.append("--- SETUP ---")
    lines.append(f"Dataset: {dataset_mode}")
    lines.append(f'Data path: {paths.get_shard_base_path(dataset_mode)}')
    lines.append(f"Test samples: {test_samples}")
    lines.append(f"Model path: {model_path}")
    lines.append(f"Vocab size: {config.get_vocab_size()}")
    lines.append(f"Embed dim: {config.get_embed_dim()}")
    lines.append(f'Num heads: {cfg.get("num_heads")}')
    lines.append(f'Num co-attention layers: {cfg.get("num_layers")}')
    lines.append(f'Max token length: {cfg.get("max_token_length")}')
    lines.append(f"Training loss mix: {training_loss_mix_string()}")
    lines.append("")

    lines.append("--- ORTHOGONAL-LOSS (reviewer field) ---")
    lines.append(
        "Definition: mean(|<synergy, difference>|) averaged over image and text branches"
    )
    if ortho_metrics:
        lines.append(f'orthogonal-loss: {ortho_metrics.get("orthogonal-loss", float("nan")):.6f}')
        lines.append(
            f'img_orthogonal_loss: {ortho_metrics.get("img_orthogonal_loss", float("nan")):.6f}'
        )
        lines.append(
            f'txt_orthogonal_loss: {ortho_metrics.get("txt_orthogonal_loss", float("nan")):.6f}'
        )
        lines.append(
            f'synergy_branch_similarity: '
            f'{ortho_metrics.get("synergy_branch_similarity", float("nan")):.6f}'
        )
        lines.append(
            f'difference_branch_similarity: '
            f'{ortho_metrics.get("difference_branch_similarity", float("nan")):.6f}'
        )
        lines.append(
            f'specialization_gap: {ortho_metrics.get("specialization_gap", float("nan")):.6f}'
        )
        lines.append(
            f'orthogonal-loss evaluated on n_samples: {ortho_metrics.get("n_samples", "N/A")}'
        )
    else:
        lines.append("orthogonal-loss: N/A (not computed)")
    lines.append("")

    lines.append("--- FULL MODEL RETRIEVAL (test set) ---")
    for key in [
        "avg_mrr",
        "avg_recall@1",
        "avg_recall@5",
        "avg_recall@10",
        "i2t_mrr",
        "t2i_mrr",
        "i2t_recall@1",
        "t2i_recall@1",
        "i2t_recall@5",
        "t2i_recall@5",
        "i2t_recall@10",
        "t2i_recall@10",
        "avg_mean_rank",
        "avg_median_rank",
    ]:
        if key in baseline:
            lines.append(f"{key}: {baseline[key]:.6f}")
    if "orthogonal-loss" in baseline:
        lines.append(f'orthogonal-loss: {baseline["orthogonal-loss"]:.6f}')
    lines.append("")

    lines.append("--- ABLATION VARIANT RESULTS ---")
    lines.append(
        f'{"variant":<32} {"MRR":>8} {"R@1":>8} {"R@5":>8} {"R@10":>8} '
        f'{"MRR_drop":>10} {"drop_%":>8}'
    )
    for variant, results in ablation_results.items():
        mrr = results.get("avg_mrr", 0.0)
        r1 = results.get("avg_recall@1", 0.0)
        r5 = results.get("avg_recall@5", 0.0)
        r10 = results.get("avg_recall@10", 0.0)
        drop = baseline_mrr - mrr if variant != "full_model" else 0.0
        drop_pct = (
            (drop / baseline_mrr * 100.0)
            if baseline_mrr > 0 and variant != "full_model"
            else 0.0
        )
        lines.append(
            f"{variant:<32} {mrr:8.4f} {r1:8.4f} {r5:8.4f} {r10:8.4f} "
            f"{drop:10.4f} {drop_pct:7.1f}%"
        )
    lines.append("")

    contributions = {
        v: baseline_mrr - r.get("avg_mrr", 0.0)
        for v, r in ablation_results.items()
        if v != "full_model"
    }
    if contributions:
        most_critical = max(contributions.items(), key=lambda x: x[1])
        least_critical = min(contributions.items(), key=lambda x: x[1])
        lines.append("--- COMPONENT IMPORTANCE ---")
        lines.append(
            f"Most critical (largest MRR drop): {most_critical[0]} "
            f"({most_critical[1]:.4f})"
        )
        lines.append(
            f"Least critical (smallest MRR drop): {least_critical[0]} "
            f"({least_critical[1]:.4f})"
        )
        lines.append("")

    lines.append("--- NOTES FOR REVIEWERS ---")
    lines.append(
        "- Plots may hide no_local_attention and single_coattention_layer; raw scores are above."
    )
    lines.append(
        "- no_local_attention currently skips co-attention rather than keeping global-only; "
        "treat that variant cautiously."
    )
    lines.append(
        "- orthogonal-loss near 0 means synergy/difference embeddings are more orthogonal."
    )
    lines.append("")

    if "no_orthogonal_loss" in ablation_results:
        no_ortho = ablation_results["no_orthogonal_loss"]
        lines.append(
            "--- ISOLATED ORTHOGONAL-LOSS ABLATION (with vs without training ortho) ---"
        )
        lines.append(
            "This compares the full model trained WITH orthogonal-loss "
            "against a matched model trained with ortho_weight=0."
        )
        lines.append(
            f"full_model MRR / R@1: "
            f'{baseline.get("avg_mrr", float("nan")):.4f} / '
            f'{baseline.get("avg_recall@1", float("nan")):.4f}'
        )
        lines.append(
            f"no_orthogonal_loss MRR / R@1: "
            f'{no_ortho.get("avg_mrr", float("nan")):.4f} / '
            f'{no_ortho.get("avg_recall@1", float("nan")):.4f}'
        )
        mrr_drop = baseline_mrr - no_ortho.get("avg_mrr", 0.0)
        r1_drop = baseline.get("avg_recall@1", 0.0) - no_ortho.get("avg_recall@1", 0.0)
        lines.append(f"MRR drop when removing ortho from training: {mrr_drop:.4f}")
        lines.append(f"Recall@1 drop when removing ortho from training: {r1_drop:.4f}")
        if "orthogonal-loss" in no_ortho:
            lines.append(
                f"no_orthogonal_loss model test orthogonal-loss: "
                f'{no_ortho["orthogonal-loss"]:.6f}'
            )
        if "specialization_gap" in no_ortho:
            lines.append(
                f"no_orthogonal_loss specialization_gap: "
                f'{no_ortho["specialization_gap"]:.6f}'
            )
        lines.append("")

    if extra_context:
        lines.append("--- ADDITIONAL CONTEXT ---")
        for k, v in extra_context.items():
            lines.append(f"{k}: {v}")
        lines.append("")

    lines.append("--- OUTPUT ARTIFACTS ---")
    lines.append(f"Performance plot: ablation_performance_comparison_{timestamp}.png")
    lines.append(f"Contribution plot: ablation_contribution_analysis_{timestamp}.png")
    lines.append(f"This log: {os.path.basename(log_path)}")
    lines.append("=" * 78)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Reviewer report saved to: {log_path}")
    return log_path


class ComprehensiveAblatedModel(MultimodalFusion):
    """Branch-level ablations (synergy / difference)."""

    def __init__(self, base_model, ablation_type="none"):
        super().__init__(
            vocab_size=config.get_vocab_size(),
            embed_dim=config.get_embed_dim(),
            num_heads=config.get_current_config()["num_heads"],
            num_layers=config.get_current_config()["num_layers"],
        )
        self.load_state_dict(base_model.state_dict())
        self.ablation_type = ablation_type

    def forward(self, inputs, training=True, verbose=False, return_branch_embeddings=False):
        images, texts = inputs
        image_tokens = self.image_encoder(images, training=training, verbose=verbose)
        text_tokens = self.text_encoder(texts, training=training, verbose=verbose)

        synergy_img_emb, synergy_txt_emb = self.synergy_branch(image_tokens, text_tokens)
        difference_img_emb, difference_txt_emb = self.difference_branch(
            image_tokens, text_tokens
        )

        if self.ablation_type == "no_synergy_branch":
            synergy_img_emb = torch.zeros_like(synergy_img_emb)
            synergy_txt_emb = torch.zeros_like(synergy_txt_emb)
        elif self.ablation_type == "no_difference_branch":
            difference_img_emb = torch.zeros_like(difference_img_emb)
            difference_txt_emb = torch.zeros_like(difference_txt_emb)
        elif self.ablation_type == "synergy_only":
            difference_img_emb = torch.zeros_like(difference_img_emb)
            difference_txt_emb = torch.zeros_like(difference_txt_emb)
        elif self.ablation_type == "difference_only":
            synergy_img_emb = torch.zeros_like(synergy_img_emb)
            synergy_txt_emb = torch.zeros_like(synergy_txt_emb)

        if self.ablation_type in ("synergy_only", "no_difference_branch"):
            final_img_emb = synergy_img_emb
            final_txt_emb = synergy_txt_emb
        elif self.ablation_type in ("difference_only", "no_synergy_branch"):
            final_img_emb = difference_img_emb
            final_txt_emb = difference_txt_emb
        else:
            final_img_emb = F.normalize(
                (synergy_img_emb + difference_img_emb) / 2, p=2, dim=-1
            )
            final_txt_emb = F.normalize(
                (synergy_txt_emb + difference_txt_emb) / 2, p=2, dim=-1
            )

        if return_branch_embeddings:
            return (
                final_img_emb,
                final_txt_emb,
                synergy_img_emb,
                synergy_txt_emb,
                difference_img_emb,
                difference_txt_emb,
            )
        return final_img_emb, final_txt_emb


class DetailedAblatedModel(MultimodalFusion):
    """Attention, gating, feedback, and depth ablations."""

    def __init__(self, base_model, ablation_type="none"):
        super().__init__(
            vocab_size=config.get_vocab_size(),
            embed_dim=config.get_embed_dim(),
            num_heads=config.get_current_config()["num_heads"],
            num_layers=config.get_current_config()["num_layers"],
        )
        self.load_state_dict(base_model.state_dict())
        self.ablation_type = ablation_type

    def forward(self, inputs, training=True, verbose=False, return_branch_embeddings=False):
        images, texts = inputs
        image_tokens = self.image_encoder(images, training=training, verbose=verbose)
        text_tokens = self.text_encoder(texts, training=training, verbose=verbose)

        synergy_img_emb, synergy_txt_emb = self._process_branch_with_ablation(
            self.synergy_branch, image_tokens, text_tokens
        )
        difference_img_emb, difference_txt_emb = self._process_branch_with_ablation(
            self.difference_branch, image_tokens, text_tokens
        )

        final_img_emb = F.normalize(
            (synergy_img_emb + difference_img_emb) / 2, p=2, dim=-1
        )
        final_txt_emb = F.normalize(
            (synergy_txt_emb + difference_txt_emb) / 2, p=2, dim=-1
        )

        if return_branch_embeddings:
            return (
                final_img_emb,
                final_txt_emb,
                synergy_img_emb,
                synergy_txt_emb,
                difference_img_emb,
                difference_txt_emb,
            )
        return final_img_emb, final_txt_emb

    def _process_branch_with_ablation(self, branch, image_tokens, text_tokens):
        if self.ablation_type == "single_coattention_layer":
            # Keep first co-attention layer only
            for i, layer in enumerate(branch.co_attn_layers):
                if i > 0:
                    break
                image_tokens, text_tokens = self._apply_coattention_ablation(
                    layer, image_tokens, text_tokens
                )
            image_emb = torch.mean(image_tokens, dim=1)
            text_emb = torch.mean(text_tokens, dim=1)
            return branch.image_proj(image_emb), branch.text_proj(text_emb)

        for layer in branch.co_attn_layers:
            image_tokens, text_tokens = self._apply_coattention_ablation(
                layer, image_tokens, text_tokens
            )

        image_emb = torch.mean(image_tokens, dim=1)
        text_emb = torch.mean(text_tokens, dim=1)
        return branch.image_proj(image_emb), branch.text_proj(text_emb)

    def _apply_coattention_ablation(self, layer, image_tokens, text_tokens):
        if self.ablation_type == "no_local_attention":
            # Historical behaviour: skip local co-attention (identity pass-through).
            return image_tokens, text_tokens
        if self.ablation_type == "no_global_attention":
            return self._apply_local_attention_only(layer, image_tokens, text_tokens)
        return self._apply_gating_ablations(layer, image_tokens, text_tokens)

    def _apply_local_attention_only(self, layer, image_tokens, text_tokens):
        attended_image, _ = layer.cross_attention1(
            query=image_tokens, key=text_tokens, value=text_tokens
        )
        image_tokens = layer.norm1(image_tokens + attended_image)
        image_tokens = layer.norm2(image_tokens + layer.ffn1(image_tokens))

        attended_text, _ = layer.cross_attention2(
            query=text_tokens, key=image_tokens, value=image_tokens
        )
        text_tokens = layer.norm3(text_tokens + attended_text)
        text_tokens = layer.norm4(text_tokens + layer.ffn2(text_tokens))
        return image_tokens, text_tokens

    def _apply_gating_ablations(self, layer, image_tokens, text_tokens):
        attended_image, _ = layer.cross_attention1(
            query=image_tokens, key=text_tokens, value=text_tokens
        )
        if self.ablation_type in ("no_gating_mechanism", "no_local_gating"):
            gated_image = attended_image
        else:
            local_image_gate = torch.sigmoid(layer.local_image_gate_weights).view(
                1, 1, layer.embed_dim
            )
            gated_image = (
                local_image_gate * attended_image + (1 - local_image_gate) * image_tokens
            )

        image_tokens = layer.norm1(image_tokens + gated_image)
        image_tokens = layer.norm2(image_tokens + layer.ffn1(image_tokens))

        attended_text, _ = layer.cross_attention2(
            query=text_tokens, key=image_tokens, value=image_tokens
        )
        if self.ablation_type in ("no_gating_mechanism", "no_local_gating"):
            gated_text = attended_text
        else:
            local_text_gate = torch.sigmoid(layer.local_text_gate_weights).view(
                1, 1, layer.embed_dim
            )
            gated_text = (
                local_text_gate * attended_text + (1 - local_text_gate) * text_tokens
            )

        text_tokens = layer.norm3(text_tokens + gated_text)
        text_tokens = layer.norm4(text_tokens + layer.ffn2(text_tokens))

        global_image_token = torch.mean(image_tokens, dim=1, keepdim=True)
        global_text_token = torch.mean(text_tokens, dim=1, keepdim=True)

        attended_global_image, _ = layer.global_cross_attention1(
            query=global_image_token, key=text_tokens, value=text_tokens
        )
        if self.ablation_type in ("no_gating_mechanism", "no_global_gating"):
            gated_global_image = attended_global_image
        else:
            global_image_gate = torch.sigmoid(layer.global_image_gate_weights).view(
                1, 1, layer.embed_dim
            )
            gated_global_image = (
                global_image_gate * attended_global_image
                + (1 - global_image_gate) * global_image_token
            )

        global_image_token = layer.global_norm1(global_image_token + gated_global_image)
        global_image_token = layer.global_norm2(
            global_image_token + layer.global_ffn1(global_image_token)
        )

        attended_global_text, _ = layer.global_cross_attention2(
            query=global_text_token, key=image_tokens, value=image_tokens
        )
        if self.ablation_type in ("no_gating_mechanism", "no_global_gating"):
            gated_global_text = attended_global_text
        else:
            global_text_gate = torch.sigmoid(layer.global_text_gate_weights).view(
                1, 1, layer.embed_dim
            )
            gated_global_text = (
                global_text_gate * attended_global_text
                + (1 - global_text_gate) * global_text_token
            )

        global_text_token = layer.global_norm3(global_text_token + gated_global_text)
        global_text_token = layer.global_norm4(
            global_text_token + layer.global_ffn2(global_text_token)
        )

        if self.ablation_type != "no_global_to_local_feedback":
            image_tokens = image_tokens + global_image_token.expand(
                -1, image_tokens.size(1), -1
            )
            text_tokens = text_tokens + global_text_token.expand(
                -1, text_tokens.size(1), -1
            )

        return image_tokens, text_tokens


def build_ablated_model(base_model, variant_name, ablation_type):
    if variant_name == "full_model" or ablation_type in ("none", None):
        return base_model
    if ablation_type in DETAILED_ABLATION_TYPES:
        model = DetailedAblatedModel(base_model, ablation_type)
    else:
        model = ComprehensiveAblatedModel(base_model, ablation_type)
    model.eval()
    return model


def load_base_model(model_path, device):
    model = MultimodalFusion(
        vocab_size=config.get_vocab_size(),
        embed_dim=config.get_embed_dim(),
        num_heads=config.get_current_config()["num_heads"],
        num_layers=config.get_current_config()["num_layers"],
    )
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def prepare_test_loader(dataset_mode, batch_size, test_samples=None):
    data_loader = CXRDataLoader(
        batch_size=batch_size,
        use_shards=True,
        shard_subfolder=dataset_mode,
    )
    data_loader.tokenizer = load_tokenizer_from_metadata(dataset_mode)
    data_loader.load_data(max_samples=None, skip_processing=True)
    test_dataset = data_loader.get_test_data(num_samples=test_samples)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return test_dataset, test_loader


def run_essential_ablation_study(
    model_path,
    dataset_mode=None,
    batch_size=32,
    device=None,
    save_dir="comprehensive_ablation_results",
    test_samples=None,
):
    """Run essential architecture ablations on a trained checkpoint."""
    dataset_mode = dataset_mode or config.DATASET_MODE
    if dataset_mode != config.DATASET_MODE:
        config.switch_dataset(dataset_mode)

    config.print_current_config()
    paths.print_paths(dataset_mode)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    print(f"\nLoading test data ({dataset_mode})...")
    test_dataset, test_loader = prepare_test_loader(
        dataset_mode, batch_size, test_samples=test_samples
    )
    print(f"Test data loaded: {len(test_dataset)} samples")

    print(f"\nLoading model: {model_path}")
    base_model = load_base_model(model_path, device)
    n_params = sum(p.numel() for p in base_model.parameters())
    print(f"Model loaded: {n_params:,} parameters on {device}")

    print("\nComputing orthogonal-loss on full model (test set)...")
    ortho_metrics = compute_orthogonal_loss(base_model, test_loader, device=device)
    print(f"  orthogonal-loss: {ortho_metrics['orthogonal-loss']:.6f}")
    print(f"  synergy_branch_similarity: {ortho_metrics['synergy_branch_similarity']:.6f}")
    print(
        f"  difference_branch_similarity: "
        f"{ortho_metrics['difference_branch_similarity']:.6f}"
    )
    print(f"  specialization_gap: {ortho_metrics['specialization_gap']:.6f}")

    ablation_results = {}
    for variant_name, ablation_type in ESSENTIAL_VARIANTS:
        print(f"\n{'=' * 60}")
        print(f"Evaluating: {variant_name}")
        print(f"{'=' * 60}")

        try:
            model_to_test = build_ablated_model(base_model, variant_name, ablation_type)
            if model_to_test is not base_model:
                model_to_test = model_to_test.to(device)

            results = evaluate_cross_modal_retrieval_streaming(
                model=model_to_test,
                test_dataset=test_loader,
                k_values=[1, 5, 10],
                batch_size=batch_size,
                visualize=False,
                num_vis_examples=0,
            )
            ablation_results[variant_name] = results

            if variant_name == "full_model":
                ablation_results[variant_name].update(
                    {
                        "orthogonal-loss": ortho_metrics["orthogonal-loss"],
                        "img_orthogonal_loss": ortho_metrics["img_orthogonal_loss"],
                        "txt_orthogonal_loss": ortho_metrics["txt_orthogonal_loss"],
                        "synergy_branch_similarity": ortho_metrics[
                            "synergy_branch_similarity"
                        ],
                        "difference_branch_similarity": ortho_metrics[
                            "difference_branch_similarity"
                        ],
                        "specialization_gap": ortho_metrics["specialization_gap"],
                    }
                )

            print(f"Results for {variant_name}:")
            print(f"  MRR:      {results['avg_mrr']:.4f}")
            print(f"  Recall@1: {results['avg_recall@1']:.4f}")
            print(f"  Recall@5: {results['avg_recall@5']:.4f}")
            print(f"  Recall@10:{results['avg_recall@10']:.4f}")
            if variant_name == "full_model":
                print(
                    f"  orthogonal-loss: "
                    f"{ablation_results[variant_name]['orthogonal-loss']:.6f}"
                )
        except Exception as exc:
            print(f"Error evaluating {variant_name}: {exc}")
            print("  Skipping this variant.")
            continue

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    create_ablation_performance_comparison(ablation_results, save_dir, timestamp)
    create_ablation_contribution_analysis(ablation_results, save_dir, timestamp)

    results_json = os.path.join(save_dir, f"ablation_results_{timestamp}.json")
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)
    print(f"Results JSON saved to: {results_json}")

    write_reviewer_ablation_log(
        ablation_results=ablation_results,
        ortho_metrics=ortho_metrics,
        save_dir=save_dir,
        timestamp=timestamp,
        model_path=model_path,
        test_samples=len(test_dataset),
        dataset_mode=dataset_mode,
        extra_context={
            "paired_performance_plot": f"ablation_performance_comparison_{timestamp}.png",
            "paired_contribution_plot": f"ablation_contribution_analysis_{timestamp}.png",
            "results_json": os.path.basename(results_json),
            "training_loss_mix": training_loss_mix_string(),
        },
    )

    print(f"\n{'=' * 70}")
    print("ESSENTIAL ABLATION STUDY SUMMARY")
    print(f"{'=' * 70}")
    if "full_model" not in ablation_results:
        print("No successful evaluations.")
        return ablation_results

    baseline_mrr = ablation_results["full_model"]["avg_mrr"]
    print(f"full_model MRR = {baseline_mrr:.4f}")
    if "orthogonal-loss" in ablation_results["full_model"]:
        print(
            f"orthogonal-loss: {ablation_results['full_model']['orthogonal-loss']:.6f}"
        )
    print(f"Dataset: {dataset_mode}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Variants evaluated: {len(ablation_results)}")

    print("\nComponent contributions (MRR drop vs full_model):")
    contributions = {}
    for variant, results in ablation_results.items():
        if variant == "full_model":
            continue
        contribution = baseline_mrr - results["avg_mrr"]
        contributions[variant] = contribution
        percentage = (contribution / baseline_mrr) * 100 if baseline_mrr else 0.0
        print(f"  {variant}: {contribution:.4f} ({percentage:+.1f}%)")

    if contributions:
        most_critical = max(contributions.items(), key=lambda x: x[1])
        print(
            f"\nMost critical component: {most_critical[0]} "
            f"({most_critical[1]:.4f} MRR drop)"
        )

    print(f"\nResults saved to: {save_dir}/")
    return ablation_results


def run_difference_branch_analysis(
    model_path,
    dataset_mode=None,
    batch_size=32,
    device=None,
    save_dir="comprehensive_ablation_results",
    test_samples=2000,
):
    """Optional focused analysis of the difference branch."""
    dataset_mode = dataset_mode or config.DATASET_MODE
    if dataset_mode != config.DATASET_MODE:
        config.switch_dataset(dataset_mode)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    test_dataset, test_loader = prepare_test_loader(
        dataset_mode, batch_size, test_samples=test_samples
    )
    base_model = load_base_model(model_path, device)

    variants = [
        ("full_model", "none"),
        ("no_difference_branch", "no_difference_branch"),
        ("difference_only", "difference_only"),
    ]
    ablation_results = {}
    for variant_name, ablation_type in variants:
        print(f"Evaluating: {variant_name}")
        model_to_test = build_ablated_model(base_model, variant_name, ablation_type)
        if model_to_test is not base_model:
            model_to_test = model_to_test.to(device)
        results = evaluate_cross_modal_retrieval_streaming(
            model=model_to_test,
            test_dataset=test_loader,
            k_values=[1, 5, 10],
            batch_size=batch_size,
            visualize=False,
            num_vis_examples=0,
        )
        ablation_results[variant_name] = results
        print(
            f"  MRR={results['avg_mrr']:.4f} "
            f"R@1={results['avg_recall@1']:.4f}"
        )

    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    create_ablation_performance_comparison(ablation_results, save_dir, timestamp)

    full_mrr = ablation_results.get("full_model", {}).get("avg_mrr", 0.0)
    no_diff_mrr = ablation_results.get("no_difference_branch", {}).get("avg_mrr", 0.0)
    print(f"\nFull model MRR: {full_mrr:.4f}")
    print(f"No difference branch MRR: {no_diff_mrr:.4f}")
    print(f"MRR change without difference branch: {no_diff_mrr - full_mrr:+.4f}")
    print(f"Results saved to: {save_dir}/")
    return ablation_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Essential architecture ablation study for dual-branch CXR retrieval."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path to trained model_weights.pth",
    )
    parser.add_argument(
        "--dataset",
        default=config.DATASET_MODE,
        help=f"Dataset mode (default: {config.DATASET_MODE})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Evaluation batch size (default: 32)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device string, e.g. cuda or cpu (default: auto)",
    )
    parser.add_argument(
        "--save-dir",
        default="comprehensive_ablation_results",
        help="Directory for plots, JSON, and reviewer log",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=None,
        help="Limit test samples (default: all)",
    )
    parser.add_argument(
        "--difference-branch-analysis",
        action="store_true",
        help="Run focused difference-branch analysis instead of the essential study",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    if args.difference_branch_analysis:
        run_difference_branch_analysis(
            model_path=args.model_path,
            dataset_mode=args.dataset,
            batch_size=args.batch_size,
            device=args.device,
            save_dir=args.save_dir,
            test_samples=args.test_samples if args.test_samples is not None else 2000,
        )
    else:
        run_essential_ablation_study(
            model_path=args.model_path,
            dataset_mode=args.dataset,
            batch_size=args.batch_size,
            device=args.device,
            save_dir=args.save_dir,
            test_samples=args.test_samples,
        )


if __name__ == "__main__":
    main()
