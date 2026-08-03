#!/usr/bin/env python3
"""Essential visualization utilities for dual-branch CXR retrieval."""

import os
import random
import textwrap
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec

plt.switch_backend("Agg")

try:
    import seaborn as sns

    seaborn_available = True
except ImportError:
    seaborn_available = False
    print("Seaborn not available. Some visualizations will use basic matplotlib.")


def decode_caption(caption_seq, tokenizer=None):
    """Decode tokenized caption back to text (supports word_index and word2idx)."""
    if tokenizer is None:
        return f"Caption sequence: {caption_seq[:10]}..."

    try:
        if isinstance(caption_seq, torch.Tensor):
            caption_seq = caption_seq.cpu().numpy()

        valid_tokens = caption_seq[caption_seq > 0]

        if hasattr(tokenizer, "word_index"):
            word_map = tokenizer.word_index
        elif hasattr(tokenizer, "word2idx"):
            word_map = tokenizer.word2idx
        else:
            return f"Caption sequence: {caption_seq[:10]}..."

        reverse_word_index = {v: k for k, v in word_map.items()}
        reverse_word_index[0] = "<PAD>"

        words = []
        for token_id in valid_tokens:
            word = reverse_word_index.get(int(token_id), f"<UNK_{token_id}>")
            if word not in ["<start>", "<end>", "<pad>", "<PAD>"]:
                words.append(word)

        return " ".join(words)
    except Exception as e:
        return f"DECODE_ERROR: {str(e)[:50]}"


def _display_image(ax, image):
    """Show an image, handling NCHW / NHWC / grayscale layouts."""
    img = image
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()

    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = np.transpose(img, (1, 2, 0))

    if img.ndim == 3 and img.shape[-1] == 3:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        if np.allclose(r, g) and np.allclose(g, b):
            ax.imshow(r, cmap="gray")
        else:
            ax.imshow(img)
    else:
        ax.imshow(np.squeeze(img), cmap="gray")


def visualize_retrieval_examples(model, test_data, num_examples=3, k=3, output_dir=None):
    """Visualize image-to-text and text-to-image retrieval examples."""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        for filename in os.listdir(output_dir):
            if filename.startswith("retrieval_example_") and filename.endswith(".png"):
                os.remove(os.path.join(output_dir, filename))
        print(f"Visualizations will be saved to {output_dir}")

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.titlesize"] = 9
    plt.rcParams["axes.labelsize"] = 8
    plt.rcParams["xtick.labelsize"] = 7
    plt.rcParams["ytick.labelsize"] = 7
    plt.rcParams["legend.fontsize"] = 7

    colors = {
        "blue": "#1f77b4",
        "red": "#d62728",
        "green": "#2ca02c",
        "orange": "#ff7f0e",
        "white": "#ffffff",
        "black": "#000000",
        "gray": "#cccccc",
    }

    model.eval()
    with torch.no_grad():
        if isinstance(test_data["images"], np.ndarray):
            images = torch.FloatTensor(test_data["images"])
            if len(images.shape) == 4 and images.shape[-1] == 3:
                images = images.permute(0, 3, 1, 2)
        else:
            images = test_data["images"]

        if isinstance(test_data["captions"], np.ndarray):
            captions = torch.LongTensor(test_data["captions"])
        else:
            captions = test_data["captions"]

        device = next(model.parameters()).device
        images = images.to(device)
        captions = captions.to(device)
        image_emb, text_emb = model((images, captions), training=False)

    i2t_sim = torch.matmul(image_emb, text_emb.transpose(0, 1)).cpu().numpy()
    t2i_sim = torch.matmul(text_emb, image_emb.transpose(0, 1)).cpu().numpy()

    tokenizer = test_data.get("tokenizer", None)
    original_images = test_data["images"]
    if isinstance(original_images, torch.Tensor):
        original_images = original_images.cpu().numpy()

    num_examples = min(num_examples, len(original_images))
    example_indices = random.sample(range(len(original_images)), num_examples)

    for idx in example_indices:
        fig = plt.figure(figsize=(7.2, 4.8), facecolor=colors["white"])

        title_ax = fig.add_axes([0.1, 0.92, 0.8, 0.06])
        title_ax.text(
            0.5,
            0.5,
            "Multimodal Medical Image-Text Retrieval Results",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=colors["black"],
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=colors["white"],
                edgecolor=colors["gray"],
                linewidth=1,
            ),
        )
        title_ax.axis("off")

        gs = GridSpec(
            2,
            4,
            figure=fig,
            height_ratios=[2, 1],
            width_ratios=[1.2, 1.2, 1.2, 1.2],
            hspace=0.15,
            wspace=0.12,
            top=0.85,
            bottom=0.08,
            left=0.08,
            right=0.92,
        )

        ax_query_img = fig.add_subplot(gs[0, 0])
        _display_image(ax_query_img, original_images[idx])
        ax_query_img.set_title(
            "Query Image", fontsize=8, fontweight="bold", color=colors["blue"]
        )
        ax_query_img.axis("off")
        ax_query_img.set_aspect("equal")
        for spine in ax_query_img.spines.values():
            spine.set_color(colors["black"])
            spine.set_linewidth(2)

        sim_scores_i2t = i2t_sim[idx]
        top_k_indices_i2t = np.argsort(sim_scores_i2t)[::-1][:k]

        for rank_pos in range(min(k, 3)):
            ax_text = fig.add_subplot(gs[0, rank_pos + 1])
            ax_text.axis("off")
            match_idx = top_k_indices_i2t[rank_pos]
            caption = decode_caption(test_data["captions"][match_idx], tokenizer)
            score = sim_scores_i2t[match_idx]
            is_correct = match_idx == idx
            wrapped_text = textwrap.fill(caption, width=25)

            ax_text.text(
                0.5,
                0.98,
                f"Rank {rank_pos + 1}\nScore: {score:.4f}",
                ha="center",
                va="top",
                fontsize=8,
                fontweight="bold",
                color=colors["green"] if is_correct else colors["black"],
                transform=ax_text.transAxes,
            )
            text_box_props = dict(
                boxstyle="round,pad=0.3",
                facecolor=colors["white"],
                edgecolor=colors["gray"],
                linewidth=1,
                alpha=0.9,
            )
            ax_text.text(
                0.5,
                0.5,
                wrapped_text,
                ha="center",
                va="center",
                fontsize=7,
                bbox=text_box_props,
                transform=ax_text.transAxes,
                clip_on=True,
                wrap=True,
            )

        ax_query_text = fig.add_subplot(gs[1, 0])
        ax_query_text.axis("off")
        query_caption = decode_caption(test_data["captions"][idx], tokenizer)
        wrapped_query = textwrap.fill(query_caption, width=25)
        ax_query_text.text(
            0.5,
            0.98,
            "Query Text",
            ha="center",
            va="top",
            fontsize=8,
            fontweight="bold",
            color=colors["blue"],
            transform=ax_query_text.transAxes,
        )
        text_box_props = dict(
            boxstyle="round,pad=0.3",
            facecolor=colors["white"],
            edgecolor=colors["gray"],
            linewidth=1,
            alpha=0.9,
        )
        ax_query_text.text(
            0.5,
            0.5,
            wrapped_query,
            ha="center",
            va="center",
            fontsize=7,
            bbox=text_box_props,
            transform=ax_query_text.transAxes,
            clip_on=True,
            wrap=True,
        )

        sim_scores_t2i = t2i_sim[idx]
        top_k_indices_t2i = np.argsort(sim_scores_t2i)[::-1][:k]

        for rank_pos in range(min(k, 3)):
            ax_img = fig.add_subplot(gs[1, rank_pos + 1])
            match_idx = top_k_indices_t2i[rank_pos]
            score = sim_scores_t2i[match_idx]
            is_correct = match_idx == idx
            _display_image(ax_img, original_images[match_idx])
            ax_img.set_title(
                f"Rank {rank_pos + 1}\nScore: {score:.4f}",
                fontsize=8,
                fontweight="bold",
                color=colors["green"] if is_correct else colors["black"],
            )
            ax_img.axis("off")
            ax_img.set_aspect("equal")
            for spine in ax_img.spines.values():
                spine.set_color(colors["black"])
                spine.set_linewidth(1)

        fig.patch.set_edgecolor(colors["black"])
        fig.patch.set_linewidth(1)

        if output_dir:
            filename = os.path.join(output_dir, f"retrieval_example_{idx}.png")
            plt.savefig(
                filename,
                dpi=300,
                bbox_inches="tight",
                facecolor=colors["white"],
                edgecolor=colors["black"],
                format="png",
                transparent=False,
            )
            print(f"Saved retrieval visualization to {filename}")
        plt.close()


class TrainingVisualizer:
    """Training progress and dual-branch loss visualization."""

    def __init__(self, save_dir="visualizations"):
        self.save_dir = save_dir
        self.history = defaultdict(list)
        os.makedirs(save_dir, exist_ok=True)

    def update_history(self, epoch_metrics):
        for key, value in epoch_metrics.items():
            self.history[key].append(value)

    def plot_training_progress(self):
        if not self.history:
            print("No training history to plot!")
            return

        plt.figure(figsize=(16, 6))

        loss_key = None
        for key in ["total_loss", "loss"]:
            if key in self.history and len(self.history[key]) > 0:
                loss_key = key
                break

        if loss_key is None:
            print("No loss data found in training history!")
            print(f"Available keys: {list(self.history.keys())}")
            return

        epochs = list(range(1, len(self.history[loss_key]) + 1))

        plt.subplot(1, 2, 1)
        plt.plot(
            epochs,
            self.history[loss_key],
            "b-",
            linewidth=2,
            marker="o",
            markersize=4,
        )
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Training Loss", fontsize=14, fontweight="bold")
        plt.grid(True, alpha=0.3)

        if len(epochs) > 0:
            last_loss = self.history[loss_key][-1]
            plt.annotate(
                f"{last_loss:.4f}",
                xy=(epochs[-1], last_loss),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
            )

        plt.subplot(1, 2, 2)
        recall_keys = [k for k in self.history.keys() if "recall" in k.lower()]
        plot_colors = ["red", "green", "blue", "orange", "purple"]

        for i, key in enumerate(recall_keys):
            if len(self.history[key]) > 0:
                color = plot_colors[i % len(plot_colors)]
                plt.plot(
                    epochs,
                    self.history[key],
                    color=color,
                    linewidth=2,
                    marker="s",
                    markersize=3,
                    label=key,
                )
                if len(epochs) > 0:
                    last_recall = self.history[key][-1]
                    plt.annotate(
                        f"{last_recall:.3f}",
                        xy=(epochs[-1], last_recall),
                        xytext=(10, 5),
                        textcoords="offset points",
                        fontsize=9,
                        color=color,
                    )

        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Recall", fontsize=12)
        plt.title("Validation Recall", fontsize=14, fontweight="bold")
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.05)

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(self.save_dir, f"training_progress_{timestamp}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Training progress saved to: {out_path}")

    def plot_dual_branch_losses(self):
        if not self.history:
            print("No training history to plot!")
            return

        has_synergy = (
            "synergy_loss" in self.history and len(self.history["synergy_loss"]) > 0
        )
        has_difference = (
            "difference_loss" in self.history
            and len(self.history["difference_loss"]) > 0
        )

        if not (has_synergy and has_difference):
            print("Dual branch loss data not found!")
            print(f"Available keys: {list(self.history.keys())}")
            return

        epochs = list(range(1, len(self.history["synergy_loss"]) + 1))
        plt.figure(figsize=(12, 5))
        plt.suptitle(
            "Dual Branch Architecture Loss Analysis",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

        plt.subplot(1, 2, 1)
        plt.plot(
            epochs,
            self.history["synergy_loss"],
            "g-",
            linewidth=2,
            marker="o",
            markersize=4,
            label="Synergy Loss",
        )
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Synergy Branch Loss", fontsize=12, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(
            epochs,
            self.history["difference_loss"],
            "r-",
            linewidth=2,
            marker="s",
            markersize=4,
            label="Difference Loss",
        )
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title("Difference Branch Loss", fontsize=12, fontweight="bold")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(self.save_dir, f"dual_branch_losses_{timestamp}.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Dual branch losses saved to: {out_path}")

    def plot_similarity_matrix(self, similarity_matrix, title="Similarity Matrix", k=20):
        try:
            if isinstance(similarity_matrix, torch.Tensor):
                similarity_matrix = similarity_matrix.cpu().numpy()

            if not isinstance(similarity_matrix, np.ndarray):
                print(
                    f"Error: similarity_matrix must be torch.Tensor or numpy.ndarray, "
                    f"got {type(similarity_matrix)}"
                )
                return

            if similarity_matrix.size == 0:
                print("Error: similarity_matrix is empty")
                return

            if np.isnan(similarity_matrix).any() or np.isinf(similarity_matrix).any():
                print("Error: similarity_matrix contains NaN or Inf values")
                return

            k = min(k, similarity_matrix.shape[0], similarity_matrix.shape[1])
            if k <= 0:
                print("Error: k must be positive")
                return

            sim_subset = similarity_matrix[:k, :k]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"similarity_matrix_{timestamp}.png"

            plt.figure(figsize=(10, 8))
            plt.suptitle(
                "Cross-Modal Similarity Matrix Analysis",
                fontsize=14,
                fontweight="bold",
                y=0.98,
            )

            if seaborn_available:
                sns.heatmap(
                    sim_subset,
                    annot=False,
                    cmap="viridis",
                    square=True,
                    cbar_kws={"label": "Cosine Similarity Score"},
                )
            else:
                im = plt.imshow(sim_subset, cmap="viridis", aspect="auto")
                plt.colorbar(im, label="Cosine Similarity Score")

            plt.title(
                f"{title}\n(Matrix Size: {k}x{k})",
                fontsize=12,
                fontweight="bold",
                pad=20,
            )
            plt.xlabel("Text Index", fontsize=12)
            plt.ylabel("Image Index", fontsize=12)

            for i in range(min(k, sim_subset.shape[0])):
                plt.plot(
                    i + 0.5,
                    i + 0.5,
                    "ro",
                    markersize=6,
                    markeredgecolor="white",
                    markeredgewidth=1,
                )

            plt.text(
                0.02,
                0.98,
                "Red dots = Correct matches\n"
                "Yellow = High similarity\n"
                "Blue = Low similarity",
                transform=plt.gca().transAxes,
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
                verticalalignment="top",
            )

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            out_path = os.path.join(self.save_dir, filename)
            plt.savefig(out_path, dpi=200, bbox_inches="tight")
            plt.close()
            print(f"Similarity matrix saved to: {out_path}")

        except Exception as e:
            print(f"Error plotting similarity matrix: {e}")
            print(
                f"Matrix shape: "
                f"{similarity_matrix.shape if hasattr(similarity_matrix, 'shape') else 'unknown'}"
            )
            print(f"Matrix type: {type(similarity_matrix)}")

    def create_comprehensive_analysis(self, model, val_loader, epoch=None):
        print(f"Creating comprehensive analysis for epoch {epoch}...")
        model.eval()

        sample_images = []
        sample_texts = []
        sample_count = 0
        max_samples = 100

        try:
            for batch in val_loader:
                if sample_count >= max_samples:
                    break

                if isinstance(batch, dict):
                    batch_images = batch["images"]
                    batch_texts = batch["captions"]
                elif isinstance(batch, (list, tuple)) and len(batch) == 2:
                    batch_images, batch_texts = batch
                else:
                    print(f"Unexpected batch format: {type(batch)}")
                    continue

                if len(batch_images.shape) == 4 and batch_images.shape[1] != 3:
                    batch_images = batch_images.permute(0, 3, 1, 2)

                sample_images.append(batch_images)
                sample_texts.append(batch_texts)
                sample_count += len(batch_images)

                if sample_count >= max_samples:
                    break

            if len(sample_images) == 0:
                print("No validation data available for analysis")
                return

            images = torch.cat(sample_images, dim=0)[:max_samples]
            texts = torch.cat(sample_texts, dim=0)[:max_samples]
            print(f"Processing {len(images)} samples for similarity matrix...")

            with torch.no_grad():
                device = next(model.parameters()).device
                images = images.to(device)
                texts = texts.to(device)
                image_emb, text_emb = model((images, texts), training=False)
                image_emb = torch.nn.functional.normalize(image_emb, p=2, dim=1)
                text_emb = torch.nn.functional.normalize(text_emb, p=2, dim=1)
                similarity_matrix = torch.matmul(
                    image_emb, text_emb.transpose(0, 1)
                )

            print(f"Similarity matrix shape: {similarity_matrix.shape}")
            print(
                f"Similarity matrix range: "
                f"[{similarity_matrix.min():.4f}, {similarity_matrix.max():.4f}]"
            )
            self.plot_similarity_matrix(
                similarity_matrix, f"Similarity Matrix - Epoch {epoch}"
            )

        except Exception as e:
            print(f"Error creating comprehensive analysis: {e}")
            import traceback

            traceback.print_exc()
            print("Skipping visualization generation")


def create_ablation_performance_comparison(results, save_dir, timestamp):
    """Create performance comparison charts for ablation study."""
    excluded_variants = ["no_local_attention", "single_coattention_layer"]
    variants = [v for v in results.keys() if v not in excluded_variants]

    metrics = ["avg_mrr", "avg_recall@1"]
    metric_labels = ["Average MRR", "Recall@1"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    plot_colors = plt.cm.Set3(np.linspace(0, 1, len(variants)))

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]
        values = []
        labels = []
        plotted_variants = []

        for variant in variants:
            if metric in results[variant]:
                values.append(results[variant][metric])
                plotted_variants.append(variant)
                short_label = variant.replace("_", " ").title()
                if variant == "no_orthogonal_loss":
                    short_label = "No\nOrthogonal Loss"
                elif "No " in short_label:
                    short_label = short_label.replace("No ", "No\n")
                labels.append(short_label)

        bars = ax.bar(labels, values, color=plot_colors[: len(values)])
        ax.set_title(label, fontsize=16, fontweight="bold", pad=20)
        ax.set_ylabel("Score", fontsize=14)
        ax.tick_params(axis="x", rotation=45, labelsize=10)
        ax.tick_params(axis="y", labelsize=12)

        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.005,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=11,
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
            )

        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, ha="right")
        max_val = max(values) if values else 1
        ax.set_ylim(0, max_val * 1.2)
        ax.grid(True, alpha=0.3)

        for bar, variant in zip(bars, plotted_variants):
            if variant == "no_orthogonal_loss":
                bar.set_edgecolor("red")
                bar.set_linewidth(2.0)
                bar.set_hatch("//")
            else:
                bar.set_edgecolor("black")
                bar.set_linewidth(0.5)

    plt.tight_layout(pad=3.0)
    save_path = os.path.join(
        save_dir, f"ablation_performance_comparison_{timestamp}.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Ablation performance comparison saved to: {save_path}")


def create_ablation_contribution_analysis(results, save_dir, timestamp):
    """Create contribution analysis for ablation study."""
    baseline_key = None
    if "full_model" in results:
        baseline_key = "full_model"
    elif "Baseline" in results:
        baseline_key = "Baseline"
    else:
        print("No baseline model results found for contribution analysis")
        return

    baseline_mrr = results[baseline_key].get("avg_mrr", 0)

    contributions = {}
    for variant, result in results.items():
        if variant != baseline_key and "avg_mrr" in result:
            contributions[variant] = baseline_mrr - result["avg_mrr"]

    if not contributions:
        print("No valid ablation variants found for contribution analysis")
        return

    sorted_contributions = sorted(
        contributions.items(), key=lambda x: abs(x[1]), reverse=True
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    variants, values = zip(*sorted_contributions)
    bar_colors = ["red" if v > 0 else "blue" for v in values]

    short_labels = []
    for variant in variants:
        short_label = variant.replace("_", " ").title()
        if "No " in short_label:
            short_label = short_label.replace("No ", "No\n")
        short_labels.append(short_label)

    bars = ax1.barh(short_labels, values, color=bar_colors, alpha=0.7)
    ax1.set_xlabel("MRR Contribution (Baseline - Ablated Model)", fontsize=14)
    ax1.set_title("Component Contribution Analysis", fontsize=16, fontweight="bold")
    ax1.axvline(x=0, color="black", linestyle="-", alpha=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis="both", labelsize=12)

    for bar, value in zip(bars, values):
        width = bar.get_width()
        ax1.text(
            width + (0.001 if width >= 0 else -0.001),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            ha="left" if width >= 0 else "right",
            va="center",
            fontweight="bold",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )

    all_variants = [baseline_key] + list(variants)
    all_mrrs = [baseline_mrr] + [results[v]["avg_mrr"] for v in variants]

    short_all_labels = []
    for variant in all_variants:
        short_label = variant.replace("_", " ").title()
        if "No " in short_label:
            short_label = short_label.replace("No ", "No\n")
        short_all_labels.append(short_label)

    bars2 = ax2.bar(
        range(len(all_variants)),
        all_mrrs,
        color=["green"] + ["lightcoral"] * (len(all_variants) - 1),
    )
    ax2.set_xlabel("Model Variant", fontsize=14)
    ax2.set_ylabel("Average MRR", fontsize=14)
    ax2.set_title("Performance Comparison", fontsize=16, fontweight="bold")
    ax2.set_xticks(range(len(all_variants)))
    ax2.set_xticklabels(short_all_labels, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis="both", labelsize=12)

    for bar, value in zip(bars2, all_mrrs):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.005,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )

    for bar in bars2:
        bar.set_edgecolor("black")
        bar.set_linewidth(0.5)

    plt.tight_layout(pad=3.0)
    save_path = os.path.join(
        save_dir, f"ablation_contribution_analysis_{timestamp}.png"
    )
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Ablation contribution analysis saved to: {save_path}")
