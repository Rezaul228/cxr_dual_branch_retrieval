#!/usr/bin/env python3
"""Cross-modal retrieval evaluation metrics."""

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from visualize import visualize_retrieval_examples

    visualization_available = True
except ImportError:
    visualization_available = False
    print("Visualization module not available. Install matplotlib for visualization features.")


def _compute_direction_metrics(sim_matrix, direction, k_values, results):
    """Compute MRR, ranks, recall@K, and precision@K for one retrieval direction."""
    correct_indices = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
    sim_np = sim_matrix.cpu().numpy()

    ranks = []
    for i in range(len(sim_np)):
        similarities = sim_np[i]
        correct_score = similarities[i]
        rank = 1 + np.sum(similarities > correct_score)

        tie_count = np.sum(np.isclose(similarities, correct_score)) - 1
        if tie_count > 0:
            rank = rank + tie_count / 2

        ranks.append(rank)

    ranks = np.array(ranks)

    mrr = np.mean(1.0 / ranks)
    results[f"{direction}_mrr"] = float(mrr)

    mean_rank = np.mean(ranks)
    results[f"{direction}_mean_rank"] = float(mean_rank)

    median_rank = np.median(ranks)
    results[f"{direction}_median_rank"] = float(median_rank)

    rank_thresholds = [1, 5, 10, 50, 100]
    for threshold in rank_thresholds:
        pct = 100 * np.mean(ranks <= threshold)
        results[f"{direction}_rank<={threshold}"] = float(pct)

    for k in k_values:
        _, top_k_indices = torch.topk(sim_matrix, k=k, dim=1)

        correct_in_topk = torch.any(
            top_k_indices == correct_indices.unsqueeze(1),
            dim=1,
        )
        recall = torch.mean(correct_in_topk.float())
        results[f"{direction}_recall@{k}"] = float(recall.cpu())

        precision_scores = []
        for i in range(sim_matrix.size(0)):
            correct_count = torch.sum(top_k_indices[i] == correct_indices[i]).float()
            precision_scores.append(correct_count / k)

        precision = torch.mean(torch.stack(precision_scores))
        results[f"{direction}_precision@{k}"] = float(precision.cpu())

    return mrr, mean_rank, median_rank


def _print_direction_results(direction, direction_name, results, k_values, mrr, mean_rank, median_rank):
    print(f"Evaluating {direction_name} Retrieval:")
    print(f"  MRR: {float(mrr):.4f}")
    print(f"  Mean Rank: {float(mean_rank):.2f}")
    print(f"  Median Rank: {float(median_rank):.2f}")

    for k in k_values:
        print(f"  Recall@{k}: {float(results[f'{direction}_recall@{k}']):.4f}")
        print(f"  Precision@{k}: {float(results[f'{direction}_precision@{k}']):.4f}")

    print("  Rank percentages:")
    for threshold in [1, 5, 10, 50, 100]:
        print(f"    <= {threshold}: {float(results[f'{direction}_rank<={threshold}']):.2f}%")


def _aggregate_overall_metrics(results, k_values, include_precision=False):
    print("\nOverall Cross-Modal Retrieval Performance:")

    avg_mrr = (results["i2t_mrr"] + results["t2i_mrr"]) / 2
    results["avg_mrr"] = avg_mrr
    print(f"  Average MRR: {avg_mrr:.4f}")

    avg_mean_rank = (results["i2t_mean_rank"] + results["t2i_mean_rank"]) / 2
    results["avg_mean_rank"] = avg_mean_rank
    print(f"  Average Mean Rank: {avg_mean_rank:.2f}")

    avg_median_rank = (results["i2t_median_rank"] + results["t2i_median_rank"]) / 2
    results["avg_median_rank"] = avg_median_rank
    print(f"  Average Median Rank: {avg_median_rank:.2f}")

    for k in k_values:
        avg_recall = (results[f"i2t_recall@{k}"] + results[f"t2i_recall@{k}"]) / 2
        results[f"avg_recall@{k}"] = avg_recall
        print(f"  Average Recall@{k}: {avg_recall:.4f}")

        if include_precision:
            avg_precision = (
                results[f"i2t_precision@{k}"] + results[f"t2i_precision@{k}"]
            ) / 2
            results[f"avg_precision@{k}"] = avg_precision
            print(f"  Average Precision@{k}: {avg_precision:.4f}")

    try:
        summary = (
            f"MRR: {avg_mrr:.4f}, "
            f"Mean Rank: {avg_mean_rank:.1f}, "
            f"Median Rank: {avg_median_rank:.1f}, "
            + ", ".join([f"R@{k}: {results[f'avg_recall@{k}']:.4f}" for k in k_values])
        )
    except Exception as e:
        summary = f"Summary generation error: {str(e)}"

    results["summary"] = summary
    return results


def evaluate_cross_modal_retrieval(
    model,
    test_data,
    k_values=None,
    visualize=False,
    num_vis_examples=3,
    output_dir=None,
):
    """Evaluate cross-modal retrieval on in-memory image/caption arrays."""
    if k_values is None:
        k_values = [1, 5, 10]

    print("Evaluating cross-modal retrieval performance...")
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

    results = {}
    print("Computing similarity matrices...")
    i2t_sim = torch.matmul(image_emb, text_emb.transpose(0, 1))
    t2i_sim = torch.matmul(text_emb, image_emb.transpose(0, 1))

    print("Calculating metrics...")
    for direction, sim_matrix, direction_name in [
        ("i2t", i2t_sim, "Image-to-Text"),
        ("t2i", t2i_sim, "Text-to-Image"),
    ]:
        mrr, mean_rank, median_rank = _compute_direction_metrics(
            sim_matrix, direction, k_values, results
        )
        _print_direction_results(
            direction, direction_name, results, k_values, mrr, mean_rank, median_rank
        )

    _aggregate_overall_metrics(results, k_values, include_precision=False)

    if visualize and visualization_available and num_vis_examples > 0:
        if "tokenizer" not in test_data and hasattr(test_data, "get"):
            pass
        visualize_retrieval_examples(
            model, test_data, num_examples=num_vis_examples, k=3, output_dir=output_dir
        )

    return results


def evaluate_cross_modal_retrieval_streaming(
    model,
    test_dataset,
    k_values=None,
    batch_size=64,
    visualize=False,
    num_vis_examples=3,
    output_dir=None,
):
    """Evaluate cross-modal retrieval on a Dataset/DataLoader in batches."""
    if k_values is None:
        k_values = [1, 5, 10]

    print(f"Processing in batches of {batch_size}...")
    model.eval()
    device = next(model.parameters()).device

    all_image_embeddings = []
    all_text_embeddings = []
    all_images_for_viz = []
    all_captions_for_viz = []
    all_study_ids = []

    sample_count = 0
    batch_count = 0

    if not isinstance(test_dataset, DataLoader):
        batched_dataset = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        batched_dataset = test_dataset

    with torch.no_grad():
        for batch in batched_dataset:
            if hasattr(batch, "keys"):
                batch_images = batch["images"]
                batch_captions = batch["captions"]
                batch_study_ids = batch["study_ids"]
            else:
                batch_images, batch_captions, batch_study_ids = batch

            if not isinstance(batch_images, torch.Tensor):
                batch_images = torch.FloatTensor(batch_images)
            if not isinstance(batch_captions, torch.Tensor):
                batch_captions = torch.LongTensor(batch_captions)

            if len(batch_images.shape) == 4 and batch_images.shape[-1] == 3:
                batch_images = batch_images.permute(0, 3, 1, 2)

            batch_images = batch_images.to(device)
            batch_captions = batch_captions.to(device)

            batch_image_emb, batch_text_emb = model(
                (batch_images, batch_captions), training=False
            )

            all_image_embeddings.append(batch_image_emb.cpu().numpy())
            all_text_embeddings.append(batch_text_emb.cpu().numpy())

            if len(all_images_for_viz) < num_vis_examples * 2:
                all_images_for_viz.extend(batch_images.cpu().numpy())
                all_captions_for_viz.extend(batch_captions.cpu().numpy())

            if isinstance(batch_study_ids, torch.Tensor):
                study_ids_batch = [str(sid.item()) for sid in batch_study_ids]
            elif hasattr(batch_study_ids, "__iter__") and not isinstance(batch_study_ids, str):
                study_ids_batch = []
                for sid in batch_study_ids:
                    if isinstance(sid, str):
                        study_ids_batch.append(sid)
                    elif hasattr(sid, "numpy"):
                        sid_val = sid.numpy()
                        if isinstance(sid_val, bytes):
                            study_ids_batch.append(sid_val.decode("utf-8"))
                        else:
                            study_ids_batch.append(str(sid_val))
                    else:
                        study_ids_batch.append(str(sid))
            else:
                study_ids_batch = [str(batch_study_ids)]
            all_study_ids.extend(study_ids_batch)

            sample_count += len(batch_images)
            batch_count += 1

            if batch_count % 10 == 0:
                print(f"  Processed {sample_count} samples in {batch_count} batches...")

    print(f"Computed embeddings for {sample_count} test samples")
    print("Concatenating embeddings and computing similarity matrices...")
    image_embeddings = np.concatenate(all_image_embeddings, axis=0)
    text_embeddings = np.concatenate(all_text_embeddings, axis=0)

    print(f"   Image embeddings shape: {image_embeddings.shape}")
    print(f"   Text embeddings shape: {text_embeddings.shape}")

    image_emb_torch = torch.FloatTensor(image_embeddings).to(device)
    text_emb_torch = torch.FloatTensor(text_embeddings).to(device)

    print("Computing similarity matrices...")
    i2t_sim = torch.matmul(image_emb_torch, text_emb_torch.transpose(0, 1))
    t2i_sim = torch.matmul(text_emb_torch, image_emb_torch.transpose(0, 1))

    results = {}
    print("Calculating metrics...")
    for direction, sim_matrix, direction_name in [
        ("i2t", i2t_sim, "Image-to-Text"),
        ("t2i", t2i_sim, "Text-to-Image"),
    ]:
        mrr, mean_rank, median_rank = _compute_direction_metrics(
            sim_matrix, direction, k_values, results
        )
        _print_direction_results(
            direction, direction_name, results, k_values, mrr, mean_rank, median_rank
        )

    print("Overall Performance:")
    _aggregate_overall_metrics(results, k_values, include_precision=True)

    if visualize and visualization_available and num_vis_examples > 0:
        print("\nGenerating retrieval visualizations...")
        viz_test_data = {
            "images": np.array(all_images_for_viz[: num_vis_examples * 2]),
            "captions": np.array(all_captions_for_viz[: num_vis_examples * 2]),
            "study_ids": np.array(all_study_ids[: num_vis_examples * 2]),
        }
        visualize_retrieval_examples(
            model, viz_test_data, num_examples=num_vis_examples, k=3, output_dir=output_dir
        )

    print(f"   Total samples evaluated: {sample_count}")
    print(f"   Memory efficient: Processed in {batch_count} batches")
    return results
