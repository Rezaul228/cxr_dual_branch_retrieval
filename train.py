#!/usr/bin/env python3
"""
Training script for dual-branch multimodal chest X-ray retrieval.
"""

import argparse
import gc
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
import paths
from data_loader import CXRDataLoader
from models import ContrastiveLoss, DifferenceLoss, MultimodalFusion, SynergyLoss
from visualize import TrainingVisualizer

VOCAB_SIZE = config.get_vocab_size()
MAX_TOKEN_LENGTH = config.get_max_token_length()
EMBED_DIM = config.get_embed_dim()


def setup_gpu_memory():
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"CUDA available with {device_count} GPU(s)")
        for i in range(device_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        torch.cuda.empty_cache()
        print("GPU memory cache cleared")
    else:
        print("CUDA not available, using CPU")


def setup_directories(experiment_name):
    train_viz_dir = os.path.join(paths.OUTPUTS_DIR, experiment_name)
    models_dir = os.path.join(paths.SAVED_MODELS_DIR, experiment_name)
    results_dir = os.path.join(paths.OUTPUTS_DIR, experiment_name)

    paths.ensure_output_dirs()
    for directory in [train_viz_dir, models_dir, results_dir]:
        os.makedirs(directory, exist_ok=True)

    return train_viz_dir, models_dir, results_dir


def compute_recall_k(similarity_matrix, k):
    batch_size = similarity_matrix.size(0)
    _, top_k_indices = torch.topk(similarity_matrix, k=k, dim=1)
    target_indices = torch.arange(batch_size, device=similarity_matrix.device)
    target_indices = target_indices.unsqueeze(1).expand(-1, k)
    correct = (top_k_indices == target_indices).any(dim=1)
    return correct.float().mean().item()


class EnhancedRetrievalTrainer:
    """Trainer for dual-branch synergy / difference architecture."""

    def __init__(
        self,
        model,
        temperature=None,
        learning_rate=1e-5,
        viz_dir="visualizations",
        model_save_path=None,
        experiment_name="dual_branch_exp",
        device=None,
        synergy_weight=0.64,
        main_weight=0.20,
        ortho_weight=0.15,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = model.to(device)
        self.synergy_weight = float(synergy_weight)
        self.main_weight = float(main_weight)
        self.ortho_weight = float(ortho_weight)

        if temperature is None:
            temperature = config.get_current_config()["temperature"]

        self.main_loss_fn = ContrastiveLoss(temperature)
        self.synergy_loss_fn = SynergyLoss(temperature)
        self.difference_loss_fn = DifferenceLoss(temperature)

        weight_decay = config.get_default_weight_decay()
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        print("Optimizer configured with:")
        print(f"  Learning Rate: {learning_rate}")
        print(f"  Weight Decay: {weight_decay}")
        print(f"  Temperature: {temperature}")
        print(
            f"  Loss weights: synergy={self.synergy_weight:.4f}, "
            f"main={self.main_weight:.4f}, ortho={self.ortho_weight:.4f}"
        )

        self.viz = TrainingVisualizer(save_dir=viz_dir)
        self.model_save_path = model_save_path
        self.experiment_name = experiment_name

        self._ensure_model_built()

        self.training_history = {
            "epochs": [],
            "total_losses": [],
            "synergy_losses": [],
            "difference_losses": [],
        }

    def _ensure_model_built(self):
        print("Building model with BranchEncoder architecture...")
        dummy_images = torch.zeros((1, 3, 224, 224), device=self.device)
        dummy_texts = torch.zeros(
            (1, config.get_max_token_length()), dtype=torch.long, device=self.device
        )

        with torch.no_grad():
            _ = self.model((dummy_images, dummy_texts), training=False)

        has_synergy_branch = hasattr(self.model, "synergy_branch")
        has_difference_branch = hasattr(self.model, "difference_branch")
        if not (has_synergy_branch and has_difference_branch):
            raise ValueError(
                "Model must have synergy_branch and difference_branch attributes"
            )

    def train_step(self, batch):
        self.model.train()
        images, texts = batch

        if images.shape[1] != 3:
            images = images.permute(0, 3, 1, 2)

        (
            image_emb,
            text_emb,
            synergy_img_emb,
            synergy_txt_emb,
            diff_img_emb,
            diff_txt_emb,
        ) = self.model((images, texts), training=True, return_branch_embeddings=True)

        synergy_loss = self.synergy_loss_fn(None, (synergy_img_emb, synergy_txt_emb))
        # Difference loss is computed for logging only (not included in total_loss)
        difference_loss = self.difference_loss_fn(None, (diff_img_emb, diff_txt_emb))
        main_loss = self.main_loss_fn(None, (image_emb, text_emb))

        img_orthogonal_loss = torch.abs(
            torch.sum(synergy_img_emb * diff_img_emb, dim=1)
        ).mean()
        txt_orthogonal_loss = torch.abs(
            torch.sum(synergy_txt_emb * diff_txt_emb, dim=1)
        ).mean()
        orthogonal_loss = (img_orthogonal_loss + txt_orthogonal_loss) / 2

        total_loss = (
            self.synergy_weight * synergy_loss
            + self.main_weight * main_loss
            + self.ortho_weight * orthogonal_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return (
            total_loss.item(),
            synergy_loss.item(),
            difference_loss.item(),
            orthogonal_loss.item(),
            main_loss.item(),
        )

    def verify_branch_specialization(self, batch):
        self.model.eval()
        with torch.no_grad():
            images, texts = batch
            if images.shape[1] != 3:
                images = images.permute(0, 3, 1, 2)

            _, _, synergy_img_emb, synergy_txt_emb, diff_img_emb, diff_txt_emb = (
                self.model(
                    (images, texts), training=False, return_branch_embeddings=True
                )
            )

            synergy_similarity = F.cosine_similarity(
                synergy_img_emb, synergy_txt_emb, dim=1
            )
            diff_similarity = F.cosine_similarity(diff_img_emb, diff_txt_emb, dim=1)

            synergy_mean = synergy_similarity.mean().item()
            diff_mean = diff_similarity.mean().item()

            print("\nBranch Specialization Check:")
            print(f"  Synergy Branch Similarity: {synergy_mean:.4f}")
            print(f"  Difference Branch Similarity: {diff_mean:.4f}")
            print(f"  Specialization Gap: {synergy_mean - diff_mean:.4f}")

            is_specialized = synergy_mean > diff_mean
            print(f"  Branches Specialized: {'YES' if is_specialized else 'NO'}")
            return is_specialized, synergy_mean, diff_mean

    def evaluate_recall_k_batched(self, data_loader, k=None, max_samples=None):
        if k is None:
            k = [1, 5, 10]

        self.model.eval()
        all_image_emb = []
        all_text_emb = []
        total_processed = 0

        for batch in data_loader:
            if max_samples and total_processed >= max_samples:
                break

            if isinstance(batch, dict):
                batch_images = batch["images"]
                batch_texts = batch["captions"]
            else:
                batch_images, batch_texts = batch

            if batch_images.shape[1] != 3:
                batch_images = batch_images.permute(0, 3, 1, 2)

            batch_images = batch_images.to(self.device)
            batch_texts = batch_texts.to(self.device)

            with torch.no_grad():
                img_emb, txt_emb = self.model(
                    (batch_images, batch_texts), training=False
                )
                all_image_emb.append(img_emb.cpu())
                all_text_emb.append(txt_emb.cpu())
                total_processed += len(batch_images)

        print(f"Processed {total_processed} validation samples in batches")

        all_image_emb = torch.cat(all_image_emb, dim=0)
        all_text_emb = torch.cat(all_text_emb, dim=0)
        similarity_matrix = torch.matmul(all_image_emb, all_text_emb.transpose(0, 1))

        recalls = {}
        for k_val in k:
            i2t_recall = compute_recall_k(similarity_matrix, k=k_val)
            t2i_recall = compute_recall_k(similarity_matrix.transpose(0, 1), k=k_val)
            recalls[f"recall@{k_val}"] = (i2t_recall + t2i_recall) / 2

        ranks = []
        for i in range(similarity_matrix.size(0)):
            sim_scores = similarity_matrix[i]
            correct_rank = (
                torch.where(torch.argsort(sim_scores, descending=True) == i)[0][0] + 1
            )
            ranks.append(correct_rank.item())

        for i in range(similarity_matrix.size(1)):
            sim_scores = similarity_matrix[:, i]
            correct_rank = (
                torch.where(torch.argsort(sim_scores, descending=True) == i)[0][0] + 1
            )
            ranks.append(correct_rank.item())

        mrr = np.mean(1.0 / np.array(ranks))
        recalls["mrr"] = float(mrr)
        return recalls

    def evaluate_recall_k(self, val_data, k=None):
        if k is None:
            k = [1, 5, 10]

        self.model.eval()
        val_images = torch.tensor(
            val_data["images"], dtype=torch.float32, device=self.device
        )
        val_texts = torch.tensor(
            val_data["captions"], dtype=torch.long, device=self.device
        )

        with torch.no_grad():
            image_emb, text_emb = self.model((val_images, val_texts), training=False)
            similarity_matrix = torch.matmul(image_emb, text_emb.transpose(0, 1))

        recalls = {}
        for k_val in k:
            _, top_k_indices = torch.topk(similarity_matrix, k=k_val, dim=1)
            correct_indices = torch.arange(
                similarity_matrix.size(0), device=self.device
            )
            correct_in_topk = torch.any(
                top_k_indices == correct_indices.unsqueeze(1),
                dim=1,
            )
            recall = torch.mean(correct_in_topk.float())
            recalls[k_val] = float(recall.cpu())
        return recalls

    def train(self, train_loader, val_loader, num_epochs):
        steps_per_epoch = len(train_loader)

        print("\nStarting Dual Branch Training")
        print("Using BranchEncoder architecture")
        print(f"{num_epochs} epochs, {steps_per_epoch} steps per epoch")
        print(f"Device: {self.device}")
        print("=" * 60)

        self.history = {
            "epochs": [],
            "total_loss": [],
            "synergy_loss": [],
            "difference_loss": [],
            "loss_ratio": [],
            "recall@1": [],
            "recall@5": [],
            "recall@10": [],
            "mrr": [],
        }

        epoch_total_loss = 0.0
        epoch_synergy_loss = 0.0
        epoch_difference_loss = 0.0

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            self.model.train()

            epoch_total_losses = []
            epoch_synergy_losses = []
            epoch_difference_losses = []
            epoch_orthogonal_losses = []
            epoch_main_losses = []

            pbar = tqdm(
                enumerate(train_loader),
                total=min(steps_per_epoch, len(train_loader)),
                desc="Training",
            )

            for _step, batch in pbar:
                if isinstance(batch, dict):
                    images = batch["images"]
                    texts = batch["captions"]
                else:
                    images, texts = batch

                images = images.to(self.device)
                texts = texts.to(self.device)

                total_loss, synergy_loss, difference_loss, orthogonal_loss, main_loss = (
                    self.train_step((images, texts))
                )

                epoch_total_losses.append(total_loss)
                epoch_synergy_losses.append(synergy_loss)
                epoch_difference_losses.append(difference_loss)
                epoch_orthogonal_losses.append(orthogonal_loss)
                epoch_main_losses.append(main_loss)

                pbar.set_description(
                    f"Total: {total_loss:.4f}, Syn: {synergy_loss:.4f}, "
                    f"Main: {main_loss:.4f}, Ortho: {orthogonal_loss:.4f}"
                )

            epoch_total_loss = np.mean(epoch_total_losses)
            epoch_synergy_loss = np.mean(epoch_synergy_losses)
            epoch_difference_loss = np.mean(epoch_difference_losses)
            epoch_orthogonal_loss = np.mean(epoch_orthogonal_losses)
            epoch_main_loss = np.mean(epoch_main_losses)
            loss_ratio = epoch_synergy_loss / (epoch_difference_loss + 1e-8)

            recalls = {}
            if val_loader is not None:
                recalls = self.evaluate_recall_k_batched(val_loader, k=[1, 5, 10])
                print(f"\nProcessed {len(val_loader.dataset)} validation samples in batches")
                print(f"Validation on {len(val_loader.dataset)} samples")

            print(f"\nEpoch {epoch + 1} Results:")
            print(f"   Total Loss:      {epoch_total_loss:.6f}")
            print(f"   Synergy Loss:    {epoch_synergy_loss:.6f}")
            print(f"   Main Loss:       {epoch_main_loss:.6f}")
            print(f"   Orthogonal Loss: {epoch_orthogonal_loss:.6f}")
            print(f"   Difference Loss: {epoch_difference_loss:.6f}")
            print(f"   Loss Ratio:      {loss_ratio:.3f}")

            if recalls:
                print("   Validation:")
                for key, recall in recalls.items():
                    print(f"     {key}: {recall:.4f}")

            if (epoch + 1) % 5 == 0 and val_loader is not None:
                sample_batch = next(iter(val_loader))
                if isinstance(sample_batch, dict):
                    sample_images = sample_batch["images"].to(self.device)
                    sample_texts = sample_batch["captions"].to(self.device)
                else:
                    sample_images, sample_texts = sample_batch
                    sample_images = sample_images.to(self.device)
                    sample_texts = sample_texts.to(self.device)
                self.verify_branch_specialization((sample_images, sample_texts))

            viz_metrics = {
                "total_loss": epoch_total_loss,
                "synergy_loss": epoch_synergy_loss,
                "difference_loss": epoch_difference_loss,
                "loss_ratio": loss_ratio,
                **recalls,
            }
            self.viz.update_history(viz_metrics)

            self.history["epochs"].append(epoch + 1)
            self.history["total_loss"].append(epoch_total_loss)
            self.history["synergy_loss"].append(epoch_synergy_loss)
            self.history["difference_loss"].append(epoch_difference_loss)
            self.history["loss_ratio"].append(loss_ratio)
            for key, value in recalls.items():
                if key not in self.history:
                    self.history[key] = []
                self.history[key].append(value)

        if val_loader is not None:
            print("\nFINAL COMPREHENSIVE VALIDATION ON ALL DATA...")
            print("=" * 60)

            final_recalls = self.evaluate_recall_k_batched(val_loader, k=[1, 5, 10])
            print("\nFINAL VALIDATION RESULTS:")
            for key, recall in final_recalls.items():
                print(f"   Final {key}: {recall:.4f}")
            print("=" * 60)

            print("\nTRAINING SUMMARY")
            print("=" * 60)
            print("Final Losses:")
            print(f"   Total: {epoch_total_loss:.6f}")
            print(f"   Synergy: {epoch_synergy_loss:.6f}")
            print(f"   Difference: {epoch_difference_loss:.6f}")

            print("\nDual Branch Verification:")
            print(f"   Synergy learning: {'YES' if epoch_synergy_loss < 3.5 else 'NO'}")
            print(
                f"   Difference learning: "
                f"{'YES' if epoch_difference_loss < 3.5 else 'NO'}"
            )

            try:
                self.save_model()
                print("\nModel saved successfully!")
            except Exception as e:
                print(f"\nError saving model: {str(e)}")

        return epoch_total_loss, epoch_synergy_loss, epoch_difference_loss

    def save_model(self):
        if not self.model_save_path:
            print("No model save path specified!")
            return False

        try:
            base_dir = os.path.dirname(self.model_save_path)
            export_dir = os.path.join(base_dir, "export")
            os.makedirs(export_dir, exist_ok=True)

            model_path = os.path.join(export_dir, "model.pth")
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "training_history": self.training_history,
                    "experiment_name": self.experiment_name,
                },
                model_path,
            )
            print(f"Model saved to: {model_path}")

            weights_path = os.path.join(export_dir, "model_weights.pth")
            torch.save(self.model.state_dict(), weights_path)
            print(f"Model weights saved to: {weights_path}")

            torch.save(self.model.state_dict(), self.model_save_path)
            print(f"Model weights also saved to (backup): {self.model_save_path}")

            model_size = os.path.getsize(model_path) / (1024 * 1024)
            weights_size = os.path.getsize(weights_path) / (1024 * 1024)
            backup_weights_size = os.path.getsize(self.model_save_path) / (1024 * 1024)

            print("\nModel file sizes:")
            print(f"  Complete model: {model_size:.2f} MB")
            print(f"  Weights file (export): {weights_size:.2f} MB")
            print(f"  Weights file (backup): {backup_weights_size:.2f} MB")

            has_synergy = hasattr(self.model, "synergy_branch")
            has_difference = hasattr(self.model, "difference_branch")
            print("\nBranch verification:")
            print(f"   Synergy branch: {'YES' if has_synergy else 'NO'}")
            print(f"   Difference branch: {'YES' if has_difference else 'NO'}")

            try:
                checkpoint = torch.load(model_path, map_location="cpu")
                print("\nSuccessfully verified saved model can be loaded")
                if "training_history" in checkpoint:
                    print("Training history saved")
            except Exception as e:
                print(f"\nError verifying saved model: {e}")
                return False

            return has_synergy and has_difference and model_size > 1
        except Exception as e:
            print(f"Error saving model: {e}")
            return False


def main():
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    default_synergy = config.DEFAULT_TRAINING_CONFIG["synergy_weight"]
    default_main = config.DEFAULT_TRAINING_CONFIG["main_weight"]
    default_ortho = config.DEFAULT_TRAINING_CONFIG["ortho_weight"]

    parser = argparse.ArgumentParser(
        description="Dual-branch multimodal retrieval training"
    )
    parser.add_argument(
        "--experiment_name", type=str, required=True, help="Experiment name (required)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Batch size (default: from config)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (default: from config)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Number of epochs (default: from config)"
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=None,
        help="Number of training samples (None = all)",
    )
    parser.add_argument(
        "--val_samples",
        type=int,
        default=None,
        help="Number of validation samples (None = all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=default_device,
        help=f"Device to use (default: {default_device})",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--synergy_weight",
        type=float,
        default=default_synergy,
        help=f"Weight for synergy loss (default: {default_synergy})",
    )
    parser.add_argument(
        "--main_weight",
        type=float,
        default=default_main,
        help=f"Weight for main contrastive loss (default: {default_main})",
    )
    parser.add_argument(
        "--ortho_weight",
        type=float,
        default=default_ortho,
        help=f"Weight for orthogonal loss (default: {default_ortho})",
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Setting random seed: {args.seed}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Random seeds set for reproducible results")

    batch_size = (
        args.batch_size if args.batch_size is not None else config.get_default_batch_size()
    )
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else config.get_default_learning_rate()
    )
    epochs = args.epochs if args.epochs is not None else config.get_default_epochs()
    train_samples = (
        args.train_samples
        if args.train_samples is not None
        else config.get_default_train_samples()
    )
    val_samples = (
        args.val_samples
        if args.val_samples is not None
        else config.get_default_val_samples()
    )

    config.print_current_config()
    print(f"Experiment: {args.experiment_name}")
    print(f"Device: {device}")
    print(f"Batch Size: {batch_size} (config: {config.get_default_batch_size()})")
    print(f"Learning Rate: {learning_rate} (config: {config.get_default_learning_rate()})")
    print(f"Epochs: {epochs} (config: {config.get_default_epochs()})")
    print(
        f"Loss weights: synergy={args.synergy_weight}, "
        f"main={args.main_weight}, ortho={args.ortho_weight}"
    )

    if train_samples is None:
        print("Training: Using ALL available samples")
    else:
        print(f"Training: Using {train_samples} samples")

    if val_samples is None:
        print("Validation: Using ALL available samples")
    else:
        print(f"Validation: Using {val_samples} samples")

    setup_gpu_memory()
    viz_dir, models_dir, _results_dir = setup_directories(args.experiment_name)

    data_loader = CXRDataLoader(
        batch_size=batch_size,
        use_shards=True,
        shard_size=config.get_current_config()["shard_size"],
        shard_subfolder=config.DATASET_MODE,
    )

    data_loader.load_data(max_samples=train_samples, skip_processing=True)
    train_dataset = data_loader.get_data(max_samples=train_samples)
    val_dataset = data_loader.get_validation_data(num_samples=val_samples)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    print(
        f"Data loaded: {len(train_dataset)} train, {len(val_dataset)} val, "
        f"vocab: {VOCAB_SIZE}"
    )
    gc.collect()

    print("Building model with BranchEncoder architecture...")
    fusion_model = MultimodalFusion(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=config.get_current_config()["num_heads"],
        num_layers=config.get_current_config()["num_layers"],
    ).to(device)

    trainer = EnhancedRetrievalTrainer(
        model=fusion_model,
        learning_rate=learning_rate,
        device=device,
        experiment_name=args.experiment_name,
        model_save_path=os.path.join(
            models_dir, f"model_{args.experiment_name}.pth"
        ),
        viz_dir=viz_dir,
        synergy_weight=args.synergy_weight,
        main_weight=args.main_weight,
        ortho_weight=args.ortho_weight,
    )

    print("\nStarting Dual Branch Training")
    print("Using BranchEncoder architecture")
    print(f"{epochs} epochs, {len(train_loader)} steps per epoch")
    print(f"Device: {device}")
    print("=" * 60 + "\n")

    trainer.train(train_loader=train_loader, val_loader=val_loader, num_epochs=epochs)

    print("\nGenerating training visualizations...")
    trainer.viz.plot_training_progress()

    print("\nGenerating dual branch loss visualizations...")
    trainer.viz.plot_dual_branch_losses()

    if val_loader is not None:
        print("\nGenerating comprehensive analysis...")
        trainer.viz.create_comprehensive_analysis(
            fusion_model, val_loader, epoch=epochs
        )

    print("\nDUAL BRANCH TRAINING COMPLETED!")


if __name__ == "__main__":
    main()
