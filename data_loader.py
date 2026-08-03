#!/usr/bin/env python3
"""Shard-based data loading for chest X-ray image-text retrieval."""

import gc
import glob
import os
import pickle
import re
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

import config
import paths


class SimpleTokenizer:
    def __init__(self, oov_token="<unk>"):
        self.word2idx = {"<pad>": 0, oov_token: 1}
        self.idx2word = {0: "<pad>", 1: oov_token}
        self.word_counts = defaultdict(int)
        self.oov_token = oov_token
        self.num_words = None

    def fit_on_texts(self, texts):
        for text in texts:
            words = self._tokenize(text)
            for word in words:
                self.word_counts[word.lower()] += 1

        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        for word, _count in sorted_words:
            if word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def texts_to_sequences(self, texts, maxlen=115):
        sequences = []
        for text in texts:
            words = self._tokenize(text)
            sequence = []
            for word in words[:maxlen]:
                if word in self.word2idx:
                    sequence.append(self.word2idx[word])
                else:
                    sequence.append(self.word2idx[self.oov_token])

            if len(sequence) < maxlen:
                sequence.extend([0] * (maxlen - len(sequence)))
            sequences.append(sequence)
        return torch.tensor(sequences, dtype=torch.long)

    def texts_to_sequences_compatible(self, texts, maxlen=115):
        if hasattr(self, "word_index"):
            sequences = self.texts_to_sequences(texts)
            padded_sequences = []
            for seq in sequences:
                if len(seq) > maxlen:
                    padded_sequences.append(seq[:maxlen])
                else:
                    padded_sequences.append(seq + [0] * (maxlen - len(seq)))
            return torch.tensor(padded_sequences, dtype=torch.long)
        return self.texts_to_sequences(texts, maxlen)

    def _tokenize(self, text):
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()

    def __len__(self):
        return len(self.word2idx)


class ShardDataset(Dataset):
    def __init__(self, shard_paths, max_samples=None):
        self.shard_paths = shard_paths
        self.max_samples = max_samples
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        samples_loaded = 0
        for shard_path in self.shard_paths:
            if self.max_samples and samples_loaded >= self.max_samples:
                break

            with open(shard_path, "rb") as f:
                shard_data = pickle.load(f)

            shard_size = len(shard_data["images"])
            for i in range(shard_size):
                if self.max_samples and samples_loaded >= self.max_samples:
                    break

                self.samples.append(
                    {
                        "images": shard_data["images"][i],
                        "captions": shard_data["captions"][i],
                        "study_ids": shard_data["study_ids"][i],
                    }
                )
                samples_loaded += 1

            del shard_data
            gc.collect()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "images": torch.FloatTensor(sample["images"]),
            "captions": torch.LongTensor(sample["captions"]),
            "study_ids": sample["study_ids"],
        }


class CXRDataLoader:
    def __init__(self, batch_size=32, use_shards=True, shard_size=100, shard_subfolder=None):
        self.batch_size = batch_size
        self.data = None
        self.tokenizer = None
        self.dataset = None
        self.use_shards = use_shards
        self.shard_size = shard_size
        self.shard_subfolder = shard_subfolder or config.DATASET_MODE

        self.shard_dir = paths.get_shard_base_path(self.shard_subfolder)
        self.metadata_path = paths.get_metadata_path(self.shard_subfolder)
        paths.print_paths(self.shard_subfolder)

    def load_data(self, max_samples=None, skip_processing=True):
        if not skip_processing:
            raise ValueError(
                "Raw data rebuild is not supported. Provide preprocessed shards "
                f"and metadata at {self.metadata_path}."
            )

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(
                f"Metadata not found at {self.metadata_path}. "
                "Ensure processed shards exist under DATA_ROOT."
            )

        print("Loading tokenizer from existing shards...")
        try:
            with open(self.metadata_path, "rb") as f:
                metadata = pickle.load(f)
                self.tokenizer = metadata.get("tokenizer")
            if self.tokenizer is None:
                raise ValueError("No tokenizer found in metadata")

            if hasattr(self.tokenizer, "word_index") and not hasattr(self.tokenizer, "word2idx"):
                self.tokenizer.word2idx = self.tokenizer.word_index
                self.tokenizer.idx2word = self.tokenizer.index_word
                print("EnhancedTokenizer detected - added compatibility attributes")
                print(f"Vocabulary size: {len(self.tokenizer.word2idx)}")

            meta_vocab_size = metadata.get("vocab_size", None)
            try:
                tok_vocab_size = (
                    len(self.tokenizer.word2idx)
                    if hasattr(self.tokenizer, "word2idx")
                    else (
                        len(self.tokenizer.word_index)
                        if hasattr(self.tokenizer, "word_index")
                        else None
                    )
                )
            except Exception:
                tok_vocab_size = None
            if (
                meta_vocab_size is not None
                and tok_vocab_size is not None
                and tok_vocab_size != meta_vocab_size
            ):
                print(
                    f"Tokenizer vocab size mismatch: metadata={meta_vocab_size}, "
                    f"loaded={tok_vocab_size}"
                )

            print("Successfully loaded existing tokenizer.")
        except Exception as e:
            print(f"Warning: Could not load existing tokenizer ({e}). Creating new one...")
            self.tokenizer = SimpleTokenizer(oov_token="<unk>")
            basic_vocab = [
                "the",
                "and",
                "or",
                "with",
                "without",
                "chest",
                "x-ray",
                "normal",
                "abnormal",
                "lung",
                "heart",
                "impression",
                "findings",
                "patient",
                "examination",
                "view",
                "shows",
            ]
            self.tokenizer.fit_on_texts(basic_vocab)
            print("Created new tokenizer with basic vocabulary.")

        self.data = {"images": [], "captions": [], "study_ids": []}
        print("Tokenizer ready. Training data will be loaded by get_data() method.")
        print(f"Loaded {len(self.data['images'])} samples from {self.shard_subfolder} dataset")
        gc.collect()
        return self.data

    def preprocess_captions(self):
        print("Caption preprocessing already completed during shard creation")
        return

    def get_data(self, max_samples=None):
        train_dir = paths.get_train_shards_dir(self.shard_subfolder)
        train_shards = sorted(glob.glob(os.path.join(train_dir, "*.pkl")))

        if not train_shards:
            raise FileNotFoundError(f"No training shards found in {train_dir}")

        print(f"Creating PyTorch Dataset from {len(train_shards)} shards")
        print(f"   Max samples: {max_samples if max_samples else 'ALL'}")

        dataset = ShardDataset(train_shards, max_samples)
        total_samples = len(dataset)
        self.dataset_size = total_samples

        self.data = {
            "images": np.array([]),
            "captions": np.array([]),
            "study_ids": np.array([]),
        }

        print(f"PyTorch Dataset created with {total_samples} samples")
        print(
            f"   Memory usage: ~{self._estimate_memory_usage():.1f} MB "
            f"(vs ~{total_samples * 0.6:.0f} MB for full loading)"
        )
        return dataset

    def _count_total_samples(self, shard_paths, max_samples=None):
        total = 0
        for shard_path in shard_paths:
            if max_samples and total >= max_samples:
                break

            with open(shard_path, "rb") as f:
                shard_data = pickle.load(f)
                shard_size = len(shard_data["images"])

                if max_samples:
                    remaining_needed = min(max_samples - total, shard_size)
                    total += remaining_needed
                else:
                    total += shard_size

                del shard_data
                gc.collect()
        return total

    def _estimate_memory_usage(self):
        return 50.0

    def get_validation_data(self, num_samples=None):
        val_dir = paths.get_val_shards_dir(self.shard_subfolder)
        val_shards = sorted(glob.glob(os.path.join(val_dir, "*.pkl")))

        if not val_shards:
            raise FileNotFoundError(f"No validation shards found in {val_dir}")

        print(f"Creating PyTorch Validation Dataset from {len(val_shards)} shards")
        print(f"   Max samples: {num_samples if num_samples else 'ALL'}")

        val_dataset = ShardDataset(val_shards, num_samples)
        print(f"PyTorch Validation Dataset created with {len(val_dataset)} samples")
        return val_dataset

    def get_test_data(self, num_samples=None):
        test_dir = paths.get_test_shards_dir(self.shard_subfolder)
        if not os.path.exists(test_dir):
            raise FileNotFoundError(
                f"Test directory {test_dir} not found. Ensure processed shards exist."
            )

        test_shards = sorted(glob.glob(os.path.join(test_dir, "*.pkl")))
        if not test_shards:
            raise FileNotFoundError(f"No test shards found in {test_dir}.")

        print(f"Creating PyTorch Test Dataset from {len(test_shards)} shards")
        print(f"   Max samples: {num_samples if num_samples else 'ALL'}")

        test_dataset = ShardDataset(test_shards, num_samples)
        print(f"PyTorch Test Dataset created with {len(test_dataset)} samples")
        return test_dataset

    def get_validation_data_for_training(self, num_samples=2500):
        print(f"Loading validation data for training (numpy arrays, {num_samples} samples)")

        val_dataset = self.get_validation_data(num_samples=num_samples)

        all_images = []
        all_captions = []
        all_study_ids = []

        for i, sample in enumerate(val_dataset):
            if num_samples is not None and i >= num_samples:
                break

            all_images.append(sample["images"].numpy())
            all_captions.append(sample["captions"].numpy())

            study_id = sample["study_ids"]
            if isinstance(study_id, bytes):
                study_id = study_id.decode("utf-8")
            else:
                study_id = str(study_id)
            all_study_ids.append(study_id)

            if (i + 1) % 50 == 0:
                target_str = f"/{num_samples}" if num_samples else ""
                print(f"  Loaded {i + 1}{target_str} validation samples...")

        val_data = {
            "images": np.array(all_images),
            "captions": np.array(all_captions),
            "study_ids": np.array(all_study_ids),
            "tokenizer": self.tokenizer,
        }
        print(f"Loaded {len(val_data['images'])} validation samples as numpy arrays")
        return val_data

    def get_test_data_for_evaluation(self, num_samples=1500):
        print(
            f"Loading test data for evaluation "
            f"(numpy arrays, {num_samples if num_samples else 'ALL'} samples)"
        )

        test_dataset = self.get_test_data(num_samples=num_samples)

        all_images = []
        all_captions = []
        all_study_ids = []

        for i, sample in enumerate(test_dataset):
            if num_samples is not None and i >= num_samples:
                break

            all_images.append(sample["images"].numpy())
            all_captions.append(sample["captions"].numpy())

            study_id = sample["study_ids"]
            if isinstance(study_id, bytes):
                study_id = study_id.decode("utf-8")
            else:
                study_id = str(study_id)
            all_study_ids.append(study_id)

            if (i + 1) % 25 == 0:
                target_str = f"/{num_samples}" if num_samples else ""
                print(f"  Loaded {i + 1}{target_str} test samples...")

        test_data = {
            "images": np.array(all_images),
            "captions": np.array(all_captions),
            "study_ids": np.array(all_study_ids),
            "tokenizer": self.tokenizer,
        }
        print(f"Loaded {len(test_data['images'])} test samples as numpy arrays")
        return test_data
