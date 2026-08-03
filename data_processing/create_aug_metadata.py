#!/usr/bin/env python3
"""
Create a single root metadata.pkl for an augmented shards directory.
This writes a tokenizer (from vocab JSON) and shard counts so downstream
code can load the augmented dataset like a standard shards dataset.

Usage:
  python create_aug_metadata.py \
    --aug_dir path/to/augmented_dir \
    --vocab_file mimic_frontal_complete_vocab_extended_vocab.json \
    --index_word_file mimic_frontal_complete_vocab_extended_index_word.json
"""

import os
import argparse
import pickle
import glob
import json

class SimpleTokenizer:
    def __init__(self, word_index, index_word):
        self.word_index = word_index
        self.index_word = index_word
        self.vocab_size = len(word_index) + 1

    def texts_to_sequences(self, texts):
        sequences = []
        for text in texts:
            seq = []
            for word in text.lower().split():
                seq.append(self.word_index.get(word, 1))  # 1 = <unk>
            sequences.append(seq)
        return sequences

    def sequences_to_texts(self, sequences):
        texts = []
        for seq in sequences:
            words = []
            for tid in seq:
                if tid == 0:
                    continue
                words.append(self.index_word.get(str(tid), '<unk>'))
            texts.append(' '.join(words))
        return texts

def main():
    parser = argparse.ArgumentParser(description='Create root metadata.pkl for augmented shards directory')
    parser.add_argument('--aug_dir', required=True, type=str, help='Augmented shards directory (contains train/val/test)')
    parser.add_argument('--vocab_file', required=True, type=str, help='Vocabulary JSON file (word->id)')
    parser.add_argument('--index_word_file', required=True, type=str, help='Index word JSON file (id->word)')
    args = parser.parse_args()

    # Validate paths
    if not os.path.isdir(args.aug_dir):
        raise FileNotFoundError(f'Augmented directory not found: {args.aug_dir}')
    if not os.path.exists(args.vocab_file):
        raise FileNotFoundError(f'Vocab JSON not found: {args.vocab_file}')
    if not os.path.exists(args.index_word_file):
        raise FileNotFoundError(f'Index-word JSON not found: {args.index_word_file}')

    # Load vocab
    with open(args.vocab_file, 'r') as f:
        word_index = json.load(f)
    with open(args.index_word_file, 'r') as f:
        index_word = json.load(f)

    tokenizer = SimpleTokenizer(word_index, index_word)

    # Count shards
    num_train_shards = len(glob.glob(os.path.join(args.aug_dir, 'train', 'shard_*.pkl')))
    num_val_shards = len(glob.glob(os.path.join(args.aug_dir, 'val', 'shard_*.pkl')))
    num_test_shards = len(glob.glob(os.path.join(args.aug_dir, 'test', 'shard_*.pkl')))

    metadata = {
        'tokenizer': tokenizer,
        'vocab_size': tokenizer.vocab_size,
        'num_train_shards': num_train_shards,
        'num_val_shards': num_val_shards,
        'num_test_shards': num_test_shards,
        'vocabulary_source': 'extended_json',
        'vocab_json_paths': {
            'vocab': os.path.abspath(args.vocab_file),
            'index_word': os.path.abspath(args.index_word_file),
        }
    }

    out_path = os.path.join(args.aug_dir, 'metadata.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    print('Wrote metadata:')
    print(f'  {out_path}')
    print(f"  vocab_size: {metadata['vocab_size']}")
    print(f"  train shards: {metadata['num_train_shards']}")
    print(f"  val shards: {metadata['num_val_shards']}")
    print(f"  test shards: {metadata['num_test_shards']}")

if __name__ == '__main__':
    main() 