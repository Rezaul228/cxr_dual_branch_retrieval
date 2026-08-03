#!/usr/bin/env python3
"""
Indiana Dataset Processing Script - MIMIC Vocabulary Version

This script processes the Indiana University chest X-ray dataset using the MIMIC-CXR vocabulary
for cross-dataset evaluation:

Key features:
1. Uses MIMIC-CXR vocabulary and tokenizer
2. Same preprocessing as MIMIC-CXR (sequence length 128)
3. Compatible with MIMIC-trained models
4. Cross-dataset evaluation capability
5. Vocabulary coverage statistics

This approach ensures that Indiana data is processed exactly like MIMIC-CXR data,
allowing direct comparison and cross-dataset model evaluation.
"""

import os
import sys
import argparse
import time
import numpy as np
from datetime import datetime

# Add current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_set_loader_simplified import IndianaDatasetLoaderSimplified

def main():
    parser = argparse.ArgumentParser(description='Process Indiana dataset with MIMIC vocabulary for cross-dataset evaluation')
    parser.add_argument('--reports_csv', type=str, required=True,
                       help='Path to indiana_reports.csv')
    parser.add_argument('--projections_csv', type=str, required=True,
                       help='Path to indiana_projections.csv')
    parser.add_argument('--image_dir', type=str, required=True,
                       help='Directory containing the image files')
    parser.add_argument('--output_dir', type=str, default='shards_indiana_mimic',
                       help='Output directory for shards (default: shards_indiana_mimic)')
    parser.add_argument('--max_studies', type=int, default=None,
                       help='Maximum number of studies to process (default: all)')
    parser.add_argument('--max_sequence_length', type=int, default=128,
                       help='Maximum sequence length for text (default: 128 to match MIMIC-CXR)')
    parser.add_argument('--shard_size', type=int, default=100,
                       help='Number of studies per shard (default: 100)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Training data ratio (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Validation data ratio (default: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='Test data ratio (default: 0.1)')
    parser.add_argument('--min_test_samples', type=int, default=1000,
                       help='Minimum number of test samples to ensure (default: 1000)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--skip_metadata', action='store_true',
                       help='Skip metadata processing if shards already exist')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize sample data after processing')
    
    # MIMIC vocabulary parameters
    parser.add_argument('--vocab_path', type=str, default='mimic_frontal_complete_vocab_vocab.json',
                       help='Path to MIMIC vocabulary JSON file (default: mimic_frontal_complete_vocab_vocab.json)')
    parser.add_argument('--index_word_path', type=str, default='mimic_frontal_complete_vocab_index_word.json',
                       help='Path to MIMIC index_word JSON file (default: mimic_frontal_complete_vocab_index_word.json)')
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.reports_csv):
        print(f"Error: Reports CSV file not found: {args.reports_csv}")
        sys.exit(1)
    
    if not os.path.exists(args.projections_csv):
        print(f"Error: Projections CSV file not found: {args.projections_csv}")
        sys.exit(1)
    
    if not os.path.exists(args.image_dir):
        print(f"Error: Image directory not found: {args.image_dir}")
        sys.exit(1)
    
    # Validate MIMIC vocabulary files
    if not os.path.exists(args.vocab_path):
        print(f"Error: MIMIC vocabulary file not found: {args.vocab_path}")
        sys.exit(1)
    
    if not os.path.exists(args.index_word_path):
        print(f"Error: MIMIC index_word file not found: {args.index_word_path}")
        sys.exit(1)
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-5:
        print(f"Error: Train, validation, and test ratios must sum to 1.0. Current sum: {total_ratio}")
        sys.exit(1)
    
    print("=" * 80)
    print("Indiana Dataset Processing - MIMIC Vocabulary for Cross-Dataset Evaluation")
    print("=" * 80)
    print(f"Reports CSV: {args.reports_csv}")
    print(f"Projections CSV: {args.projections_csv}")
    print(f"Image Directory: {args.image_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Max Studies: {args.max_studies if args.max_studies else 'All'}")
    print(f"Max Sequence Length: {args.max_sequence_length}")
    print(f"Shard Size: {args.shard_size}")
    print(f"Train/Val/Test Split: {args.train_ratio:.1%}/{args.val_ratio:.1%}/{args.test_ratio:.1%}")
    print(f"Random Seed: {args.seed}")
    print(f"Skip Metadata: {args.skip_metadata}")
    print(f"MIMIC Vocabulary: {args.vocab_path}")
    print(f"MIMIC Index Word: {args.index_word_path}")
    print("=" * 80)
    
    # Record start time
    start_time = time.time()
    
    try:
        # Initialize the dataset loader with MIMIC vocabulary
        print("\nInitializing dataset loader with MIMIC vocabulary...")
        loader = IndianaDatasetLoaderSimplified(
            reports_csv_path=args.reports_csv,
            projections_csv_path=args.projections_csv,
            image_dir=args.image_dir,
            max_studies=args.max_studies,
            max_sequence_length=args.max_sequence_length,
            shard_size=args.shard_size,
            shard_dir=args.output_dir,
            skip_metadata_processing=args.skip_metadata,
            vocab_path=args.vocab_path,
            index_word_path=args.index_word_path
        )
        
        print(f"Loaded {len(loader.study_entries)} study entries")
        print(f"Vocabulary size: {len(loader.tokenizer.word_index) + 1}")
        print(f"Number of labels: {len(loader.label_names)}")
        
        # Create shards
        print("\nCreating shards with MIMIC vocabulary...")
        loader.create_shards_with_test_split(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            min_test_samples=args.min_test_samples
        )
        
        # Test data loading
        print("\nTesting data loading...")
        
        # Test training data
        train_images, train_captions, train_study_ids = loader.get_training_data(num_samples=5)
        if train_images is not None:
            print(f"✓ Training data loaded successfully: {train_images.shape}")
        
        # Test validation data
        val_images, val_captions, val_study_ids = loader.get_validation_data(num_samples=5)
        if val_images is not None:
            print(f"✓ Validation data loaded successfully: {val_images.shape}")
        
        # Test test data
        test_images, test_captions, test_study_ids = loader.get_test_data(num_samples=5)
        if test_images is not None:
            print(f"✓ Test data loaded successfully: {test_images.shape}")
        
        # Save study data summary
        print("\nSaving study data summary...")
        loader.save_study_data(f'study_data_indiana_mimic_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        
        # Visualize samples if requested
        if args.visualize:
            print("\nVisualizing sample data...")
            loader.visualize_samples(split='val', num_samples=3)
        
        # Print summary statistics
        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE - MIMIC Vocabulary Cross-Dataset Evaluation")
        print("=" * 80)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        print(f"Total processing time: {processing_time:.2f} seconds")
        
        # Print vocabulary statistics
        vocab_size = len(loader.tokenizer.word_index) + 1
        print(f"Vocabulary size: {vocab_size}")
        
        # Count UNK tokens in sample data
        if train_captions is not None:
            unk_count = np.sum(train_captions == 1)  # 1 is UNK token index
            total_tokens = train_captions.size
            unk_ratio = unk_count / total_tokens if total_tokens > 0 else 0
            print(f"UNK token ratio in training sample: {unk_ratio:.2%}")
        
        # Print sample decoded text
        if train_captions is not None and len(train_captions) > 0:
            print("\nSample decoded text (first training sample):")
            caption_tokens = train_captions[0]
            decoded_words = []
            for token in caption_tokens:
                if token != 0:  # Skip padding
                    word = loader.tokenizer.index_word.get(token, '<unk>')
                    decoded_words.append(word)
            sample_text = ' '.join(decoded_words)
            print(f"Text: {sample_text[:200]}{'...' if len(sample_text) > 200 else ''}")
        
        print("\nKey features for cross-dataset evaluation:")
        print("- Uses MIMIC-CXR vocabulary and tokenizer")
        print("- Same preprocessing as MIMIC-CXR (sequence length 128)")
        print("- Compatible with MIMIC-trained models")
        print("- Enables direct cross-dataset comparison")
        print("- Maintains vocabulary coverage statistics")
        
        print(f"\nOutput directory: {args.output_dir}")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 