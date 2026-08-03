#!/usr/bin/env python3
"""
Indiana Chest X-ray Dataset Processing Script

This script processes the Indiana University Chest X-ray dataset and converts it 
to MIMIC-CXR compatible shard format for efficient loading and processing.

Features:
- Processes CSV files (reports and projections)
- Creates train/validation/test splits
- Converts images to normalized arrays
- Tokenizes text (findings + impressions)
- Saves data in MIMIC-CXR compatible shard format
- Optional enhanced vocabulary loading for better text processing

Usage:
    python process_indiana_data.py --reports_csv path/to/reports.csv \
                                   --projections_csv path/to/projections.csv \
                                   --image_dir path/to/images

Enhanced Usage (with pre-built vocabulary):
    python process_indiana_data.py --reports_csv path/to/reports.csv \
                                   --projections_csv path/to/projections.csv \
                                   --image_dir path/to/images \
                                   --enhanced_loader \
                                   --vocab_path path/to/vocab.json

Output:
    shards/
    ├── metadata.pkl
    ├── train/
    │   └── shard_0000.pkl, shard_0001.pkl, ...
    ├── val/
    │   └── shard_0000.pkl, shard_0001.pkl, ...
    └── test/
        └── shard_0000.pkl, shard_0001.pkl, ...
"""

import os
import sys
import argparse
import numpy as np
import random
import pickle
from data_set_loader import IndianaDatasetLoader

# Try to import enhanced loader
try:
    from enhanced_data_loader import EnhancedIndianaDatasetLoader
    ENHANCED_LOADER_AVAILABLE = True
except ImportError:
    ENHANCED_LOADER_AVAILABLE = False
    print("Warning: Enhanced data loader not available. Using basic loader.")

def main(args):
    """Main data processing function"""
    print("=== Indiana Chest X-ray Dataset Processing ===")
    print(f"Reports CSV: {args.reports_csv}")
    print(f"Projections CSV: {args.projections_csv}")
    print(f"Image Directory: {args.image_dir}")
    print(f"Output Directory: {args.output_dir}")
    
    if args.enhanced_loader:
        if not ENHANCED_LOADER_AVAILABLE:
            print("✗ Error: Enhanced loader requested but not available.")
            print("Please ensure enhanced_data_loader.py is in the same directory.")
            return False
        print(f"Enhanced Loader: Enabled")
        if args.vocab_path:
            print(f"Vocabulary Path: {args.vocab_path}")
        if args.index_word_path:
            print(f"Index Word Path: {args.index_word_path}")
    else:
        print(f"Enhanced Loader: Disabled (using basic loader)")
    
    print()
    
    # Set seed for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Initialize the dataset loader
    print("Initializing Indiana dataset loader...")
    if args.skip_processing:
        print("Skip processing flag is set. Will try to use existing processed data.")
    
    # Choose loader based on arguments
    if args.enhanced_loader and ENHANCED_LOADER_AVAILABLE:
        loader = EnhancedIndianaDatasetLoader(
            reports_csv_path=args.reports_csv,
            projections_csv_path=args.projections_csv,
            image_dir=args.image_dir,
            image_size=(args.image_size, args.image_size),
            batch_size=args.batch_size,
            max_studies=args.max_studies,
            max_sequence_length=args.max_seq_length,
            shard_size=args.shard_size,
            shard_dir=args.output_dir,
            skip_metadata_processing=args.skip_processing,
            vocab_path=args.vocab_path,
            index_word_path=args.index_word_path
        )
    else:
        loader = IndianaDatasetLoader(
            reports_csv_path=args.reports_csv,
            projections_csv_path=args.projections_csv,
            image_dir=args.image_dir,
            image_size=(args.image_size, args.image_size),
            batch_size=args.batch_size,
            max_studies=args.max_studies,
            max_sequence_length=args.max_seq_length,
            shard_size=args.shard_size,
            shard_dir=args.output_dir,
            skip_metadata_processing=args.skip_processing
        )
    
    # Check if we should skip processing
    metadata_path = os.path.join(loader.shard_base_dir, 'metadata.pkl')
    if args.skip_processing and os.path.exists(metadata_path):
        print("Existing data shards found. Skipping data processing...")
        # Mark shards as created to prevent reprocessing
        loader.shards_created = True
        
        # Load metadata to get tokenizer and other info
        try:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                loader.tokenizer = metadata.get('tokenizer')
                loader.label_names = metadata.get('label_names', [])
            print("Successfully loaded metadata from existing shards.")
        except Exception as e:
            print(f"Error loading metadata: {e}")
            print("Will process data from scratch.")
            loader.create_shards_with_test_split()
    else:
        # Process data from scratch
        print("Creating data shards with train/val/test split...")
        if not hasattr(loader, 'test_shard_dir') or not os.path.exists(loader.test_shard_dir):
            loader.create_shards_with_test_split()
    
    # Get data statistics
    print("\n=== Data Processing Complete ===")
    
    # Load and display statistics
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    print(f"Vocabulary size: {metadata['vocab_size']}")
    print(f"Number of train shards: {metadata['num_train_shards']}")
    print(f"Number of validation shards: {metadata['num_val_shards']}")
    print(f"Number of test shards: {metadata['num_test_shards']}")
    
    # Test loading a small sample of data
    if args.test_loading:
        print("\n=== Testing Data Loading ===")
        
        try:
            # Test validation data loading
            val_data = loader.get_validation_data(num_samples=5)
            print(f"Successfully loaded {len(val_data['images'])} validation samples")
            print(f"Image shape: {val_data['images'][0].shape}")
            print(f"Caption shape: {val_data['captions'][0].shape}")
            
            # Test training data loading
            train_data = loader.get_training_data(num_samples=5)
            print(f"Training data contains {len(train_data['images'])} samples")
            
            # Test test data loading
            test_data = loader.get_test_data(num_samples=5)
            print(f"Successfully loaded {len(test_data['images'])} test samples")
            
            print("✓ All data loading tests passed!")
            
        except Exception as e:
            print(f"✗ Error during data loading test: {e}")
            return False
    
    print(f"\n✓ Data processing completed successfully!")
    print(f"Data saved to: {loader.shard_base_dir}")
    print(f"Directory structure:")
    print(f"  {loader.shard_base_dir}/")
    print(f"    ├── metadata.pkl")
    print(f"    ├── train/")
    print(f"    │   └── shard_0000.pkl, shard_0001.pkl, ...")
    print(f"    ├── val/")
    print(f"    │   └── shard_0000.pkl, shard_0001.pkl, ...")
    print(f"    └── test/")
    print(f"        └── shard_0000.pkl, shard_0001.pkl, ...")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process Indiana Chest X-ray dataset into MIMIC-CXR compatible format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input data parameters
    parser.add_argument("--reports_csv", type=str, 
                        default="archive/indiana_reports.csv",
                        help="Path to indiana_reports.csv file")
    parser.add_argument("--projections_csv", type=str, 
                        default="archive/indiana_projections.csv",
                        help="Path to indiana_projections.csv file")
    parser.add_argument("--image_dir", type=str, 
                        default="archive/images/images_normalized",
                        help="Directory containing the image files")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="shards",
                        help="Output directory for processed shards")
    parser.add_argument("--max_studies", type=int, default=None,
                        help="Maximum number of studies to include (None for all)")
    
    # Processing parameters
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for data loading")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # Data format parameters
    parser.add_argument("--image_size", type=int, default=224,
                        help="Size to resize images to (square)")
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="Maximum length of text sequences")
    parser.add_argument("--shard_size", type=int, default=100,
                        help="Number of samples per shard")
    
    # Enhanced loader parameters
    parser.add_argument("--enhanced_loader", action="store_true",
                        help="Use enhanced data loader with better text processing")
    parser.add_argument("--vocab_path", type=str, default=None,
                        help="Path to pre-built vocabulary JSON file")
    parser.add_argument("--index_word_path", type=str, default=None,
                        help="Path to pre-built index_word JSON file")
    
    # Control parameters
    parser.add_argument("--skip_processing", action="store_true",
                        help="Skip processing if data shards already exist")
    parser.add_argument("--test_loading", action="store_true",
                        help="Test data loading after processing")
    
    args = parser.parse_args()
    
    # Validate input files exist
    if not args.skip_processing:
        if not os.path.exists(args.reports_csv):
            print(f"✗ Error: Reports CSV file not found: {args.reports_csv}")
            sys.exit(1)
        if not os.path.exists(args.projections_csv):
            print(f"✗ Error: Projections CSV file not found: {args.projections_csv}")
            sys.exit(1)
        if not os.path.exists(args.image_dir):
            print(f"✗ Error: Image directory not found: {args.image_dir}")
            sys.exit(1)
    
    # Validate enhanced loader arguments
    if args.enhanced_loader:
        if args.vocab_path and not os.path.exists(args.vocab_path):
            print(f"✗ Error: Vocabulary file not found: {args.vocab_path}")
            sys.exit(1)
        if args.index_word_path and not os.path.exists(args.index_word_path):
            print(f"✗ Error: Index word file not found: {args.index_word_path}")
            sys.exit(1)
    
    # Run the main processing
    success = main(args)
    sys.exit(0 if success else 1) 