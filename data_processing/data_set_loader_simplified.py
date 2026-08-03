import os
import pandas as pd
import numpy as np
import random
import pickle
import glob
from PIL import Image
import matplotlib.pyplot as plt
import gc
import re
import json

# Import the EnhancedTokenizer from the enhanced data loader
from enhanced_data_loader import EnhancedTokenizer

def pad_sequences(sequences, maxlen=None, padding='post', truncating='post', value=0):
    """Pad sequences to same length, compatible with Keras pad_sequences"""
    if maxlen is None:
        maxlen = max(len(seq) for seq in sequences)
    
    padded = np.zeros((len(sequences), maxlen), dtype=np.int32)
    
    for i, seq in enumerate(sequences):
        if len(seq) > maxlen:
            # Truncate
            if truncating == 'post':
                padded[i] = seq[:maxlen]
            else:
                padded[i] = seq[-maxlen:]
        else:
            # Pad
            if padding == 'post':
                padded[i, :len(seq)] = seq
            else:
                padded[i, -len(seq):] = seq
    
    return padded

class IndianaDatasetLoaderSimplified:
    """
    Indiana University Chest X-ray Dataset Loader - Simplified Version
    
    This version uses the MIMIC-CXR vocabulary and tokenizer for cross-dataset evaluation:
    - Uses pre-built MIMIC vocabulary (vocab_path and index_word_path)
    - Same tokenizer as MIMIC-CXR processing
    - Same sequence length (128) and preprocessing
    - Compatible with MIMIC-trained models
    
    Processes Indiana University chest X-ray reports and images into MIMIC-CXR 
    compatible shard format for efficient data loading and processing.
    
    Features:
    - Loads and merges reports with projections data
    - Processes and tokenizes text using MIMIC vocabulary
    - Loads and preprocesses chest X-ray images
    - Creates train/validation/test splits at patient level
    - Saves data in memory-efficient shard format
    
    Data Format:
    Each shard contains: {'images': np.array, 'captions': np.array, 'study_ids': np.array}
    - images: (N, 224, 224, 3) normalized image arrays
    - captions: (N, 128) tokenized text sequences (matches MIMIC-CXR format)
    - study_ids: (N,) study identifier strings
    """
    
    def __init__(
        self,
        reports_csv_path,
        projections_csv_path,
        image_dir,
        image_size=(224, 224),
        batch_size=4,
        shuffle=True,
        max_studies=None,
        max_sequence_length=128,  # Changed to 128 to match MIMIC-CXR
        shard_size=100,
        shard_dir='shards_simplified',
        skip_metadata_processing=False,
        vocab_path=None,  # Path to MIMIC vocabulary file
        index_word_path=None  # Path to MIMIC index_word file
    ):
        """
        Initialize the Indiana dataset loader with MIMIC vocabulary
        
        Args:
            reports_csv_path: Path to indiana_reports.csv
            projections_csv_path: Path to indiana_projections.csv
            image_dir: Directory containing the image files
            image_size: Tuple of (height, width) to resize images to
            batch_size: Batch size for the dataset
            shuffle: Whether to shuffle the dataset
            max_studies: Maximum number of studies to include
            max_sequence_length: Maximum length of text sequences (128 to match MIMIC-CXR)
            shard_size: Number of studies per shard
            shard_dir: Directory to store shards (different from original)
            skip_metadata_processing: If True, skip metadata processing step
            vocab_path: Path to MIMIC vocabulary JSON file
            index_word_path: Path to MIMIC index_word JSON file
        """
        self.image_dir = image_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_studies = max_studies
        self.max_sequence_length = max_sequence_length
        self.shard_size = shard_size
        self.skip_metadata_processing = skip_metadata_processing
        self.vocab_path = vocab_path
        self.index_word_path = index_word_path
        
        # Create shard directory if it doesn't exist
        self.shard_base_dir = shard_dir
        os.makedirs(self.shard_base_dir, exist_ok=True)
        
        # Create subdirectories for different data splits
        self.train_shard_dir = os.path.join(self.shard_base_dir, 'train')
        self.val_shard_dir = os.path.join(self.shard_base_dir, 'val')
        self.test_shard_dir = os.path.join(self.shard_base_dir, 'test')
        
        os.makedirs(self.train_shard_dir, exist_ok=True)
        os.makedirs(self.val_shard_dir, exist_ok=True)
        os.makedirs(self.test_shard_dir, exist_ok=True)
        
        # Initialize tokenizer for text processing (uses MIMIC vocabulary)
        self.tokenizer = None
        
        # Track if shards have been created
        self.shards_created = False
        
        # Store CSV paths
        self.reports_csv_path = reports_csv_path
        self.projections_csv_path = projections_csv_path

        # Check for existing metadata
        metadata_path = os.path.join(self.shard_base_dir, 'metadata.pkl')
        if skip_metadata_processing and os.path.exists(metadata_path):
            # Load metadata only if skip_metadata_processing flag is set and metadata exists
            print("Loading existing metadata and skipping CSV processing...")
            try:
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    self.tokenizer = metadata.get('tokenizer')
                    self.label_names = metadata.get('label_names', [])
                    self.label_to_idx = {label: i for i, label in enumerate(self.label_names)}
                    self.shards_created = True
                    
                    # Set up test directory path when loading from metadata
                    self.test_shard_dir = os.path.join(self.shard_base_dir, 'test')
                    
                print("Successfully loaded metadata from existing shards.")
            except Exception as e:
                print(f"Error loading metadata: {e}")
                print("Will process metadata from scratch.")
                self.process_metadata(reports_csv_path, projections_csv_path)
        else:
            # Load and process the metadata
            self.process_metadata(reports_csv_path, projections_csv_path)
        
    def process_metadata(self, reports_csv_path, projections_csv_path):
        """Process the CSV files and create the study data"""
        # Read the CSV files
        self.reports_df = pd.read_csv(reports_csv_path)
        self.projections_df = pd.read_csv(projections_csv_path)
        
        # Merge reports with projections
        self.study_data = pd.merge(self.reports_df, self.projections_df, on='uid')
        
        # Get unique labels across all studies
        all_labels = set()
        for mesh in self.reports_df['MeSH'].dropna():
            all_labels.update(mesh.split(';'))
        self.label_names = sorted(list(all_labels))
        self.label_to_idx = {label: i for i, label in enumerate(self.label_names)}
        
        # Process study groups
        self.process_study_groups()
        
        # Process text using MIMIC vocabulary
        self.process_text_with_mimic_vocab()
        
    def process_text_with_mimic_vocab(self):
        """Process the text data using MIMIC vocabulary"""
        # Load MIMIC vocabulary if provided
        if self.vocab_path and self.index_word_path and os.path.exists(self.vocab_path) and os.path.exists(self.index_word_path):
            print(f"Loading MIMIC vocabulary from: {self.vocab_path}")
            self.tokenizer = EnhancedTokenizer()
            self.tokenizer.load_from_files(self.vocab_path, self.index_word_path)
            print(f"Loaded MIMIC vocabulary with {len(self.tokenizer.word_index)} tokens")
        else:
            print("Warning: MIMIC vocabulary files not found. Creating new tokenizer.")
            # Fallback to creating new tokenizer (not recommended for cross-dataset evaluation)
            self.process_text_simplified()
            return
        
        # Define only the most obvious noise words to remove
        noise_patterns = [
            r'\bxxxx+\b',  # XXXX, XXXXX, etc.
            r'\bxxx\b',    # XXX
            r'\bxx\b',     # XX
            r'\bx\b',      # Single X
            r'\bna\b',     # NA
            r'\bn/a\b',    # N/A
            r'\bnone\b',   # None
            r'\bnull\b',   # Null
            r'\bunknown\b', # Unknown
        ]
        
        # Combine findings and impressions for text
        text_data = []
        for entry in self.study_entries:
            # Simple concatenation with minimal processing
            findings = entry['findings'] if entry['findings'] else ''
            impression = entry['impression'] if entry['impression'] else ''
            combined_text = findings + ' ' + impression
            
            # Only remove obvious noise words
            for pattern in noise_patterns:
                combined_text = re.sub(pattern, '', combined_text, flags=re.IGNORECASE)
            
            # Clean up extra whitespace
            combined_text = ' '.join(combined_text.split())
            combined_text = combined_text.strip()
            
            text_data.append(combined_text)
        
        # Convert text to sequences using MIMIC vocabulary
        sequences = self.tokenizer.texts_to_sequences(text_data)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')
        
        # Add tokenized sequences to study entries
        for i, entry in enumerate(self.study_entries):
            entry['caption_seq'] = padded_sequences[i]
        
        # Print vocabulary coverage statistics
        total_tokens = sum(len(seq) for seq in sequences)
        unk_tokens = sum(sum(1 for token in seq if token == 1) for seq in sequences)  # 1 is UNK token
        coverage = (total_tokens - unk_tokens) / total_tokens if total_tokens > 0 else 0
        print(f"MIMIC vocabulary coverage on Indiana data: {coverage:.2%}")
        print(f"Total tokens: {total_tokens}, UNK tokens: {unk_tokens}")
        
    def process_text_simplified(self):
        """Process the text data with minimal cleaning (fallback method)"""
        # Define only the most obvious noise words to remove
        noise_patterns = [
            r'\bxxxx+\b',  # XXXX, XXXXX, etc.
            r'\bxxx\b',    # XXX
            r'\bxx\b',     # XX
            r'\bx\b',      # Single X
            r'\bna\b',     # NA
            r'\bn/a\b',    # N/A
            r'\bnone\b',   # None
            r'\bnull\b',   # Null
            r'\bunknown\b', # Unknown
        ]
        
        # Combine findings and impressions for text
        text_data = []
        for entry in self.study_entries:
            # Simple concatenation with minimal processing
            findings = entry['findings'] if entry['findings'] else ''
            impression = entry['impression'] if entry['impression'] else ''
            combined_text = findings + ' ' + impression
            
            # Only remove obvious noise words
            for pattern in noise_patterns:
                combined_text = re.sub(pattern, '', combined_text, flags=re.IGNORECASE)
            
            # Clean up extra whitespace
            combined_text = ' '.join(combined_text.split())
            combined_text = combined_text.strip()
            
            text_data.append(combined_text)
            
        # Create and fit the EnhancedTokenizer (fallback method)
        self.tokenizer = EnhancedTokenizer()
        self.tokenizer.fit_on_texts(text_data)
        
        # Convert text to sequences and pad
        sequences = self.tokenizer.texts_to_sequences(text_data)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')
        
        # Add tokenized sequences to study entries
        for i, entry in enumerate(self.study_entries):
            entry['caption_seq'] = padded_sequences[i]
        
    def process_study_groups(self):
        """Process study groups and create study entries"""
        study_groups = self.study_data.groupby('uid')
        self.study_entries = []
        
        for study_id, group in study_groups:
            if self.max_studies and len(self.study_entries) >= self.max_studies:
                break
                    
            frontal_view = group[group['projection'] == 'Frontal'].iloc[0] if any(group['projection'] == 'Frontal') else None
            
            # Only process if frontal view exists
            if frontal_view is not None:
                # Create one-hot encoded labels
                label_vector = np.zeros(len(self.label_names), dtype=np.float32)
                if pd.notna(frontal_view['MeSH']):
                    for label in frontal_view['MeSH'].split(';'):
                        label_vector[self.label_to_idx[label]] = 1.0
                
                self.study_entries.append({
                    'study_id': study_id,
                    'frontal_path': os.path.join(self.image_dir, frontal_view['filename']),
                    'labels': label_vector,
                    'findings': frontal_view['findings'] if pd.notna(frontal_view['findings']) else '',
                    'impression': frontal_view['impression'] if pd.notna(frontal_view['impression']) else ''
                })
        
    def load_and_preprocess_image(self, image_path):
        """Load and preprocess a single image"""
        try:
            img = Image.open(image_path)
            img = img.resize(self.image_size)
            
            # Convert to RGB if not already (handles grayscale X-rays)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to numpy array and normalize
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            return img_array
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def create_shards_with_test_split(self, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42, min_test_samples=1000):
        """
        Create sharded data files with a separate test set for comprehensive evaluation
        
        Args:
            train_ratio: Ratio of data to use for training
            val_ratio: Ratio of data to use for validation
            test_ratio: Ratio of data to use for testing
            seed: Random seed for reproducibility
            min_test_samples: Minimum number of test samples to ensure
        """
        # Skip if shards are already created
        if self.shards_created:
            print("Shards already exist. Skipping shard creation.")
            return
            
        # Check ratios sum to 1.0
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
        
        print("Creating sharded data files with train/val/test split (simplified processing)...")
        
        # First, split studies by patient ID to prevent data leakage
        study_ids = [entry['study_id'] for entry in self.study_entries]
        unique_studies = np.unique(study_ids)
        
        # Shuffle and split studies for train/val/test
        np.random.seed(seed)
        shuffled_studies = np.random.permutation(unique_studies)
        
        # Calculate split indices with minimum test samples guarantee
        total_samples = len(shuffled_studies)
        
        # Ensure minimum test samples
        min_test_studies = max(int(min_test_samples * 0.8), 1)  # Estimate studies needed for min_test_samples
        min_test_ratio = min_test_studies / total_samples
        
        # Adjust ratios if needed to ensure minimum test samples
        if test_ratio < min_test_ratio:
            print(f"Adjusting test ratio from {test_ratio:.1%} to {min_test_ratio:.1%} to ensure minimum {min_test_samples} test samples")
            test_ratio = min_test_ratio
            # Adjust train and val ratios proportionally
            remaining_ratio = 1.0 - test_ratio
            train_ratio = train_ratio / (train_ratio + val_ratio) * remaining_ratio
            val_ratio = val_ratio / (train_ratio + val_ratio) * remaining_ratio
        
        train_idx = int(len(shuffled_studies) * train_ratio)
        val_idx = int(len(shuffled_studies) * (train_ratio + val_ratio))
        
        # Split the studies
        train_studies = set(shuffled_studies[:train_idx])
        val_studies = set(shuffled_studies[train_idx:val_idx])
        test_studies = set(shuffled_studies[val_idx:])
        
        # Group studies by train/val/test split
        train_entries = [entry for entry in self.study_entries if entry['study_id'] in train_studies]
        val_entries = [entry for entry in self.study_entries if entry['study_id'] in val_studies]
        test_entries = [entry for entry in self.study_entries if entry['study_id'] in test_studies]
        
        print(f"Split data: {len(train_entries)} training samples, {len(val_entries)} validation samples, {len(test_entries)} test samples")
        
        # Verify minimum test samples
        if len(test_entries) < min_test_samples:
            print(f"Warning: Only {len(test_entries)} test samples available, requested minimum {min_test_samples}")
        else:
            print(f"✓ Test set has {len(test_entries)} samples (minimum requested: {min_test_samples})")
        
        # Create shards for training data
        self._create_shards_for_split(train_entries, self.train_shard_dir, "train")
        
        # Create shards for validation data
        self._create_shards_for_split(val_entries, self.val_shard_dir, "val")
        
        # Create shards for test data
        self._create_shards_for_split(test_entries, self.test_shard_dir, "test")
        
        # Create metadata file with tokenizer and other necessary info (matches expected format)
        metadata = {
            'tokenizer': self.tokenizer,
            'label_names': self.label_names,
            'vocab_size': len(self.tokenizer.word_index) + 1,
            'num_train_shards': len(glob.glob(os.path.join(self.train_shard_dir, "*.pkl"))),
            'num_val_shards': len(glob.glob(os.path.join(self.val_shard_dir, "*.pkl"))),
            'num_test_shards': len(glob.glob(os.path.join(self.test_shard_dir, "*.pkl"))),
            'train_studies': list(train_studies),
            'val_studies': list(val_studies),
            'test_studies': list(test_studies),
            'processing_type': 'simplified_old_style'
        }
        
        with open(os.path.join(self.shard_base_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Created {metadata['num_train_shards']} training shards, "
              f"{metadata['num_val_shards']} validation shards, and "
              f"{metadata['num_test_shards']} test shards (simplified processing)")
        
        self.shards_created = True
    
    def _create_shards_for_split(self, entries, shard_dir, split_name):
        """
        Create sharded pickle files for a specific data split in MIMIC-CXR format
        Format: {'images': np.array, 'captions': np.array, 'study_ids': np.array}
        """
        for shard_idx in range(0, len(entries), self.shard_size):
            shard_entries = entries[shard_idx:shard_idx+self.shard_size]
            
            # Initialize lists to collect data
            images_list = []
            captions_list = []
            study_ids_list = []
            
            print(f"Processing {split_name} shard {shard_idx//self.shard_size + 1}/{(len(entries)-1)//self.shard_size + 1}")
            
            for entry in shard_entries:
                # Check if entry already has processed image
                if 'frontal_img' in entry:
                    frontal_img = entry['frontal_img']
                elif 'frontal_path' in entry:
                    # Load and process image if we have the path
                    frontal_img = self.load_and_preprocess_image(entry['frontal_path'])
                else:
                    print(f"Warning: Entry missing both frontal_img and frontal_path. Skipping.")
                    continue
                
                if frontal_img is not None:
                    # Get caption sequence from entry
                    caption_seq = entry['caption_seq']
                    
                    images_list.append(frontal_img)
                    captions_list.append(caption_seq)
                    study_ids_list.append(entry['study_id'])
            
            # Convert lists to numpy arrays
            if images_list:
                images_array = np.array(images_list, dtype=np.float32)
                captions_array = np.array(captions_list, dtype=np.int32)
                study_ids_array = np.array(study_ids_list, dtype=object)
                
                # Create shard data
                shard_data = {
                    'images': images_array,
                    'captions': captions_array,
                    'study_ids': study_ids_array
                }
                
                # Save shard
                shard_path = os.path.join(shard_dir, f'shard_{shard_idx//self.shard_size:04d}.pkl')
                with open(shard_path, 'wb') as f:
                    pickle.dump(shard_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                print(f"Saved {split_name} shard {shard_idx//self.shard_size + 1} with {len(images_list)} samples")
            else:
                print(f"Warning: No valid samples in {split_name} shard {shard_idx//self.shard_size + 1}")
    
    def get_training_data(self, num_samples=None):
        """
        Get training data from shards
        
        Args:
            num_samples: Number of samples to return (None for all)
            
        Returns:
            tuple: (images, captions, study_ids) as numpy arrays
        """
        if not self.shards_created:
            print("Shards not created yet. Call create_shards_with_test_split() first.")
            return None, None, None
        
        # Get all training shard files
        shard_files = sorted(glob.glob(os.path.join(self.train_shard_dir, "*.pkl")))
        
        if not shard_files:
            print("No training shards found.")
            return None, None, None
        
        # Load all shards
        all_images = []
        all_captions = []
        all_study_ids = []
        
        for shard_file in shard_files:
            with open(shard_file, 'rb') as f:
                shard_data = pickle.load(f)
                all_images.append(shard_data['images'])
                all_captions.append(shard_data['captions'])
                all_study_ids.append(shard_data['study_ids'])
        
        # Concatenate all shards
        images = np.concatenate(all_images, axis=0)
        captions = np.concatenate(all_captions, axis=0)
        study_ids = np.concatenate(all_study_ids, axis=0)
        
        # Limit samples if requested
        if num_samples is not None:
            indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
            images = images[indices]
            captions = captions[indices]
            study_ids = study_ids[indices]
        
        return images, captions, study_ids
    
    def get_validation_data(self, num_samples=None):
        """
        Get validation data from shards
        
        Args:
            num_samples: Number of samples to return (None for all)
            
        Returns:
            tuple: (images, captions, study_ids) as numpy arrays
        """
        if not self.shards_created:
            print("Shards not created yet. Call create_shards_with_test_split() first.")
            return None, None, None
        
        # Get all validation shard files
        shard_files = sorted(glob.glob(os.path.join(self.val_shard_dir, "*.pkl")))
        
        if not shard_files:
            print("No validation shards found.")
            return None, None, None
        
        # Load all shards
        all_images = []
        all_captions = []
        all_study_ids = []
        
        for shard_file in shard_files:
            with open(shard_file, 'rb') as f:
                shard_data = pickle.load(f)
                all_images.append(shard_data['images'])
                all_captions.append(shard_data['captions'])
                all_study_ids.append(shard_data['study_ids'])
        
        # Concatenate all shards
        images = np.concatenate(all_images, axis=0)
        captions = np.concatenate(all_captions, axis=0)
        study_ids = np.concatenate(all_study_ids, axis=0)
        
        # Limit samples if requested
        if num_samples is not None:
            indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
            images = images[indices]
            captions = captions[indices]
            study_ids = study_ids[indices]
        
        return images, captions, study_ids
    
    def get_test_data(self, num_samples=None):
        """
        Get test data from shards
        
        Args:
            num_samples: Number of samples to return (None for all)
            
        Returns:
            tuple: (images, captions, study_ids) as numpy arrays
        """
        if not self.shards_created:
            print("Shards not created yet. Call create_shards_with_test_split() first.")
            return None, None, None
        
        # Get all test shard files
        shard_files = sorted(glob.glob(os.path.join(self.test_shard_dir, "*.pkl")))
        
        if not shard_files:
            print("No test shards found.")
            return None, None, None
        
        # Load all shards
        all_images = []
        all_captions = []
        all_study_ids = []
        
        for shard_file in shard_files:
            with open(shard_file, 'rb') as f:
                shard_data = pickle.load(f)
                all_images.append(shard_data['images'])
                all_captions.append(shard_data['captions'])
                all_study_ids.append(shard_data['study_ids'])
        
        # Concatenate all shards
        images = np.concatenate(all_images, axis=0)
        captions = np.concatenate(all_captions, axis=0)
        study_ids = np.concatenate(all_study_ids, axis=0)
        
        # Limit samples if requested
        if num_samples is not None:
            indices = np.random.choice(len(images), min(num_samples, len(images)), replace=False)
            images = images[indices]
            captions = captions[indices]
            study_ids = study_ids[indices]
        
        return images, captions, study_ids
    
    def visualize_samples(self, split='val', num_samples=2):
        """
        Visualize sample images and their captions
        
        Args:
            split: Which split to visualize ('train', 'val', 'test')
            num_samples: Number of samples to visualize
        """
        if split == 'train':
            images, captions, study_ids = self.get_training_data(num_samples)
        elif split == 'val':
            images, captions, study_ids = self.get_validation_data(num_samples)
        elif split == 'test':
            images, captions, study_ids = self.get_test_data(num_samples)
        else:
            print("Invalid split. Use 'train', 'val', or 'test'.")
            return
        
        if images is None:
            print(f"No {split} data available.")
            return
        
        fig, axes = plt.subplots(1, num_samples, figsize=(4*num_samples, 4))
        if num_samples == 1:
            axes = [axes]
        
        for i in range(num_samples):
            axes[i].imshow(images[i])
            axes[i].set_title(f'Study ID: {study_ids[i]}')
            axes[i].axis('off')
            
            # Decode caption
            caption_tokens = captions[i]
            decoded_words = []
            for token in caption_tokens:
                if token != 0:  # Skip padding
                    word = self.tokenizer.index_word.get(token, '<unk>')
                    decoded_words.append(word)
            
            caption_text = ' '.join(decoded_words)
            axes[i].set_xlabel(caption_text[:100] + '...' if len(caption_text) > 100 else caption_text, 
                             fontsize=8, wrap=True)
        
        plt.tight_layout()
        plt.show()
    
    def save_study_data(self, output_path='study_data_simplified.csv'):
        """Save the processed study data to a CSV file"""
        if not hasattr(self, 'study_entries'):
            print("No study data to save.")
            return
        
        # Convert to DataFrame
        data = []
        for entry in self.study_entries:
            data.append({
                'study_id': entry['study_id'],
                'frontal_path': entry['frontal_path'],
                'findings': entry['findings'],
                'impression': entry['impression'],
                'labels': ';'.join([self.label_names[i] for i, val in enumerate(entry['labels']) if val == 1.0])
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Saved study data to {output_path}")

def create_train_val_test_split(study_ids, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Create train/validation/test split for study IDs
    
    Args:
        study_ids: List of study IDs
        train_ratio: Ratio for training data
        val_ratio: Ratio for validation data
        test_ratio: Ratio for test data
        seed: Random seed
        
    Returns:
        tuple: (train_ids, val_ids, test_ids)
    """
    unique_studies = np.unique(study_ids)
    np.random.seed(seed)
    shuffled_studies = np.random.permutation(unique_studies)
    
    train_idx = int(len(shuffled_studies) * train_ratio)
    val_idx = int(len(shuffled_studies) * (train_ratio + val_ratio))
    
    train_studies = shuffled_studies[:train_idx]
    val_studies = shuffled_studies[train_idx:val_idx]
    test_studies = shuffled_studies[val_idx:]
    
    return train_studies, val_studies, test_studies 