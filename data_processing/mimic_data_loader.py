"""
MIMIC-CXR Dataset Loader

This module provides a data loader for the MIMIC-CXR dataset that follows
the same format as the Indiana dataset loader but adapted for MIMIC-CXR structure.

Features:
- Loads MIMIC-CXR metadata CSV and report text files
- Creates train/validation/test splits
- Converts images to normalized arrays
- Tokenizes text (findings + impressions)
- Saves data in MIMIC-CXR compatible shard format
- Optional enhanced vocabulary loading for better text processing
"""

import os
import pandas as pd
import numpy as np
import random
import pickle
import glob
import json
from PIL import Image
import matplotlib.pyplot as plt
import gc
import re
from tqdm import tqdm

# Try to import enhanced tokenizer
try:
    from enhanced_data_loader import clean_medical_text, EnhancedTokenizer
    ENHANCED_TOKENIZER_AVAILABLE = True
except ImportError:
    ENHANCED_TOKENIZER_AVAILABLE = False
    print("Warning: Enhanced tokenizer not available. Using basic tokenization.")

def extract_text_from_report(report_path):
    """
    Extract findings and impression from MIMIC-CXR report file
    
    Args:
        report_path: Path to the report text file
        
    Returns:
        Tuple of (findings, impression) text
    """
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        findings = ""
        impression = ""
        
        # Extract findings section
        findings_match = re.search(r'FINDINGS:(.*?)(?:IMPRESSION:|$)', content, re.DOTALL | re.IGNORECASE)
        if findings_match:
            findings = findings_match.group(1).strip()
        
        # Extract impression section
        impression_match = re.search(r'IMPRESSION:(.*?)$', content, re.DOTALL | re.IGNORECASE)
        if impression_match:
            impression = impression_match.group(1).strip()
        
        return findings, impression
    except Exception as e:
        print(f"Error reading report {report_path}: {e}")
        return "", ""

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

class SimpleTokenizer:
    """Simple tokenizer for compatibility"""
    def __init__(self, oov_token="<unk>"):
        self.word_index = {}
        self.index_word = {}
        self.oov_token = oov_token
        self.oov_index = 1
    
    def fit_on_texts(self, texts):
        """Build vocabulary from texts"""
        word_freq = {}
        for text in texts:
            words = text.lower().split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and assign indices (1-based like Keras)
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Reserve index 1 for OOV token
        self.word_index[self.oov_token] = 1
        self.index_word[1] = self.oov_token
        
        # Assign indices starting from 2
        for i, (word, count) in enumerate(sorted_words, 2):
            self.word_index[word] = i
            self.index_word[i] = word
    
    def texts_to_sequences(self, texts):
        """Convert texts to sequences"""
        sequences = []
        for text in texts:
            words = text.lower().split()
            sequence = [self.word_index.get(word, self.oov_index) for word in words]
            sequences.append(sequence)
        return sequences

class MIMICDatasetLoader:
    """
    MIMIC-CXR Dataset Loader
    
    Loads MIMIC-CXR dataset and converts it to the same format as Indiana dataset
    for compatibility with existing processing pipelines.
    
    Features:
    - Loads metadata CSV and report text files
    - Creates train/validation/test splits
    - Converts images to normalized arrays
    - Tokenizes text (findings + impressions)
    - Saves data in MIMIC-CXR compatible shard format
    - Optional enhanced vocabulary loading
    """
    
    def __init__(
        self,
        metadata_csv_path,
        reports_dir,
        images_dir,
        image_size=(224, 224),
        batch_size=4,
        shuffle=True,
        max_studies=None,
        max_sequence_length=128,
        shard_size=100,
        shard_dir='mimic_shards',
        skip_metadata_processing=False,
        vocab_path=None,
        index_word_path=None
    ):
        """
        Initialize the MIMIC-CXR dataset loader
        
        Args:
            metadata_csv_path: Path to metadata.csv
            reports_dir: Directory containing report text files
            images_dir: Directory containing image files
            image_size: Tuple of (height, width) to resize images to
            batch_size: Batch size for the dataset
            shuffle: Whether to shuffle the dataset
            max_studies: Maximum number of studies to include
            max_sequence_length: Maximum length of text sequences
            shard_size: Number of studies per shard
            shard_dir: Directory to store shards
            skip_metadata_processing: If True, skip metadata processing step
            vocab_path: Path to pre-built vocabulary JSON file
            index_word_path: Path to pre-built index_word JSON file
        """
        self.metadata_csv_path = metadata_csv_path
        self.reports_dir = reports_dir
        self.images_dir = images_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_studies = max_studies
        self.max_sequence_length = max_sequence_length
        self.shard_size = shard_size
        self.skip_metadata_processing = skip_metadata_processing
        
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
        
        # Initialize tokenizer
        if vocab_path and os.path.exists(vocab_path):
            print(f"Loading pre-built vocabulary from: {vocab_path}")
            if ENHANCED_TOKENIZER_AVAILABLE:
                self.tokenizer = EnhancedTokenizer()
                self.tokenizer.load_from_files(vocab_path, index_word_path)
                # Store original tokenizer type for metadata compatibility
                self.tokenizer.original_tokenizer_type = EnhancedTokenizer
            else:
                # Load vocabulary manually for SimpleTokenizer
                with open(vocab_path, 'r') as f:
                    vocab = json.load(f)
                with open(index_word_path, 'r') as f:
                    index_word = json.load(f)
                self.tokenizer = SimpleTokenizer()
                self.tokenizer.word_index = vocab
                self.tokenizer.index_word = index_word
                # Store original tokenizer type for metadata compatibility
                self.tokenizer.original_tokenizer_type = SimpleTokenizer
                print(f"Loaded vocabulary with {len(vocab)} tokens")
        else:
            self.tokenizer = SimpleTokenizer()
            # Store original tokenizer type for metadata compatibility
            self.tokenizer.original_tokenizer_type = SimpleTokenizer
        
        # Track if shards have been created
        self.shards_created = False
        
        # Check for existing metadata
        metadata_path = os.path.join(self.shard_base_dir, 'metadata.pkl')
        if skip_metadata_processing and os.path.exists(metadata_path):
            print("Loading existing metadata and skipping CSV processing...")
            try:
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    loaded_tokenizer = metadata.get('tokenizer', self.tokenizer)
                    self.tokenizer = loaded_tokenizer
                    # Preserve original tokenizer type for compatibility
                    if hasattr(loaded_tokenizer, 'original_tokenizer_type'):
                        self.tokenizer.original_tokenizer_type = loaded_tokenizer.original_tokenizer_type
                    else:
                        # Set default original tokenizer type based on current tokenizer
                        if isinstance(loaded_tokenizer, EnhancedTokenizer):
                            self.tokenizer.original_tokenizer_type = EnhancedTokenizer
                        else:
                            self.tokenizer.original_tokenizer_type = SimpleTokenizer
                    self.shards_created = True
                    
                print("Successfully loaded metadata from existing shards.")
            except Exception as e:
                print(f"Error loading metadata: {e}")
                print("Will process metadata from scratch.")
                self.process_metadata()
        else:
            # Load and process the metadata
            self.process_metadata()
    
    def process_metadata(self):
        """Process the metadata CSV and create the study data"""
        # Read the metadata CSV
        self.metadata_df = pd.read_csv(self.metadata_csv_path)
        
        print(f"Loaded {len(self.metadata_df)} study entries from metadata")
        
        # Process study entries
        self.process_study_entries()
        
        # Process text for the tokenizer (only if not pre-loaded)
        if not hasattr(self.tokenizer, 'word_index') or not self.tokenizer.word_index:
            self.process_text()
        
        # Print DataFrame head
        print("\nStudy Data Head:")
        print(self.metadata_df.head())
        
        # Create shards
        self.create_shards_with_test_split()
    
    def process_study_entries(self):
        """Process study entries and extract text and image paths"""
        self.study_entries = []
        processed_count = 0
        
        for _, row in tqdm(self.metadata_df.iterrows(), total=len(self.metadata_df), desc="Processing studies"):
            if self.max_studies and len(self.study_entries) >= self.max_studies:
                break
            
            study_id = row['study_id']
            image_file = row['image_file']
            report_file = row['report_file']
            
            # Construct paths
            image_path = os.path.join(self.images_dir, image_file)
            report_path = os.path.join(self.reports_dir, report_file)
            
            # Check if files exist
            if not os.path.exists(image_path):
                print(f"Warning: Image file not found: {image_path}")
                continue
            if not os.path.exists(report_path):
                print(f"Warning: Report file not found: {report_path}")
                continue
            
            # Extract text from report
            findings, impression = extract_text_from_report(report_path)
            
            # Include if we have either findings OR impression (more flexible)
            if findings or impression:
                # Combine findings and impression, handling cases where one might be empty
                combined_text = ""
                if findings:
                    combined_text += findings
                if impression:
                    if combined_text:
                        combined_text += " " + impression
                    else:
                        combined_text = impression
                
                self.study_entries.append({
                    'study_id': study_id,
                    'image_path': image_path,
                    'findings': findings,
                    'impression': impression,
                    'combined_text': combined_text,  # Store combined text for processing
                    'official_split': row.get('official_split', 'train'),  # Include official split
                    'hybrid_split': row.get('hybrid_split', 'train') # Include hybrid split
                })
                processed_count += 1
        
        print(f"Successfully processed {processed_count} studies with valid text and images")
    
    def process_text(self):
        """Process the text data (findings and impressions) for tokenization"""
        # Use combined text (findings + impression, or just one if available)
        text_data = []
        for entry in self.study_entries:
            text_data.append(entry['combined_text'].strip())
            
        # Create and fit the tokenizer
        self.tokenizer.fit_on_texts(text_data)
        
        # Convert text to sequences and pad
        sequences = self.tokenizer.texts_to_sequences(text_data)
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')
        
        # Add tokenized sequences to study entries
        for i, entry in enumerate(self.study_entries):
            entry['caption_seq'] = padded_sequences[i]
    
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
    
    def create_shards_with_test_split(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        """Create sharded data files using hybrid MIMIC-CXR split"""
        # Skip if shards are already created
        if self.shards_created:
            print("Shards already exist. Skipping shard creation.")
            return
        
        print("Creating sharded data files with HYBRID MIMIC-CXR split...")
        
        # Use hybrid split from metadata instead of random splitting
        train_entries = []
        val_entries = []
        test_entries = []
        
        for entry in self.study_entries:
            hybrid_split = entry.get('hybrid_split', 'train')  # Use hybrid_split instead of official_split
            
            if hybrid_split == 'train':
                train_entries.append(entry)
            elif hybrid_split == 'validate':
                val_entries.append(entry)
            elif hybrid_split == 'test':
                test_entries.append(entry)
            else:
                # Default to train for unknown splits
                train_entries.append(entry)
        
        print(f"Hybrid split data: {len(train_entries)} training samples, {len(val_entries)} validation samples, {len(test_entries)} test samples")
        
        # Create shards for training data
        self._create_shards_for_split(train_entries, self.train_shard_dir, "train")
        
        # Create shards for validation data
        self._create_shards_for_split(val_entries, self.val_shard_dir, "val")
        
        # Create shards for test data
        self._create_shards_for_split(test_entries, self.test_shard_dir, "test")
        
        # Create metadata file with tokenizer and other necessary info
        # Always save the actual tokenizer instance, not the class type
        tokenizer_for_metadata = self.tokenizer
        
        metadata = {
            'tokenizer': tokenizer_for_metadata,
            'vocab_size': len(self.tokenizer.word_index) + 1,
            'num_train_shards': len(glob.glob(os.path.join(self.train_shard_dir, "*.pkl"))),
            'num_val_shards': len(glob.glob(os.path.join(self.val_shard_dir, "*.pkl"))),
            'num_test_shards': len(glob.glob(os.path.join(self.test_shard_dir, "*.pkl"))),
            'train_studies': [entry['study_id'] for entry in train_entries],
            'val_studies': [entry['study_id'] for entry in val_entries],
            'test_studies': [entry['study_id'] for entry in test_entries],
            'split_type': 'official_mimic_cxr_split'
        }
        
        with open(os.path.join(self.shard_base_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        print(f"Created {metadata['num_train_shards']} training shards, "
              f"{metadata['num_val_shards']} validation shards, and "
              f"{metadata['num_test_shards']} test shards")
        
        self.shards_created = True
    
    def _create_shards_for_split(self, entries, shard_dir, split_name):
        """Create sharded pickle files for a specific data split in MIMIC-CXR format"""
        for shard_idx in range(0, len(entries), self.shard_size):
            shard_entries = entries[shard_idx:shard_idx+self.shard_size]
            
            # Initialize lists to collect data
            images_list = []
            captions_list = []
            study_ids_list = []
            
            print(f"Processing {split_name} shard {shard_idx//self.shard_size + 1}/{(len(entries)-1)//self.shard_size + 1}")
            
            for entry in shard_entries:
                # Load and process image
                image = self.load_and_preprocess_image(entry['image_path'])
                
                if image is not None:
                    # Get caption sequence from entry
                    if 'caption_seq' in entry:
                        caption_seq = entry['caption_seq']
                    else:
                        # Process text if not already done
                        combined_text = entry['findings'] + ' ' + entry['impression']
                        sequences = self.tokenizer.texts_to_sequences([combined_text.strip()])
                        caption_seq = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')[0]
                    
                    # Ensure study_id is string with proper dtype
                    study_id = str(entry['study_id'])
                    
                    images_list.append(image)
                    captions_list.append(caption_seq)
                    study_ids_list.append(study_id)
            
            if images_list:
                # Convert to numpy arrays
                images = np.array(images_list, dtype=np.float32)
                captions = np.array(captions_list, dtype=np.int32)
                study_ids = np.array(study_ids_list, dtype='<U50')  # String array with proper dtype
                
                # Create shard data
                shard_data = {
                    'images': images,
                    'captions': captions,
                    'study_ids': study_ids
                }
                
                # Save shard
                shard_path = os.path.join(shard_dir, f'shard_{shard_idx//self.shard_size:04d}.pkl')
                with open(shard_path, 'wb') as f:
                    pickle.dump(shard_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                print(f"Saved {split_name} shard {shard_idx//self.shard_size + 1} with {len(images)} samples")
    
    def get_training_data(self, num_samples=None):
        """Get training data from shards"""
        return self._get_data_from_shards(self.train_shard_dir, num_samples, "training")
    
    def get_validation_data(self, num_samples=None):
        """Get validation data from shards"""
        return self._get_data_from_shards(self.val_shard_dir, num_samples, "validation")
    
    def get_test_data(self, num_samples=None):
        """Get test data from shards"""
        return self._get_data_from_shards(self.test_shard_dir, num_samples, "test")
    
    def _get_data_from_shards(self, shard_dir, num_samples, split_name):
        """Load data from shards in the specified directory"""
        if not os.path.exists(shard_dir):
            raise FileNotFoundError(f"Shard directory {shard_dir} not found")
        
        # Find all shard files
        shard_files = sorted(glob.glob(os.path.join(shard_dir, "*.pkl")))
        if not shard_files:
            raise FileNotFoundError(f"No shard files found in {shard_dir}")
        
        # Initialize data containers
        all_images = []
        all_captions = []
        all_study_ids = []
        
        # Load data from each shard
        print(f"Loading {split_name} data from {len(shard_files)} shards...")
        for shard_file in shard_files:
            with open(shard_file, 'rb') as f:
                shard_data = pickle.load(f)
            
            all_images.append(shard_data['images'])
            all_captions.append(shard_data['captions'])
            all_study_ids.extend(shard_data['study_ids'])
        
        # Concatenate all data
        images = np.concatenate(all_images, axis=0)
        captions = np.concatenate(all_captions, axis=0)
        study_ids = np.array(all_study_ids, dtype='<U50')
        
        # Limit samples if requested
        if num_samples and num_samples < len(images):
            indices = np.random.choice(len(images), num_samples, replace=False)
            images = images[indices]
            captions = captions[indices]
            study_ids = study_ids[indices]
        
        print(f"Loaded {len(images)} {split_name} samples")
        
        return {
            'images': images,
            'captions': captions,
            'study_ids': study_ids,
            'tokenizer': self.tokenizer,
            'vocab_size': len(self.tokenizer.word_index) + 1
        }
    
    def visualize_samples(self, split='val', num_samples=2):
        """Visualize sample images and their captions"""
        if split == 'train':
            data = self.get_training_data(num_samples)
        elif split == 'val':
            data = self.get_validation_data(num_samples)
        elif split == 'test':
            data = self.get_test_data(num_samples)
        else:
            raise ValueError("split must be 'train', 'val', or 'test'")
        
        images = data['images']
        captions = data['captions']
        study_ids = data['study_ids']
        tokenizer = data['tokenizer']
        
        fig, axes = plt.subplots(1, num_samples, figsize=(4*num_samples, 4))
        if num_samples == 1:
            axes = [axes]
        
        for i in range(num_samples):
            # Display image
            axes[i].imshow(images[i])
            axes[i].set_title(f'Study ID: {study_ids[i]}')
            axes[i].axis('off')
            
            # Decode caption
            caption_seq = captions[i]
            words = []
            for token_id in caption_seq:
                if token_id == 0:  # Skip padding
                    continue
                word = tokenizer.index_word.get(token_id, '<UNK>')
                if word in ['<START>', '<END>', '<PAD>', '<UNK>']:
                    continue
                words.append(word)
            
            caption_text = " ".join(words)
            axes[i].set_xlabel(caption_text[:50] + "..." if len(caption_text) > 50 else caption_text, 
                             fontsize=8, wrap=True)
        
        plt.tight_layout()
        plt.show()
    
    def save_study_data(self, output_path='mimic_study_data.csv'):
        """Save processed study data to CSV for inspection"""
        study_data_list = []
        for entry in self.study_entries:
            study_data_list.append({
                'study_id': entry['study_id'],
                'findings': entry['findings'],
                'impression': entry['impression'],
                'image_path': entry['image_path']
            })
        
        study_df = pd.DataFrame(study_data_list)
        study_df.to_csv(output_path, index=False)
        print(f"Saved study data to {output_path}")

def create_train_val_test_split(study_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Create train/validation/test split for study IDs"""
    # Check ratios sum to 1.0
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
    
    unique_studies = np.unique(study_ids)
    
    # Shuffle and split studies
    np.random.seed(seed)
    shuffled_studies = np.random.permutation(unique_studies)
    
    # Calculate split indices
    train_idx = int(len(shuffled_studies) * train_ratio)
    val_idx = int(len(shuffled_studies) * (train_ratio + val_ratio))
    
    # Split the studies
    train_studies = set(shuffled_studies[:train_idx])
    val_studies = set(shuffled_studies[train_idx:val_idx])
    test_studies = set(shuffled_studies[val_idx:])
    
    return train_studies, val_studies, test_studies 