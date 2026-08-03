"""
Enhanced Data Loader for Indiana Chest X-ray Dataset

This module provides an enhanced version of the IndianaDatasetLoader that can use
the improved vocabulary builder for better text processing and tokenization.

Features:
- Integration with enhanced vocabulary builder
- Better text preprocessing for medical reports
- Improved tokenization with NLTK
- Stopword removal and medical term preservation
- Configurable vocabulary loading from pre-built vocab files
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
import nltk
from collections import Counter
from nltk.corpus import stopwords
import re

# Try to download NLTK data
try:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
except:
    print("NLTK download failed, using fallback text processing")

def clean_medical_text(text):
    """
    Clean and preprocess medical text for tokenization
    
    Args:
        text: Raw medical text string
        
    Returns:
        List of cleaned tokens
    """
    if pd.isna(text) or text == '':
        return []
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep important medical punctuation
    text = re.sub(r'[^\w\s\-\.]', ' ', text)
    
    # Handle common medical abbreviations and terms
    text = re.sub(r'\bvs\.\b', 'versus', text)
    text = re.sub(r'\betc\.\b', 'etc', text)
    text = re.sub(r'\bdr\.\b', 'doctor', text)
    text = re.sub(r'\bpt\.\b', 'patient', text)
    
    # Tokenize using NLTK for better accuracy
    try:
        tokens = nltk.word_tokenize(text)
    except:
        # Fallback to simple splitting
        tokens = text.split()
    
    # Remove stopwords (but keep medical terms)
    try:
        stop_words = set(stopwords.words('english'))
        # Keep important medical words that might be in stopwords
        medical_keep_words = {
            'no', 'not', 'normal', 'abnormal', 'present', 'absent',
            'mild', 'moderate', 'severe', 'large', 'small', 'right', 'left'
        }
        stop_words = stop_words - medical_keep_words
        
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
    except:
        # Fallback: just remove very short tokens
        tokens = [t for t in tokens if len(t) > 1]
    
    return tokens

class EnhancedTokenizer:
    """
    Enhanced tokenizer with better medical text processing
    Compatible with the SimpleTokenizer interface
    """
    def __init__(self, vocab=None, index_word=None, oov_token="<unk>"):
        self.word_index = vocab or {}
        self.index_word = index_word or {}
        self.oov_token = oov_token
        self.oov_index = self.word_index.get(oov_token, 1)
        
    def load_from_files(self, vocab_path, index_word_path=None):
        """Load vocabulary from saved files"""
        # Load vocabulary
        with open(vocab_path, 'r') as f:
            self.word_index = json.load(f)
        
        # Load index_word mapping if provided
        if index_word_path and os.path.exists(index_word_path):
            with open(index_word_path, 'r') as f:
                self.index_word = json.load(f)
        else:
            # Create reverse mapping
            self.index_word = {int(idx): word for word, idx in self.word_index.items()}
        
        self.oov_index = self.word_index.get(self.oov_token, 1)
        print(f"Loaded vocabulary with {len(self.word_index)} tokens")
        
    def fit_on_texts(self, texts):
        """Build vocabulary from texts (if not pre-loaded)"""
        if self.word_index:
            print("Vocabulary already loaded, skipping fit_on_texts")
            return
            
        word_freq = Counter()
        for text in texts:
            tokens = clean_medical_text(text)
            word_freq.update(tokens)
        
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
        """Convert texts to sequences using enhanced processing"""
        sequences = []
        for text in texts:
            tokens = clean_medical_text(text)
            sequence = [self.word_index.get(token, self.oov_index) for token in tokens]
            sequences.append(sequence)
        return sequences

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

class EnhancedIndianaDatasetLoader:
    """
    Enhanced Indiana University Chest X-ray Dataset Loader
    
    Enhanced version with better text processing and vocabulary management.
    Can use pre-built vocabularies or build new ones with improved processing.
    
    Features:
    - Enhanced text preprocessing for medical reports
    - NLTK-based tokenization with stopword removal
    - Medical term preservation
    - Pre-built vocabulary loading
    - Better text cleaning and normalization
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
        max_sequence_length=64,
        shard_size=100,
        shard_dir='shards',
        skip_metadata_processing=False,
        vocab_path=None,
        index_word_path=None
    ):
        """
        Initialize the enhanced Indiana dataset loader
        
        Args:
            reports_csv_path: Path to indiana_reports.csv
            projections_csv_path: Path to indiana_projections.csv
            image_dir: Directory containing the image files
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
        self.image_dir = image_dir
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
        
        # Initialize enhanced tokenizer
        self.tokenizer = EnhancedTokenizer()
        
        # Load pre-built vocabulary if provided
        if vocab_path and os.path.exists(vocab_path):
            print(f"Loading pre-built vocabulary from: {vocab_path}")
            self.tokenizer.load_from_files(vocab_path, index_word_path)
        
        # Track if shards have been created
        self.shards_created = False
        
        # Store CSV paths
        self.reports_csv_path = reports_csv_path
        self.projections_csv_path = projections_csv_path

        # Check for existing metadata
        metadata_path = os.path.join(self.shard_base_dir, 'metadata.pkl')
        if skip_metadata_processing and os.path.exists(metadata_path):
            print("Loading existing metadata and skipping CSV processing...")
            try:
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                    # Use enhanced tokenizer if available, otherwise fall back to simple one
                    if hasattr(metadata.get('tokenizer'), 'word_index'):
                        self.tokenizer = metadata.get('tokenizer')
                    self.label_names = metadata.get('label_names', [])
                    self.label_to_idx = {label: i for i, label in enumerate(self.label_names)}
                    self.shards_created = True
                    
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
        
        # Process text for the tokenizer (only if not pre-loaded)
        if not self.tokenizer.word_index:
            self.process_text()
        
        # Print DataFrame head
        print("\nStudy Data Head:")
        print(self.study_data.head())
        
        # Create shards
        self.create_shards_with_test_split()
    
    def process_text(self):
        """Process the text data (findings and impressions) for tokenization"""
        # Combine findings and impressions for text
        text_data = []
        for entry in self.study_entries:
            combined_text = entry['findings'] + ' ' + entry['impression']
            text_data.append(combined_text.strip())
            
        # Create and fit the enhanced tokenizer
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
    
    def create_shards_with_test_split(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
        """Create sharded data files with a separate test set"""
        # Skip if shards are already created
        if self.shards_created:
            print("Shards already exist. Skipping shard creation.")
            return
            
        # Check ratios sum to 1.0
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
        
        print("Creating sharded data files with train/val/test split...")
        
        # First, split studies by patient ID to prevent data leakage
        study_ids = [entry['study_id'] for entry in self.study_entries]
        unique_studies = np.unique(study_ids)
        
        # Shuffle and split studies for train/val/test
        np.random.seed(seed)
        shuffled_studies = np.random.permutation(unique_studies)
        
        # Calculate split indices
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
        
        # Create shards for training data
        self._create_shards_for_split(train_entries, self.train_shard_dir, "train")
        
        # Create shards for validation data
        self._create_shards_for_split(val_entries, self.val_shard_dir, "val")
        
        # Create shards for test data
        self._create_shards_for_split(test_entries, self.test_shard_dir, "test")
        
        # Create metadata file with tokenizer and other necessary info
        metadata = {
            'tokenizer': self.tokenizer,
            'label_names': self.label_names,
            'vocab_size': len(self.tokenizer.word_index) + 1,
            'num_train_shards': len(glob.glob(os.path.join(self.train_shard_dir, "*.pkl"))),
            'num_val_shards': len(glob.glob(os.path.join(self.val_shard_dir, "*.pkl"))),
            'num_test_shards': len(glob.glob(os.path.join(self.test_shard_dir, "*.pkl"))),
            'train_studies': list(train_studies),
            'val_studies': list(val_studies),
            'test_studies': list(test_studies)
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
                    if 'caption_seq' in entry:
                        caption_seq = entry['caption_seq']
                    else:
                        # Process text if not already done
                        combined_text = entry['findings'] + ' ' + entry['impression']
                        sequences = self.tokenizer.texts_to_sequences([combined_text.strip()])
                        caption_seq = pad_sequences(sequences, maxlen=self.max_sequence_length, padding='post')[0]
                    
                    # Ensure study_id is string with proper dtype
                    study_id = str(entry['study_id'])
                    
                    images_list.append(frontal_img)
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
    
    def save_study_data(self, output_path='study_data.csv'):
        """Save processed study data to CSV for inspection"""
        study_data_list = []
        for entry in self.study_entries:
            study_data_list.append({
                'study_id': entry['study_id'],
                'findings': entry['findings'],
                'impression': entry['impression'],
                'frontal_path': entry['frontal_path']
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