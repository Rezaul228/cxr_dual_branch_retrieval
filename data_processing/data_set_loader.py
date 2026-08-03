import os
import pandas as pd
import numpy as np
import random
import pickle
import glob
from PIL import Image
import matplotlib.pyplot as plt
import gc

# Simple tokenizer to replace Keras tokenizer with same interface
class SimpleTokenizer:
    def __init__(self, oov_token="<unk>"):
        self.word_index = {}
        self.index_word = {}
        self.oov_token = oov_token
        self.word_counts = {}
        
    def fit_on_texts(self, texts):
        """Fit tokenizer on texts, compatible with Keras tokenizer interface"""
        word_freq = {}
        for text in texts:
            for word in text.lower().split():
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
            self.word_counts[word] = count
    
    def texts_to_sequences(self, texts):
        """Convert texts to sequences, compatible with Keras tokenizer interface"""
        sequences = []
        for text in texts:
            sequence = []
            for word in text.lower().split():
                sequence.append(self.word_index.get(word, 1))  # 1 is OOV token index
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

class IndianaDatasetLoader:
    """
    Indiana University Chest X-ray Dataset Loader
    
    Processes Indiana University chest X-ray reports and images into MIMIC-CXR 
    compatible shard format for efficient data loading and processing.
    
    Features:
    - Loads and merges reports with projections data
    - Processes and tokenizes text (findings + impressions)
    - Loads and preprocesses chest X-ray images
    - Creates train/validation/test splits at patient level
    - Saves data in memory-efficient shard format
    
    Data Format:
    Each shard contains: {'images': np.array, 'captions': np.array, 'study_ids': np.array}
    - images: (N, 224, 224, 3) normalized image arrays
    - captions: (N, max_seq_length) tokenized text sequences  
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
        max_sequence_length=64,
        shard_size=100,
        shard_dir='shards',
        skip_metadata_processing=False
    ):
        """
        Initialize the Indiana dataset loader
        
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
        
        # Initialize tokenizer for text processing
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
        
        # Process text for the tokenizer
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
            
        # Create and fit the tokenizer
        self.tokenizer = SimpleTokenizer()
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
            
            # Only process if frontal view exists (removed lateral view check)
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
        

        '''
        """Process study groups and create study entries"""
        study_groups = self.study_data.groupby('uid')
        self.study_entries = []
        
        for study_id, group in study_groups:
            if self.max_studies and len(self.study_entries) >= self.max_studies:
                break
                
            frontal_view = group[group['projection'] == 'Frontal'].iloc[0] if any(group['projection'] == 'Frontal') else None
            lateral_view = group[group['projection'] == 'Lateral'].iloc[0] if any(group['projection'] == 'Lateral') else None
            
            if frontal_view is not None and lateral_view is not None:
                # Create one-hot encoded labels
                label_vector = np.zeros(len(self.label_names), dtype=np.float32)
                if pd.notna(frontal_view['MeSH']):
                    for label in frontal_view['MeSH'].split(';'):
                        label_vector[self.label_to_idx[label]] = 1.0
                
                self.study_entries.append({
                    'study_id': study_id,
                    'frontal_path': os.path.join(self.image_dir, frontal_view['filename']),
                    'lateral_path': os.path.join(self.image_dir, lateral_view['filename']),
                    'labels': label_vector,
                    'findings': frontal_view['findings'] if pd.notna(frontal_view['findings']) else '',
                    'impression': frontal_view['impression'] if pd.notna(frontal_view['impression']) else ''
                })
        '''
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
        """
        Create sharded data files with a separate test set for comprehensive evaluation
        
        Args:
            train_ratio: Ratio of data to use for training
            val_ratio: Ratio of data to use for validation
            test_ratio: Ratio of data to use for testing
            seed: Random seed for reproducibility
        """
        # Skip if shards are already created
        if self.shards_created:
            print("Shards already exist. Skipping shard creation.")
            return
            
        # Check ratios sum to 1.0
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1.0"
        
        # We'll use the study_entries that were already processed in process_study_groups
        # No need to call a non-existent method
        
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
                    caption_seq = entry.get('caption_seq', None)
                    if caption_seq is None and 'findings' in entry and 'impression' in entry:
                        # Process text if needed
                        combined_text = entry['findings'] + ' ' + entry['impression']
                        caption_seq = self.tokenizer.texts_to_sequences([combined_text])[0]
                        caption_seq = pad_sequences([caption_seq], maxlen=self.max_sequence_length, padding='post')[0]
                    
                    # Collect data for stacking
                    images_list.append(frontal_img)
                    captions_list.append(caption_seq)
                    # Convert study_id to string to ensure TensorFlow compatibility
                    study_ids_list.append(str(entry['study_id']))
            
            # Stack data into numpy arrays (MIMIC-CXR format)
            shard_data = {
                'images': np.stack(images_list, axis=0),
                'captions': np.stack(captions_list, axis=0), 
                # Use proper string dtype for study_ids array
                'study_ids': np.array(study_ids_list, dtype='<U50')
            }
            
            # Save the shard with proper naming convention
            shard_path = os.path.join(shard_dir, f"shard_{shard_idx//self.shard_size:04d}.pkl")
            with open(shard_path, 'wb') as f:
                pickle.dump(shard_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            print(f"Saved shard with {len(images_list)} samples: {shard_path}")
            
            # Free memory
            del shard_data, images_list, captions_list, study_ids_list
            gc.collect()
       
       
       
    '''
        """
        Create sharded pickle files for a specific data split
        
        Args:
            entries: List of study entries for this split
            shard_dir: Directory to save shards
            split_name: Name of the split (train, val, etc.)
        """
        # Process in batches of shard_size
        for shard_idx in range(0, len(entries), self.shard_size):
            shard_entries = entries[shard_idx:shard_idx+self.shard_size]
            shard_data = []
            
            print(f"Processing {split_name} shard {shard_idx//self.shard_size + 1}/{(len(entries)-1)//self.shard_size + 1}")
            
            for entry in shard_entries:
                # Load and preprocess images
                frontal_img = self.load_and_preprocess_image(entry['frontal_path'])
                lateral_img = self.load_and_preprocess_image(entry['lateral_path'])
                
                if frontal_img is not None and lateral_img is not None:
                    # Store processed data including images, captions, and study ID
                    shard_data.append({
                        'frontal_img': frontal_img,
                        'lateral_img': lateral_img,
                        'caption_seq': entry['caption_seq'],
                        'labels': entry['labels'],
                        'study_id': entry['study_id']
                    })
            
            # Save the shard
            shard_path = os.path.join(shard_dir, f"{split_name}_shard_{shard_idx//self.shard_size}.pkl")
            with open(shard_path, 'wb') as f:
                pickle.dump(shard_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Free memory
            del shard_data
            gc.collect()
    '''   
    def get_training_data(self, num_samples=None):
        """
        Get training data using the sharded approach
        
        Args:
            num_samples: Maximum number of samples to include (None for all)
        
        Returns:
            Dictionary with training data
        """
        # Ensure shards are created
        if not self.shards_created:
            self.create_shards_with_test_split()
        
        # Load metadata
        with open(os.path.join(self.shard_base_dir, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        # Load training shards
        train_shards = sorted(glob.glob(os.path.join(self.train_shard_dir, "*.pkl")))
        
        # Prepare lists to collect data from all shards
        all_images = []
        all_captions = []
        all_study_ids = []
        
        # Load and process each shard (new format)
        samples_loaded = 0
        for shard_path in train_shards:
            with open(shard_path, 'rb') as f:
                shard_data = pickle.load(f)
                
                # Shard now contains: {'images': np.array, 'captions': np.array, 'study_ids': np.array}
                shard_size = len(shard_data['images'])
                
                if num_samples is not None:
                    # Limit samples if specified
                    remaining_samples = num_samples - samples_loaded
                    if remaining_samples <= 0:
                        break
                    take_samples = min(remaining_samples, shard_size)
                    
                    all_images.append(shard_data['images'][:take_samples])
                    all_captions.append(shard_data['captions'][:take_samples])
                    # Ensure study_ids are converted to list and maintained as strings
                    study_ids_batch = shard_data['study_ids'][:take_samples]
                    all_study_ids.extend([str(sid) for sid in study_ids_batch])
                    samples_loaded += take_samples
                else:
                    # Take all samples
                    all_images.append(shard_data['images'])
                    all_captions.append(shard_data['captions'])
                    # Ensure study_ids are converted to list and maintained as strings
                    study_ids_batch = shard_data['study_ids']
                    all_study_ids.extend([str(sid) for sid in study_ids_batch])
            
            # Free memory after each shard
            gc.collect()
        
        # Concatenate all data
        if all_images:
            images = np.concatenate(all_images, axis=0)
            captions = np.concatenate(all_captions, axis=0)
        else:
            images = np.array([])
            captions = np.array([])
        
        return {
            'images': images,
            'captions': captions,
            # Ensure study_ids are returned as string array with proper dtype
            'study_ids': np.array(all_study_ids, dtype='<U50'),
            'vocab_size': metadata['vocab_size']
        }
    
    def get_validation_data(self, num_samples=None):
        """
        Get validation data using the sharded approach
        
        Args:
            num_samples: Maximum number of samples to include (None for all)
            
        Returns:
            Dictionary with validation data
        """
        # Ensure shards are created
        if not self.shards_created:
            self.create_shards_with_test_split()
        
        # Load metadata
        with open(os.path.join(self.shard_base_dir, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        # Load validation shards
        val_shards = sorted(glob.glob(os.path.join(self.val_shard_dir, "*.pkl")))
        
        # Prepare lists to collect data from all shards
        all_images = []
        all_captions = []
        all_study_ids = []
        
        # Load and process each shard (new format)
        samples_loaded = 0
        for shard_path in val_shards:
            with open(shard_path, 'rb') as f:
                shard_data = pickle.load(f)
                
                # Shard now contains: {'images': np.array, 'captions': np.array, 'study_ids': np.array}
                shard_size = len(shard_data['images'])
                
                if num_samples is not None:
                    # Limit samples if specified
                    remaining_samples = num_samples - samples_loaded
                    if remaining_samples <= 0:
                        break
                    take_samples = min(remaining_samples, shard_size)
                    
                    all_images.append(shard_data['images'][:take_samples])
                    all_captions.append(shard_data['captions'][:take_samples])
                    # Ensure study_ids are converted to list and maintained as strings
                    study_ids_batch = shard_data['study_ids'][:take_samples]
                    all_study_ids.extend([str(sid) for sid in study_ids_batch])
                    samples_loaded += take_samples
                else:
                    # Take all samples
                    all_images.append(shard_data['images'])
                    all_captions.append(shard_data['captions'])
                    # Ensure study_ids are converted to list and maintained as strings
                    study_ids_batch = shard_data['study_ids']
                    all_study_ids.extend([str(sid) for sid in study_ids_batch])
                
            # Free memory after each shard
            gc.collect()
        
        # Concatenate all data
        if all_images:
            images = np.concatenate(all_images, axis=0)
            captions = np.concatenate(all_captions, axis=0)
        else:
            images = np.array([])
            captions = np.array([])
        
        return {
            'images': images,
            'captions': captions,
            # Ensure study_ids are returned as string array with proper dtype
            'study_ids': np.array(all_study_ids, dtype='<U50'),
            'vocab_size': metadata['vocab_size']
        }
    
    def visualize_samples(self, split='val', num_samples=2):
        """
        Visualize samples from the processed dataset
        
        Args:
            split: Which split to visualize ('train', 'val', 'test')
            num_samples: Number of samples to display
        """
        print(f"Visualizing {num_samples} samples from {split} split...")
        
        # Get data from the specified split
        if split == 'train':
            data = self.get_training_data(num_samples=num_samples)
        elif split == 'val':
            data = self.get_validation_data(num_samples=num_samples)
        elif split == 'test':
            data = self.get_test_data(num_samples=num_samples)
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        
        if len(data['images']) == 0:
            print(f"No data found in {split} split.")
            return
            
        # Create visualization
        fig, axes = plt.subplots(num_samples, 1, figsize=(10, 4 * num_samples))
        if num_samples == 1:
            axes = [axes]
        
        for i in range(min(num_samples, len(data['images']))):
            # Display image
            axes[i].imshow(data['images'][i])
            axes[i].set_title(f'Study ID: {data["study_ids"][i]}')
            axes[i].axis('off')
            
            # Add caption information
            caption_tokens = data['captions'][i]
            non_zero_tokens = caption_tokens[caption_tokens > 0]
            caption_info = f"Caption tokens: {len(non_zero_tokens)} non-zero tokens out of {len(caption_tokens)}"
            axes[i].text(0, -20, caption_info, fontsize=8, transform=axes[i].transData)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Successfully visualized {min(num_samples, len(data['images']))} samples from {split} split.")

    def save_study_data(self, output_path='study_data.csv'):
        """
        Save the merged study data to a CSV file
        
        Args:
            output_path: Path where the CSV file will be saved
        """
        if hasattr(self, 'study_data'):
            self.study_data.to_csv(output_path, index=False)
            print(f"Study data saved to {output_path}")
        else:
            print("No study data available to save")

    def get_test_data(self, num_samples=None):
        """
        Get test data using the sharded approach for comprehensive evaluation
        
        Args:
            num_samples: Maximum number of samples to include (None for all)
            
        Returns:
            Dictionary with test data formatted for evaluation
        """
        # Ensure shards are created with test split
        if not hasattr(self, 'test_shard_dir') or not os.path.exists(self.test_shard_dir):
            print("Test shards not found. Creating shards with test split...")
            self.create_shards_with_test_split()
        
        # Load metadata
        with open(os.path.join(self.shard_base_dir, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
        
        # Load test shards (new naming convention)
        test_shards = sorted(glob.glob(os.path.join(self.test_shard_dir, "shard_*.pkl")))
        
        if not test_shards:
            raise ValueError("No test shards found. Please check the sharding process.")
        
        # Prepare lists to collect data from all shards
        all_images = []
        all_captions = []
        all_study_ids = []
        
        # Load and process each shard (new format)
        samples_loaded = 0
        for shard_path in test_shards:
            with open(shard_path, 'rb') as f:
                shard_data = pickle.load(f)
                
                # Shard now contains: {'images': np.array, 'captions': np.array, 'study_ids': np.array}
                shard_size = len(shard_data['images'])
                
                if num_samples is not None:
                    # Limit samples if specified
                    remaining_samples = num_samples - samples_loaded
                    if remaining_samples <= 0:
                        break
                    take_samples = min(remaining_samples, shard_size)
                    
                    all_images.append(shard_data['images'][:take_samples])
                    all_captions.append(shard_data['captions'][:take_samples])
                    # Ensure study_ids are converted to list and maintained as strings
                    study_ids_batch = shard_data['study_ids'][:take_samples]
                    all_study_ids.extend([str(sid) for sid in study_ids_batch])
                    samples_loaded += take_samples
                else:
                    # Take all samples
                    all_images.append(shard_data['images'])
                    all_captions.append(shard_data['captions'])
                    # Ensure study_ids are converted to list and maintained as strings
                    study_ids_batch = shard_data['study_ids']
                    all_study_ids.extend([str(sid) for sid in study_ids_batch])
                
            # Free memory after each shard
            gc.collect()
        
        # Concatenate all data
        if all_images:
            images = np.concatenate(all_images, axis=0)
            captions = np.concatenate(all_captions, axis=0)
        else:
            images = np.array([])
            captions = np.array([])
        
        # Construct the test data dictionary
        test_data = {
            'images': images,
            'captions': captions,
            # Ensure study_ids are returned as string array with proper dtype
            'study_ids': np.array(all_study_ids, dtype='<U50'),
            'vocab_size': metadata['vocab_size']
        }
        
        print(f"Loaded {len(images)} test samples")
        return test_data

def create_train_val_test_split(study_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Create a patient-level split for training, validation, and testing
    
    Args:
        study_ids: Array of study IDs
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        seed: Random seed for reproducibility
        
    Returns:
        train_ids, val_ids, test_ids: Sets of study IDs for each split
    """
    import numpy as np
    
    # Verify ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1"
    
    # Get unique patient/study IDs
    unique_studies = np.unique(study_ids)
    
    # Shuffle the studies
    np.random.seed(seed)
    shuffled_studies = np.random.permutation(unique_studies)
    
    # Calculate split indices
    train_idx = int(len(shuffled_studies) * train_ratio)
    val_idx = int(len(shuffled_studies) * (train_ratio + val_ratio))
    
    # Split the studies
    train_studies = set(shuffled_studies[:train_idx])
    val_studies = set(shuffled_studies[train_idx:val_idx])
    test_studies = set(shuffled_studies[val_idx:])
    
    # Verify no overlap between sets
    assert len(train_studies.intersection(val_studies)) == 0, "Train and validation sets overlap"
    assert len(train_studies.intersection(test_studies)) == 0, "Train and test sets overlap"
    assert len(val_studies.intersection(test_studies)) == 0, "Validation and test sets overlap"
    
    print(f"Split {len(unique_studies)} patients into: {len(train_studies)} train, "
          f"{len(val_studies)} validation, {len(test_studies)} test")
    
    return train_studies, val_studies, test_studies

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test IndianaDatasetLoader")
    parser.add_argument("--reports_csv", required=True)
    parser.add_argument("--projections_csv", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_csv", default="merged_study_data.csv")
    args = parser.parse_args()

    try:
        loader = IndianaDatasetLoader(
            reports_csv_path=args.reports_csv,
            projections_csv_path=args.projections_csv,
            image_dir=args.image_dir,
            image_size=(224, 224),
            batch_size=4,
            max_sequence_length=30,
            shard_size=50,
        )

        loader.save_study_data(args.output_csv)
        loader.create_shards_with_test_split()

        val_data = loader.get_validation_data(num_samples=5)
        print(f"Loaded validation data with {len(val_data['images'])} samples")

        # Visualize samples from validation data
        loader.visualize_samples(split='val', num_samples=2)
        
        print("Sharded data preparation completed successfully!")
        
    except Exception as e:
        print(f"Error creating dataset: {e}")