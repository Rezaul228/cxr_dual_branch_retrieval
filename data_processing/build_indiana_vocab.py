"""
Enhanced Vocabulary Builder for Indiana Chest X-ray Dataset

This script builds a comprehensive vocabulary from Indiana chest X-ray reports
using advanced text processing techniques similar to MIMIC-CXR but adapted
for the Indiana dataset format.

Features:
- NLTK-based tokenization for better accuracy
- Stopword removal to focus on medical content
- Frequency filtering to remove rare/irrelevant words
- Medical text cleaning and preprocessing
- Configurable vocabulary size and minimum frequency
- Special token handling for sequence modeling
"""

import json
import pandas as pd
import nltk
from collections import Counter
from nltk.corpus import stopwords
from tqdm import tqdm
import re
import argparse
import os
import pickle

# Download required NLTK data
try:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
    print("NLTK data downloaded successfully")
except Exception as e:
    print(f"Warning: NLTK download failed: {e}")
    print("Will use fallback text processing")

def clean_medical_text(text):
    """
    Clean and preprocess medical text for vocabulary building
    
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
    # Keep hyphens (important for medical terms like "chest-wall")
    # Keep periods (important for abbreviations)
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

def build_vocab_from_indiana_data(reports_csv_path, projections_csv_path, 
                                 min_freq=5, max_vocab_size=1500, 
                                 sections=['findings', 'impression']):
    """
    Build vocabulary from Indiana chest X-ray reports
    
    Args:
        reports_csv_path: Path to indiana_reports.csv
        projections_csv_path: Path to indiana_projections.csv
        min_freq: Minimum word frequency to include in vocabulary
        max_vocab_size: Maximum vocabulary size
        sections: Text sections to extract ('findings', 'impression')
        
    Returns:
        Dictionary with vocabulary mapping and statistics
    """
    print(f"Loading Indiana dataset from:")
    print(f"  Reports: {reports_csv_path}")
    print(f"  Projections: {projections_csv_path}")
    
    # Load CSV files
    reports_df = pd.read_csv(reports_csv_path)
    projections_df = pd.read_csv(projections_csv_path)
    
    # Merge reports with projections
    study_data = pd.merge(reports_df, projections_df, on='uid')
    
    print(f"Loaded {len(study_data)} study entries")
    
    # Extract text from specified sections
    all_texts = []
    for section in sections:
        if section in study_data.columns:
            section_texts = study_data[section].dropna().astype(str).tolist()
            all_texts.extend(section_texts)
            print(f"Extracted {len(section_texts)} {section} texts")
        else:
            print(f"Warning: Section '{section}' not found in dataset")
    
    print(f"Total texts to process: {len(all_texts)}")
    
    # Build word frequency counter
    counter = Counter()
    for text in tqdm(all_texts, desc="Processing texts"):
        tokens = clean_medical_text(text)
        counter.update(tokens)
    
    print(f"Found {len(counter)} unique tokens")
    
    # Filter by minimum frequency
    filtered = {word: freq for word, freq in counter.items() if freq >= min_freq}
    print(f"Tokens with frequency >= {min_freq}: {len(filtered)}")
    
    # Get most frequent tokens up to max_vocab_size
    most_common = sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:max_vocab_size]
    
    # Build vocabulary with special tokens
    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "<start>": 2,
        "<end>": 3,
    }
    
    for idx, (word, freq) in enumerate(most_common, start=4):
        vocab[word] = idx
    
    # Create reverse mapping
    index_word = {idx: word for word, idx in vocab.items()}
    
    # Calculate statistics
    total_words = sum(counter.values())
    vocab_words = sum(freq for word, freq in most_common)
    coverage = vocab_words / total_words if total_words > 0 else 0
    
    stats = {
        'total_words': total_words,
        'vocab_words': vocab_words,
        'coverage': coverage,
        'min_freq': min_freq,
        'max_vocab_size': max_vocab_size,
        'sections_processed': sections
    }
    
    return vocab, index_word, stats, counter

def save_vocab(vocab, index_word, stats, output_path):
    """Save vocabulary and statistics to files"""
    
    # Save vocabulary as JSON
    vocab_path = output_path.replace('.json', '_vocab.json')
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f, indent=2)
    
    # Save reverse mapping
    index_word_path = output_path.replace('.json', '_index_word.json')
    with open(index_word_path, 'w') as f:
        json.dump(index_word, f, indent=2)
    
    # Save statistics
    stats_path = output_path.replace('.json', '_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Saved vocabulary to: {vocab_path}")
    print(f"Saved index mapping to: {index_word_path}")
    print(f"Saved statistics to: {stats_path}")
    
    return vocab_path, index_word_path, stats_path

def create_enhanced_tokenizer(vocab, index_word):
    """
    Create an enhanced tokenizer compatible with your existing SimpleTokenizer interface
    but with better text processing
    """
    class EnhancedTokenizer:
        def __init__(self, vocab, index_word, oov_token="<unk>"):
            self.word_index = vocab
            self.index_word = index_word
            self.oov_token = oov_token
            self.oov_index = vocab.get(oov_token, 1)
        
        def fit_on_texts(self, texts):
            """Compatibility method - vocabulary already built"""
            pass
        
        def texts_to_sequences(self, texts):
            """Convert texts to sequences using enhanced processing"""
            sequences = []
            for text in texts:
                tokens = clean_medical_text(text)
                sequence = [self.word_index.get(token, self.oov_index) for token in tokens]
                sequences.append(sequence)
            return sequences
    
    return EnhancedTokenizer(vocab, index_word)

def main():
    parser = argparse.ArgumentParser(description="Build vocabulary for Indiana Chest X-ray dataset")
    parser.add_argument("--reports_csv", type=str, required=True, 
                       help="Path to indiana_reports.csv")
    parser.add_argument("--projections_csv", type=str, required=True,
                       help="Path to indiana_projections.csv")
    parser.add_argument("--output_vocab", type=str, default="indiana_vocab.json",
                       help="Output vocabulary file path")
    parser.add_argument("--min_freq", type=int, default=5,
                       help="Minimum word frequency to include")
    parser.add_argument("--vocab_size", type=int, default=1500,
                       help="Maximum vocabulary size")
    parser.add_argument("--sections", nargs='+', default=['findings', 'impression'],
                       help="Text sections to extract")
    parser.add_argument("--save_tokenizer", action='store_true',
                       help="Save enhanced tokenizer as pickle file")
    
    args = parser.parse_args()
    
    # Build vocabulary
    vocab, index_word, stats, counter = build_vocab_from_indiana_data(
        args.reports_csv, args.projections_csv,
        min_freq=args.min_freq, max_vocab_size=args.vocab_size,
        sections=args.sections
    )
    
    # Save vocabulary files
    vocab_path, index_word_path, stats_path = save_vocab(vocab, index_word, stats, args.output_vocab)
    
    # Print statistics
    print(f"\n=== Vocabulary Statistics ===")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Total words in corpus: {stats['total_words']:,}")
    print(f"Words in vocabulary: {stats['vocab_words']:,}")
    print(f"Coverage: {stats['coverage']:.2%}")
    print(f"Minimum frequency: {stats['min_freq']}")
    print(f"Maximum vocabulary size: {stats['max_vocab_size']}")
    print(f"Sections processed: {stats['sections_processed']}")
    
    # Show some example tokens
    print(f"\n=== Example Vocabulary Entries ===")
    example_words = list(vocab.keys())[:20]  # First 20 tokens
    for word in example_words:
        if word not in ['<pad>', '<unk>', '<start>', '<end>']:
            print(f"  '{word}': {vocab[word]}")
    
    # Create and save enhanced tokenizer if requested
    if args.save_tokenizer:
        enhanced_tokenizer = create_enhanced_tokenizer(vocab, index_word)
        tokenizer_path = args.output_vocab.replace('.json', '_tokenizer.pkl')
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(enhanced_tokenizer, f)
        print(f"\nSaved enhanced tokenizer to: {tokenizer_path}")
    
    print(f"\n✅ Vocabulary building complete!")
    print(f"Use the vocabulary files to enhance your data processing pipeline.")

if __name__ == "__main__":
    main() 