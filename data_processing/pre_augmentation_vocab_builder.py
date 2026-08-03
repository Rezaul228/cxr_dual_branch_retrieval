#!/usr/bin/env python3
"""
Pre-Augmentation Vocabulary Builder

This script implements the research community's best practice for data augmentation:
1. Generate all possible augmented texts first (without tokenizing)
2. Build vocabulary from the complete augmented corpus
3. Then process and tokenize everything

This ensures that all augmented words are included in the vocabulary.
"""

import os
import sys
import argparse
import json
import pickle
import numpy as np
import pandas as pd
from collections import Counter
import random
from tqdm import tqdm

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from adv_aug_text import ADVANCED_MEDICAL_TERMS, apply_advanced_text_augmentation
from adv_aug_config import AdvAugConfig

def generate_augmented_text_corpus(original_texts, config, num_augmentations=3):
    """
    Generate augmented text corpus without tokenization
    
    Args:
        original_texts: List of original text strings
        config: Augmentation configuration
        num_augmentations: Number of augmentations per text
        
    Returns:
        List of all texts (original + augmented)
    """
    print(f"Generating augmented text corpus with {num_augmentations} augmentations per text...")
    
    all_texts = []
    
    # Add original texts
    all_texts.extend(original_texts)
    
    # Generate augmented texts
    for text in tqdm(original_texts, desc="Generating augmented texts"):
        for aug_idx in range(num_augmentations):
            # Apply text augmentation techniques directly to raw text
            augmented_text = apply_text_augmentation_to_raw_text(text, config)
            all_texts.append(augmented_text)
    
    print(f"Generated {len(all_texts)} total texts ({len(original_texts)} original + {len(all_texts) - len(original_texts)} augmented)")
    return all_texts

def apply_text_augmentation_to_raw_text(text, config):
    """
    Apply text augmentation directly to raw text (not tokenized)
    
    Args:
        text: Raw text string
        config: Augmentation configuration
        
    Returns:
        Augmented text string
    """
    # Split into sentences
    sentences = text.split('.')
    augmented_sentences = []
    
    for sentence in sentences:
        if not sentence.strip():
            continue
        
        # Sentence structure variation
        if random.random() < config.sentence_restructure_prob:
            sentence = restructure_sentence(sentence)
        
        # Style variation
        if random.random() < config.terminology_style_prob:
            style = random.choice(["academic", "community"])
            sentence = change_terminology_style(sentence, style)
        
        # Certainty modification
        if random.random() < config.certainty_modifier_prob:
            direction = random.choice(['increase', 'decrease'])
            sentence = modify_certainty(sentence, direction)
        
        # Word-level synonym replacement
        words = sentence.split()
        for i, word in enumerate(words):
            # Clean the word of punctuation for matching
            clean_word = word.lower().strip(".,;:()")
            
            if clean_word in ADVANCED_MEDICAL_TERMS and random.random() < config.synonym_replacement_prob:
                synonyms = ADVANCED_MEDICAL_TERMS[clean_word]
                if synonyms:
                    # Choose a random synonym
                    new_word = random.choice(synonyms)
                    # Preserve capitalization and punctuation
                    if word[0].isupper():
                        new_word = new_word.capitalize()
                    # Preserve trailing punctuation
                    if not word[-1].isalnum():
                        new_word += word[-1]
                    words[i] = new_word
        
        augmented_sentences.append(" ".join(words))
    
    # Permute findings order if enabled
    if random.random() < config.finding_order_prob:
        augmented_text = permute_findings_order(' '.join(augmented_sentences))
    else:
        augmented_text = ' '.join(augmented_sentences)
    
    # Add period at end if needed
    if text.endswith('.') and not augmented_text.endswith('.'):
        augmented_text += '.'
    
    return augmented_text

def restructure_sentence(sentence):
    """Restructure a radiological sentence while preserving meaning"""
    if sentence.startswith("There is "):
        return sentence[9:].strip().capitalize() + " is present."
    elif sentence.startswith("There are "):
        return sentence[10:].strip().capitalize() + " are present."
    elif sentence.startswith("No "):
        return sentence[3:].strip().capitalize() + " is not identified."
    elif " is seen" in sentence:
        return sentence.replace(" is seen", " is noted")
    elif " are seen" in sentence:
        return sentence.replace(" are seen", " are noted")
    elif "demonstrates" in sentence:
        return sentence.replace("demonstrates", "shows")
    elif "reveals" in sentence:
        return sentence.replace("reveals", "shows")
    elif "shows" in sentence:
        return sentence.replace("shows", "demonstrates")
    elif "compatible with" in sentence:
        return sentence.replace("compatible with", "consistent with")
    return sentence

def change_terminology_style(text, style="academic"):
    """Convert text to different radiological reporting style"""
    import re
    
    style_variations = {
        "academic": {
            "opacity": "parenchymal opacification",
            "pneumonia": "infectious pneumonitis",
            "findings": "imaging features",
            "normal": "within normal radiographic limits",
            "lungs": "lung parenchyma",
            "effusion": "pleural fluid collection",
            "clear": "aerated"
        },
        "community": {
            "parenchymal opacification": "opacity",
            "infectious pneumonitis": "pneumonia",
            "imaging features": "findings",
            "within normal radiographic limits": "normal",
            "lung parenchyma": "lungs",
            "pleural fluid collection": "effusion",
            "aerated": "clear"
        }
    }
    
    if style not in ["academic", "community"]:
        return text
    
    styled_text = text
    style_dict = style_variations[style]
    
    for original, replacement in style_dict.items():
        pattern = re.compile(re.escape(original), re.IGNORECASE)
        styled_text = pattern.sub(lambda m: replacement if m.group(0).islower() else replacement.capitalize(), styled_text)
    
    return styled_text

def permute_findings_order(text):
    """Change the order in which findings are reported"""
    sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
    
    if len(sentences) <= 2:
        return text
    
    first_sentence = sentences[0]
    middle_sentences = sentences[1:-1] if len(sentences) > 2 else sentences[1:]
    random.shuffle(middle_sentences)
    
    if len(sentences) > 2:
        permuted_sentences = [first_sentence] + middle_sentences + [sentences[-1]]
    else:
        permuted_sentences = [first_sentence] + middle_sentences
    
    return ' '.join(permuted_sentences)

def modify_certainty(text, direction=None):
    """Add or remove certainty modifiers to radiological findings"""
    import re
    
    if direction is None:
        direction = random.choice(['increase', 'decrease'])
    
    increase_certainty = [
        (r'may represent', 'represents'),
        (r'possibly', 'definitely'),
        (r'could be', 'is'),
        (r'suggestive of', 'consistent with'),
        (r'cannot exclude', 'demonstrates'),
        (r'cannot rule out', 'demonstrates')
    ]
    
    decrease_certainty = [
        (r'represents', 'may represent'),
        (r'definitely', 'possibly'),
        (r'is consistent with', 'is suggestive of'),
        (r'demonstrates', 'is suggestive of'),
        (r'shows', 'may show')
    ]
    
    modifiers = increase_certainty if direction == 'increase' else decrease_certainty
    modified_text = text
    
    for pattern, replacement in modifiers:
        modified_text = re.sub(pattern, replacement, modified_text, flags=re.IGNORECASE)
    
    return modified_text

def build_vocabulary_from_augmented_corpus(texts, min_freq=1, max_vocab_size=None):
    """
    Build vocabulary from the complete augmented corpus
    
    Args:
        texts: List of all texts (original + augmented)
        min_freq: Minimum word frequency
        max_vocab_size: Maximum vocabulary size
        
    Returns:
        vocab, index_word, stats
    """
    print("Building vocabulary from augmented corpus...")
    
    # Count word frequencies
    word_counts = Counter()
    for text in tqdm(texts, desc="Counting words"):
        words = text.lower().split()
        word_counts.update(words)
    
    print(f"Total unique words: {len(word_counts)}")
    
    # Filter by minimum frequency
    filtered_words = {word: count for word, count in word_counts.items() 
                     if count >= min_freq}
    
    print(f"Words with frequency >= {min_freq}: {len(filtered_words)}")
    
    # Sort by frequency
    sorted_words = sorted(filtered_words.items(), key=lambda x: x[1], reverse=True)
    
    # Create vocabulary
    vocab = {}
    
    # Add special tokens first
    special_tokens = ['<pad>', '<unk>', '<start>', '<end>']
    for i, token in enumerate(special_tokens):
        vocab[token] = i
    
    # Add words up to max_vocab_size
    max_words = max_vocab_size - len(special_tokens) if max_vocab_size else len(sorted_words)
    for i, (word, count) in enumerate(sorted_words[:max_words]):
        vocab[word] = i + len(special_tokens)
    
    # Create reverse mapping
    index_word = {v: k for k, v in vocab.items()}
    
    # Calculate statistics
    total_words = sum(word_counts.values())
    vocab_words = sum(word_counts[word] for word in vocab if word not in special_tokens)
    coverage = vocab_words / total_words if total_words > 0 else 0
    
    stats = {
        'total_words': total_words,
        'vocab_words': vocab_words,
        'coverage': coverage,
        'vocab_size': len(vocab),
        'min_freq': min_freq,
        'max_vocab_size': max_vocab_size
    }
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Coverage: {coverage:.2%}")
    
    return vocab, index_word, stats

def main():
    parser = argparse.ArgumentParser(description="Build vocabulary from augmented corpus")
    parser.add_argument("--reports_csv", type=str, required=True,
                       help="Path to Indiana reports CSV")
    parser.add_argument("--projections_csv", type=str, required=True,
                       help="Path to Indiana projections CSV")
    parser.add_argument("--output_vocab", type=str, default="augmented_vocab.json",
                       help="Output vocabulary file path")
    parser.add_argument("--num_augmentations", type=int, default=3,
                       help="Number of augmentations per text")
    parser.add_argument("--min_freq", type=int, default=1,
                       help="Minimum word frequency")
    parser.add_argument("--max_vocab_size", type=int, default=3000,
                       help="Maximum vocabulary size")
    parser.add_argument("--save_tokenizer", action='store_true',
                       help="Save enhanced tokenizer")
    
    args = parser.parse_args()
    
    # Load original texts
    print("Loading original texts...")
    df = pd.read_csv(args.reports_csv)
    original_texts = []
    
    for _, row in df.iterrows():
        findings = row.get('findings', '')
        impression = row.get('impression', '')
        combined_text = f"{findings} {impression}".strip()
        if combined_text:
            original_texts.append(combined_text)
    
    print(f"Loaded {len(original_texts)} original texts")
    
    # Create augmentation config
    config = AdvAugConfig()
    config.sentence_restructure_prob = 0.3
    config.terminology_style_prob = 0.3
    config.certainty_modifier_prob = 0.3
    config.synonym_replacement_prob = 0.4
    config.finding_order_prob = 0.2
    
    # Generate augmented corpus
    all_texts = generate_augmented_text_corpus(original_texts, config, args.num_augmentations)
    
    # Build vocabulary from complete corpus
    vocab, index_word, stats = build_vocabulary_from_augmented_corpus(
        all_texts, args.min_freq, args.max_vocab_size
    )
    
    # Save vocabulary files
    vocab_filename = f"{args.output_vocab}_vocab.json"
    index_word_filename = f"{args.output_vocab}_index_word.json"
    stats_filename = f"{args.output_vocab}_stats.json"
    
    with open(vocab_filename, 'w') as f:
        json.dump(vocab, f, indent=2)
    
    with open(index_word_filename, 'w') as f:
        json.dump(index_word, f, indent=2)
    
    with open(stats_filename, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nVocabulary files saved:")
    print(f"  - {vocab_filename}")
    print(f"  - {index_word_filename}")
    print(f"  - {stats_filename}")
    
    # Create and save enhanced tokenizer if requested
    if args.save_tokenizer:
        from enhanced_data_loader import EnhancedTokenizer
        
        tokenizer = EnhancedTokenizer(vocab, index_word)
        tokenizer_path = f"{args.output_vocab}_tokenizer.pkl"
        
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f)
        
        print(f"  - {tokenizer_path}")
    
    # Show some example words from the vocabulary
    print(f"\nSample words from augmented vocabulary:")
    example_words = [word for word in list(vocab.keys())[:30] 
                    if word not in ['<pad>', '<unk>', '<start>', '<end>']]
    for word in example_words:
        print(f"  '{word}': {vocab[word]}")
    
    print(f"\n✅ Pre-augmentation vocabulary building complete!")
    print(f"This vocabulary includes all original and augmented words.")
    print(f"Use this vocabulary for your data processing pipeline.")

if __name__ == "__main__":
    main() 