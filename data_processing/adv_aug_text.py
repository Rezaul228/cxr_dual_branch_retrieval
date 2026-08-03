"""
Advanced text augmentation for medical reports.
Implements more sophisticated text augmentation techniques while preserving clinical meaning.
"""

import random
import re
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

# Try to download NLTK data, with graceful fallback
try:
    nltk.download('punkt', quiet=True)
except:
    print("NLTK download failed, using simpler text processing")

# Enhanced medical terminology dictionary
ADVANCED_MEDICAL_TERMS = {
    "opacity": ["consolidation", "infiltrate", "density", "haziness", "shadowing", "opacification"],
    "cardiomegaly": ["enlarged heart", "cardiac enlargement", "heart enlargement", "enlarged cardiac silhouette", "prominent cardiac silhouette"],
    "pneumonia": ["pulmonary infection", "lung infection", "pneumonitis", "infectious process", "inflammatory process", "parenchymal infection"],
    "effusion": ["fluid collection", "pleural fluid", "fluid accumulation", "pleural effusion", "fluid in pleural space"],
    "atelectasis": ["lung collapse", "collapsed lung", "lung volume loss", "subsegmental atelectasis", "compressive atelectasis"],
    "pneumothorax": ["collapsed lung", "air in pleural space", "pleural air", "air collection", "pleural gas"],
    "normal": ["unremarkable", "no acute abnormality", "within normal limits", "no significant abnormality", "no acute cardiopulmonary process", "no active disease"],
    "abnormal": ["pathologic", "unusual", "remarkable", "notable finding", "abnormality", "pathology"],
    "mild": ["slight", "minimal", "minor", "subtle", "trace"],
    "moderate": ["intermediate", "medium", "moderate-sized", "moderately severe"],
    "severe": ["marked", "pronounced", "significant", "extensive", "profound", "striking", "large"],
    "bilateral": ["affecting both sides", "on both sides", "involving both lungs", "in both lungs", "both left and right"],
    "unilateral": ["affecting one side", "on one side", "involving one lung", "in one lung", "one-sided"],
    "likely": ["probable", "suggestive of", "consistent with", "suspicious for", "compatible with", "concerning for"],
    "possible": ["may represent", "cannot exclude", "cannot rule out", "potentially", "possibly", "could represent", "may be due to"],
    "no": ["absent", "not seen", "not identified", "not detected", "not present", "not noted", "not visualized"],
    "present": ["seen", "identified", "noted", "visualized", "detected", "demonstrated", "observed", "appreciated"],
    "increased": ["elevated", "enhanced", "prominent", "accentuated", "pronounced", "more pronounced"],
    "decreased": ["reduced", "diminished", "less prominent", "subtle", "limited", "faint"]
}

# Add academic vs community style terminologies
STYLE_VARIATIONS = {
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

def restructure_sentence(sentence):
    """
    Restructure a radiological sentence while preserving meaning
    
    Args:
        sentence: Input sentence string
        
    Returns:
        Restructured sentence
    """
    # Common radiological sentence patterns and restructuring options
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
    # Default: return original
    return sentence

def change_terminology_style(text, style="academic"):
    """
    Convert text to different radiological reporting style
    
    Args:
        text: Input text
        style: "academic" or "community"
        
    Returns:
        Styled text
    """
    if style not in ["academic", "community"]:
        return text
    
    styled_text = text
    style_dict = STYLE_VARIATIONS[style]
    
    for original, replacement in style_dict.items():
        # Case insensitive replacement while preserving case
        pattern = re.compile(re.escape(original), re.IGNORECASE)
        styled_text = pattern.sub(lambda m: replacement if m.group(0).islower() else replacement.capitalize(), styled_text)
    
    return styled_text

def permute_findings_order(text):
    """
    Change the order in which findings are reported
    
    Args:
        text: Input radiological report
        
    Returns:
        Text with findings in different order
    """
    try:
        # Try to split into sentences using NLTK
        sentences = sent_tokenize(text)
    except:
        # Fallback to simpler method
        sentences = [s.strip() + '.' for s in text.split('.') if s.strip()]
    
    # Only permute if we have enough sentences
    if len(sentences) <= 2:
        return text
    
    # Keep the first sentence (often summary) and last sentence (often conclusion)
    first_sentence = sentences[0]
    
    # Shuffle the middle findings
    middle_sentences = sentences[1:-1] if len(sentences) > 2 else sentences[1:]
    random.shuffle(middle_sentences)
    
    # Recombine
    if len(sentences) > 2:
        permuted_sentences = [first_sentence] + middle_sentences + [sentences[-1]]
    else:
        permuted_sentences = [first_sentence] + middle_sentences
    
    # Join back together
    return ' '.join(permuted_sentences)

def modify_certainty(text, direction=None):
    """
    Add or remove certainty modifiers to radiological findings
    
    Args:
        text: Input text
        direction: 'increase', 'decrease', or None (random)
        
    Returns:
        Text with modified certainty
    """
    if direction is None:
        direction = random.choice(['increase', 'decrease'])
    
    # Define certainty modifier patterns
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
    
    # Apply modifications
    for pattern, replacement in modifiers:
        modified_text = re.sub(pattern, replacement, modified_text, flags=re.IGNORECASE)
    
    return modified_text

def apply_advanced_text_augmentation(text, tokenizer, config):
    """
    Apply advanced augmentations to a radiological report
    
    Args:
        text: Tokenized text sequence
        tokenizer: Tokenizer for encoding/decoding
        config: AdvAugConfig with augmentation parameters
        
    Returns:
        Augmented tokenized sequence
    """
    # Decode the tokenized text to words
    original_text = decode_caption(text, tokenizer)
    
    # Split into sentences
    try:
        sentences = sent_tokenize(original_text)
    except:
        # Fallback to simpler splitting
        sentences = [s.strip() + '.' for s in original_text.split('.') if s.strip()]
    
    augmented_sentences = []
    
    # Process each sentence
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
    if original_text.endswith('.') and not augmented_text.endswith('.'):
        augmented_text += '.'
    
    # Convert back to token sequence
    augmented_seq = tokenizer.texts_to_sequences([augmented_text])[0]
    
    # Pad to original length
    if len(augmented_seq) < len(text):
        augmented_seq = np.pad(augmented_seq, (0, len(text) - len(augmented_seq)))
    elif len(augmented_seq) > len(text):
        augmented_seq = augmented_seq[:len(text)]
    
    return np.array(augmented_seq)

def decode_caption(caption_seq, tokenizer):
    """
    Decode a caption from its tokenized sequence
    
    Args:
        caption_seq: Tokenized caption sequence
        tokenizer: Tokenizer used to decode the sequence
        
    Returns:
        String representation of the caption
    """
    words = []
    for token_id in caption_seq:
        if token_id == 0:  # Skip padding
            continue
        
        # Convert token_id to string for string-keyed tokenizers
        token_key = str(int(token_id))
        
        # Get the word for this token ID
        word = tokenizer.index_word.get(token_key, '<UNK>')
        if word in ['<START>', '<END>', '<PAD>', '<UNK>']:
            continue
            
        words.append(word)
    
    return " ".join(words)
