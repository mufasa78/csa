import jieba
import re
import numpy as np
import os
import pickle

# Set up environment for jieba
jieba.setLogLevel(20)  # Suppress jieba's debug info

def clean_text(text):
    """
    Clean the input text by removing special characters and normalizing
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text string
    """
    # Convert to lowercase if needed (for mixed Chinese-English text)
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^\u4e00-\u9fff\s]', '', text)  # Keep only Chinese characters and spaces
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def segment_text(text):
    """
    Segment Chinese text into words using jieba
    
    Args:
        text: Input text string
        
    Returns:
        List of segmented words
    """
    segmented = jieba.cut(text)
    return list(segmented)

def preprocess_text(text, max_length=100):
    """
    Preprocess text for sentiment analysis
    
    Args:
        text: Input text string
        max_length: Maximum sequence length
        
    Returns:
        Preprocessed text sequence
    """
    # Clean text
    cleaned_text = clean_text(text)
    
    # Segment text
    segmented_text = segment_text(cleaned_text)
    
    # Join back to string for tokenization
    processed_text = ' '.join(segmented_text)
    
    # In a real application, we would tokenize and pad the sequence here
    # For this demo, we'll return the processed text as is
    return processed_text

class SimpleTokenizer:
    """
    A simple tokenizer for Chinese text
    
    This is a basic implementation to replace TensorFlow's Tokenizer
    """
    def __init__(self, num_words=10000, oov_token='<OOV>'):
        self.num_words = num_words
        self.oov_token = oov_token
        self.word_index = {oov_token: 1}  # 0 is reserved for padding
        self.index_word = {1: oov_token, 0: '<PAD>'}
        self.word_counts = {}
        self.document_count = 0
    
    def fit_on_texts(self, texts):
        """
        Build vocabulary from texts
        
        Args:
            texts: List of text strings
        """
        for text in texts:
            self.document_count += 1
            words = text.split()
            for word in words:
                if word in self.word_counts:
                    self.word_counts[word] += 1
                else:
                    self.word_counts[word] = 1
        
        # Sort words by frequency and keep only top num_words
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        sorted_words = sorted_words[:self.num_words - 2]  # -2 for OOV and PAD
        
        # Build word_index
        for i, (word, _) in enumerate(sorted_words):
            idx = i + 2  # +2 because 0=PAD, 1=OOV
            self.word_index[word] = idx
            self.index_word[idx] = word
    
    def texts_to_sequences(self, texts):
        """
        Convert texts to sequences of integers
        
        Args:
            texts: List of text strings
            
        Returns:
            List of lists of integers
        """
        sequences = []
        for text in texts:
            sequence = []
            words = text.split()
            for word in words:
                if word in self.word_index:
                    sequence.append(self.word_index[word])
                else:
                    sequence.append(self.word_index[self.oov_token])
            sequences.append(sequence)
        return sequences

def load_tokenizer(tokenizer_path=None):
    """
    Load tokenizer or create a new one
    
    Args:
        tokenizer_path: Path to saved tokenizer
        
    Returns:
        Tokenizer object
    """
    if tokenizer_path and os.path.exists(tokenizer_path):
        # Load tokenizer from file
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
    else:
        # Create a new tokenizer
        tokenizer = SimpleTokenizer(num_words=10000, oov_token='<OOV>')
    
    return tokenizer

def tokenize_text(tokenizer, texts, max_length=100):
    """
    Tokenize and pad text sequences
    
    Args:
        tokenizer: Tokenizer object
        texts: List of text strings
        max_length: Maximum sequence length
        
    Returns:
        Padded sequences
    """
    # Convert texts to sequences
    sequences = tokenizer.texts_to_sequences(texts)
    
    # Pad sequences manually
    padded_sequences = []
    for sequence in sequences:
        if len(sequence) > max_length:
            # Truncate
            padded = sequence[:max_length]
        else:
            # Pad with zeros
            padded = sequence + [0] * (max_length - len(sequence))
        padded_sequences.append(padded)
    
    return np.array(padded_sequences)

def build_vocab_from_texts(texts, num_words=10000):
    """
    Build vocabulary from texts
    
    Args:
        texts: List of text strings
        num_words: Maximum number of words to keep
        
    Returns:
        Tokenizer object with fitted vocabulary
    """
    tokenizer = SimpleTokenizer(num_words=num_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    
    return tokenizer

def prepare_dataset(df, text_column, label_column, test_size=0.2, random_state=42):
    """
    Prepare dataset for training (simplified version without scikit-learn)
    
    Args:
        df: DataFrame with text and labels
        text_column: Name of the text column
        label_column: Name of the label column
        test_size: Proportion of test set
        random_state: Random seed
        
    Returns:
        Train and test data
    """
    # Preprocess texts
    texts = []
    for text in df[text_column]:
        texts.append(preprocess_text(text))
    
    # Convert labels to indices
    label_map = {'消极': 0, '中性': 1, '积极': 2}
    labels = []
    for label in df[label_column]:
        if label in label_map:
            labels.append(label_map[label])
        else:
            # Default to neutral for unknown labels
            labels.append(1)
    
    # Simple manual split (no stratification)
    np.random.seed(random_state)
    indices = np.random.permutation(len(texts))
    test_size_int = int(len(texts) * test_size)
    test_indices = indices[:test_size_int]
    train_indices = indices[test_size_int:]
    
    # Create train and test sets
    X_train = [texts[i] for i in train_indices]
    X_test = [texts[i] for i in test_indices]
    y_train = [labels[i] for i in train_indices]
    y_test = [labels[i] for i in test_indices]
    
    # One-hot encode labels
    y_train_onehot = []
    y_test_onehot = []
    
    for label in y_train:
        one_hot = [0] * 3  # For 3 classes (negative, neutral, positive)
        one_hot[label] = 1
        y_train_onehot.append(one_hot)
    
    for label in y_test:
        one_hot = [0] * 3
        one_hot[label] = 1
        y_test_onehot.append(one_hot)
    
    # Build vocabulary from training data
    tokenizer = build_vocab_from_texts(X_train)
    
    # Tokenize and pad sequences
    X_train_seq = tokenize_text(tokenizer, X_train)
    X_test_seq = tokenize_text(tokenizer, X_test)
    
    return X_train_seq, X_test_seq, np.array(y_train_onehot), np.array(y_test_onehot), tokenizer
