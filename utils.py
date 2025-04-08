import os
import time
import json
import pandas as pd
import numpy as np
import streamlit as st
from data_security import encrypt_data, decrypt_data, anonymize_text, secure_logging
import pickle

def save_model(model, model_name):
    """
    Save a trained model to disk
    
    Args:
        model: Trained model
        model_name: Name for the saved model
    """
    model_path = f"{model_name.lower()}_sentiment_model.pkl"
    
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        st.success(f"Model saved as {model_path}")
    except Exception as e:
        st.error(f"Error saving model: {str(e)}")

def load_model(model_name):
    """
    Load a trained model from disk
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        Loaded model or None if not found
    """
    from models import create_cnn_model, create_lstm_model
    
    model_path = f"{model_name.lower()}_sentiment_model.pkl"
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except:
        st.info(f"Creating new {model_name} model")
        if model_name.upper() == "CNN":
            return create_cnn_model()
        elif model_name.upper() == "LSTM":
            return create_lstm_model()
        else:
            st.error(f"Unknown model type: {model_name}")
            return None

def save_tokenizer(tokenizer, path="tokenizer.pickle"):
    """
    Save tokenizer to disk
    
    Args:
        tokenizer: Tokenizer object
        path: Path to save the tokenizer
    """
    with open(path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_tokenizer(path="tokenizer.pickle"):
    """
    Load tokenizer from disk
    
    Args:
        path: Path to the saved tokenizer
        
    Returns:
        Loaded tokenizer or None if not found
    """
    try:
        with open(path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        return tokenizer
    except:
        st.error(f"Could not load tokenizer from {path}")
        return None

def save_analysis_history(history, path="analysis_history.json"):
    """
    Save analysis history to disk
    
    Args:
        history: List of analysis results
        path: Path to save the history
    """
    # Convert history to JSON-serializable format
    serializable_history = []
    
    for item in history:
        serializable_item = {
            'text': encrypt_data(item['text']),  # Encrypt sensitive text
            'sentiment': item['sentiment'],
            'confidence': float(item['confidence']),
            'model': item['model'],
            'timestamp': item['timestamp']
        }
        serializable_history.append(serializable_item)
    
    # Save to file
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(serializable_history, f, ensure_ascii=False, indent=2)

def load_analysis_history(path="analysis_history.json"):
    """
    Load analysis history from disk
    
    Args:
        path: Path to the saved history
        
    Returns:
        Loaded history or empty list if not found
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            serializable_history = json.load(f)
        
        # Convert back to original format
        history = []
        
        for item in serializable_history:
            original_item = {
                'text': decrypt_data(item['text']),  # Decrypt text
                'sentiment': item['sentiment'],
                'confidence': item['confidence'],
                'model': item['model'],
                'timestamp': item['timestamp']
            }
            history.append(original_item)
        
        return history
    except:
        # Return empty list if file not found or error
        return []

def export_results(df, format="csv"):
    """
    Export analysis results to a file
    
    Args:
        df: DataFrame with results
        format: Export format ('csv', 'json', or 'excel')
        
    Returns:
        File data for download
    """
    # Anonymize text before export
    df_export = df.copy()
    if 'text' in df_export.columns:
        df_export['text'] = df_export['text'].apply(anonymize_text)
    
    # Export based on format
    if format == "csv":
        return df_export.to_csv(index=False).encode('utf-8')
    elif format == "json":
        return df_export.to_json(orient="records", force_ascii=False).encode('utf-8')
    elif format == "excel":
        # Use BytesIO for binary formats
        from io import BytesIO
        buffer = BytesIO()
        df_export.to_excel(buffer, index=False)
        buffer.seek(0)
        return buffer.read()

def get_sentiment_color(sentiment):
    """
    Get color for a sentiment
    
    Args:
        sentiment: Sentiment string
        
    Returns:
        Color hex code
    """
    colors = {
        '积极': '#4CAF50',  # Green
        '中性': '#2196F3',  # Blue
        '消极': '#F44336'   # Red
    }
    
    return colors.get(sentiment, '#9E9E9E')  # Default gray

def format_time(seconds):
    """
    Format seconds into human-readable time
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{hours}小时{minutes}分钟{seconds}秒"
    elif minutes > 0:
        return f"{minutes}分钟{seconds}秒"
    else:
        return f"{seconds}秒"

def get_system_stats():
    """
    Get system statistics
    
    Returns:
        Dictionary with system stats
    """
    import psutil
    
    stats = {
        'cpu_percent': psutil.cpu_percent(),
        'memory_percent': psutil.virtual_memory().percent,
        'start_time': time.time() - psutil.Process(os.getpid()).create_time(),
    }
    
    return stats
