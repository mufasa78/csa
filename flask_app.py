from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import os
import sys
import logging
import numpy as np
import pandas as pd
from models import create_cnn_model, create_lstm_model
from preprocessing import preprocess_text
from data_security import encrypt_data, decrypt_data, anonymize_text
from utils import save_analysis_history, load_analysis_history
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default_secret_key_for_flask_session")

# Mock user database with simple passwords for demo
# In a real application, this would be stored in a secure database with hashed passwords
USERS = {
    'admin': {
        'password': 'hello',  # Simple plain text for demo
        'role': 'admin'
    },
    'user': {
        'password': 'hello',  # Simple plain text for demo
        'role': 'user'
    }
}

# Global variables to store models and analysis history
models = {}
analysis_history = load_analysis_history()

# Load models
def load_models():
    """Load sentiment analysis models"""
    global models
    logging.info('Loading CNN model...')
    models['CNN'] = create_cnn_model()
    logging.info('CNN model loaded successfully')

    logging.info('Loading LSTM model...')
    models['LSTM'] = create_lstm_model()
    logging.info('LSTM model loaded successfully')

    logging.info('All models loaded successfully')
    return models

# Initialize models
logging.info('Initializing models...')
load_models()

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in USERS and USERS[username]['password'] == password:
            session['username'] = username
            session['role'] = USERS[username]['role']
            return redirect(url_for('dashboard'))
        else:
            error = 'Invalid username or password'

    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('role', None)
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    if 'username' not in session:
        return redirect(url_for('login'))

    return render_template('dashboard.html',
                          username=session['username'],
                          role=session['role'])

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    if 'username' not in session:
        return redirect(url_for('login'))

    result = None
    if request.method == 'POST':
        text = request.form['text']
        model_type = request.form['model_type']

        # Preprocess text
        preprocessed_text = preprocess_text(text)

        # Encrypt data for privacy
        encrypted_text = encrypt_data(text)

        # Get prediction from model
        model = models[model_type]
        prediction = model.predict(text)

        # Get sentiment label and confidence
        sentiment_labels = ['消极', '中性', '积极']
        sentiment_idx = np.argmax(prediction[0])
        sentiment = sentiment_labels[sentiment_idx]
        confidence = float(prediction[0][sentiment_idx])

        # Store in history
        analysis_result = {
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence,
            'model': model_type,
            'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        analysis_history.append(analysis_result)
        save_analysis_history(analysis_history)

        # Prepare result for display
        result = {
            'text': text,
            'preprocessed_text': preprocessed_text,
            'sentiment': sentiment,
            'confidence': confidence,
            'model': model_type,
            'prediction': prediction[0].tolist()
        }

    return render_template('analyze.html',
                          result=result,
                          username=session['username'],
                          role=session['role'])

@app.route('/history')
def history():
    if 'username' not in session:
        return redirect(url_for('login'))

    return render_template('history.html',
                          history=analysis_history,
                          username=session['username'],
                          role=session['role'])

@app.route('/admin')
def admin():
    if 'username' not in session or session['role'] != 'admin':
        return redirect(url_for('dashboard'))

    users = [{"username": username, "role": data['role']} for username, data in USERS.items()]

    # Calculate some statistics
    stats = {
        'total_analyses': len(analysis_history),
        'cnn_analyses': len([a for a in analysis_history if a['model'] == 'CNN']),
        'lstm_analyses': len([a for a in analysis_history if a['model'] == 'LSTM']),
        'positive_sentiments': len([a for a in analysis_history if a['sentiment'] == '积极']),
        'neutral_sentiments': len([a for a in analysis_history if a['sentiment'] == '中性']),
        'negative_sentiments': len([a for a in analysis_history if a['sentiment'] == '消极'])
    }

    return render_template('admin.html',
                          users=users,
                          stats=stats,
                          username=session['username'],
                          role=session['role'])

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    if 'text' not in data or 'model_type' not in data:
        return jsonify({"error": "Missing text or model_type"}), 400

    text = data['text']
    model_type = data['model_type']

    if model_type not in models:
        return jsonify({"error": f"Unknown model type: {model_type}"}), 400

    # Preprocess text
    preprocessed_text = preprocess_text(text)

    # Get prediction from model
    model = models[model_type]
    prediction = model.predict(text)

    # Get sentiment label and confidence
    sentiment_labels = ['消极', '中性', '积极']
    sentiment_idx = np.argmax(prediction[0])
    sentiment = sentiment_labels[sentiment_idx]
    confidence = float(prediction[0][sentiment_idx])

    # Store in history
    analysis_result = {
        'text': text,
        'sentiment': sentiment,
        'confidence': confidence,
        'model': model_type,
        'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    analysis_history.append(analysis_result)
    save_analysis_history(analysis_history)

    # Return result
    return jsonify({
        'text': text,
        'preprocessed_text': preprocessed_text,
        'sentiment': sentiment,
        'confidence': confidence,
        'model': model_type,
        'prediction': prediction[0].tolist()
    })

if __name__ == '__main__':
    # Add logging for startup
    logging.info('Starting Flask application on port 5000')
    logging.info('Models loaded: %s', list(models.keys()))

    # Run with threaded=True for better handling of multiple requests
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000,
        threaded=True,
        use_reloader=False  # Disable reloader to prevent duplicate processes
    )

    logging.info('Flask application shutdown')
