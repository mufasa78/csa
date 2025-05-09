# ChineseSentimentGuard (CSG)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Overview

ChineseSentimentGuard is a comprehensive sentiment analysis system designed for Chinese and English text. It provides accurate sentiment classification using advanced deep learning models while ensuring data privacy and security.

![ChineseSentimentGuard](project/generated-icon.png)

## ✨ Features

- **Dual-Model Analysis**: Choose between CNN and LSTM models for sentiment analysis
- **Bilingual Support**: Analyze both Chinese and English text
- **Privacy Protection**: Built-in data encryption and anonymization
- **User Management**: Role-based access control (admin/user)
- **Visualization**: Comprehensive data visualization and analysis history
- **Dual Interface**: Both web-based Flask interface and Streamlit dashboard
- **API Access**: RESTful API for integration with other systems

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip or conda for package management

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ChineseSentimentGuard.git
   cd ChineseSentimentGuard
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🖥️ Usage

### Flask Web Interface

Start the Flask web application:

```bash
python project/flask_app.py
```

Access the web interface at http://localhost:5000

### Streamlit Dashboard

Start the Streamlit dashboard:

```bash
streamlit run project/app.py
```

Access the dashboard at http://localhost:8501

### Default Login Credentials

- **Admin User**:
  - Username: admin
  - Password: hello
- **Regular User**:
  - Username: user
  - Password: hello

## 📊 Features in Detail

### Sentiment Analysis

The system uses two deep learning models for sentiment analysis:

1. **CNN Model**: Fast and efficient for shorter texts
2. **LSTM Model**: Better for capturing long-range dependencies in text

Sentiment is classified into three categories:
- 消极 (Negative)
- 中性 (Neutral)
- 积极 (Positive)

### Data Privacy

The system implements several privacy-preserving features:

- **Data Encryption**: Sensitive text is encrypted using AES-256
- **Text Anonymization**: Personal identifiers can be automatically removed
- **Secure Logging**: Privacy-aware logging system

### Admin Features

Administrators have access to additional features:

- **User Management**: View and manage user accounts
- **System Statistics**: Monitor system usage and performance
- **Model Management**: Train, evaluate, and manage sentiment models
- **Data Visualization**: Advanced analytics and visualization tools

## 🔧 API Reference

The system provides a RESTful API for sentiment analysis:

### Analyze Sentiment

```
POST /api/analyze
```

**Request Body:**
```json
{
  "text": "这个产品非常好用，我很喜欢！",
  "model_type": "CNN"
}
```

**Response:**
```json
{
  "text": "这个产品非常好用，我很喜欢！",
  "preprocessed_text": "产品 好用 喜欢",
  "sentiment": "积极",
  "confidence": 0.92,
  "model": "CNN",
  "prediction": [0.05, 0.03, 0.92]
}
```

## 📁 Project Structure

```
project/
├── .config/               # Configuration files
├── .streamlit/           # Streamlit configuration
├── pages/                # Streamlit pages
│   ├── admin.py          # Admin dashboard
│   └── user.py           # User dashboard
├── templates/            # Flask HTML templates
│   ├── admin.html        # Admin interface
│   ├── analyze.html      # Analysis page
│   ├── base.html         # Base template
│   ├── dashboard.html    # Main dashboard
│   ├── history.html      # Analysis history
│   └── login.html        # Login page
├── app.py                # Streamlit application
├── flask_app.py          # Flask web application
├── models.py             # Sentiment analysis models
├── preprocessing.py      # Text preprocessing utilities
├── data_security.py      # Data encryption and privacy
├── utils.py              # Utility functions
├── visualization.py      # Data visualization
└── requirements.txt      # Project dependencies
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

Your Name - your.email@example.com

Project Link: [https://github.com/yourusername/ChineseSentimentGuard](https://github.com/yourusername/ChineseSentimentGuard)
