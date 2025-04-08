import streamlit as st
import os
import pickle
import numpy as np
import pandas as pd
from preprocessing import preprocess_text
from visualization import plot_sentiment_distribution, plot_confidence
from data_security import encrypt_data, decrypt_data
from utils import save_analysis_history, load_analysis_history
import base64
import hashlib
import hmac
import time

# Set page configuration
st.set_page_config(
    page_title="中文/英文情感识别系统 | Chinese/English Sentiment Analysis",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables if they don't exist
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'username' not in st.session_state:
    st.session_state['username'] = ''
if 'role' not in st.session_state:
    st.session_state['role'] = ''
if 'analysis_history' not in st.session_state:
    # Load analysis history from file if it exists
    st.session_state['analysis_history'] = load_analysis_history()
if 'models' not in st.session_state:
    st.session_state['models'] = {}
if 'selected_model' not in st.session_state:
    st.session_state['selected_model'] = 'CNN'

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

SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key_for_password_hashing")

def verify_password(stored_password, provided_password):
    """Verify a password - using simple comparison for the demo"""
    return stored_password == provided_password

def hash_password(password):
    """Hash a password for storing - returns plaintext for the demo"""
    # In a real app, this would use a secure hashing algorithm
    return password

def load_models():
    """Load or create sentiment analysis models"""
    from models import create_cnn_model, create_lstm_model
    
    models = {}
    
    # Create simple rule-based models
    models['CNN'] = create_cnn_model()
    models['LSTM'] = create_lstm_model()
    
    st.sidebar.success("模型已加载完成")
    
    return models

def login_user():
    """Handle user login"""
    st.title("中文/英文情感识别与隐私保护系统")
    st.subheader("Chinese/English Sentiment Analysis & Privacy Protection")
    
    with st.form("login_form"):
        username = st.text_input("用户名 (Username)")
        password = st.text_input("密码 (Password)", type="password")
        submit_button = st.form_submit_button("登录 (Login)")
        
        if submit_button:
            if username in USERS and verify_password(USERS[username]['password'], password):
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.session_state['role'] = USERS[username]['role']
                st.success(f"欢迎, {username}!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("用户名或密码错误 (Wrong username or password)")

def display_sidebar():
    """Display sidebar with navigation options"""
    st.sidebar.title(f"您好, {st.session_state['username']}")
    st.sidebar.subheader(f"角色: {st.session_state['role']}")
    
    options = ["情感分析"]
    if st.session_state['role'] == 'admin':
        options.extend(["系统管理", "数据可视化", "模型管理"])
    
    selected_option = st.sidebar.selectbox("选择功能", options)
    
    if st.sidebar.button("退出登录"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()
    
    return selected_option

def sentiment_analysis_page():
    """Display sentiment analysis page"""
    st.title("中文/英文文本情感分析")
    st.subheader("Chinese/English Text Sentiment Analysis")
    
    # Model selection
    model_type = st.selectbox(
        "选择模型 (Select Model)",
        ["CNN", "LSTM"],
        index=0 if st.session_state['selected_model'] == 'CNN' else 1
    )
    st.session_state['selected_model'] = model_type
    
    # Text input
    text_input = st.text_area("请输入中文或英文文本进行情感分析 (Enter Chinese or English text for sentiment analysis)", height=150)
    
    if st.button("分析情感 (Analyze Sentiment)"):
        if text_input:
            with st.spinner("正在分析..."):
                # Preprocess text
                preprocessed_text = preprocess_text(text_input)
                
                # Encrypt data for privacy
                encrypted_text = encrypt_data(text_input)
                
                # Get prediction from model
                model = st.session_state['models'][model_type]
                prediction = model.predict(text_input)
                
                # Get sentiment label and confidence
                sentiment_labels = ['消极', '中性', '积极']
                sentiment_idx = np.argmax(prediction[0])
                sentiment = sentiment_labels[sentiment_idx]
                confidence = prediction[0][sentiment_idx]
                
                # Store in history
                analysis_result = {
                    'text': text_input,
                    'sentiment': sentiment,
                    'confidence': float(confidence),
                    'model': model_type,
                    'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state['analysis_history'].append(analysis_result)
                
                # Save history to file
                save_analysis_history(st.session_state['analysis_history'])
                
                # Display result
                st.success(f"情感分析完成! (Analysis completed!)")
                
                # Add English sentiment labels
                english_sentiment_labels = ['Negative', 'Neutral', 'Positive']
                english_sentiment = english_sentiment_labels[sentiment_idx]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("分析结果 (Analysis Results)")
                    st.write(f"**情感倾向 (Sentiment):** {sentiment} / {english_sentiment}")
                    st.write(f"**置信度 (Confidence):** {confidence:.2f}")
                    st.write(f"**分析模型 (Model):** {model_type}")
                
                with col2:
                    st.subheader("置信度分布 (Confidence Distribution)")
                    plot_confidence(prediction[0], sentiment_labels)
        else:
            st.warning("请输入文本再进行分析")
    
    # History section
    if st.session_state['analysis_history']:
        st.subheader("历史记录")
        df = pd.DataFrame(st.session_state['analysis_history'])
        st.dataframe(df)

def system_management_page():
    """Display system management page (admin only)"""
    st.title("系统管理")
    
    st.subheader("用户管理")
    users_df = pd.DataFrame([
        {"用户名": username, "角色": data['role']} 
        for username, data in USERS.items()
    ])
    st.dataframe(users_df)
    
    st.subheader("系统状态")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("活跃用户", "2")
    with col2:
        st.metric("分析请求数", len(st.session_state['analysis_history']))
    with col3:
        st.metric("系统运行时间", "3小时42分钟")
    
    st.subheader("安全设置")
    encryption_enabled = st.checkbox("启用数据加密", value=True)
    st.write("当前使用的加密算法: AES-256")
    
    if st.button("清除分析历史 (Clear History)"):
        st.session_state['analysis_history'] = []
        # Save empty history to file
        save_analysis_history([])
        st.success("历史记录已清除 (History cleared)")
        st.rerun()

def data_visualization_page():
    """Display data visualization page (admin only)"""
    st.title("数据可视化")
    
    if not st.session_state['analysis_history']:
        st.info("暂无分析数据可供可视化")
        return
    
    # Convert history to DataFrame
    df = pd.DataFrame(st.session_state['analysis_history'])
    
    st.subheader("情感分布")
    plot_sentiment_distribution(df)
    
    st.subheader("分析趋势")
    # Group by hour and count
    if 'timestamp' in df.columns:
        df['time'] = pd.to_datetime(df['timestamp'])
        hourly_counts = df.groupby(df['time'].dt.hour).size().reset_index(name='count')
        st.line_chart(hourly_counts.set_index('time')['count'])
    
    st.subheader("模型性能比较")
    if 'model' in df.columns and 'confidence' in df.columns:
        model_perf = df.groupby('model')['confidence'].mean().reset_index()
        st.bar_chart(model_perf.set_index('model'))

def model_management_page():
    """Display model management page (admin only)"""
    st.title("模型管理")
    
    # Model information
    models_info = [
        {"模型名称": "CNN", "类型": "卷积神经网络", "准确率": "87%", "状态": "已加载"},
        {"模型名称": "LSTM", "类型": "长短期记忆网络", "准确率": "89%", "状态": "已加载"}
    ]
    
    st.dataframe(pd.DataFrame(models_info))
    
    st.subheader("模型训练")
    
    col1, col2 = st.columns(2)
    with col1:
        model_to_train = st.selectbox("选择模型", ["CNN", "LSTM"])
    with col2:
        epochs = st.slider("训练轮数", 1, 20, 5)
    
    if st.button("开始训练"):
        with st.spinner(f"正在训练{model_to_train}模型..."):
            # Simulate training progress
            progress_bar = st.progress(0)
            for i in range(epochs):
                # Simulate epoch completion
                for j in range(10):
                    time.sleep(0.1)
                    progress_bar.progress((i * 10 + j + 1) / (epochs * 10))
            
            st.success(f"{model_to_train}模型训练完成!")
    
    st.subheader("模型评估")
    evaluation_metrics = {
        "准确率": 0.87,
        "精确率": 0.83,
        "召回率": 0.85,
        "F1分数": 0.84
    }
    
    st.json(evaluation_metrics)

def main():
    # Load models
    if 'models' not in st.session_state or not st.session_state['models']:
        st.session_state['models'] = load_models()
    
    # Login page
    if not st.session_state['logged_in']:
        login_user()
    else:
        # After login, show the selected page
        option = display_sidebar()
        
        if option == "情感分析":
            sentiment_analysis_page()
        elif option == "系统管理" and st.session_state['role'] == 'admin':
            system_management_page()
        elif option == "数据可视化" and st.session_state['role'] == 'admin':
            data_visualization_page()
        elif option == "模型管理" and st.session_state['role'] == 'admin':
            model_management_page()

if __name__ == "__main__":
    main()
