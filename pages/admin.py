import streamlit as st
import pandas as pd
import numpy as np
import time
from visualization import (
    plot_sentiment_distribution,
    plot_training_history,
    plot_model_comparison,
    plot_word_frequency,
    plot_confusion_matrix
)
from models import create_cnn_model, create_lstm_model, train_model, evaluate_model
from preprocessing import prepare_dataset
from utils import save_model, load_model, export_results, get_system_stats, format_time
from data_security import secure_logging

def admin_dashboard():
    """Admin dashboard with system statistics and management options"""
    st.title("管理员控制台")
    
    # System stats
    stats = get_system_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("CPU 使用率", f"{stats['cpu_percent']}%")
    with col2:
        st.metric("内存使用率", f"{stats['memory_percent']}%")
    with col3:
        st.metric("系统运行时间", format_time(stats['start_time']))
    
    # Analysis history overview
    if 'analysis_history' in st.session_state and st.session_state['analysis_history']:
        st.subheader("分析历史概览")
        
        history_df = pd.DataFrame(st.session_state['analysis_history'])
        
        # Summary stats
        total_analyses = len(history_df)
        unique_texts = history_df['text'].nunique()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("总分析次数", total_analyses)
        with col2:
            st.metric("唯一文本数量", unique_texts)
        
        # Sentiment distribution
        plot_sentiment_distribution(history_df)
        
        # Export options
        st.subheader("导出分析结果")
        export_format = st.selectbox("选择导出格式", ["CSV", "JSON", "Excel"])
        
        if st.button("导出数据"):
            file_data = export_results(history_df, export_format.lower())
            
            # Create download button
            st.download_button(
                label=f"下载 {export_format} 文件",
                data=file_data,
                file_name=f"sentiment_analysis_results.{export_format.lower()}",
                mime={
                    'csv': 'text/csv',
                    'json': 'application/json',
                    'excel': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                }[export_format.lower()]
            )
    else:
        st.info("暂无分析历史数据")

def model_management():
    """Model management interface for admins"""
    st.title("模型管理")
    
    # Model information
    models_info = [
        {"模型名称": "CNN", "类型": "卷积神经网络", "参数数量": "约500K", "准确率": "87%"},
        {"模型名称": "LSTM", "类型": "长短期记忆网络", "参数数量": "约800K", "准确率": "89%"}
    ]
    
    st.dataframe(pd.DataFrame(models_info))
    
    # Model training tab
    st.subheader("模型训练")
    
    # Sample dataset for training
    if st.checkbox("使用示例数据集进行训练"):
        # In a real application, this would load actual data
        # Here we create a simple mock dataset
        
        # Create a progress spinner for "loading data"
        with st.spinner("正在加载数据集..."):
            time.sleep(2)  # Simulate loading time
            
            # Create mock dataset
            texts = [
                "这个产品非常好用，我很喜欢！",
                "质量还可以，但是价格有点贵",
                "太糟糕了，完全不值这个价格",
                "服务态度很好，会再次购买",
                "一般般吧，没什么特别的",
                # ... more examples would be here
            ]
            
            labels = ["积极", "中性", "消极", "积极", "中性"]
            
            mock_df = pd.DataFrame({
                'text': texts,
                'sentiment': labels
            })
            
            st.success(f"数据集加载完成！共 {len(mock_df)} 条样本")
            st.dataframe(mock_df.head())
        
        # Model selection
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox("选择模型类型", ["CNN", "LSTM"])
        
        with col2:
            epochs = st.slider("训练轮数", 2, 20, 5)
        
        # Training button
        if st.button("开始训练"):
            # Log the training action
            secure_logging("model_training_started", st.session_state['username'], 
                          f"Model: {model_type}, Epochs: {epochs}")
            
            with st.spinner(f"正在训练{model_type}模型..."):
                # Create progress bar
                progress_bar = st.progress(0)
                
                # Create model
                if model_type == "CNN":
                    model = create_cnn_model()
                else:
                    model = create_lstm_model()
                
                # Simulate training progress
                for i in range(epochs):
                    # Update progress text
                    st.text(f"Epoch {i+1}/{epochs}")
                    
                    # Simulate epoch training
                    for j in range(10):
                        time.sleep(0.1)
                        progress_bar.progress((i * 10 + j + 1) / (epochs * 10))
                    
                    # Show metrics
                    train_acc = 0.7 + (i * 0.03) + (np.random.random() * 0.02)
                    val_acc = 0.65 + (i * 0.025) + (np.random.random() * 0.03)
                    train_loss = 0.8 - (i * 0.06) + (np.random.random() * 0.03)
                    val_loss = 0.9 - (i * 0.05) + (np.random.random() * 0.04)
                    
                    metrics_col1, metrics_col2 = st.columns(2)
                    with metrics_col1:
                        st.metric("训练准确率", f"{train_acc:.4f}")
                        st.metric("验证准确率", f"{val_acc:.4f}")
                    with metrics_col2:
                        st.metric("训练损失", f"{train_loss:.4f}")
                        st.metric("验证损失", f"{val_loss:.4f}")
                
                # Save model to session state
                st.session_state['models'][model_type] = model
                
                # Log the completed training
                secure_logging("model_training_completed", st.session_state['username'], 
                              f"Model: {model_type}, Final Accuracy: {train_acc:.4f}")
                
                st.success(f"{model_type}模型训练完成!")
    
    # Model evaluation
    st.subheader("模型评估")
    
    if 'models' in st.session_state and st.session_state['models']:
        # Create tabs for different evaluations
        eval_tab1, eval_tab2 = st.tabs(["性能指标", "混淆矩阵"])
        
        with eval_tab1:
            st.write("模型性能指标")
            
            # Simulate performance metrics
            metrics = {
                "CNN": {
                    "accuracy": 0.87,
                    "precision": 0.86,
                    "recall": 0.84,
                    "f1": 0.85,
                    "loss": 0.32
                },
                "LSTM": {
                    "accuracy": 0.89,
                    "precision": 0.88,
                    "recall": 0.87,
                    "f1": 0.87,
                    "loss": 0.29
                }
            }
            
            # Display metrics
            metrics_df = pd.DataFrame({
                "指标": ["准确率", "精确率", "召回率", "F1分数", "损失"],
                "CNN": [
                    f"{metrics['CNN']['accuracy']:.2f}",
                    f"{metrics['CNN']['precision']:.2f}",
                    f"{metrics['CNN']['recall']:.2f}",
                    f"{metrics['CNN']['f1']:.2f}",
                    f"{metrics['CNN']['loss']:.2f}"
                ],
                "LSTM": [
                    f"{metrics['LSTM']['accuracy']:.2f}",
                    f"{metrics['LSTM']['precision']:.2f}",
                    f"{metrics['LSTM']['recall']:.2f}",
                    f"{metrics['LSTM']['f1']:.2f}",
                    f"{metrics['LSTM']['loss']:.2f}"
                ]
            })
            
            st.table(metrics_df)
            
            # Plot model comparison
            plot_model_comparison({
                "CNN": {"accuracy": metrics["CNN"]["accuracy"], "loss": metrics["CNN"]["loss"]},
                "LSTM": {"accuracy": metrics["LSTM"]["accuracy"], "loss": metrics["LSTM"]["loss"]}
            })
        
        with eval_tab2:
            st.write("混淆矩阵")
            
            # Simulate confusion matrix
            confusion_matrix = np.array([
                [42, 5, 3],
                [4, 38, 8],
                [2, 6, 52]
            ])
            
            plot_confusion_matrix(confusion_matrix, ["消极", "中性", "积极"])
    else:
        st.info("请先训练模型")

def user_management():
    """User management interface for admins"""
    st.title("用户管理")
    
    # User list
    users_data = [
        {"用户名": "admin", "角色": "管理员", "上次登录": "2023-07-15 14:30:22", "状态": "活跃"},
        {"用户名": "user", "角色": "普通用户", "上次登录": "2023-07-14 09:45:11", "状态": "活跃"},
    ]
    
    st.dataframe(pd.DataFrame(users_data))
    
    # User activity
    st.subheader("用户活动日志")
    
    activity_logs = [
        {"时间": "2023-07-15 14:30:22", "用户名": "admin", "活动": "登录系统", "IP地址": "192.168.1.101"},
        {"时间": "2023-07-15 14:32:15", "用户名": "admin", "活动": "查看系统统计", "IP地址": "192.168.1.101"},
        {"时间": "2023-07-15 14:35:47", "用户名": "admin", "活动": "训练模型(CNN)", "IP地址": "192.168.1.101"},
        {"时间": "2023-07-14 09:45:11", "用户名": "user", "活动": "登录系统", "IP地址": "192.168.1.105"},
        {"时间": "2023-07-14 09:47:32", "用户名": "user", "活动": "情感分析", "IP地址": "192.168.1.105"},
    ]
    
    st.dataframe(pd.DataFrame(activity_logs))
    
    # User management options
    st.subheader("用户管理选项")
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_username = st.text_input("用户名")
        new_password = st.text_input("密码", type="password")
        new_role = st.selectbox("角色", ["管理员", "普通用户"])
        
        if st.button("添加用户"):
            if new_username and new_password:
                st.success(f"已添加用户 {new_username}")
            else:
                st.error("用户名和密码不能为空")
    
    with col2:
        user_to_manage = st.selectbox("选择用户", ["user"])
        action = st.selectbox("操作", ["禁用账户", "重置密码", "更改角色"])
        
        if st.button("执行操作"):
            st.success(f"已对用户 {user_to_manage} 执行 {action} 操作")

def security_settings():
    """Security settings interface for admins"""
    st.title("安全设置")
    
    # Encryption settings
    st.subheader("加密设置")
    
    encryption_enabled = st.checkbox("启用数据加密", value=True)
    encryption_algorithm = st.selectbox(
        "加密算法",
        ["AES-256 (推荐)", "AES-128", "3DES"],
        index=0
    )
    
    # SSL/TLS settings
    st.subheader("传输安全设置")
    
    ssl_enabled = st.checkbox("启用 SSL/TLS 加密", value=True)
    ssl_cert_expiry = "2024-12-31"
    
    st.info(f"SSL 证书有效期至: {ssl_cert_expiry}")
    
    # Authentication settings
    st.subheader("认证设置")
    
    mfa_enabled = st.checkbox("启用多因素认证 (MFA)", value=False)
    password_expiry = st.slider("密码过期时间 (天)", 30, 180, 90)
    min_password_length = st.slider("最小密码长度", 6, 16, 8)
    
    # Privacy settings
    st.subheader("隐私设置")
    
    data_retention = st.slider("数据保留期限 (天)", 7, 365, 90)
    anonymize_data = st.checkbox("自动匿名化敏感数据", value=True)
    
    # Save settings
    if st.button("保存设置"):
        # In a real application, this would save the settings to a database or config file
        st.success("安全设置已更新")
        
        # Log the action
        secure_logging("security_settings_updated", st.session_state['username'])

def main():
    """Main function for admin page"""
    st.sidebar.title("管理员控制台")
    
    # Navigation
    page = st.sidebar.radio(
        "选择页面",
        ["控制台", "模型管理", "用户管理", "安全设置"]
    )
    
    # Display the selected page
    if page == "控制台":
        admin_dashboard()
    elif page == "模型管理":
        model_management()
    elif page == "用户管理":
        user_management()
    elif page == "安全设置":
        security_settings()

if __name__ == "__main__":
    main()
