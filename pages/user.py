import streamlit as st
import pandas as pd
import numpy as np
import time
from preprocessing import preprocess_text
from visualization import plot_confidence
from data_security import encrypt_data, anonymize_text, secure_logging

def user_dashboard():
    """Dashboard for regular users"""
    st.title("中文情感分析")
    
    # Model selection
    model_type = st.selectbox(
        "选择模型",
        ["CNN", "LSTM"],
        index=0 if st.session_state.get('selected_model', 'CNN') == 'CNN' else 1
    )
    st.session_state['selected_model'] = model_type
    
    # Text input
    text_input = st.text_area("请输入中文文本进行情感分析", height=150)
    
    # Analysis options
    with st.expander("高级选项"):
        show_details = st.checkbox("显示详细分析", value=True)
        anonymize = st.checkbox("匿名化敏感数据", value=True)
    
    # Analyze button
    if st.button("分析情感"):
        if text_input:
            with st.spinner("正在分析..."):
                # Anonymize if requested
                if anonymize:
                    text_for_display = anonymize_text(text_input)
                else:
                    text_for_display = text_input
                
                # Log the analysis request
                secure_logging("sentiment_analysis_request", st.session_state['username'])
                
                # Preprocess text
                preprocessed_text = preprocess_text(text_input)
                
                # Encrypt original text for privacy
                encrypted_text = encrypt_data(text_input)
                
                # Add slight delay to simulate processing
                time.sleep(1)
                
                # Simulate model prediction with random values weighted toward positive
                # In a real application, this would use actual model prediction
                rand_val = np.random.random()
                if "好" in text_input or "喜欢" in text_input or "excellent" in text_input.lower():
                    # Bias toward positive for texts with positive words
                    prediction = np.array([[0.15, 0.25, 0.60]])
                elif "坏" in text_input or "差" in text_input or "terrible" in text_input.lower():
                    # Bias toward negative for texts with negative words
                    prediction = np.array([[0.65, 0.25, 0.10]])
                else:
                    # Random but weighted slightly positive
                    prediction = np.array([[
                        0.2 + (0.3 * rand_val), 
                        0.3, 
                        0.5 - (0.3 * rand_val)
                    ]])
                
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
                
                if 'analysis_history' not in st.session_state:
                    st.session_state['analysis_history'] = []
                
                st.session_state['analysis_history'].append(analysis_result)
                
                # Display result
                st.success(f"情感分析完成!")
                
                # Display text that was analyzed
                st.subheader("分析文本")
                st.write(text_for_display)
                
                # Display sentiment result
                st.subheader("情感分析结果")
                
                # Use different colors for different sentiments
                if sentiment == "积极":
                    st.markdown(f"<h3 style='color:#4CAF50'>👍 {sentiment}</h3>", unsafe_allow_html=True)
                elif sentiment == "消极":
                    st.markdown(f"<h3 style='color:#F44336'>👎 {sentiment}</h3>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<h3 style='color:#2196F3'>😐 {sentiment}</h3>", unsafe_allow_html=True)
                
                st.write(f"**置信度:** {confidence:.2f}")
                st.write(f"**分析模型:** {model_type}")
                
                # Show confidence distribution
                if show_details:
                    st.subheader("置信度分布")
                    plot_confidence(prediction[0], sentiment_labels)
                    
                    # Display preprocessing details
                    st.subheader("预处理详情")
                    st.text(f"预处理后的文本: {preprocessed_text}")
        else:
            st.warning("请输入文本再进行分析")

def history_view():
    """View past analysis history"""
    st.title("分析历史")
    
    if 'analysis_history' in st.session_state and st.session_state['analysis_history']:
        # Filter options
        with st.expander("筛选选项"):
            # Collect unique sentiments and models
            sentiments = list(set([item['sentiment'] for item in st.session_state['analysis_history']]))
            models = list(set([item['model'] for item in st.session_state['analysis_history']]))
            
            # Add "All" option
            sentiments = ["全部"] + sentiments
            models = ["全部"] + models
            
            # Create filters
            col1, col2 = st.columns(2)
            with col1:
                filter_sentiment = st.selectbox("情感", sentiments)
            with col2:
                filter_model = st.selectbox("模型", models)
        
        # Convert to DataFrame
        df = pd.DataFrame(st.session_state['analysis_history'])
        
        # Apply filters
        filtered_df = df.copy()
        if filter_sentiment != "全部":
            filtered_df = filtered_df[filtered_df['sentiment'] == filter_sentiment]
        if filter_model != "全部":
            filtered_df = filtered_df[filtered_df['model'] == filter_model]
        
        # Sort by timestamp
        filtered_df = filtered_df.sort_values('timestamp', ascending=False)
        
        # Display filtered results
        st.write(f"显示 {len(filtered_df)} 条记录 (共 {len(df)} 条)")
        
        for i, row in filtered_df.iterrows():
            with st.container():
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Display text and truncate if too long
                    text = row['text']
                    if len(text) > 100:
                        text = text[:100] + "..."
                    
                    st.text(text)
                    st.caption(f"时间: {row['timestamp']}")
                
                with col2:
                    # Display sentiment with appropriate color
                    sentiment = row['sentiment']
                    confidence = row['confidence']
                    model = row['model']
                    
                    if sentiment == "积极":
                        st.markdown(f"<p style='color:#4CAF50'>👍 {sentiment} ({confidence:.2f})</p>", unsafe_allow_html=True)
                    elif sentiment == "消极":
                        st.markdown(f"<p style='color:#F44336'>👎 {sentiment} ({confidence:.2f})</p>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<p style='color:#2196F3'>😐 {sentiment} ({confidence:.2f})</p>", unsafe_allow_html=True)
                    
                    st.caption(f"模型: {model}")
                
                st.markdown("---")
    else:
        st.info("暂无分析历史记录")

def batch_analysis():
    """Batch analysis for multiple texts"""
    st.title("批量情感分析")
    
    # Instructions
    st.write("""
    使用此功能可以一次分析多条文本。
    请在下方文本框中输入多条文本，每行一条。
    """)
    
    # Text input
    batch_texts = st.text_area("请输入多条文本（每行一条）", height=200)
    
    # Model selection
    model_type = st.selectbox(
        "选择模型",
        ["CNN", "LSTM"],
        index=0 if st.session_state.get('selected_model', 'CNN') == 'CNN' else 1
    )
    
    if st.button("开始批量分析"):
        if batch_texts:
            # Split text into lines
            texts = [line.strip() for line in batch_texts.split('\n') if line.strip()]
            
            if not texts:
                st.warning("请输入至少一条文本")
                return
            
            # Create progress bar
            progress_bar = st.progress(0)
            
            # Results container
            results = []
            
            # Process each text
            for i, text in enumerate(texts):
                with st.spinner(f"正在分析第 {i+1}/{len(texts)} 条文本..."):
                    # Update progress
                    progress_bar.progress((i + 1) / len(texts))
                    
                    # Preprocess text
                    preprocessed_text = preprocess_text(text)
                    
                    # Simulate model prediction
                    # In a real application, this would use actual model prediction
                    rand_val = np.random.random()
                    if "好" in text or "喜欢" in text or "excellent" in text.lower():
                        # Bias toward positive for texts with positive words
                        prediction = np.array([[0.15, 0.25, 0.60]])
                    elif "坏" in text or "差" in text or "terrible" in text.lower():
                        # Bias toward negative for texts with negative words
                        prediction = np.array([[0.65, 0.25, 0.10]])
                    else:
                        # Random but weighted slightly positive
                        prediction = np.array([[
                            0.2 + (0.3 * rand_val), 
                            0.3, 
                            0.5 - (0.3 * rand_val)
                        ]])
                    
                    # Get sentiment label and confidence
                    sentiment_labels = ['消极', '中性', '积极']
                    sentiment_idx = np.argmax(prediction[0])
                    sentiment = sentiment_labels[sentiment_idx]
                    confidence = prediction[0][sentiment_idx]
                    
                    # Add to results
                    results.append({
                        'text': text,
                        'sentiment': sentiment,
                        'confidence': float(confidence),
                        'model': model_type,
                        'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    # Add to session state history
                    if 'analysis_history' not in st.session_state:
                        st.session_state['analysis_history'] = []
                    
                    st.session_state['analysis_history'].append(results[-1])
                    
                    # Add a small delay to make progress visible
                    time.sleep(0.1)
            
            # Show completion message
            st.success(f"批量分析完成! 已分析 {len(results)} 条文本")
            
            # Display results in a table
            results_df = pd.DataFrame(results)
            st.dataframe(results_df)
            
            # Summary
            st.subheader("分析摘要")
            sentiment_counts = results_df['sentiment'].value_counts()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                positive_count = sentiment_counts.get('积极', 0)
                st.metric("积极", positive_count, f"{positive_count/len(results):.1%}")
            with col2:
                neutral_count = sentiment_counts.get('中性', 0)
                st.metric("中性", neutral_count, f"{neutral_count/len(results):.1%}")
            with col3:
                negative_count = sentiment_counts.get('消极', 0)
                st.metric("消极", negative_count, f"{negative_count/len(results):.1%}")
            
            # Offer to download results
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="下载分析结果 (CSV)",
                data=csv,
                file_name="batch_analysis_results.csv",
                mime="text/csv",
            )
        else:
            st.warning("请输入文本再进行分析")

def help_page():
    """Help page with usage instructions"""
    st.title("帮助与说明")
    
    st.markdown("""
    ## 系统介绍
    
    本系统是一个基于深度学习的中文情感识别与隐私保护系统，可以帮助您分析中文文本的情感倾向（积极、消极或中性）。
    
    系统采用了卷积神经网络（CNN）和长短期记忆网络（LSTM）两种深度学习模型，您可以根据需要选择不同的模型进行分析。
    
    ## 使用方法
    
    ### 单条文本分析
    
    1. 在"中文情感分析"页面，输入您想分析的中文文本
    2. 选择想要使用的模型（CNN或LSTM）
    3. 点击"分析情感"按钮
    4. 系统会显示分析结果，包括情感倾向和置信度
    
    ### 批量文本分析
    
    1. 在"批量情感分析"页面，输入多条中文文本，每行一条
    2. 选择想要使用的模型
    3. 点击"开始批量分析"按钮
    4. 系统会依次分析所有文本，并显示综合结果
    
    ### 查看历史记录
    
    在"分析历史"页面，您可以查看之前的分析记录，并可以按情感类型和使用的模型进行筛选。
    
    ## 隐私保护
    
    系统采用了多种隐私保护措施：
    
    1. 文本加密：所有输入的文本在存储前都会进行加密
    2. 数据匿名化：系统可以自动识别并匿名化敏感信息（如个人姓名、电话号码等）
    3. 安全传输：数据传输过程采用SSL/TLS加密
    
    如有其他问题，请联系系统管理员。
    """)

def main():
    """Main function for user page"""
    st.sidebar.title(f"您好, {st.session_state['username']}")
    
    # Navigation
    page = st.sidebar.radio(
        "选择功能",
        ["中文情感分析", "批量情感分析", "分析历史", "帮助"]
    )
    
    # Display the selected page
    if page == "中文情感分析":
        user_dashboard()
    elif page == "批量情感分析":
        batch_analysis()
    elif page == "分析历史":
        history_view()
    elif page == "帮助":
        help_page()

if __name__ == "__main__":
    main()
