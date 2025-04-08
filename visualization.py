import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

def plot_sentiment_distribution(df):
    """
    Plot the distribution of sentiment classes
    
    Args:
        df: DataFrame with 'sentiment' column
    """
    if 'sentiment' not in df.columns:
        st.error("DataFrame does not contain a 'sentiment' column")
        return
    
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    
    fig = px.pie(
        sentiment_counts, 
        values='Count', 
        names='Sentiment',
        title='情感分布',
        color='Sentiment',
        color_discrete_map={
            '积极': '#4CAF50',
            '中性': '#2196F3',
            '消极': '#F44336'
        }
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    st.plotly_chart(fig, use_container_width=True)

def plot_confidence(prediction, labels):
    """
    Plot confidence scores for each sentiment class
    
    Args:
        prediction: Array of confidence scores
        labels: List of sentiment labels
    """
    fig = go.Figure()
    
    colors = ['#F44336', '#2196F3', '#4CAF50']
    
    for i, (label, score) in enumerate(zip(labels, prediction)):
        fig.add_trace(go.Bar(
            x=[label],
            y=[score],
            name=label,
            marker_color=colors[i]
        ))
    
    fig.update_layout(
        title='情感置信度',
        yaxis=dict(
            title='置信度',
            range=[0, 1]
        ),
        xaxis_title='情感类别',
        legend_title='情感类别'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_word_frequency(texts, top_n=20):
    """
    Plot the frequency of top N words
    
    Args:
        texts: List of tokenized texts
        top_n: Number of top words to display
    """
    # Flatten the list of tokenized texts
    all_words = [word for text in texts for word in text]
    
    # Count word frequencies
    word_counts = Counter(all_words)
    
    # Get top N words
    top_words = word_counts.most_common(top_n)
    
    # Create DataFrame
    df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
    
    # Plot
    fig = px.bar(
        df, 
        x='Word', 
        y='Frequency',
        title=f'Top {top_n} Words by Frequency',
        labels={'x': 'Word', 'y': 'Frequency'}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_confusion_matrix(confusion_matrix, classes):
    """
    Plot confusion matrix
    
    Args:
        confusion_matrix: Confusion matrix array
        classes: List of class labels
    """
    fig = px.imshow(
        confusion_matrix,
        labels=dict(x="Predicted", y="True", color="Count"),
        x=classes,
        y=classes,
        title="Confusion Matrix",
        color_continuous_scale='Blues'
    )
    
    # Add text annotations
    for i in range(len(classes)):
        for j in range(len(classes)):
            fig.add_annotation(
                x=j, 
                y=i,
                text=str(confusion_matrix[i, j]),
                showarrow=False,
                font=dict(color='black' if confusion_matrix[i, j] < confusion_matrix.max() / 2 else 'white')
            )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_training_history(history):
    """
    Plot training history
    
    Args:
        history: Training history object
    """
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add accuracy
    fig.add_trace(go.Scatter(
        x=list(range(1, len(history.history['accuracy']) + 1)),
        y=history.history['accuracy'],
        name='Training Accuracy',
        mode='lines+markers',
        line=dict(color='#2196F3')
    ))
    
    # Add validation accuracy if available
    if 'val_accuracy' in history.history:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(history.history['val_accuracy']) + 1)),
            y=history.history['val_accuracy'],
            name='Validation Accuracy',
            mode='lines+markers',
            line=dict(color='#4CAF50', dash='dash')
        ))
    
    # Add loss
    fig.add_trace(go.Scatter(
        x=list(range(1, len(history.history['loss']) + 1)),
        y=history.history['loss'],
        name='Training Loss',
        mode='lines+markers',
        line=dict(color='#F44336'),
        yaxis='y2'
    ))
    
    # Add validation loss if available
    if 'val_loss' in history.history:
        fig.add_trace(go.Scatter(
            x=list(range(1, len(history.history['val_loss']) + 1)),
            y=history.history['val_loss'],
            name='Validation Loss',
            mode='lines+markers',
            line=dict(color='#FF9800', dash='dash'),
            yaxis='y2'
        ))
    
    # Set layout with dual y-axes
    fig.update_layout(
        title='Training History',
        xaxis_title='Epoch',
        yaxis=dict(
            title='Accuracy',
            range=[0, 1]
        ),
        yaxis2=dict(
            title='Loss',
            overlaying='y',
            side='right',
            showgrid=False
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_model_comparison(models_metrics):
    """
    Plot comparison of multiple models
    
    Args:
        models_metrics: Dictionary with model names as keys and metrics as values
    """
    # Prepare data
    models = list(models_metrics.keys())
    accuracy = [metrics['accuracy'] for metrics in models_metrics.values()]
    loss = [metrics['loss'] for metrics in models_metrics.values()]
    
    # Create figure
    fig = go.Figure()
    
    # Add accuracy bars
    fig.add_trace(go.Bar(
        x=models,
        y=accuracy,
        name='Accuracy',
        marker_color='#2196F3',
        text=[f"{acc:.2f}" for acc in accuracy],
        textposition='auto'
    ))
    
    # Add loss bars
    fig.add_trace(go.Bar(
        x=models,
        y=loss,
        name='Loss',
        marker_color='#F44336',
        text=[f"{l:.2f}" for l in loss],
        textposition='auto'
    ))
    
    # Update layout
    fig.update_layout(
        title='Model Comparison',
        xaxis_title='Model',
        yaxis_title='Value',
        barmode='group',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
