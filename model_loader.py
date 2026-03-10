import streamlit as st
from transformers import pipeline
import time

@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    start_time = time.time()
    try:
        model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        print(f"Sentiment model loaded in {time.time() - start_time:.2f}s")
        return model
    except Exception as e:
        print(f"Error loading sentiment model: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_aspect_model():
    start_time = time.time()
    try:
        model = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        print(f"Aspect model loaded in {time.time() - start_time:.2f}s")
        return model
    except Exception as e:
        print(f"Error loading aspect model: {e}")
        return None
