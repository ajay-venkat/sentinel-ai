import streamlit as st
import requests
import os

HF_API_URL_SENTIMENT = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
HF_API_URL_ASPECT = "https://api-inference.huggingface.co/models/facebook/bart-large-mnli"

def get_headers():
    token = os.getenv("HF_TOKEN", "")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}

def hf_sentiment(text: str) -> dict:
    """Call HF Inference API for sentiment. Returns {'label': ..., 'score': ...}"""
    try:
        resp = requests.post(HF_API_URL_SENTIMENT, headers=get_headers(), json={"inputs": text}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # HF returns [[{label, score}, ...]] — pick the top result
        if isinstance(data, list) and len(data) > 0:
            if isinstance(data[0], list) and len(data[0]) > 0:
                return data[0][0]
            elif isinstance(data[0], dict):
                return data[0]
    except Exception as e:
        print(f"HF Sentiment API error: {e}")
    return {"label": "neutral", "score": 0.5}

def hf_zero_shot(text: str, candidate_labels: list) -> dict:
    """Call HF Inference API for zero-shot classification."""
    try:
        payload = {"inputs": text, "parameters": {"candidate_labels": candidate_labels}}
        resp = requests.post(HF_API_URL_ASPECT, headers=get_headers(), json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and "labels" in data:
            return data
    except Exception as e:
        print(f"HF Zero-shot API error: {e}")
    return {"labels": candidate_labels, "scores": [0.25] * len(candidate_labels)}

# --- Fallback: Local pipeline loading (only if running locally with enough RAM) ---
_local_sentiment = None
_local_aspect = None

def _try_load_local():
    global _local_sentiment, _local_aspect
    try:
        from transformers import pipeline
        _local_sentiment = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        _local_aspect = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        return True
    except Exception:
        return False

def local_sentiment(text: str) -> dict:
    if _local_sentiment is None:
        return hf_sentiment(text)
    try:
        result = _local_sentiment(text)[0]
        return result
    except:
        return hf_sentiment(text)

def local_zero_shot(text: str, labels: list) -> dict:
    if _local_aspect is None:
        return hf_zero_shot(text, labels)
    try:
        return _local_aspect(text, labels)
    except:
        return hf_zero_shot(text, labels)
