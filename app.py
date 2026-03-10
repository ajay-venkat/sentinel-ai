import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
from model_loader import hf_sentiment, hf_zero_shot
from logic import (
    preprocess_text, analyze_sentiment, detect_aspect,
    calculate_crisis_score, generate_responses, ASPECT_CATEGORIES
)

# --- 1. Page Config & CSS ---
st.set_page_config(page_title="Sentinel AI Dashboard", layout="wide", page_icon="🛡️")

def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except Exception:
        pass

load_css("style.css")

# --- 2. Wrapper functions for the API calls ---
def sentiment_fn(text):
    return hf_sentiment(text)

def aspect_fn(text, labels):
    return hf_zero_shot(text, labels)

# --- 3. UI Layout ---
st.markdown("""
<div style="text-align:center; padding: 10px 0;">
    <h1 style="margin-bottom:0;">🛡️ Sentinel AI</h1>
    <p style="color:#94a3b8; font-size:1.1rem; margin-top:5px;">Aspect-Based Sentiment & Crisis Prediction Engine</p>
</div>
""", unsafe_allow_html=True)

# Create placeholders for dynamic content
header_placeholder = st.empty()
metrics_placeholder = st.empty()
charts_placeholder = st.empty()
gen_placeholder = st.empty()
feed_placeholder = st.empty()
progress_placeholder = st.empty()

# --- 4. State Initialization ---
if 'data_history' not in st.session_state:
    st.session_state.data_history = pd.DataFrame(columns=[
        'timestamp', 'username', 'text', 'sentiment', 'confidence', 'aspect', 'risk_score'
    ])
if 'negative_mentions' not in st.session_state:
    st.session_state.negative_mentions = []
if 'stream_running' not in st.session_state:
    st.session_state.stream_running = False

# --- 5. Render functions ---
def render_gauge(health_score):
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Live Health Score", 'font': {'color': 'white', 'size': 20}},
        number={'font': {'color': 'white', 'size': 40}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': "white", 'tickfont': {'color': 'white'}},
            'bar': {'color': "#3b82f6"},
            'bgcolor': "#1e293b",
            'steps': [
                {'range': [0, 30], 'color': "#ef4444"},
                {'range': [30, 70], 'color': "#f59e0b"},
                {'range': [70, 100], 'color': "#22c55e"}
            ],
        }
    ))
    fig_gauge.update_layout(
        height=250,
        margin=dict(l=10, r=10, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"}
    )
    return fig_gauge

COLOR_MAP = {
    'Positive': '#22c55e',
    'Negative': '#ef4444',
    'Informational': '#94a3b8',
    'Sarcastic/Critical': '#f59e0b'
}

def render_feed_html(df):
    feed_html = '<div class="feed-container">'
    for _, r in df.tail(10).iloc[::-1].iterrows():
        css_class = str(r['sentiment']).replace("/", "").replace(" ", "")
        badge_bg = COLOR_MAP.get(r['sentiment'], '#94a3b8')
        feed_html += f"""
        <div class="feed-item feed-{css_class}">
            <div class="feed-meta">
                <span>👤 <b>{r['username']}</b> · {r['aspect']}</span>
                <span class="sentiment-badge" style="background-color: {badge_bg}; color: white;">{r['sentiment']}</span>
            </div>
            <div style="margin-top:6px;">{r['text']}</div>
        </div>
        """
    feed_html += '</div>'
    return feed_html

# --- 6. Simulation Logic ---
def simulate_live_stream():
    try:
        df_stream = pd.read_csv("mock_stream.csv")
    except Exception as e:
        st.error(f"Could not load mock_stream.csv: {e}")
        return

    total_rows = len(df_stream)

    for i, row in df_stream.iterrows():
        raw_text = str(row.get('text', ''))
        clean_text = preprocess_text(raw_text)

        if not clean_text:
            clean_text = raw_text

        # --- ML Prediction via API ---
        sentiment_label, confidence = analyze_sentiment(clean_text, sentiment_fn)
        aspect = detect_aspect(clean_text, aspect_fn, sentiment_label)

        # --- Analytics ---
        sentiment_intensity = confidence if sentiment_label in ['Negative', 'Sarcastic/Critical'] else 0.0
        reach_factor = np.random.uniform(0.5, 2.5)
        recent_history = st.session_state.data_history.tail(10)
        recent_negs = len(recent_history[recent_history['sentiment'].isin(['Negative', 'Sarcastic/Critical'])])
        velocity = max(1, recent_negs)
        risk_score = calculate_crisis_score(sentiment_intensity, reach_factor, velocity)

        if sentiment_label in ['Negative', 'Sarcastic/Critical']:
            st.session_state.negative_mentions.insert(0, raw_text)

        # Store result
        new_row = pd.DataFrame([{
            'timestamp': row.get('timestamp', ''),
            'username': row.get('username', f'user{i}'),
            'text': raw_text,
            'sentiment': sentiment_label,
            'confidence': confidence,
            'aspect': aspect,
            'risk_score': risk_score
        }])
        st.session_state.data_history = pd.concat(
            [st.session_state.data_history, new_row], ignore_index=True
        )

        # Compute metrics
        df_current = st.session_state.data_history
        total_mentions = len(df_current)
        total_pos = len(df_current[df_current['sentiment'] == 'Positive'])
        total_neg = len(df_current[df_current['sentiment'].isin(['Negative', 'Sarcastic/Critical'])])
        nps = int(((total_pos - total_neg) / total_mentions) * 100) if total_mentions > 0 else 0
        avg_risk = df_current['risk_score'].mean()
        health_score = max(0, min(100, 100 - avg_risk))
        predicted_virality = min(100, reach_factor * velocity * 15)

        # --- UPDATE UI ---
        with progress_placeholder.container():
            st.progress((i + 1) / total_rows, text=f"Processing post {i + 1}/{total_rows}...")

        with header_placeholder.container():
            st.plotly_chart(render_gauge(health_score), use_container_width=True, key=f"gauge_{i}")

        with metrics_placeholder.container():
            col1, col2, col3 = st.columns(3)
            col1.metric("📊 Total Mentions", total_mentions, delta="+1")
            col2.metric("📈 Net Sentiment (NPS)", f"{nps}%",
                        delta=f"{nps}%", delta_color="normal" if nps >= 0 else "inverse")
            col3.metric("🔥 Predicted Virality", f"{int(predicted_virality)}%",
                        delta=f"{int(velocity)} vel", delta_color="inverse")

        with charts_placeholder.container():
            col_a, col_b = st.columns(2)
            with col_a:
                df_counts = df_current.groupby('sentiment').size().reset_index(name='counts')
                fig_bar = px.bar(df_counts, x='sentiment', y='counts', color='sentiment',
                                 color_discrete_map=COLOR_MAP,
                                 title="📊 Sentiment Distribution")
                fig_bar.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white", showlegend=False
                )
                st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{i}")
            with col_b:
                fig_aspect = px.histogram(df_current, x='aspect', color='sentiment',
                                          color_discrete_map=COLOR_MAP, barmode='group',
                                          title="🏷️ Topic / Aspect Breakdown")
                fig_aspect.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font_color="white"
                )
                st.plotly_chart(fig_aspect, use_container_width=True, key=f"aspect_{i}")

        # Generative insights every 5 posts or on the last post
        if (i + 1) % 5 == 0 or i == total_rows - 1:
            with gen_placeholder.container():
                st.markdown("### 🤖 AI-Powered Actionable Insights")
                summary, responses = generate_responses(st.session_state.negative_mentions)
                st.info(f"**🔎 Root Cause Summary:** {summary}")
                c1, c2, c3 = st.columns(3)
                c1.warning(f"**🏢 Professional:**\n\n{responses[0]}")
                c2.success(f"**💚 Empathic:**\n\n{responses[1]}")
                c3.info(f"**😎 Brand-Witty:**\n\n{responses[2]}")

        # Live Feed
        with feed_placeholder.container():
            st.markdown("### 📡 Live Scrolling Feed")
            st.markdown(render_feed_html(df_current), unsafe_allow_html=True)

        # Simulate real-time delay
        time.sleep(3)

    with progress_placeholder.container():
        st.success("✅ Stream simulation complete! All posts have been processed.")

# --- 7. Main App ---
st.divider()

col_btn1, col_btn2 = st.columns([1, 1])
with col_btn1:
    start = st.button("▶️ Start Live Stream Simulation", type="primary", use_container_width=True)
with col_btn2:
    reset = st.button("🔄 Reset Dashboard", use_container_width=True)

if reset:
    st.session_state.data_history = pd.DataFrame(columns=[
        'timestamp', 'username', 'text', 'sentiment', 'confidence', 'aspect', 'risk_score'
    ])
    st.session_state.negative_mentions = []
    st.rerun()

if start:
    simulate_live_stream()
elif len(st.session_state.data_history) == 0:
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px; color: #64748b;">
        <h2>👋 Welcome to Sentinel AI</h2>
        <p style="font-size:1.1rem;">Click <b>"Start Live Stream Simulation"</b> to begin analyzing mock social media data in real-time.</p>
        <p style="font-size:0.9rem; color:#475569;">The dashboard will process 20 posts at 3-second intervals, showing live sentiment analysis, aspect detection, and crisis scoring.</p>
    </div>
    """, unsafe_allow_html=True)
