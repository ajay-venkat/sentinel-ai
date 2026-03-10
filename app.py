import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
from model_loader import load_sentiment_model, load_aspect_model
from logic import preprocess_text, analyze_sentiment, detect_aspect, calculate_crisis_score, generate_responses

# --- 1. Page Config & CSS ---
st.set_page_config(page_title="Sentinel AI Dashboard", layout="wide", page_icon="🛡️")

def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except:
        pass

load_css("style.css")

# --- 2. Load Models ---
@st.cache_resource(show_spinner=False)
def init_models():
    st.toast("Loading NLP Models... (This might take a minute initially)", icon="⏳")
    s_model = load_sentiment_model()
    a_model = load_aspect_model()
    st.toast("Models Loaded Successfully!", icon="✅")
    return s_model, a_model

sentiment_model, aspect_model = init_models()

# --- 3. UI Layout ---
st.title("🛡️ Sentinel AI: Aspect-Based Sentiment & Crisis Prediction")

# Create placeholders for dynamic content
header_placeholder = st.empty()
metrics_placeholder = st.empty()
charts_placeholder = st.empty()
gen_placeholder = st.empty()
feed_placeholder = st.empty()

# --- 4. State Initialization ---
if 'data_history' not in st.session_state:
    st.session_state.data_history = pd.DataFrame(columns=[
        'timestamp', 'username', 'text', 'sentiment', 'confidence', 'aspect', 'risk_score'
    ])
if 'negative_mentions' not in st.session_state:
    st.session_state.negative_mentions = []

# --- 5. Simulation Logic ---
def simulate_live_stream():
    try:
        df_stream = pd.read_csv("mock_stream.csv")
    except Exception as e:
        st.error(f"Could not load mock_stream.csv: {e}")
        return

    # To calculate moving metrics
    sentiment_history = {'Positive': 0, 'Negative': 0, 'Informational': 0, 'Sarcastic/Critical': 0}
    
    for i, row in df_stream.iterrows():
        # --- Preprocess ---
        raw_text = str(row['text'])
        clean_text = preprocess_text(raw_text)
        
        # --- ML Prediction ---
        sentiment_label, confidence = analyze_sentiment(clean_text, sentiment_model)
        
        # Sarcastic Layer re-classification done inside logic.py
        
        aspect = detect_aspect(clean_text, aspect_model, sentiment_label)
        
        # --- Analytics ---
        sentiment_history[sentiment_label] = sentiment_history.get(sentiment_label, 0) + 1
        
        # Risk Score metrics
        sentiment_intensity = confidence if sentiment_label in ['Negative', 'Sarcastic/Critical'] else 0.0
        reach_factor = np.random.uniform(0.5, 2.5) # Simulated reach 
        # Velocity = recent negative posts count (last 10 elements in history approximation)
        recent_history = st.session_state.data_history.tail(10)
        recent_negs = len(recent_history[recent_history['sentiment'].isin(['Negative', 'Sarcastic/Critical'])])
        velocity = max(1, recent_negs)
        
        risk_score = calculate_crisis_score(sentiment_intensity, reach_factor, velocity)
        
        # Track negative mentions
        if sentiment_label in ['Negative', 'Sarcastic/Critical']:
            st.session_state.negative_mentions.insert(0, raw_text)
            
        # Store in session state
        new_row = pd.DataFrame([{
            'timestamp': row['timestamp'],
            'username': row['username'],
            'text': raw_text,
            'sentiment': sentiment_label,
            'confidence': confidence,
            'aspect': aspect,
            'risk_score': risk_score
        }])
        st.session_state.data_history = pd.concat([st.session_state.data_history, new_row], ignore_index=True)
        
        # Prepare Data for UI
        df_current = st.session_state.data_history
        total_mentions = len(df_current)
        total_pos = len(df_current[df_current['sentiment'] == 'Positive'])
        total_neg = len(df_current[df_current['sentiment'].isin(['Negative', 'Sarcastic/Critical'])])
        
        # NPS Proxy (Net Sentiment)
        nps = int(((total_pos - total_neg) / total_mentions) * 100) if total_mentions > 0 else 0
        
        # Average Crisis Score
        avg_risk = df_current['risk_score'].mean()
        health_score = max(0, 100 - avg_risk)
        
        predicted_virality = min(100, reach_factor * velocity * 15)

        # --- UPDATE UI PLACEHOLDERS ---

        # Gauge Chart
        with header_placeholder.container():
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = health_score,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Live Health Score", 'font': {'color': 'white'}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickcolor': "white"},
                    'bar': {'color': "#3b82f6"},
                    'steps': [
                        {'range': [0, 30], 'color': "#ef4444"},
                        {'range': [30, 70], 'color': "#f59e0b"},
                        {'range': [70, 100], 'color': "#22c55e"}],
                }
            ))
            fig_gauge.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10), paper_bgcolor="#0e1117", font={'color': "white"})
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Core Metrics
        with metrics_placeholder.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Mentions", total_mentions, delta=f"+1", delta_color="normal")
            with col2:
                st.metric("Net Sentiment (NPS)", f"{nps}%", delta=f"{nps}%", delta_color="normal" if nps >=0 else "inverse")
            with col3:
                st.metric("Predicted Virality", f"{int(predicted_virality)}%", delta=f"{int(velocity)} vel", delta_color="inverse")

        # Charts Row
        with charts_placeholder.container():
            col_a, col_b = st.columns(2)
            with col_a:
                # Line Chart
                df_counts = df_current.groupby('sentiment').size().reset_index(name='counts')
                color_map = {'Positive': '#22c55e', 'Negative': '#ef4444', 'Informational': '#94a3b8', 'Sarcastic/Critical': '#f59e0b'}
                fig_line = px.bar(df_counts, x='sentiment', y='counts', color='sentiment', 
                                   color_discrete_map=color_map, title="Sentiment Distribution")
                fig_line.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white")
                st.plotly_chart(fig_line, use_container_width=True)
            with col_b:
                # Aspect Breakdown
                fig_aspect = px.histogram(df_current, x='aspect', color='sentiment', 
                                          color_discrete_map=color_map, barmode='group',
                                          title="Topic/Aspect Breakdown")
                fig_aspect.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#0e1117", font_color="white")
                st.plotly_chart(fig_aspect, use_container_width=True)

        # Generative Stubs (Every 5 posts updating)
        if i % 5 == 0 or i == len(df_stream) - 1:
            with gen_placeholder.container():
                st.markdown("### 🤖 Generative AI: Actionable Insights")
                summary, responses = generate_responses(st.session_state.negative_mentions)
                st.info(f"**Root Cause Summary:** {summary}")
                c1, c2, c3 = st.columns(3)
                c1.warning(f"**Professional Reply:**\n\n{responses[0]}")
                c2.success(f"**Empathic Reply:**\n\n{responses[1]}")
                c3.info(f"**Witty Reply:**\n\n{responses[2]}")

        # Live Feed
        with feed_placeholder.container():
            st.markdown("### 📡 Live Scrolling Feed")
            feed_html = '<div class="feed-container">'
            # Reverse order to show newest first
            for idx, r in df_current.tail(10).iloc[::-1].iterrows():
                css_class = str(r['sentiment']).replace("/", "").replace(" ", "")
                badge_bg = "#22c55e" if "Pos" in css_class else "#ef4444" if "Neg" in css_class else "#f59e0b" if "Sarc" in css_class else "#94a3b8"
                feed_html += f"""
                <div class="feed-item feed-{css_class}">
                    <div class="feed-meta">
                        <span>👤 {r['username']} in <b>{r['aspect']}</b></span>
                        <span class="sentiment-badge" style="background-color: {badge_bg}">{r['sentiment']}</span>
                    </div>
                    <div>{r['text']}</div>
                </div>
                """
            feed_html += '</div>'
            st.markdown(feed_html, unsafe_allow_html=True)

        # Sleep to simulate interval
        time.sleep(3)

if st.button("▶️ Start Live Stream Simulation", type="primary"):
    simulate_live_stream()
else:
    st.info("Click 'Start Live Stream Simulation' to begin feeding data.")


