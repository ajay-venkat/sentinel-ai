# 🛡️ Sentinel AI: Aspect-Based Sentiment & Crisis Prediction Engine

A real-time social media sentiment analysis dashboard that goes beyond simple Positive/Negative classification. Built for hackathons.

## ✨ Features

- **Aspect-Based Sentiment Analysis** — Identifies *why* users feel a certain way (Product Quality, Price, Customer Support, Speed)
- **Sarcasm Detection Layer** — Catches sarcastic posts that fool traditional models
- **Crisis Risk Scoring** — Real-time risk metric based on sentiment intensity, reach, and velocity
- **Generative AI Insights** — Root-cause summaries and auto-drafted response styles
- **Live Stream Simulation** — Real-time data feed with animated UI updates
- **Dark Mode Dashboard** — Professional, modern UI with color-coded sentiment tags

## 🧠 Tech Stack

| Component | Technology |
|---|---|
| Dashboard | Streamlit |
| Sentiment Model | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Aspect Detection | `facebook/bart-large-mnli` (Zero-Shot) |
| Visualizations | Plotly Express |
| Data Handling | Pandas, NumPy |
| Generative AI | Google Gemini API (optional) |

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

## 📁 Project Structure

```
├── app.py              # Streamlit frontend dashboard
├── model_loader.py     # Cached NLP model loading
├── logic.py            # Sentiment, aspect, sarcasm & crisis scoring logic
├── style.css           # Dark mode CSS injection
├── mock_stream.csv     # Simulated social media data
└── requirements.txt    # Python dependencies
```

## 📊 Dashboard Layout

1. **Health Score Gauge** — Real-time 0–100 system health indicator
2. **KPI Metrics** — Total Mentions, Net Sentiment (NPS), Predicted Virality
3. **Trend Charts** — Sentiment distribution & topic/aspect breakdown
4. **Live Feed** — Scrolling, color-coded post stream with sentiment badges

## 📝 License

MIT
