import re

EMOJI_MAP = {
    "🚀": "bullish excited",
    "📉": "falling bad",
    "😡": "angry mad",
    "🙄": "annoyed sarcastic",
    "📦": "package delivery",
    "💨": "fast speed",
    "😍": "love excellent",
    "🔥": "amazing perfect",
    "🗑️": "trash garbage",
    "💀": "dead terrible",
    "😂": "funny laughing",
    "👎": "bad dislike",
    "👍": "good like",
}

NEGATIVE_WORDS = [
    "garbage", "fail", "broken", "worst", "trash", "slow",
    "terrible", "bad", "awful", "cheap", "sucks", "horrible",
    "useless", "scam", "ripoff", "disappointing"
]

ASPECT_CATEGORIES = ["Product Quality", "Price", "Customer Support", "Speed"]

def preprocess_text(text: str) -> str:
    if not text or not text.strip():
        return ""
    # Remove RTs
    text = re.sub(r'^RT\s+', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text, flags=re.MULTILINE)
    # Map emojis to text
    for emoji, meaning in EMOJI_MAP.items():
        if emoji in text:
            text = text.replace(emoji, f" {meaning} ")
    # Clean excessive whitespace
    text = " ".join(text.split())
    return text.strip()

def analyze_sentiment(text: str, sentiment_fn) -> tuple:
    """Returns (label, score). sentiment_fn is a callable that takes text and returns {'label':..., 'score':...}"""
    if not text.strip():
        return ("Informational", 0.5)

    try:
        result = sentiment_fn(text)
        label = result.get('label', 'neutral')
        score = result.get('score', 0.5)

        # Normalize label names from the HF model
        label_lower = label.lower()
        if 'positive' in label_lower:
            label = 'Positive'
        elif 'negative' in label_lower:
            label = 'Negative'
        else:
            label = 'Informational'

        # --- Sarcasm Layer ---
        lower_text = text.lower()
        negative_word_count = sum(1 for word in NEGATIVE_WORDS if word in lower_text)

        if label == 'Positive':
            sarcasm_cues = any(cue in lower_text for cue in ["sure", "great", "love how", "perfect", "brilliant", "fantastic", "oh wow", "yeah right"])
            has_eyeroll = "🙄" in text or "sarcastic" in lower_text or "annoyed" in lower_text
            if negative_word_count >= 1 and (sarcasm_cues or has_eyeroll):
                label = "Sarcastic/Critical"
            elif negative_word_count >= 2:
                label = "Sarcastic/Critical"

        return (label, score)
    except Exception as e:
        print(f"Sentiment error: {e}")
        return ("Informational", 0.5)

def detect_aspect(text: str, zero_shot_fn, sentiment_label: str) -> str:
    """Returns the detected aspect category."""
    if not text.strip():
        return "General"
    if len(text.split()) < 3:
        return "General"

    try:
        result = zero_shot_fn(text, ASPECT_CATEGORIES)
        return result['labels'][0]
    except Exception as e:
        print(f"Aspect error: {e}")
        return "General"

def calculate_crisis_score(sentiment_intensity: float, reach_factor: float, velocity: float) -> float:
    score = (sentiment_intensity * reach_factor) * velocity
    return min(100.0, max(0.0, score))

def generate_responses(negative_posts: list) -> tuple:
    """Returns (summary, [professional, empathic, witty])"""
    if not negative_posts:
        return ("No negative mentions detected yet.", [
            "We're monitoring the conversation.",
            "We appreciate your engagement!",
            "All clear on the radar! 🛡️"
        ])

    # Smart keyword-based heuristic
    context = " ".join(negative_posts[:20]).lower()
    keywords = set()
    for w in NEGATIVE_WORDS:
        if w in context:
            keywords.add(w)

    # Detect themes
    themes = []
    if any(w in context for w in ["support", "help", "service", "hung up", "hold", "wait"]):
        themes.append("customer support")
    if any(w in context for w in ["price", "expensive", "cost", "cheap", "money", "pay"]):
        themes.append("pricing")
    if any(w in context for w in ["quality", "broken", "defect", "build", "material"]):
        themes.append("product quality")
    if any(w in context for w in ["slow", "speed", "fast", "lag", "loading", "optimization"]):
        themes.append("performance/speed")
    if any(w in context for w in ["crash", "bug", "update", "fix", "broken"]):
        themes.append("software stability")

    theme_str = ", ".join(themes) if themes else ", ".join(list(keywords)[:3]) if keywords else "general dissatisfaction"

    summary = f"Users are primarily expressing frustration about: {theme_str}."

    prof = "Thank you for your feedback. Our team is actively investigating these concerns and will provide an update shortly."
    emp = "We sincerely apologize for the experience you're having. Your concerns matter deeply to us — please DM us so we can make this right."
    wit = "Yikes, that wasn't supposed to happen! 😅 Our best engineers are already on it — expect improvements soon! 🚀"

    return (summary, [prof, emp, wit])
