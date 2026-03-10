import re
import random
try:
    import google.generativeai as genai
    import os
    if os.getenv("GEMINI_API_KEY"):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        HAS_GENAI = True
    else:
        HAS_GENAI = False
except ImportError:
    HAS_GENAI = False

EMOJI_MAP = {
    "🚀": "bullish excited",
    "📉": "falling bad",
    "😡": "angry mad",
    "🙄": "annoyed sarcastic",
    "📦": "package delivery",
    "💨": "fast speed",
    "😍": "love excellent",
    "🔥": "amazing perfect",
    "🗑️": "trash garbage"
}

NEGATIVE_WORDS = ["garbage", "fail", "broken", "worst", "trash", "slow", "terrible", "bad", "awful", "cheap", "sucks"]
ASPECT_CATEGORIES = ["Product Quality", "Price", "Customer Support", "Speed"]

def preprocess_text(text: str) -> str:
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text, flags=re.MULTILINE)
    # Map emojis
    for emoji, meaning in EMOJI_MAP.items():
        if emoji in text:
            text = text.replace(emoji, f" {meaning} ")
    # Clean excessive whitespace
    text = " ".join(text.split())
    return text

def analyze_sentiment(text: str, sentiment_model) -> tuple:
    if not text.strip():
        return ("Neutral", 0.0)
    
    try:
        result = sentiment_model(text)[0]
        label = result['label']
        score = result['score']
        
        # Sarcasm Layer
        if label.lower() == 'positive' or label.lower() == 'neutral':
            lower_text = text.lower()
            negative_word_count = sum(1 for word in NEGATIVE_WORDS if word in lower_text)
            if negative_word_count >= 1 and ("🙄" in text or "sure" in lower_text or "great" in lower_text):
                # Heuristic for sarcasm if positive but has negative context words or eyeroll
                label = "Sarcastic/Critical"
            elif negative_word_count >= 2:
                label = "Sarcastic/Critical"
        
        if label.lower() == 'neutral':
            label = "Informational"

        # Capitolize correctly
        if label.lower() == 'positive': label = 'Positive'
        if label.lower() == 'negative': label = 'Negative'
            
        return (label, score)
    except Exception as e:
        print(f"Sentiment error: {e}")
        return ("Neutral", 0.0)

def detect_aspect(text: str, aspect_model, sentiment_label: str) -> str:
    if not text.strip():
        return "Unknown"
    
    # If the user is just saying hi or something irrelevant
    if len(text.split()) < 3 and sentiment_label in ["Positive", "Informational"]:
         return "General"

    try:
        result = aspect_model(text, ASPECT_CATEGORIES)
        # return the top category
        return result['labels'][0]
    except Exception as e:
        print(f"Aspect error: {e}")
        return "Unknown"

def calculate_crisis_score(sentiment_intensity: float, reach_factor: float, velocity: float) -> float:
    # R = (Sentiment_Intensity * Reach_Factor) * Velocity
    # Bound it between 0 and 100
    score = (sentiment_intensity * reach_factor) * velocity
    return min(100.0, max(0.0, score))

def generate_responses(negative_posts: list) -> tuple:
    """Returns (summary, [professional, empathic, witty])"""
    if not negative_posts:
        return ("No negative mentions to summarize.", ["", "", ""])
    
    context = " ".join(negative_posts[:10])
    
    if HAS_GENAI:
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"Summarize the last few negative mentions into a 1-sentence root cause based on: {context}"
            summary_resp = model.generate_content(prompt)
            summary = summary_resp.text.strip()
            
            resp_prompt = f"Based on this issue: '{summary}', write 3 short responses. \n1) Professional\n2) Empathic\n3) Brand-Witty\nFormat: just the 3 responses separated by ||"
            reply_resp = model.generate_content(resp_prompt)
            replies = reply_resp.text.split("||")
            if len(replies) != 3:
                replies = ["We apologize for the inconvenience and are working on it.", "We hear your frustration and want to make it right.", "Whoops! Our bad. We're on it! 🚀"]
            return (summary, [r.strip() for r in replies])
        except Exception as e:
            print(f"GenAI error: {e}")
            pass

    # Fallback if no LLM
    keywords = set()
    for w in NEGATIVE_WORDS:
        if w in context.lower():
            keywords.add(w)
    
    kb_str = ", ".join(list(keywords)[:3]) if keywords else "general issues"
    summary = f"Users are frequently complaining about matters related to: {kb_str}."
    
    prof = "Thank you for bringing this to our attention. Our team is investigating."
    emp = "We are so sorry to hear you're experiencing this. Please DM us so we can resolve this immediately."
    wit = "Oof, that wasn't supposed to happen! We're sending our best engineers to fix it right now 🏃‍♂️💨"
    
    return (summary, [prof, emp, wit])
