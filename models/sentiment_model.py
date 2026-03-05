"""
models/sentiment_model.py
Medical-grade sentiment engine for mental health analysis.
VADER + TextBlob ensemble with mental health signal detection.
Gracefully handles missing packages.
"""

import re
import statistics

# ── Graceful imports ──────────────────────────────────────────────────────────
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    _VADER_OK = True
except ImportError:
    _VADER_OK = False
    _vader = None

try:
    from textblob import TextBlob
    _TEXTBLOB_OK = True
except ImportError:
    _TEXTBLOB_OK = False

# ── Emoji sentiment sets ──────────────────────────────────────────────────────
_POS_EMOJI = set("😀😃😄😁😊😇🙂😍🥰😘🤗🥳🎉❤️💕💯👍✅🙏😂🤣")
_NEG_EMOJI = set("😢😭😡🤬😠😤😔😟☹️😞😓😥😰😨😱💔👎❌😶🥺😮‍💨")

# ── Tone keywords ─────────────────────────────────────────────────────────────
_TONE_KEYWORDS = {
    "joy":      ["happy","excited","love","great","wonderful","yay","haha","lol","fun","enjoy","glad","blessed","amazing","fantastic"],
    "anger":    ["hate","angry","furious","irritated","frustrated","mad","annoyed","rage","stupid","idiot","useless","pathetic"],
    "sadness":  ["sad","cry","miss","lonely","depressed","unhappy","hurt","tears","broken","lost","hopeless","heartbroken"],
    "fear":     ["scared","afraid","anxious","worried","nervous","panic","terror","dread","uneasy","terrified","stressed"],
    "disgust":  ["disgusting","gross","sick","nasty","horrible","awful","terrible","yuck","revolting","pathetic"],
}

# ── Mental health indicator patterns ─────────────────────────────────────────
_MH_POSITIVE = [
    "feeling better","getting better","much better","doing well","making progress",
    "thank you for","appreciate you","grateful","proud of","looking forward",
    "excited about","cant wait","happy to","love you","miss you","support",
]
_MH_NEGATIVE = [
    "not okay","not good","really bad","very bad","feel terrible","feel awful",
    "dont know anymore","give up","whats the point","what's the point",
    "no one cares","nobody cares","dont matter","doesn't matter","pointless",
    "so tired","exhausted","drained","empty inside","cant feel","can't feel",
]

_SARCASM_CAPS = re.compile(r'\b[A-Z]{4,}\b')
_SARCASM_PUNCT = re.compile(r'[!?]{2,}')


def _emoji_mod(text: str) -> float:
    pos = sum(1 for c in text if c in _POS_EMOJI)
    neg = sum(1 for c in text if c in _NEG_EMOJI)
    total = pos + neg
    return (pos - neg) / total * 0.2 if total else 0.0


def _sarcasm_penalty(text: str, compound: float) -> float:
    if compound > 0.3:
        if len(_SARCASM_CAPS.findall(text)) >= 2 or len(_SARCASM_PUNCT.findall(text)) >= 2:
            return -0.15
    return 0.0


def _detect_tone(text: str) -> str:
    tl = text.lower()
    scores = {tone: sum(1 for kw in kws if kw in tl) for tone, kws in _TONE_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "neutral"


def _mental_health_signals(text: str) -> dict:
    """Detect positive and negative mental health language patterns."""
    tl = text.lower()
    pos_signals = [p for p in _MH_POSITIVE if p in tl]
    neg_signals = [p for p in _MH_NEGATIVE if p in tl]
    return {
        "positive_signals": pos_signals,
        "negative_signals": neg_signals,
        "mh_score_adjustment": len(pos_signals) * 3 - len(neg_signals) * 5,
    }


def analyze_message(text: str) -> dict:
    """Full sentiment analysis for a single message."""
    if not text or not text.strip():
        return _empty_result()

    # VADER
    if _VADER_OK and _vader:
        v = _vader.polarity_scores(text)
        vader_compound = v["compound"]
        vader_pos, vader_neg, vader_neu = v["pos"], v["neg"], v["neu"]
    else:
        tl = text.lower()
        pos_kws = ["good","great","happy","love","nice","excellent","amazing","thanks","wonderful"]
        neg_kws = ["bad","hate","terrible","awful","sad","angry","horrible","worst","fail","problem"]
        ps = sum(1 for w in pos_kws if w in tl)
        ng = sum(1 for w in neg_kws if w in tl)
        vader_compound = (ps - ng) / max(ps + ng, 1) * 0.5
        vader_pos = ps / max(ps + ng + 1, 1)
        vader_neg = ng / max(ps + ng + 1, 1)
        vader_neu = 1 - vader_pos - vader_neg

    # TextBlob
    if _TEXTBLOB_OK:
        tb = TextBlob(text)
        tb_polarity     = tb.sentiment.polarity
        tb_subjectivity = tb.sentiment.subjectivity
    else:
        tb_polarity     = vader_compound * 0.8
        tb_subjectivity = 0.5

    emoji_mod  = _emoji_mod(text)
    sarcasm_p  = _sarcasm_penalty(text, vader_compound)

    ensemble = (vader_compound * 0.70 + tb_polarity * 0.30) + emoji_mod + sarcasm_p
    ensemble = max(-1.0, min(1.0, ensemble))

    label = "positive" if ensemble >= 0.05 else ("negative" if ensemble <= -0.05 else "neutral")
    tone  = _detect_tone(text)

    same_sign = (
        (vader_compound >= 0.05 and tb_polarity > 0) or
        (vader_compound <= -0.05 and tb_polarity < 0) or
        (abs(vader_compound) < 0.05 and abs(tb_polarity) < 0.1)
    )
    confidence = round(min((abs(vader_compound) * 0.7 + abs(tb_polarity) * 0.3) * (1.2 if same_sign else 0.7), 1.0), 4)

    return {
        "vader_compound":        round(vader_compound, 4),
        "vader_pos":             round(vader_pos, 4),
        "vader_neg":             round(vader_neg, 4),
        "vader_neu":             round(vader_neu, 4),
        "textblob_polarity":     round(tb_polarity, 4),
        "textblob_subjectivity": round(tb_subjectivity, 4),
        "emoji_modifier":        round(emoji_mod, 4),
        "ensemble_compound":     round(ensemble, 4),
        "label":                 label,
        "tone":                  tone,
        "score_0_100":           round((ensemble + 1) / 2 * 100, 1),
        "confidence":            confidence,
    }


def _empty_result() -> dict:
    return {
        "vader_compound": 0.0, "vader_pos": 0.0, "vader_neg": 0.0, "vader_neu": 1.0,
        "textblob_polarity": 0.0, "textblob_subjectivity": 0.0,
        "emoji_modifier": 0.0, "ensemble_compound": 0.0,
        "label": "neutral", "tone": "neutral",
        "score_0_100": 50.0, "confidence": 0.0,
    }


def bulk_sentiment(messages: list) -> list:
    return [analyze_message(m) for m in messages]


def aggregate_sentiment(scores: list) -> dict:
    if not scores:
        return {}
    n = len(scores)
    compounds  = [s["ensemble_compound"] for s in scores]
    avg_c      = sum(compounds) / n
    avg_score  = sum(s["score_0_100"] for s in scores) / n
    avg_conf   = sum(s["confidence"] for s in scores) / n
    label      = "positive" if avg_c >= 0.05 else ("negative" if avg_c <= -0.05 else "neutral")
    pos_c      = sum(1 for s in scores if s["label"] == "positive")
    neg_c      = sum(1 for s in scores if s["label"] == "negative")
    neu_c      = sum(1 for s in scores if s["label"] == "neutral")
    tone_dist  = {}
    for s in scores:
        tone_dist[s["tone"]] = tone_dist.get(s["tone"], 0) + 1
    volatility = round(statistics.stdev(compounds), 4) if n > 1 else 0.0

    return {
        "overall_label":        label,
        "avg_compound":         round(avg_c, 4),
        "avg_score_0_100":      round(avg_score, 1),
        "avg_confidence":       round(avg_conf, 4),
        "positive_msgs":        pos_c,
        "negative_msgs":        neg_c,
        "neutral_msgs":         neu_c,
        "positivity_ratio":     round(pos_c / n, 4),
        "negativity_ratio":     round(neg_c / n, 4),
        "sentiment_volatility": volatility,
        "tone_distribution":    tone_dist,
        "sample_size":          n,
    }
