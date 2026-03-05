"""
models/analysis_model.py
Medical-grade WhatsApp chat analysis.
No external deps beyond pandas/numpy. urlextract replaced with regex.
All NLP packages (vader, textblob, emoji) are optional.
"""

import re
import statistics
from collections import Counter
import pandas as pd
import numpy as np

# ── URL counter (pure regex — no urlextract needed) ───────────────────────────
_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.I)
def _count_urls(text: str) -> int:
    return len(_URL_RE.findall(str(text)))

# ── Safe emoji import ─────────────────────────────────────────────────────────
try:
    import emoji as _emoji_lib
    _EMOJI_OK = True
except ImportError:
    _EMOJI_OK = False

# ── Sentiment imports ─────────────────────────────────────────────────────────
try:
    from models.sentiment_model import analyze_message as _analyze_msg
    from models.sentiment_model import aggregate_sentiment as _agg_sentiment
    _SENTIMENT_OK = True
except Exception:
    _SENTIMENT_OK = False

def _safe_analyze(text: str) -> dict:
    if _SENTIMENT_OK:
        return _analyze_msg(text)
    return {"score_0_100": 50.0, "label": "neutral", "tone": "neutral",
            "ensemble_compound": 0.0, "confidence": 0.0}

def _safe_agg(scores: list) -> dict:
    if _SENTIMENT_OK and scores:
        return _agg_sentiment(scores)
    n = len(scores) if scores else 0
    return {
        "overall_label": "neutral", "avg_compound": 0.0,
        "avg_score_0_100": 50.0, "avg_confidence": 0.0,
        "positive_msgs": 0, "negative_msgs": 0, "neutral_msgs": n,
        "positivity_ratio": 0.0, "negativity_ratio": 0.0,
        "sentiment_volatility": 0.0, "tone_distribution": {}, "sample_size": n,
    }

# ── Stop words ────────────────────────────────────────────────────────────────
STOP_WORDS = {
    "the","and","for","that","this","with","are","was","you","your","have",
    "from","they","will","been","has","had","not","but","what","all","were",
    "when","can","said","there","which","she","him","her","its","also","than",
    "then","more","just","into","some","would","like","about","know","don",
    "okay","yeah","yes","lol","hey","hello","haha","ok","hi","got","get",
    "did","going","come","let","now","one","out","up","so","it","is","in",
    "of","to","a","i","he","we","our","an","my","do","how","who","why","at",
    "be","me","on","if","or","as","no","by","his","over","them","after",
    "their","use","two","way","each","time","day","lmk","omg","msg","msgs",
    "am","pm","re","ll","ve","t","s","m","d","ya","na","ka","hai","nahi",
    "bhi","kya","aur","toh","ko","ke","ki","se","mera","tera","yaar",
}

# ── Medical / Mental health keyword markers ───────────────────────────────────
_POSITIVE_MH = [
    "feeling better","much better","getting better","doing well","thank you",
    "appreciate","grateful","proud","looking forward","excited","support",
    "love you","miss you","happy","wonderful","amazing",
]
_NEGATIVE_MH = [
    "not okay","really bad","feel terrible","feel awful","give up",
    "whats the point","what's the point","no one cares","dont matter",
    "so tired","exhausted","drained","empty inside","cant feel","can't feel",
    "pointless","meaningless","worthless","hopeless","depressed",
]


# ═════════════════════════════════════════════════════════════════════════════
# BASIC STATS
# ═════════════════════════════════════════════════════════════════════════════

def fetch_stats(df: pd.DataFrame, user: str = "Overall") -> dict:
    sub = df if user == "Overall" else df[df["user"] == user]
    n   = len(sub)
    if n == 0:
        return {}
    words   = int(sub["word_count"].sum())
    media   = int(sub["is_media"].sum())
    deleted = int(sub["is_deleted"].sum())
    links   = int(sub["message"].apply(_count_urls).sum())
    emojis  = int(sub["emoji_count"].sum()) if "emoji_count" in sub.columns else 0
    d_range = (sub["date"].max() - sub["date"].min()).days if n > 1 else 0

    return {
        "total_messages":    n,
        "total_words":       words,
        "media_shared":      media,
        "links_shared":      links,
        "deleted_messages":  deleted,
        "total_emojis":      emojis,
        "avg_words_per_msg": round(words / n, 2),
        "messages_per_day":  round(n / d_range, 2) if d_range else n,
        "date_range_days":   d_range,
    }


def get_participants(df: pd.DataFrame) -> list:
    return sorted(df["user"].unique().tolist())


# ═════════════════════════════════════════════════════════════════════════════
# MEDICAL MENTAL HEALTH PROFILE
# ═════════════════════════════════════════════════════════════════════════════

def mental_health_profile(df: pd.DataFrame) -> dict:
    """
    Per-person mental health indicators derived from chat patterns.
    Returns a profile that a clinician can interpret directly.
    """
    text_df = df[~df["is_media"]].copy()
    profiles = []

    for sender in df["user"].unique():
        s_df    = text_df[text_df["user"] == sender]
        s_msgs  = s_df["message"].tolist()
        s_text  = " ".join(s_msgs).lower()
        n       = len(s_msgs)
        if n == 0:
            continue

        # Positive vs negative MH language count
        pos_hits = [p for p in _POSITIVE_MH if p in s_text]
        neg_hits = [p for p in _NEGATIVE_MH if p in s_text]

        # Emotional score: 0 (very negative) to 100 (very positive)
        raw_mh = 50 + (len(pos_hits) * 4) - (len(neg_hits) * 6)
        mh_score = max(0, min(100, raw_mh))

        # Late night messaging (possible sleep disturbance)
        late_night = int(s_df[s_df["hour"].between(0, 4)]["hour"].count())
        late_night_pct = round(late_night / n * 100, 1) if n else 0

        # Message length patterns (very short = disengaged, very long = distressed outpouring)
        avg_words = round(s_df["word_count"].mean(), 1) if n else 0

        # Deleted messages (possible regret / impulsive messaging)
        deleted = int(s_df["is_deleted"].sum())

        # Sentiment
        sample  = s_msgs[:200]
        scores  = [_safe_analyze(m) for m in sample]
        agg     = _safe_agg(scores)

        # Interpretation
        flags = []
        if mh_score < 35:
            flags.append("High volume of distress language detected")
        if late_night_pct > 20:
            flags.append(f"Frequent late-night messaging ({late_night_pct}% of messages between 12AM–5AM) — possible sleep disturbance")
        if agg.get("negativity_ratio", 0) > 0.5:
            flags.append("More than half of messages carry negative sentiment")
        if agg.get("sentiment_volatility", 0) > 0.4:
            flags.append("High emotional volatility — sentiment fluctuates significantly")
        if deleted > 5:
            flags.append(f"{deleted} deleted messages — possible impulsive communication or regret")
        if avg_words < 3 and n > 20:
            flags.append("Very short messages on average — may indicate emotional withdrawal or disengagement")

        profiles.append({
            "person":                 sender,
            "mental_health_score":    mh_score,
            "mh_score_interpretation": (
                "Positive" if mh_score >= 65 else
                "Moderate concern" if mh_score >= 40 else
                "Elevated concern"
            ),
            "positive_language_signals": pos_hits,
            "negative_language_signals": neg_hits,
            "late_night_messages":       late_night,
            "late_night_percent":        late_night_pct,
            "avg_words_per_message":     avg_words,
            "deleted_messages":          deleted,
            "sentiment_overview":        agg,
            "clinical_flags":            flags,
            "total_messages_analysed":   n,
        })

    profiles.sort(key=lambda x: x["mental_health_score"])
    return {"per_person": profiles}


# ═════════════════════════════════════════════════════════════════════════════
# SENTIMENT
# ═════════════════════════════════════════════════════════════════════════════

def sentiment_analysis(df: pd.DataFrame, user: str = "Overall") -> dict:
    sub       = df if user == "Overall" else df[df["user"] == user]
    text_msgs = sub[~sub["is_media"]]["message"].tolist()[:500]
    scores    = [_safe_analyze(m) for m in text_msgs]
    agg       = _safe_agg(scores)

    per_sender = []
    for u in df["user"].unique():
        u_msgs   = df[(df["user"] == u) & (~df["is_media"])]["message"].tolist()[:150]
        u_scores = [_safe_analyze(m) for m in u_msgs]
        per_sender.append({"sender": u, **_safe_agg(u_scores)})

    return {"aggregate": agg, "per_sender": per_sender}


def sentiment_timeline(df: pd.DataFrame) -> list:
    df = df[~df["is_media"]].copy()
    df["score"] = df["message"].apply(lambda m: _safe_analyze(m)["score_0_100"])
    tl = df.groupby("only_date")["score"].mean().reset_index()
    return [
        {"date": str(r["only_date"]), "avg_sentiment": round(r["score"], 1)}
        for _, r in tl.iterrows()
    ]


# ═════════════════════════════════════════════════════════════════════════════
# ACTIVITY
# ═════════════════════════════════════════════════════════════════════════════

def monthly_timeline(df: pd.DataFrame, user: str = "Overall") -> list:
    sub = df if user == "Overall" else df[df["user"] == user]
    tl  = sub.groupby(["year","month_num","month"]).size().reset_index(name="count")
    tl  = tl.sort_values(["year","month_num"])
    tl["period"] = tl["month"] + "-" + tl["year"].astype(str)
    return tl[["period","count"]].to_dict(orient="records")


def daily_timeline(df: pd.DataFrame, user: str = "Overall") -> list:
    sub = df if user == "Overall" else df[df["user"] == user]
    d   = sub.groupby("only_date").size().reset_index(name="count")
    d["date"] = d["only_date"].astype(str)
    return d[["date","count"]].to_dict(orient="records")


def week_activity(df: pd.DataFrame, user: str = "Overall") -> list:
    sub   = df if user == "Overall" else df[df["user"] == user]
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    counts = sub["day_name"].value_counts().reindex(order, fill_value=0)
    return [{"day": d, "count": int(c)} for d, c in counts.items()]


def hour_activity(df: pd.DataFrame, user: str = "Overall") -> list:
    sub   = df if user == "Overall" else df[df["user"] == user]
    valid = sub[sub["hour"] >= 0]
    counts = valid["hour"].value_counts().sort_index()
    return [{"hour": int(h), "count": int(c)} for h, c in counts.items()]


def heatmap_7x24(df: pd.DataFrame, user: str = "Overall") -> list:
    sub  = (df if user == "Overall" else df[df["user"] == user])
    sub  = sub[sub["hour"] >= 0]
    days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    return [
        {"day": day, "hours": {
            str(h): int(sub[sub["day_name"] == day]["hour"].value_counts().get(h, 0))
            for h in range(24)
        }} for day in days
    ]


# ═════════════════════════════════════════════════════════════════════════════
# USERS / WORDS / EMOJIS
# ═════════════════════════════════════════════════════════════════════════════

def most_active_users(df: pd.DataFrame) -> list:
    top   = df["user"].value_counts().head(10)
    total = len(df)
    return [
        {"user": u, "messages": int(c), "percent": round(c / total * 100, 2)}
        for u, c in top.items()
    ]


def word_cloud_data(df: pd.DataFrame, user: str = "Overall", top_n: int = 60) -> list:
    sub  = df if user == "Overall" else df[df["user"] == user]
    text = " ".join(sub[~sub["is_media"]]["message"].tolist()).lower()
    words = re.findall(r"\b[a-z]{3,}\b", text)
    freq  = Counter(w for w in words if w not in STOP_WORDS)
    return [{"word": w, "count": c} for w, c in freq.most_common(top_n)]


def vocabulary_richness(df: pd.DataFrame) -> list:
    result = []
    for user in df["user"].unique():
        sub  = df[(df["user"] == user) & (~df["is_media"])]
        text = " ".join(sub["message"].tolist()).lower()
        words = re.findall(r"\b[a-z]{3,}\b", text)
        meaningful = [w for w in words if w not in STOP_WORDS]
        if not meaningful:
            continue
        unique = len(set(meaningful))
        result.append({
            "sender":            user,
            "total_words":       len(meaningful),
            "unique_words":      unique,
            "lexical_diversity": round(unique / len(meaningful), 4),
        })
    return sorted(result, key=lambda x: x["lexical_diversity"], reverse=True)


def emoji_analysis(df: pd.DataFrame, user: str = "Overall") -> list:
    if not _EMOJI_OK:
        return []
    sub   = df if user == "Overall" else df[df["user"] == user]
    all_e = [ch for msg in sub["message"] for ch in str(msg) if ch in _emoji_lib.EMOJI_DATA]
    freq  = Counter(all_e)
    return [{"emoji": e, "count": c} for e, c in freq.most_common(20)]


# ═════════════════════════════════════════════════════════════════════════════
# PATTERNS
# ═════════════════════════════════════════════════════════════════════════════

def response_time_stats(df: pd.DataFrame) -> list:
    rt = df.dropna(subset=["response_time_min"])
    result = []
    for user in df["user"].unique():
        u_rt = rt[rt["user"] == user]["response_time_min"]
        if len(u_rt) < 2:
            continue
        result.append({
            "sender":               user,
            "avg_response_min":     round(float(u_rt.mean()), 2),
            "median_response_min":  round(float(u_rt.median()), 2),
            "fastest_response_min": round(float(u_rt.min()), 2),
        })
    return sorted(result, key=lambda x: x["avg_response_min"])


def conversation_initiator(df: pd.DataFrame) -> list:
    df = df.sort_values("date").copy()
    df["gap_min"]      = (df["date"].diff().dt.total_seconds() / 60).fillna(9999)
    df["is_initiator"] = df["gap_min"] > 60
    counts = df[df["is_initiator"]]["user"].value_counts()
    total  = counts.sum()
    return [
        {"sender": u, "initiations": int(c), "percent": round(c / total * 100, 2)}
        for u, c in counts.items()
    ]


def silent_periods(df: pd.DataFrame, gap_hours: int = 24) -> list:
    df = df.sort_values("date").copy()
    df["gap_h"] = (df["date"].diff().dt.total_seconds() / 3600)
    result = []
    for _, row in df[df["gap_h"] > gap_hours].iterrows():
        prev = row["date"] - pd.Timedelta(hours=row["gap_h"])
        result.append({
            "from": str(prev.date()), "to": str(row["date"].date()),
            "gap_hours": round(row["gap_h"], 1),
        })
    return result[:20]


def late_night_stats(df: pd.DataFrame) -> dict:
    late = df[(df["hour"] >= 0) & (df["hour"] <= 4)]
    return {
        "total_late_night_msgs": int(len(late)),
        "per_sender": [
            {"sender": k, "count": int(v)}
            for k, v in late["user"].value_counts().to_dict().items()
        ],
    }


# ═════════════════════════════════════════════════════════════════════════════
# FULL BUNDLE
# ═════════════════════════════════════════════════════════════════════════════

def full_analysis(df: pd.DataFrame, user: str = "Overall") -> dict:
    from models.risk_model import risk_analysis
    return {
        "stats":               fetch_stats(df, user),
        "participants":        get_participants(df),
        "most_active":         most_active_users(df),
        "mental_health":       mental_health_profile(df),
        "sentiment":           sentiment_analysis(df, user),
        "monthly_timeline":    monthly_timeline(df, user),
        "daily_timeline":      daily_timeline(df, user),
        "week_activity":       week_activity(df, user),
        "hour_activity":       hour_activity(df, user),
        "heatmap_7x24":        heatmap_7x24(df, user),
        "top_words":           word_cloud_data(df, user),
        "top_emojis":          emoji_analysis(df, user),
        "vocabulary_richness": vocabulary_richness(df),
        "response_time":       response_time_stats(df),
        "initiator_stats":     conversation_initiator(df),
        "silent_periods":      silent_periods(df),
        "late_night_stats":    late_night_stats(df),
        "risk":                risk_analysis(df),
    }
