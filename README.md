# 💬 WhatsApp Chat Analyzer

> **MindCareX · Module 1 of 4**  
> Upload a WhatsApp export → get sentiment, risk flags, activity patterns, and word frequency in one API call.

---

## Table of Contents

- [What It Does](#what-it-does)
- [How It Works](#how-it-works)
- [File Structure](#file-structure)
- [API Endpoints](#api-endpoints)
- [Request & Response Examples](#request--response-examples)
- [Sentiment Engine](#sentiment-engine)
- [Risk Detection](#risk-detection)
- [Setup & Run](#setup--run)
- [Environment Variables](#environment-variables)
- [Limits & Notes](#limits--notes)

---

## What It Does

A doctor uploads a patient's WhatsApp chat export (`.txt` file). The backend parses every message and returns:

| Output | Description |
|--------|-------------|
| 📊 **Stats** | Total messages, words, media shared, links |
| 😊 **Sentiment** | Positive / negative / neutral label + 0–100 score per sender |
| 📈 **Timeline** | Daily and monthly message volume trends |
| 🕐 **Activity** | Most active hour of day + day of week |
| ☁️ **Word frequency** | Top 50 words (stopwords removed) |
| 🎭 **Emoji analysis** | Top 20 emojis with counts |
| ⚠️ **Risk flags** | Keyword-based risk detection across 3 severity levels |

---

## How It Works

```
WhatsApp .txt export
        │
        ▼
┌───────────────────┐
│   preprocessor.py │  ← parse date / time / sender / message
│   (Pandas)        │  ← handles Android + iOS formats
└────────┬──────────┘
         │
         ▼
┌───────────────────────────────────────────┐
│          chat_analysis_service.py         │
│                                           │
│  sentiment_service.py                     │
│   ├── VADER  → compound score (−1 to +1)  │
│   └── TextBlob → polarity + subjectivity  │
│                                           │
│  Risk scan → keyword match (3 levels)     │
│  Activity  → group by hour / day / month  │
│  Word freq → Counter + stopword filter    │
│  Emoji     → emoji.EMOJI_DATA lookup      │
└────────┬──────────────────────────────────┘
         │
         ▼
  session_store.py  ← save result to Neon DB (or memory in dev)
         │
         ▼
  JSON response  → Lovable frontend
```

---

## File Structure

```
services/
├── preprocessor.py          # WhatsApp .txt → Pandas DataFrame
├── chat_analysis_service.py # All analysis logic lives here
├── sentiment_service.py     # VADER + TextBlob scoring
└── session_store.py         # Neon-backed session persistence

api/routes/
└── analysis.py              # All 9 HTTP endpoints (Flask Blueprint)
```

---

## API Endpoints

Base URL: `https://mindcarex-backend.onrender.com`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/analysis/chat/analyze` | Upload `.txt` → full analysis |
| `GET` | `/api/analysis/chat/{session_id}` | Fetch cached result |
| `GET` | `/api/analysis/chat/{session_id}/sentiment-timeline` | Daily sentiment trend |
| `GET` | `/api/analysis/chat/{session_id}/stats` | Counts only (fast) |
| `GET` | `/api/analysis/chat/{session_id}/risk` | Risk flags only |
| `GET` | `/api/analysis/chat/{session_id}/participants` | List of senders |
| `POST` | `/api/analysis/chat/realtime` | Single message instant analysis |
| `GET` | `/api/analysis/chat/patient/{patient_id}/history` | All sessions for a patient |

> 💡 **Save the `session_id`** from the first response — every other endpoint needs it.

---

## Request & Response Examples

### Upload & Analyze

```http
POST /api/analysis/chat/analyze
Content-Type: multipart/form-data

file = chat_export.txt      ← required
user = "Overall"            ← optional, filter to one sender
```

**Response:**

```json
{
  "success": true,
  "data": {
    "session_id": "3f9a2c1d-...",
    "stats": {
      "total_messages": 512,
      "total_words": 4280,
      "media_shared": 34,
      "links_shared": 7
    },
    "sentiment": {
      "aggregate": {
        "overall_label": "positive",
        "score_0_100": 72.4,
        "avg_compound": 0.448,
        "positive_msgs": 310,
        "negative_msgs": 48,
        "neutral_msgs": 154,
        "positivity_ratio": 0.605
      },
      "per_sender": [
        { "sender": "Alice", "overall_label": "positive", "score_0_100": 78.1 },
        { "sender": "Bob",   "overall_label": "neutral",  "score_0_100": 61.3 }
      ]
    },
    "risk": {
      "risk_level": "low",
      "risk_flags": { "high": [], "medium": [], "low": ["stressed"] },
      "total_flagged": 1
    },
    "top_words": [
      { "word": "meeting", "count": 42 },
      { "word": "tomorrow", "count": 31 }
    ],
    "top_emojis": [
      { "emoji": "😂", "count": 28 },
      { "emoji": "👍", "count": 19 }
    ],
    "week_activity": [
      { "day": "Monday", "count": 120 },
      { "day": "Tuesday", "count": 98 }
    ],
    "hour_activity": [
      { "hour": 9,  "count": 65 },
      { "hour": 21, "count": 88 }
    ],
    "participants": ["Alice", "Bob"],
    "monthly_timeline": [
      { "period": "January-2024", "count": 180 }
    ]
  }
}
```

---

### Realtime Single Message

```http
POST /api/analysis/chat/realtime
Content-Type: application/json

{
  "message": "I feel really anxious and can't sleep"
}
```

**Response:**

```json
{
  "success": true,
  "data": {
    "message": "I feel really anxious and can't sleep",
    "analysis": {
      "label": "negative",
      "score": 28.5,
      "vader_compound": -0.430,
      "vader_pos": 0.0,
      "vader_neg": 0.321,
      "vader_neu": 0.679,
      "textblob_polarity": -0.25,
      "textblob_subjectivity": 0.60
    }
  }
}
```

---

### Sentiment Timeline

```http
GET /api/analysis/chat/{session_id}/sentiment-timeline
```

```json
{
  "success": true,
  "data": {
    "session_id": "3f9a2c1d-...",
    "timeline": [
      { "date": "2024-01-15", "avg_sentiment": 68.2 },
      { "date": "2024-01-16", "avg_sentiment": 45.1 },
      { "date": "2024-01-17", "avg_sentiment": 71.9 }
    ]
  }
}
```

---

## Sentiment Engine

Two libraries run independently on every message, then results are combined:

### VADER *(primary)*
- Built specifically for **short social text** — works well with slang, emoji, abbreviations
- Knows `NOT good` is negative even though `good` alone is positive
- Understands `GREAT!!!` is more intense than `great`
- Returns `compound` score from **−1.0 to +1.0**

| Compound Range | Label |
|---------------|-------|
| `>= +0.05` | `positive` |
| `<= −0.05` | `negative` |
| Between | `neutral` |

The compound score is converted to **0–100** for UI display:
```
score_0_100 = (compound + 1) / 2 × 100
```

### TextBlob *(secondary)*
- General-purpose NLP scoring
- Returns `polarity` (−1 to +1) and `subjectivity` (0 = fact, 1 = pure opinion)
- Both values included in the response for charts

---

## Risk Detection

The full chat text is scanned for keyword matches across three severity levels:

```
High   → kill, threat, hurt, die, abuse, attack, danger, suicide, murder
Medium → angry, fight, argument, frustrated, scam, lie, cheat, ignore
Low    → sad, worried, stressed, tired, alone, cry, miss, sorry
```

**`risk_level`** in the response is the highest severity that triggered:

```python
# Logic:
if any HIGH keywords found    → risk_level = "high"
elif any MEDIUM keywords found → risk_level = "medium"
else                           → risk_level = "low"
```

> ⚠️ This is keyword-based — it flags presence, not context. A message like  
> *"I told him not to kill the music"* would still trigger the HIGH flag.  
> Always treat as a signal to review, not a clinical diagnosis.

---

## Setup & Run

### Local Development

```bash
# 1. Clone and enter project
cd mindcarex-flask-backend

# 2. Create virtualenv
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
python -m textblob.download_corpora

# 4. Copy env file
cp .env.example .env

# 5. Start server
python app.py
# → REST: http://localhost:5000
```

### Test this module

```bash
# Health check
curl http://localhost:5000/health

# Analyze a chat (replace with a real .txt path)
curl -X POST http://localhost:5000/api/analysis/chat/analyze \
  -F "file=@sample_chat.txt" \
  -F "user=Overall"

# Single message
curl -X POST http://localhost:5000/api/analysis/chat/realtime \
  -H "Content-Type: application/json" \
  -d '{"message": "I am feeling much better today!"}'
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_ENV` | `development` | Set to `production` on Render |
| `SECRET_KEY` | `dev-secret` | Must be set to random hex in prod |
| `DATABASE_URL` | _(empty)_ | Neon PostgreSQL connection string |
| `MAX_UPLOAD_MB` | `25` | Max file upload size |
| `SESSION_TTL_MINUTES` | `120` | How long session results are cached |
| `UPLOAD_FOLDER` | `uploads/` | Where `.txt` files are saved |

> In production on Render, `UPLOAD_FOLDER` is automatically set to `/tmp/mindcarex/uploads/`  
> (Render has an ephemeral filesystem — files don't survive redeploys, only session data in Neon persists).

---

## Limits & Notes

- **Session TTL** — results expire after 120 minutes. Re-upload to refresh.
- **Sentiment cap** — scored on first 300 messages for speed. Per-sender capped at 100.
- **File format** — only `.txt` WhatsApp exports work. Both Android and iOS export formats are supported.
- **Media messages** — `<Media omitted>` lines are counted but skipped for sentiment.
- **System messages** — group notifications and LTR-mark system messages are filtered out automatically.
- **Language** — VADER and TextBlob work best with **English**. Hindi/Telugu messages will still be parsed but sentiment accuracy drops significantly.

---

## Dependencies

```
vaderSentiment==3.3.2   # primary sentiment engine
textblob==0.18.0        # secondary sentiment + subjectivity
pandas==2.2.2           # message table, groupby, timelines
emoji==2.12.1           # emoji detection from Unicode data
urlextract==1.9.0       # URL counting in messages
numpy==1.26.4           # numerical ops (via pandas)
```

---

## Part of MindCareX

| Module | Description | Status |
|--------|-------------|--------|
| 💬 **Chat Analyzer** | WhatsApp sentiment, risk, patterns | ✅ This module |
| 🎭 **Live Emotion** | Real-time webcam emotion via DeepFace + WebSocket | ✅ Built |
| 🎙️ **Voice Stress** | Acoustic stress scoring via librosa | ✅ Built |
| 📋 **Session Summary** | Groq Whisper transcription + LLaMA3 clinical notes | ✅ Built |

---

*MindCareX Platform · Flask Backend v2.0 · Deployed on Render*
