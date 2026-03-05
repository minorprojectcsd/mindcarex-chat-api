# MindCareX — Chat Analysis API
### Production Deployment · Local Run · Frontend Integration

---

## ⚡ Run Locally Right Now

### What you need installed on your machine
```
Python 3.10+
pip
```

### Step 1 — Install packages
```bash
cd module_1_chat
pip install fastapi uvicorn[standard] python-multipart python-dotenv \
            pandas numpy vaderSentiment textblob emoji gunicorn
python -m textblob.download_corpora
```

### Step 2 — Copy environment file
```bash
cp .env.example .env
```

### Step 3 — Start the API
```bash
uvicorn main:app --reload --port 8001
```

### Step 4 — Verify it works
Open in browser:
```
http://localhost:8001/health      → {"status":"ok"}
http://localhost:8001/docs        → Swagger UI — test all endpoints here
```

---

## What Is Currently Running (Complete List)

| # | Package | Purpose | Required? |
|---|---------|---------|-----------|
| 1 | fastapi | REST API framework | Yes |
| 2 | uvicorn | ASGI web server | Yes |
| 3 | python-multipart | File upload parsing | Yes |
| 4 | pandas | Chat data processing | Yes |
| 5 | numpy | Numerical operations | Yes |
| 6 | python-dotenv | .env config loading | Yes |
| 7 | vaderSentiment | Sentiment analysis engine | Recommended |
| 8 | textblob | Sentiment + subjectivity | Recommended |
| 9 | emoji | Emoji detection/counting | Recommended |
| 10 | gunicorn | Production WSGI server | Production only |

Packages 7-9 are optional. Without them the app falls back to keyword-based scoring.
All other features (stats, activity, risk, patterns) work fine without them.

No paid APIs. No external services. Runs 100% offline.

---

## All Active API Endpoints

Base URL (local): http://localhost:8001
Base URL (production): https://your-api.onrender.com

| Method | Endpoint | What it returns |
|--------|----------|----------------|
| GET | /health | Server status check |
| GET | /docs | Swagger UI - interactive API explorer |
| POST | /api/analysis/chat/analyze | Main endpoint. Upload .txt, get full analysis + session_id |
| GET | /api/analysis/chat/{session_id} | Full cached result by session |
| GET | /api/analysis/chat/{session_id}/stats | Message counts, words, media, links, deleted |
| GET | /api/analysis/chat/{session_id}/risk | Who is at risk, who poses risk to others, flagged messages |
| GET | /api/analysis/chat/{session_id}/mental-health | Per-person MH score, clinical flags, sleep signals |
| GET | /api/analysis/chat/{session_id}/sentiment-timeline | Daily average sentiment scores |
| GET | /api/analysis/chat/{session_id}/participants | List of all senders |
| GET | /api/analysis/chat/{session_id}/words | Top words for word cloud |
| GET | /api/analysis/chat/{session_id}/emojis | Emoji frequency |
| GET | /api/analysis/chat/{session_id}/response-time | Average response time per sender |
| GET | /api/analysis/chat/{session_id}/most-active | Most active senders |
| POST | /api/analysis/chat/realtime | Instant analysis of a single message, no file needed |

---

## Deploy Backend to Render

Why Render and not Vercel?
Vercel runs serverless Node.js functions. FastAPI is Python with persistent workers.
Render supports Python natively, gives persistent processes, and has a free tier.
Your Vercel frontend calls your Render backend. This is the standard architecture.

Step 1 - Push to GitHub
```bash
cd module_1_chat
git init && git add . && git commit -m "initial"
git remote add origin https://github.com/YOUR_USERNAME/mindcarex-chat-api.git
git push -u origin main
```

Step 2 - Create Web Service on Render (render.com > New > Web Service)

| Setting | Value |
|---------|-------|
| Runtime | Python 3 |
| Build Command | pip install -r requirements.txt && python -m textblob.download_corpora |
| Start Command | uvicorn main:app --host 0.0.0.0 --port $PORT |
| Instance Type | Free tier or Starter for always-on |

Step 3 - Add environment variables on Render

| Key | Value |
|-----|-------|
| APP_ENV | production |
| DEBUG | False |
| SECRET_KEY | (generate a random string) |
| MAX_UPLOAD_MB | 10 |
| SESSION_EXPIRY_MINUTES | 60 |

Step 4 - Deploy
Your API will be live at: https://mindcarex-chat-api.onrender.com
The render.yaml file in this folder automates all of the above.

---

## Connect Backend to mindcarex.vercel.app

CORS is already configured in main.py to allow:
- https://mindcarex.vercel.app
- http://localhost:3000
- http://localhost:5173

Step 1 - Set API URL in your Vite project

Create src/config.ts:
```typescript
export const API_BASE =
  import.meta.env.VITE_API_URL ?? "http://localhost:8001";
```

In Vercel dashboard > your project > Settings > Environment Variables, add:
```
VITE_API_URL = https://mindcarex-chat-api.onrender.com
```

Step 2 - Upload and analyze a chat

```typescript
// src/api/chatApi.ts
const BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8001";

export async function analyzeChat(file: File, user = "Overall") {
  const form = new FormData();
  form.append("file", file);
  form.append("user", user);

  const res = await fetch(`${BASE}/api/analysis/chat/analyze`, {
    method: "POST",
    body: form,
    // Do NOT set Content-Type - browser sets it with boundary automatically
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.detail ?? "Analysis failed");
  }
  return res.json();
  // Returns: { session_id, stats, risk, mental_health, sentiment, ... }
}
```

Step 3 - Fetch specific sections

```typescript
export async function getRisk(sessionId: string) {
  const res = await fetch(`${BASE}/api/analysis/chat/${sessionId}/risk`);
  return res.json();
}

export async function getMentalHealth(sessionId: string) {
  const res = await fetch(`${BASE}/api/analysis/chat/${sessionId}/mental-health`);
  return res.json();
}

export async function getSentimentTimeline(sessionId: string) {
  const res = await fetch(`${BASE}/api/analysis/chat/${sessionId}/sentiment-timeline`);
  return res.json();
}

export async function analyzeMessage(message: string) {
  const res = await fetch(`${BASE}/api/analysis/chat/realtime`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message }),
  });
  return res.json();
}
```

Step 4 - Understanding the risk response

The risk response is designed to be self-explanatory:

```
result.overall_risk_level              "none"|"low"|"medium"|"high"|"critical"
result.overall_summary                 Plain English: who has what risk
result.persons_at_risk                 ["Alice"] - people who are at risk themselves
result.persons_posing_risk_to_others   ["Bob"] - people threatening others

Per person:
result.per_person_risk[0].person
result.per_person_risk[0].person_is_at_risk            true/false
result.per_person_risk[0].person_poses_risk_to_others  true/false
result.per_person_risk[0].plain_english_summary        human-readable text
result.per_person_risk[0].clinical_notes[].clinical_note
result.per_person_risk[0].flagged_messages[].message_text
result.per_person_risk[0].flagged_messages[].this_message_severity
result.per_person_risk[0].risk_is_escalating           true/false
result.per_person_risk[0].escalation_note              explains why
```

---

## Project Structure

```
module_1_chat/
├── main.py                   FastAPI app + CORS config
├── config.py                 Environment config
├── requirements.txt          Python dependencies
├── .env.example              Environment template
├── Dockerfile                Docker build
├── render.yaml               Render deployment config
│
├── models/
│   ├── preprocessor.py       WhatsApp .txt parser (handles all formats)
│   ├── sentiment_model.py    VADER + TextBlob ensemble sentiment
│   ├── risk_model.py         Medical risk per person with direction
│   └── analysis_model.py     Stats, mental health, patterns
│
├── api/
│   └── routes.py             All FastAPI endpoints
│
├── services/
│   └── session_store.py      In-memory session cache with TTL
│
└── utils/
    ├── file_utils.py         Secure file upload
    └── response.py           JSON response helpers
```
