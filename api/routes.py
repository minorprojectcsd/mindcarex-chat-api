"""
api/routes.py — All working chat analysis endpoints.
No stubs. No future placeholders. Only production routes.
"""

import pandas as pd
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from config import settings
from utils.file_utils import save_upload, read_file, allowed_file
from models.preprocessor import preprocess
from models.analysis_model import (
    full_analysis, fetch_stats, get_participants,
    sentiment_timeline, word_cloud_data, emoji_analysis,
    most_active_users, response_time_stats, mental_health_profile,
)
from models.risk_model import risk_analysis
from models.sentiment_model import analyze_message
from services import session_store

router = APIRouter(prefix="/chat", tags=["Chat Analysis"])


# ── Load df from session ──────────────────────────────────────────────────────
def _load_df(session_id: str) -> pd.DataFrame:
    data = session_store.get(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    df = pd.read_json(data["df_json"], orient="records")
    df["is_media"]   = df["is_media"].astype(bool)
    df["is_deleted"] = df["is_deleted"].astype(bool)
    df["only_date"]  = pd.to_datetime(df["only_date"]).dt.date
    df["date"]       = pd.to_datetime(df["date"])
    return df


# ════════════════════════════════════════════════════════════════════════════
# POST /api/analysis/chat/analyze
# Main endpoint — upload .txt, get full analysis back
# ════════════════════════════════════════════════════════════════════════════
@router.post("/analyze", summary="Upload WhatsApp .txt and run full medical analysis")
async def analyze(
    file: UploadFile = File(..., description="WhatsApp exported .txt file"),
    user: str = Form("Overall", description="Sender name to filter, or 'Overall'"),
):
    if not allowed_file(file.filename or ""):
        raise HTTPException(status_code=400, detail="Only .txt WhatsApp exports are supported")

    raw_bytes = await file.read()
    if len(raw_bytes) > settings.MAX_CONTENT_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.MAX_UPLOAD_MB}MB limit")

    try:
        session_id, path = save_upload(raw_bytes, file.filename or "chat.txt")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        raw_text = read_file(path)
        df       = preprocess(raw_text)
        result   = full_analysis(df, user)

        session_store.save(session_id, {
            "df_json": df.to_json(orient="records", date_format="iso"),
            "result":  result,
            "user":    user,
        })

        return {"success": True, "session_id": session_id, "user": user, **result}

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


# ════════════════════════════════════════════════════════════════════════════
# GET /api/analysis/chat/{session_id}
# ════════════════════════════════════════════════════════════════════════════
@router.get("/{session_id}", summary="Get full cached analysis by session ID")
def get_analysis(session_id: str):
    data = session_store.get(session_id)
    if not data:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return {"success": True, "session_id": session_id, **data["result"]}


# ════════════════════════════════════════════════════════════════════════════
# GET drill-downs
# ════════════════════════════════════════════════════════════════════════════
@router.get("/{session_id}/stats", summary="Message counts and basic statistics")
def get_stats(session_id: str, user: str = "Overall"):
    df = _load_df(session_id)
    return {"success": True, **fetch_stats(df, user)}


@router.get("/{session_id}/risk", summary="Risk analysis — who is at risk, who poses risk to others")
def get_risk(session_id: str):
    df = _load_df(session_id)
    return {"success": True, **risk_analysis(df)}


@router.get("/{session_id}/mental-health", summary="Mental health profile per person")
def get_mental_health(session_id: str):
    df = _load_df(session_id)
    return {"success": True, **mental_health_profile(df)}


@router.get("/{session_id}/sentiment-timeline", summary="Daily average sentiment scores")
def get_sentiment_timeline(session_id: str):
    df       = _load_df(session_id)
    timeline = sentiment_timeline(df)
    return {"success": True, "session_id": session_id, "timeline": timeline}


@router.get("/{session_id}/participants", summary="List all participants in the chat")
def get_participants_route(session_id: str):
    df = _load_df(session_id)
    return {"success": True, "participants": get_participants(df)}


@router.get("/{session_id}/words", summary="Top words for word cloud")
def get_words(session_id: str, user: str = "Overall", top: int = 60):
    df = _load_df(session_id)
    return {"success": True, "words": word_cloud_data(df, user, top)}


@router.get("/{session_id}/emojis", summary="Top emoji usage breakdown")
def get_emojis(session_id: str, user: str = "Overall"):
    df = _load_df(session_id)
    return {"success": True, "emojis": emoji_analysis(df, user)}


@router.get("/{session_id}/response-time", summary="Average response time per sender")
def get_response_time(session_id: str):
    df = _load_df(session_id)
    return {"success": True, "response_time": response_time_stats(df)}


@router.get("/{session_id}/most-active", summary="Most active senders by message count")
def get_most_active(session_id: str):
    df = _load_df(session_id)
    return {"success": True, "most_active": most_active_users(df)}


# ════════════════════════════════════════════════════════════════════════════
# POST /api/analysis/chat/realtime
# Instant single-message analysis — no file upload needed
# ════════════════════════════════════════════════════════════════════════════
class RealtimeRequest(BaseModel):
    message: str

@router.post("/realtime", summary="Instant sentiment analysis for a single message")
def realtime(body: RealtimeRequest):
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="'message' field required")
    result = analyze_message(body.message)
    return {"success": True, "message": body.message, "analysis": result}
