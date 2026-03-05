"""
main.py — MindCareX Chat Analysis Microservice
FastAPI · Production-ready · CORS configured for mindcarex.vercel.app

Local run:
    uvicorn main:app --reload --port 8001

Production:
    uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from config import settings
from api.routes import router as chat_router

app = FastAPI(
    title="MindCareX — Chat Analysis API",
    description=(
        "WhatsApp chat analysis microservice for MindCareX mental health platform.\n\n"
        "Provides: sentiment analysis, mental health profiling, risk scoring, "
        "activity patterns, word frequency, and conversation insights."
    ),
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS — allow Vercel frontend + local dev ──────────────────────────────────
ALLOWED_ORIGINS = [
    "https://mindcarex.vercel.app",
    "https://www.mindcarex.vercel.app",
    "http://localhost:3000",   # Vite default
    "http://localhost:5173",   # Vite alt
    "http://localhost:4173",   # Vite preview
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(chat_router, prefix="/api/analysis")

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Health"])
def health():
    return {
        "status":  "ok",
        "service": "mindcarex-chat-analysis",
        "version": "2.1.0",
    }

# ── Global error handler ──────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": str(exc)},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8001)),
        reload=settings.DEBUG,
    )
