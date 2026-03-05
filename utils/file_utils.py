"""
utils/file_utils.py
Secure file upload handling.
"""

import os
import uuid
from pathlib import Path
from config import settings


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in settings.ALLOWED_EXTENSIONS


def save_upload(file_bytes: bytes, original_filename: str) -> tuple[str, str]:
    """Save uploaded bytes, return (session_id, filepath)."""
    if not original_filename:
        raise ValueError("No filename provided")
    if not allowed_file(original_filename):
        raise ValueError("Only .txt WhatsApp exports are supported")

    session_id = str(uuid.uuid4())
    folder     = Path(settings.UPLOAD_FOLDER)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{session_id}.txt"
    path.write_bytes(file_bytes)
    return session_id, str(path)


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()
