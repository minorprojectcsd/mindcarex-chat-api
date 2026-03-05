import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    APP_ENV: str             = os.getenv("APP_ENV", "development")
    DEBUG: bool              = os.getenv("DEBUG", "True") == "True"
    SECRET_KEY: str          = os.getenv("SECRET_KEY", "dev-secret-123")
    MAX_UPLOAD_MB: int       = int(os.getenv("MAX_UPLOAD_MB", 10))
    MAX_CONTENT_BYTES: int   = MAX_UPLOAD_MB * 1024 * 1024
    UPLOAD_FOLDER: str       = os.getenv("UPLOAD_FOLDER", "uploads/")
    SESSION_EXPIRY: int      = int(os.getenv("SESSION_EXPIRY_MINUTES", 60))
    ALLOWED_EXTENSIONS: set  = {"txt"}

settings = Settings()
