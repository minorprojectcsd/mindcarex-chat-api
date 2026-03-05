# ---------- Base Image ----------
FROM python:3.10-slim

# ---------- Environment ----------
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---------- Working Directory ----------
WORKDIR /app

# ---------- System Dependencies ----------
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---------- Copy Dependency File ----------
COPY requirements.txt .

# ---------- Install Python Dependencies ----------
RUN pip install --no-cache-dir -r requirements.txt

# ---------- Download TextBlob Data ----------
RUN python -m textblob.download_corpora

# ---------- Copy Application Code ----------
COPY . .

# ---------- Expose Port ----------
EXPOSE 8001

# ---------- Start Server ----------
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8001", "--workers", "2"]