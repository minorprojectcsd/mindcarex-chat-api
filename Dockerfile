# Base image
FROM python:3.10-slim

# Environment settings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Working directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download TextBlob corpora
RUN python -m textblob.download_corpora

# Copy application code
COPY . .

# Expose port
EXPOSE 8001

# Start FastAPI with Gunicorn
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--timeout", "120", "--workers", "1", "--bind", "0.0.0.0:8001"]