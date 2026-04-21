FROM python:3.11-slim

# Set working directory (single source of truth)
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached layer)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# NLTK resources (build-time, deterministic image)
RUN python -m nltk.downloader punkt

# Copy application code
COPY app/ ./app/
COPY src/ ./src/

# Copy model into application namespace (IMPORTANT)
COPY models/bioclinicalbert_final/ ./models/bioclinicalbert_final/

# Ensure Python can import src modules correctly
ENV PYTHONPATH=/app:/app/src

# Cloud Run port requirement
ENV PORT=8080
EXPOSE 8080

# Start server (Cloud Run compatible)
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]