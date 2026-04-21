FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Download NLTK at build time
RUN python -m nltk.downloader punkt

# Copy the specific project files needed for the API
COPY app/ app/
COPY src/ src/
COPY models/bioclinicalbert_final/ models/bioclinicalbert_final/

# IMPORTANT: ensures Python can find src/
ENV PYTHONPATH=src

# Expose port (Cloud Run uses 8080)
EXPOSE 8080

# Run API
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]