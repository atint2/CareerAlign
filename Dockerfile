FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (needed for some ML packages)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Cloud Run expects port 8080
ENV PORT=8080

CMD uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT} --workers 1