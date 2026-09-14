FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV FORCE_CPU=1
ENV IDR_MODEL_DIR=/app/data/preprocessing/experiment_17/checkpoints

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY api /app/api
COPY id_engine /app/id_engine
COPY data/experiments /app/data/experiments
COPY data/preprocessing/experiment_17/checkpoints /app/data/preprocessing/experiment_17/checkpoints

RUN useradd --create-home --uid 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 10000

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "10000"]
