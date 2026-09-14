FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1
ENV FORCE_CPU=1
ENV IDR_MODEL_DIR=/app/data/preprocessing/experiment_17/checkpoints

WORKDIR /app

# Install requirements first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Ensure the checkpoint exists exactly where the user wants it
RUN mkdir -p /app/data/preprocessing/experiment_17/checkpoints && \
    cp /app/id_engine/assets/best_deltav_gru.pt /app/data/preprocessing/experiment_17/checkpoints/ && \
    cp /app/id_engine/assets/deltav_norm_stats.pt /app/data/preprocessing/experiment_17/checkpoints/

EXPOSE 10000

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "10000"]
