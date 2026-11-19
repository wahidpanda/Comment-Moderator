# Comment-Moderator API (FastAPI + Detoxify on CPU)
FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LR_MODEL_PATH=models/lr/v1/model.joblib \
    LR_THRESHOLDS_PATH=models/lr/v1/thresholds.json \
    USE_TRANSFORMER=true

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY engine ./engine
COPY api ./api
COPY models/lr/v1 ./models/lr/v1

RUN pip install --upgrade pip \
    && pip install -e ".[transformer]"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]