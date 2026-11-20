FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential gcc curl libgomp1 libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements_current.txt ./requirements_current.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements_current.txt

COPY . .
RUN mkdir -p /app/data/documents /app/evals/reports

EXPOSE 8000 8501

CMD ["uvicorn", "ragnexus.main:app", "--host", "0.0.0.0", "--port", "8000"]
