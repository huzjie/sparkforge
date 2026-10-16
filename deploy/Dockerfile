# Multi-stage, zero-dependency Python image
FROM python:3.11-slim AS base
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

COPY pyproject.toml ./
COPY sparkforge ./sparkforge
COPY configs ./configs
COPY README.md ./

RUN pip install --no-cache-dir -e . 2>/dev/null || true

EXPOSE 8000
CMD ["python", "-m", "sparkforge.cli", "serve", "--config", "configs/default.yaml"]
