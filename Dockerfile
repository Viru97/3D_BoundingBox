FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ src/
COPY scripts/ scripts/
COPY paths.example.json ./

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir torch --index-url "${PYTORCH_INDEX_URL}" && \
    python -m pip install --no-cache-dir ".[all]" && \
    useradd --create-home --uid 10001 app && \
    mkdir -p /app/output_visualizations /app/test_output /app/onnx_export && \
    chown -R app:app /app

USER app

CMD ["python", "scripts/inference.py", "--help"]
