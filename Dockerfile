# Use a slim, universal Python base image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the essential project files
COPY pyproject.toml README.md ./
COPY src/ src/
COPY scripts/ scripts/

# Install pip and the project (this will pull PyTorch and its bundled CUDA automagically)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

CMD ["bash"]
