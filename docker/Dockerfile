# Multi-stage Dockerfile for CaRLeT

FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    graphviz \
    libgraphviz-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements/base.txt requirements/base.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements/base.txt

# Development stage
FROM base as development

COPY requirements/dev.txt requirements/dev.txt
RUN pip install --no-cache-dir -r requirements/dev.txt

COPY . .
RUN pip install -e .

CMD ["bash"]

# Production stage
FROM base as production

COPY requirements/prod.txt requirements/prod.txt
RUN pip install --no-cache-dir -r requirements/prod.txt

# Copy only necessary files
COPY carlet/ carlet/
COPY configs/ configs/
COPY setup.py .
COPY README.md .

# Install package
RUN pip install --no-cache-dir .

# Create output directories
RUN mkdir -p /app/output /app/models

# Add non-root user
RUN useradd -m -u 1000 carlet && \
    chown -R carlet:carlet /app

USER carlet

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
  CMD python -c "import carlet; print('OK')" || exit 1

CMD ["python", "-m", "carlet.cli", "--help"]
