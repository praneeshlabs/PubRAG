
# PubMed RAG System — Dockerfile
# Base: python:3.11-slim  (Debian Bookworm, minimal footprint)

# Stage 1: dependency builder 
FROM python:3.11-slim AS builder

WORKDIR /build

# Installing build tools needed by some Python wheels (e.g. biopython, torch)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements file first so Docker can cache this layer
# Re-running `pip install` is only triggered when requirements.txt changes
COPY requirements.txt .

# Install all Python dependencies into a non-root prefix for easy copying.
RUN pip install --upgrade pip --no-cache-dir \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt


# Stage 2: runtime image
FROM python:3.11-slim AS runtime

# Metadata
LABEL maintainer="your@gmail.com"
LABEL description="PubMed RAG Research Assistant — PubMedBERT + FlashRank + Claude"
LABEL version="1.0.0"

# System dependencies for runtime (no build tools needed here)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /install /usr/local

# Copy application source
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Environment defaults
# These can be overridden at runtime via:
#   docker run -e ANTHROPIC_API_KEY = OUR_KEY -e NCBI_EMAIL = REGISTERED_MAIL
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_THEME_BASE=light

# Expose Streamlit's default port (as local host)
EXPOSE 8501

# Liveness probe — Streamlit exposes a health endpoint at /_stcore/health
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
ENTRYPOINT ["streamlit", "run", "app/main.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--server.fileWatcherType=none"]
