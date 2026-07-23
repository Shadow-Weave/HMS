# HMS Docker Image
# Supports building API-only, Control Plane-only, or both
#
# Build args:
#   INCLUDE_API=true/false         - Include API (default: true)
#   INCLUDE_CP=true/false          - Include Control Plane (default: true)
#   INCLUDE_LOCAL_MODELS=true/false - Include local ML models for embeddings/reranking (default: true)
#                                     Set to false when using external providers (TEI, OpenAI, Cohere)
#   INCLUDE_MULTIMODAL_VIDEO=true/false - Include PyAV for local video decode (default: false)
#   PRELOAD_ML_MODELS=true/false   - Pre-download ML models during build (default: true)
#                                     Only effective when INCLUDE_LOCAL_MODELS=true
#                                     NOTE: tiktoken encodings are ALWAYS preloaded (required for air-gapped deployments)
#
# Examples:
#   docker build -f deploy/containers/standalone/Dockerfile --target standalone -t hms .
#   docker build -f deploy/containers/standalone/Dockerfile --target api-only -t hms-api .
#   docker build -f deploy/containers/standalone/Dockerfile --target cp-only -t hms-cp .
#   docker build -f deploy/containers/standalone/Dockerfile --target standalone \
#     --build-arg PRELOAD_ML_MODELS=false -t hms .
#   docker build -f deploy/containers/standalone/Dockerfile --target api-only \
#     --build-arg INCLUDE_LOCAL_MODELS=false --build-arg INCLUDE_MULTIMODAL_VIDEO=true -t hms-video .

ARG INCLUDE_API=true
ARG INCLUDE_CP=true
ARG PRELOAD_ML_MODELS=true
ARG INCLUDE_LOCAL_MODELS=true
ARG INCLUDE_MULTIMODAL_VIDEO=false

# =============================================================================
# Stage: API Builder
# =============================================================================
FROM python:3.11-slim AS api-builder

ARG INCLUDE_API
ARG INCLUDE_LOCAL_MODELS
ARG INCLUDE_MULTIMODAL_VIDEO
RUN if [ "$INCLUDE_API" != "true" ]; then echo "Skipping API build" && exit 0; fi

WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

# Copy dependency files and README (required by pyproject.toml)
COPY core/dataplane/pyproject.toml ./api/
COPY core/dataplane/README.md ./api/

WORKDIR /app/api

# Sync dependencies using appropriate extras based on build-time capabilities.
# local-ml: torch, sentence-transformers, transformers, einops, flashrank, mlx (optional)
# embedded-db: pg0-embedded (always included for embedded PostgreSQL support)
# multimodal-video: PyAV for bounded local video probing/decoding (optional)
RUN set -eu; \
    case "$INCLUDE_MULTIMODAL_VIDEO" in \
        true|false) ;; \
        *) echo "INCLUDE_MULTIMODAL_VIDEO must be true or false" >&2; exit 2 ;; \
    esac; \
    if [ "$INCLUDE_LOCAL_MODELS" = "true" ] && [ "$INCLUDE_MULTIMODAL_VIDEO" = "true" ]; then \
        uv sync --extra local-ml --extra embedded-db --extra multimodal-video; \
    elif [ "$INCLUDE_LOCAL_MODELS" = "true" ]; then \
        uv sync --extra local-ml --extra embedded-db; \
    elif [ "$INCLUDE_MULTIMODAL_VIDEO" = "true" ]; then \
        uv sync --extra embedded-db --extra multimodal-video; \
    else \
        uv sync --extra embedded-db; \
    fi; \
    if [ "$INCLUDE_MULTIMODAL_VIDEO" = "true" ]; then \
        .venv/bin/python -c "import av; decoder = av.codec.CodecContext.create('h264', 'r'); print('PyAV/H.264 decoder available:', av.__version__, decoder.codec.name)"; \
    fi

# Copy source code (alembic migrations are inside hms_api/)
COPY core/dataplane/hms_api ./hms_api

# Install the local package (uv sync only installed dependencies, not the package itself)
RUN uv pip install -e .

# =============================================================================
# Stage: SDK Builder (needed for Control Plane)
# =============================================================================
FROM node:20-slim AS sdk-builder

ARG INCLUDE_CP
RUN if [ "$INCLUDE_CP" != "true" ]; then echo "Skipping SDK build" && exit 0; fi

WORKDIR /app

# Copy root package files for npm workspaces
COPY package.json package-lock.json ./
COPY interface/sdk/typescript/ ./interface/sdk/typescript/

# Install and build SDK using workspace (--ignore-scripts skips git hooks setup)
RUN npm ci --ignore-scripts -w @hms-memory/hms-client
RUN npm run build -w @hms-memory/hms-client

# =============================================================================
# Stage: Control Plane Builder
# =============================================================================
FROM node:20-slim AS cp-builder

ARG INCLUDE_CP
RUN if [ "$INCLUDE_CP" != "true" ]; then echo "Skipping CP build" && exit 0; fi

# Create directory structure matching the monorepo layout.
WORKDIR /app/hms-workspace/interface/console

# Install Control Plane dependencies
# Only copy package.json (not package-lock.json) to ensure npm installs
# correct platform-specific native bindings for lightningcss/tailwindcss
COPY interface/console/package.json ./
# Remove the file: dependency on SDK (we'll copy it directly later)
RUN sed -i '/"@hms-memory\/hms-client":/d' package.json
RUN npm install

# Copy Control Plane source (excluding node_modules via .dockerignore)
COPY interface/console/ ./
# Remove package-lock.json to avoid conflicts with installed native bindings
# Also remove the file: dependency from package.json (restored by COPY above)
RUN rm -f package-lock.json && sed -i '/"@hms-memory\/hms-client":/d' package.json

# Copy built SDK directly into node_modules (more reliable than npm link in Docker)
COPY --from=sdk-builder /app/interface/sdk/typescript ./node_modules/@hms-memory/hms-client

# Accept base path as build argument for reverse proxy deployments
# Usage: docker build --build-arg NEXT_PUBLIC_BASE_PATH=/hms ...
ARG NEXT_PUBLIC_BASE_PATH=""

# Build Control Plane - run next build first, then custom standalone copy
# (The build:standalone script expects a specific path structure that differs in Docker)
RUN npm exec -- next build

# Create standalone directory structure manually
# Note: Must exclude node_modules from find to avoid wrong server.js from next/dist/experimental/testmode/
# Note: Must explicitly copy .next since glob * doesn't match hidden directories
RUN STANDALONE_ROOT=$(find .next/standalone -path '*/node_modules' -prune -o -name 'server.js' -print | head -1 | xargs dirname) && \
    mkdir -p standalone && \
    cp -r "$STANDALONE_ROOT"/* standalone/ && \
    cp -r "$STANDALONE_ROOT"/.next standalone/.next && \
    # Copy node_modules if separate from app dir (monorepo structure)
    if [ -d ".next/standalone/node_modules" ] && [ "$STANDALONE_ROOT" != ".next/standalone" ]; then \
      cp -r .next/standalone/node_modules standalone/node_modules; \
    fi && \
    cp -r .next/static standalone/.next/static && \
    mkdir -p standalone/public && \
    cp -r public/* standalone/public/ 2>/dev/null || true && \
    # Verify required files exist
    test -f standalone/server.js || (echo "ERROR: server.js missing!" && exit 1) && \
    test -f standalone/.next/BUILD_ID || (echo "ERROR: BUILD_ID missing!" && exit 1)

# =============================================================================
# Stage: Final Image - API Only
# =============================================================================
FROM python:3.11-slim AS api-only

ARG INCLUDE_MULTIMODAL_VIDEO

WORKDIR /app

# Note: libicu version varies by Debian version - try common versions in order
RUN apt-get update && apt-get install -y \
    curl \
    procps \
    libxml2 \
    libssl3 \
    libgssapi-krb5-2 \
    libossp-uuid16 \
    && (apt-get install -y libicu72 2>/dev/null || apt-get install -y libicu74 2>/dev/null || apt-get install -y libicu76 2>/dev/null || true) \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

RUN useradd -m -s /bin/bash hms

# Copy API with virtual environment from builder
COPY --from=api-builder /app/api /app/api

# Verify the copied environment in the slim runtime stage, not only in the
# compiler-bearing builder. This catches missing shared-library dependencies.
RUN if [ "$INCLUDE_MULTIMODAL_VIDEO" = "true" ]; then \
        /app/api/.venv/bin/python -c "import av; decoder = av.codec.CodecContext.create('h264', 'r'); print('Runtime PyAV/H.264 available:', av.__version__, decoder.codec.name)"; \
    fi

# Copy startup script
COPY deploy/containers/standalone/start-all.sh /app/start-all.sh
RUN chmod +x /app/start-all.sh

RUN chown -R hms:hms /app

USER hms

# Create pg0 data directory as hms user so that Docker seeds new named
# volumes with correct ownership (UID 1000) on first use, avoiding the
# "Permission denied" error when mounting a fresh root-owned volume.
RUN mkdir -p /home/hms/.pg0

# Make /home/hms traversable when running with --user UID:GID overrides
# (default 0700 blocks traversal by non-owner UIDs needed for bind-mount ownership matching)
RUN chmod 755 /home/hms

ENV PATH="/app/api/.venv/bin:${PATH}"

# Pre-download tiktoken encoding (ALWAYS - required for token counting even in air-gapped envs)
# Tiktoken is a core runtime dependency, not an optional ML model
RUN MAX_RETRIES=3; \
    RETRY_DELAY=5; \
    for i in $(seq 1 $MAX_RETRIES); do \
      echo "Attempt $i/$MAX_RETRIES: Downloading tiktoken encoding..."; \
      /app/api/.venv/bin/python -c "\
import tiktoken; \
print('Downloading cl100k_base encoding...'); \
tiktoken.get_encoding('cl100k_base'); \
print('Tiktoken encoding cached successfully')" && break; \
      if [ $i -lt $MAX_RETRIES ]; then \
        echo "Attempt $i failed, retrying in ${RETRY_DELAY}s..."; \
        sleep $RETRY_DELAY; \
        RETRY_DELAY=$((RETRY_DELAY * 2)); \
      fi; \
    done; \
    if [ $i -eq $MAX_RETRIES ]; then \
      echo "ERROR: Failed to download tiktoken encoding after $MAX_RETRIES attempts"; \
      exit 1; \
    fi

# Pre-download ML models to avoid runtime download (conditional)
# Only runs if both PRELOAD_ML_MODELS=true AND INCLUDE_LOCAL_MODELS=true
# Includes retry logic with exponential backoff for transient network failures
ARG PRELOAD_ML_MODELS
ARG INCLUDE_LOCAL_MODELS
ENV HF_HUB_DOWNLOAD_TIMEOUT=600
RUN if [ "$PRELOAD_ML_MODELS" = "true" ] && [ "$INCLUDE_LOCAL_MODELS" = "true" ]; then \
    MAX_RETRIES=3; \
    RETRY_DELAY=10; \
    for i in $(seq 1 $MAX_RETRIES); do \
      echo "Attempt $i/$MAX_RETRIES: Downloading ML models..."; \
      /app/api/.venv/bin/python -c "\
import os; os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'; \
from sentence_transformers import SentenceTransformer, CrossEncoder; \
print('Downloading embedding model...'); \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
print('Downloading cross-encoder model...'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('Models cached successfully')" && break; \
      if [ $i -lt $MAX_RETRIES ]; then \
        echo "Attempt $i failed, retrying in ${RETRY_DELAY}s..."; \
        sleep $RETRY_DELAY; \
        RETRY_DELAY=$((RETRY_DELAY * 2)); \
      fi; \
    done; \
    if [ $i -eq $MAX_RETRIES ] && ! /app/api/.venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')" 2>/dev/null; then \
      echo "ERROR: Failed to download models after $MAX_RETRIES attempts"; \
      exit 1; \
    fi; \
    elif [ "$INCLUDE_LOCAL_MODELS" != "true" ]; then echo "Skipping ML model preload (local-models not included)"; \
    else echo "Skipping ML model preload"; fi

EXPOSE 8888

ENV HMS_API_HOST=0.0.0.0
ENV HMS_API_PORT=8888
ENV HMS_API_LOG_LEVEL=info
ENV HMS_ENABLE_API=true
ENV HMS_ENABLE_CP=false
ENV PYTHONUNBUFFERED=1
# Suppress verbose transformers/HuggingFace model loading warnings
ENV TRANSFORMERS_VERBOSITY=error
ENV HF_HUB_VERBOSITY=error
ENV TOKENIZERS_PARALLELISM=false

CMD ["/app/start-all.sh"]

# =============================================================================
# Stage: Final Image - Control Plane Only
# =============================================================================
FROM node:20-alpine AS cp-only

WORKDIR /app

# Copy built SDK
COPY --from=sdk-builder /app/interface/sdk/typescript /app/sdk

# Copy Control Plane standalone build
WORKDIR /app/control-plane
COPY --from=cp-builder /app/hms-workspace/interface/console/standalone ./
COPY --from=cp-builder /app/hms-workspace/interface/console/.next/static ./.next/static
COPY --from=cp-builder /app/hms-workspace/interface/console/public ./public

WORKDIR /app

# Copy startup script
COPY deploy/containers/standalone/start-all.sh /app/start-all.sh
RUN chmod +x /app/start-all.sh

# Install curl for health checks
RUN apk add --no-cache curl bash

EXPOSE 9999

ENV NODE_ENV=production
ENV HMS_CP_DATAPLANE_API_URL=http://localhost:8888
ENV HMS_ENABLE_API=false
ENV HMS_ENABLE_CP=true

CMD ["/app/start-all.sh"]

# =============================================================================
# Stage: Final Image - Standalone (both API and Control Plane)
# =============================================================================
FROM python:3.11-slim AS standalone

ARG INCLUDE_MULTIMODAL_VIDEO

WORKDIR /app

# Install Node.js, curl, uv, and system dependencies
# Note: libicu version varies by Debian version - try common versions in order
RUN apt-get update && apt-get install -y \
    curl \
    procps \
    libxml2 \
    libssl3 \
    libgssapi-krb5-2 \
    libossp-uuid16 \
    && (apt-get install -y libicu72 2>/dev/null || apt-get install -y libicu74 2>/dev/null || apt-get install -y libicu76 2>/dev/null || true) \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

RUN useradd -m -s /bin/bash hms

# Copy API with virtual environment from builder
COPY --from=api-builder /app/api /app/api

# Keep the combined runtime subject to the same decoder verification as the
# API-only image used by dedicated API/worker deployments.
RUN if [ "$INCLUDE_MULTIMODAL_VIDEO" = "true" ]; then \
        /app/api/.venv/bin/python -c "import av; decoder = av.codec.CodecContext.create('h264', 'r'); print('Runtime PyAV/H.264 available:', av.__version__, decoder.codec.name)"; \
    fi

# Copy built SDK
COPY --from=sdk-builder /app/interface/sdk/typescript /app/sdk

# Copy Control Plane standalone build
WORKDIR /app/control-plane
COPY --from=cp-builder /app/hms-workspace/interface/console/standalone ./
COPY --from=cp-builder /app/hms-workspace/interface/console/.next/static ./.next/static
COPY --from=cp-builder /app/hms-workspace/interface/console/public ./public

WORKDIR /app

# Copy startup script
COPY deploy/containers/standalone/start-all.sh /app/start-all.sh
RUN chmod +x /app/start-all.sh

RUN chown -R hms:hms /app

USER hms

# Create pg0 data directory as hms user so that Docker seeds new named
# volumes with correct ownership (UID 1000) on first use, avoiding the
# "Permission denied" error when mounting a fresh root-owned volume.
RUN mkdir -p /home/hms/.pg0

# Make /home/hms traversable when running with --user UID:GID overrides
# (default 0700 blocks traversal by non-owner UIDs needed for bind-mount ownership matching)
RUN chmod 755 /home/hms

ENV PATH="/app/api/.venv/bin:${PATH}"

# Pre-download tiktoken encoding (ALWAYS - required for token counting even in air-gapped envs)
# Tiktoken is a core runtime dependency, not an optional ML model
RUN MAX_RETRIES=3; \
    RETRY_DELAY=5; \
    for i in $(seq 1 $MAX_RETRIES); do \
      echo "Attempt $i/$MAX_RETRIES: Downloading tiktoken encoding..."; \
      /app/api/.venv/bin/python -c "\
import tiktoken; \
print('Downloading cl100k_base encoding...'); \
tiktoken.get_encoding('cl100k_base'); \
print('Tiktoken encoding cached successfully')" && break; \
      if [ $i -lt $MAX_RETRIES ]; then \
        echo "Attempt $i failed, retrying in ${RETRY_DELAY}s..."; \
        sleep $RETRY_DELAY; \
        RETRY_DELAY=$((RETRY_DELAY * 2)); \
      fi; \
    done; \
    if [ $i -eq $MAX_RETRIES ]; then \
      echo "ERROR: Failed to download tiktoken encoding after $MAX_RETRIES attempts"; \
      exit 1; \
    fi

# Pre-download ML models to avoid runtime download (conditional)
# Only runs if both PRELOAD_ML_MODELS=true AND INCLUDE_LOCAL_MODELS=true
# Includes retry logic with exponential backoff for transient network failures
ARG PRELOAD_ML_MODELS
ARG INCLUDE_LOCAL_MODELS
ENV HF_HUB_DOWNLOAD_TIMEOUT=600
RUN if [ "$PRELOAD_ML_MODELS" = "true" ] && [ "$INCLUDE_LOCAL_MODELS" = "true" ]; then \
    MAX_RETRIES=3; \
    RETRY_DELAY=10; \
    for i in $(seq 1 $MAX_RETRIES); do \
      echo "Attempt $i/$MAX_RETRIES: Downloading ML models..."; \
      /app/api/.venv/bin/python -c "\
import os; os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '600'; \
from sentence_transformers import SentenceTransformer, CrossEncoder; \
print('Downloading embedding model...'); \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
print('Downloading cross-encoder model...'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('Models cached successfully')" && break; \
      if [ $i -lt $MAX_RETRIES ]; then \
        echo "Attempt $i failed, retrying in ${RETRY_DELAY}s..."; \
        sleep $RETRY_DELAY; \
        RETRY_DELAY=$((RETRY_DELAY * 2)); \
      fi; \
    done; \
    if [ $i -eq $MAX_RETRIES ] && ! /app/api/.venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')" 2>/dev/null; then \
      echo "ERROR: Failed to download models after $MAX_RETRIES attempts"; \
      exit 1; \
    fi; \
    elif [ "$INCLUDE_LOCAL_MODELS" != "true" ]; then echo "Skipping ML model preload (local-models not included)"; \
    else echo "Skipping ML model preload"; fi

EXPOSE 8888 9999

ENV HMS_API_HOST=0.0.0.0
ENV HMS_API_PORT=8888
ENV HMS_API_LOG_LEVEL=info
ENV NODE_ENV=production
ENV HMS_CP_DATAPLANE_API_URL=http://localhost:8888
ENV HMS_ENABLE_API=true
ENV HMS_ENABLE_CP=true
ENV PYTHONUNBUFFERED=1
# Suppress verbose transformers/HuggingFace model loading warnings
ENV TRANSFORMERS_VERBOSITY=error
ENV HF_HUB_VERBOSITY=error
ENV TOKENIZERS_PARALLELISM=false

CMD ["/app/start-all.sh"]

# =============================================================================
# Default target selection based on build args
# =============================================================================
FROM standalone AS default-both
FROM api-only AS default-api
FROM cp-only AS default-cp

# This selects the final stage based on INCLUDE_API and INCLUDE_CP
# Use --target to override: docker build --target api-only .
FROM standalone
