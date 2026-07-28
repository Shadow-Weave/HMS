FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY core/dataplane /app/core/dataplane

# Install CPU-only PyTorch from the official CPU wheel index, then the
# reranker extra (which declares torch>=2.6.0; pip will see it is already
# satisfied and skip reinstallation).
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.6.0" \
    && pip install "/app/core/dataplane[reranker]"

RUN useradd --create-home --shell /usr/sbin/nologin hms \
    && mkdir -p /app/logs /home/hms/.cache/huggingface \
    && chown -R hms:hms /app /home/hms/.cache

USER hms

EXPOSE 18080

CMD ["hms-api", "--host", "0.0.0.0", "--port", "18080"]
