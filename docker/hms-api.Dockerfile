FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY core/dataplane /app/core/dataplane

RUN pip install --upgrade pip \
    && pip install /app/core/dataplane

RUN useradd --create-home --shell /usr/sbin/nologin hms \
    && mkdir -p /app/logs \
    && chown -R hms:hms /app

USER hms

EXPOSE 18080

CMD ["hms-api", "--host", "0.0.0.0", "--port", "18080"]
