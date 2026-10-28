# hms-elite runtime image.
#
# Single stage: there is no compiled dependency to strip (fastembed ships a pure
# ONNX runtime wheel), so a builder stage would only add concepts.
#
# The base image is pulled through the daocloud mirror FQN because
# registry-1.docker.io has no egress from the dev/prod boxes. Same content and
# digest as docker.io — keep the FQN, not docker.io.
FROM docker.m.daocloud.io/library/python:3.12.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TMPDIR=/app/model-cache/tmp

# Forward a proxy into the build (dev hosts without direct egress). Empty by
# default, so prod builds pass nothing and the image gets no proxy. Cleared
# again before the runtime stage below.
ARG HTTP_PROXY
ARG HTTPS_PROXY
ARG NO_PROXY
ENV HTTP_PROXY=$HTTP_PROXY HTTPS_PROXY=$HTTPS_PROXY NO_PROXY=$NO_PROXY \
    http_proxy=$HTTP_PROXY https_proxy=$HTTPS_PROXY no_proxy=$NO_PROXY

# Some hosts inject an *http* index-url (no TLS), which pip ignores unless its
# host is explicitly trusted, so a matching PIP_TRUSTED_HOST is forwarded too.
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/
ARG PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
ENV PIP_INDEX_URL=$PIP_INDEX_URL PIP_TRUSTED_HOST=$PIP_TRUSTED_HOST

# deb.debian.org/security.debian.org are slow or unreachable from the dev + CI
# hosts. Point apt at the Tsinghua mirror (bookworm-slim keeps its sources in
# /etc/apt/sources.list.d/debian.sources).
#
# APT_MIRROR is overridable purely so the image can be built from a network the
# Tsinghua mirror rejects (it answers 403 to some external IPs). The default is
# what the dev and CI hosts use.
ARG APT_MIRROR=https://mirrors.tuna.tsinghua.edu.cn
RUN printf '%s\n' \
    'Types: deb' \
    "URIs: $APT_MIRROR/debian" \
    'Suites: bookworm bookworm-updates' \
    'Components: main contrib non-free non-free-firmware' \
    'Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg' \
    '' \
    'Types: deb' \
    "URIs: $APT_MIRROR/debian-security" \
    'Suites: bookworm-security' \
    'Components: main contrib non-free non-free-firmware' \
    'Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg' \
    > /etc/apt/sources.list.d/debian.sources

# libgomp1 is onnxruntime's OpenMP runtime and is NOT in python:*-slim; without
# it the fastembed TextEmbedding import fails at load time.
RUN apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 install --no-install-recommends -y libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 hms \
    && useradd --uid 10001 --gid hms --create-home --shell /usr/sbin/nologin hms

WORKDIR /app
# README.md must be present: pyproject declares readme = "README.md" and
# hatchling reads it while building the wheel. Do not add it to .dockerignore.
COPY pyproject.toml README.md ./
COPY hms_elite ./hms_elite

# pip, not uv: this project has no uv.lock, so `uv sync --frozen` cannot run.
# Dropping --frozen would move resolution to build time anyway (no determinism
# gained) and add a uv bootstrap on top — same cost, zero benefit.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --index-url "$PIP_INDEX_URL" --trusted-host "$PIP_TRUSTED_HOST" \
        ".[server,fastembed,openai,numpy]"

# Bake the local text-embedding model into the image so the runtime never needs
# HF egress. hf-mirror.com is reachable (huggingface.co is not, even via the
# proxy). fastembed downloads through huggingface_hub, whose Xet CAS CDN
# (cas-server.xethub.hf.co) is not mirrored and 401s without a real HF token —
# HF_HUB_DISABLE_XET forces classic HTTPS through HF_ENDPOINT.
#
# ⚠️ The path here MUST match `embedder.config.cache_dir` in
# deploy/config.dev.json, byte for byte. hms-elite never reads
# FASTEMBED_CACHE_PATH (nothing in the package references it); the only lever is
# FastEmbedEmbedderConfig.cache_dir. Mismatch = the runtime looks in an empty
# directory and re-downloads ~130MB from hf-mirror on first search.
#
# ⚠️ `mkdir -p /data` + chown is not optional: a named volume inherits the
# ownership of the image directory at first creation, and the container runs as
# uid 10001. Miss it and SQLite fails with "unable to open database file".
ENV HF_ENDPOINT=https://hf-mirror.com \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_DOWNLOAD_TIMEOUT=60 \
    HF_HUB_ETAG_TIMEOUT=60
# The download lands in a BuildKit cache mount first, then gets copied to the
# baked path. Without the mount, this layer is invalidated by the `COPY
# hms_elite` above — i.e. every source edit would re-fetch 65MB from the mirror
# on every deploy. The cache is build-time only; the model still ships inside
# the image, so the runtime needs neither a volume nor HF egress.
RUN --mount=type=cache,target=/root/.cache/fastembed-dl \
    python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/root/.cache/fastembed-dl')" \
    && mkdir -p /app/model-cache/fastembed /app/model-cache/tmp /data \
    && cp -a /root/.cache/fastembed-dl/. /app/model-cache/fastembed/ \
    && chown -R hms:hms /app/model-cache /data

# Do not carry the build proxy into the runtime image; compose re-supplies it
# per service at runtime for genuine external egress (api.deepseek.com).
ENV HTTP_PROXY= HTTPS_PROXY= NO_PROXY= http_proxy= https_proxy= no_proxy=

USER hms
EXPOSE 8000
# --host 0.0.0.0 is mandatory: main() defaults to 127.0.0.1 (server/app.py:1063),
# which would make the container unreachable from nginx.
CMD ["hms-server", "--host", "0.0.0.0", "--port", "8000", \
     "--config", "/app/config/hms.config.json", "--log-level", "info"]
