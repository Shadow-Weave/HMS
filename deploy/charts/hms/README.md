# HMS Helm Chart

Helm chart for deploying HMS - a temporal-semantic-entity memory system for AI agents.

## Prerequisites

- Kubernetes 1.19+
- Helm 3.0+
- PostgreSQL database (external or bundled)

## Quick Start

```bash
# Update dependencies first
helm dependency update ./deploy/charts/hms

# Install (PostgreSQL included by default)
export OPENAI_API_KEY="sk-your-openai-key"
helm upgrade hms --install ./deploy/charts/hms -n hms --create-namespace \
  --set api.secrets.HMS_API_LLM_API_KEY="$OPENAI_API_KEY"
```

To use an external database instead:

```bash
helm install hms ./deploy/charts/hms -n hms --create-namespace \
  --set api.secrets.HMS_API_LLM_API_KEY="sk-your-openai-key" \
  --set postgresql.enabled=false \
  --set postgresql.external.host=my-postgres.example.com \
  --set postgresql.external.password=mypassword
```

## Installation

### Add the repository (if published)

```bash
helm repo add hms https://your-helm-repo.com
helm repo update
```

### Install with custom values file

Create a `values-override.yaml`:

```yaml
api:
  secrets:
    HMS_API_LLM_API_KEY: "sk-your-openai-key"

postgresql:
  external:
    host: "my-postgres.example.com"
    password: "mypassword"
```

Then install:

```bash
helm install hms ./deploy/charts/hms -n hms --create-namespace -f values-override.yaml
```

## Configuration

### Key Values

| Parameter | Description | Default |
|-----------|-------------|---------|
| `version` | Default image tag for all components | `0.1.0` |
| `api.enabled` | Enable the API component | `true` |
| `api.image.repository` | API image repository | `hms/api` |
| `api.image.tag` | API image tag (defaults to `version`) | - |
| `api.service.port` | API service port | `8888` |
| `controlPlane.enabled` | Enable the control plane | `true` |
| `controlPlane.image.repository` | Control plane image repository | `hms/control-plane` |
| `controlPlane.image.tag` | Control plane image tag (defaults to `version`) | - |
| `controlPlane.service.port` | Control plane service port | `3000` |
| `postgresql.enabled` | Deploy PostgreSQL as subchart | `true` |
| `postgresql.external.host` | External PostgreSQL host | `postgresql` |
| `postgresql.external.port` | External PostgreSQL port | `5432` |
| `postgresql.external.database` | Database name | `hms` |
| `postgresql.external.username` | Database username | `hms` |
| `ingress.enabled` | Enable ingress | `false` |
| `autoscaling.enabled` | Enable HPA | `false` |

### Environment Variables

All environment variables in `api.env` and `controlPlane.env` are automatically added to the respective pods. Sensitive values should go in `api.secrets` or `controlPlane.secrets`.

```yaml
api:
  env:
    HMS_API_LLM_PROVIDER: "openai"
    HMS_API_LLM_MODEL: "gpt-4"
  secrets:
    HMS_API_LLM_API_KEY: "your-api-key"
    HMS_API_LLM_BASE_URL: "https://api.openai.com/v1"

controlPlane:
  env:
    NODE_ENV: "production"
  secrets: {}
```

### Multimodal video image

The Helm chart deploys existing images; it cannot install the optional PyAV
runtime package. The default HMS API image is not evidence that video decode is
present. Build and publish an image from the repository with the explicit
extra, then use the same verified, non-moving tag for the API and all dedicated
workers:

```bash
docker build \
  --target api-only \
  --build-arg INCLUDE_LOCAL_MODELS=false \
  --build-arg INCLUDE_MULTIMODAL_VIDEO=true \
  -f deploy/containers/standalone/Dockerfile \
  -t registry.example.com/hms-api:multimodal-video \
  .
docker push registry.example.com/hms-api:multimodal-video
```

Example values keep the live-quality marker false until a separately approved
provider test passes:

```yaml
api:
  image:
    repository: registry.example.com/hms-api
    tag: multimodal-video
  env:
    HMS_API_ENABLE_FILE_UPLOAD_API: "true"
    HMS_API_MULTIMODAL_ENABLED: "true"
    HMS_API_MULTIMODAL_IMAGE_ENABLED: "true"
    HMS_API_MULTIMODAL_VIDEO_ENABLED: "true"
    HMS_API_MULTIMODAL_LIVE_VERIFIED: "false"
  secrets:
    HMS_API_MULTIMODAL_API_KEY: "replace-with-secret"

worker:
  enabled: true
  image:
    repository: registry.example.com/hms-api
    tag: multimodal-video
```

Plain `GET /version` intentionally keeps the legacy wire shape. Use
`GET /version?include_multimodal=true` for the approved additive image, video,
and live-verification capability flags. After rollout, verify the server-static
settings and confirm that PyAV can construct an H.264 decoder in both the API
and every worker pod. The API-local decoder check cannot inspect a separately
deployed worker image:

```bash
kubectl exec -n hms "$WORKER_POD" -- /app/api/.venv/bin/python -c \
  'import av; print(av.__version__, av.codec.CodecContext.create("h264", "r").codec.name)'
```

Full configuration, polling, data controls, and live-gate requirements are in the
[multimodal operator guide](../../../docs/multimodal_memory.md).

### External Database

To connect to an external PostgreSQL database:

```yaml
postgresql:
  enabled: false
  external:
    host: "my-postgres.example.com"
    port: 5432
    database: "hms"
    username: "hms"
    password: "your-password"
```

### Ingress

To expose the services via ingress:

```yaml
ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  hosts:
    - host: hms.example.com
      paths:
        - path: /
          pathType: Prefix
          service: controlPlane
        - path: /api
          pathType: Prefix
          service: api
  tls:
    - secretName: hms-tls
      hosts:
        - hms.example.com
```

## Upgrading

```bash
helm upgrade hms ./deploy/charts/hms -n hms
```

## Uninstalling

```bash
helm uninstall hms -n hms
```

## Components

The chart deploys:

- **API**: The main HMS API server for memory operations
- **Control Plane**: Web UI for managing agents and viewing memories
- **Worker**: Optional dedicated background-operation workers
- **PostgreSQL**: Optional bundled database, or connection settings for an external PostgreSQL service

## Development

### Lint the chart

```bash
helm lint ./deploy/charts/hms
```

### Template locally

```bash
helm template hms ./deploy/charts/hms --debug
```

### Dry run installation

```bash
helm install hms ./deploy/charts/hms --dry-run --debug
```
