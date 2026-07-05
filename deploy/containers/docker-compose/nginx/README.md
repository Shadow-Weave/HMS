# Nginx Reverse Proxy with Custom Base Path

Deploy HMS API under `/hms` (or any custom path) using Nginx reverse proxy.

## Quick Start (Published Image - API Only)

```bash
docker-compose up
```

- **API:** http://localhost:8080/hms/docs
- **Control Plane:** http://localhost:9999 (direct access, not proxied)

## Full Stack with Custom Base Path (Requires Build)

**Important:** You cannot rebuild from the published image with build args. You must build from source.

### Build from Source with Custom Base Path

1. **Clone the repository** (if you haven't):
```bash
git clone https://github.com/hms-memory/hms.git
cd hms
```

2. **Build with base path**:
```bash
docker build \
  --build-arg NEXT_PUBLIC_BASE_PATH=/hms \
  -f deploy/containers/standalone/Dockerfile \
  -t hms:custom \
  .
```

3. **Update docker-compose.yml** to use your built image:
```yaml
services:
  hms:
    image: hms:custom  # ← Change this
    environment:
      HMS_API_BASE_PATH: /hms
      NEXT_PUBLIC_BASE_PATH: /hms
```

4. **Update nginx.conf** to handle Control Plane routes (see below)

5. **Run**:
```bash
docker-compose up
```

### Required nginx.conf for Full Stack

Replace the current `nginx.conf` with this to proxy both API and Control Plane:

```nginx
events { worker_connections 1024; }

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    upstream hms_api { server hms:8888; }
    upstream hms_cp { server hms:9999; }

    server {
        listen 80;

        # API
        location ~ ^/hms/(docs|openapi\.json|health|metrics|v1|mcp) {
            proxy_pass http://hms_api;
            proxy_set_header Host $http_host;
        }

        # Control Plane static files
        location ~ ^/hms/_next/ {
            proxy_pass http://hms_cp;
            proxy_set_header Host $http_host;
        }

        # Control Plane UI
        location /hms {
            proxy_pass http://hms_cp;
            proxy_set_header Host $http_host;
        }

        location = / { return 301 /hms; }
    }
}
```

### Why Build is Required

Next.js requires `basePath` at **build time**. The published image was built without a custom base path, so you must rebuild from source with the `NEXT_PUBLIC_BASE_PATH` build arg to deploy the Control Plane under a subpath.

The API works without rebuild because `HMS_API_BASE_PATH` is a runtime environment variable.
