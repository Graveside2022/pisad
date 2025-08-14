# Deployment Architecture

## Deployment Strategy

**Frontend Deployment:**

- **Platform:** Static files served by FastAPI
- **Build Command:** `npm run build`
- **Output Directory:** `dist/`
- **CDN/Edge:** N/A (local only)

**Backend Deployment:**

- **Platform:** Systemd service on Raspberry Pi
- **Build Command:** `pip install -r requirements.txt`
- **Deployment Method:** Ansible playbook

## CI/CD Pipeline

```yaml

```
