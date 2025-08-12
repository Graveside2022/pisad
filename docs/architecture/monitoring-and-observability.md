# Monitoring and Observability

## Monitoring Stack
- **Frontend Monitoring:** Browser console logs + local performance API
- **Backend Monitoring:** Prometheus metrics endpoint
- **Error Tracking:** Local log aggregation with systemd journal
- **Performance Monitoring:** Custom metrics via Prometheus

## Key Metrics
**Frontend Metrics:**
- Core Web Vitals (FCP, LCP, CLS)
- JavaScript errors count
- API response times (p50, p95, p99)
- WebSocket connection drops

**Backend Metrics:**
- Request rate (req/s)
- Error rate (4xx, 5xx)
- Response time (p50, p95, p99)
- SDR sample processing latency
- MAVLink command latency
- State transition frequency
- Signal detection rate
