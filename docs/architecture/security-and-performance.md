# Security and Performance

## Security Requirements

**Frontend Security:**
- CSP Headers: `default-src 'self'; ws-src 'self' ws://localhost:*`
- XSS Prevention: React automatic escaping, input sanitization
- Secure Storage: No sensitive data in localStorage, session only

**Backend Security:**
- Input Validation: Pydantic schemas for all inputs
- Rate Limiting: 100 requests/minute per IP
- CORS Policy: Allow only localhost and configured IPs

**Authentication Security:**
- Token Storage: N/A (local network only)
- Session Management: WebSocket session timeout after 30 min idle
- Password Policy: N/A (no user accounts)

## Performance Optimization

**Frontend Performance:**
- Bundle Size Target: < 500KB gzipped
- Loading Strategy: Code splitting by route
- Caching Strategy: Service worker for static assets

**Backend Performance:**
- Response Time Target: < 50ms for API calls
- Database Optimization: Indexes on timestamp fields
- Caching Strategy: In-memory cache for RSSI data (10 sec TTL)
