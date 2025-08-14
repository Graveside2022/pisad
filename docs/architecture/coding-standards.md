# Coding Standards

## Critical Fullstack Rules

- **Type Sharing:** Always define shared types in shared/types.ts and import
- **API Calls:** Never make direct HTTP calls - use the service layer
- **Environment Variables:** Access only through config objects, never process.env directly
- **Error Handling:** All API routes must use the standard error handler
- **State Updates:** Never mutate state directly - use proper state management patterns
- **WebSocket Messages:** Use binary protocol for RSSI data, JSON for control messages
- **Safety Checks:** All velocity commands must pass through safety interlock validation
- **Logging:** Use structured logging with correlation IDs for request tracking

## Naming Conventions

| Element          | Frontend             | Backend     | Example                      |
| ---------------- | -------------------- | ----------- | ---------------------------- |
| Components       | PascalCase           | -           | `SignalMeter.tsx`            |
| Hooks            | camelCase with 'use' | -           | `useSystemState.ts`          |
| API Routes       | -                    | kebab-case  | `/api/system-status`         |
| Database Tables  | -                    | snake_case  | `signal_detections`          |
| Config Keys      | -                    | UPPER_SNAKE | `SDR_FREQUENCY`              |
| WebSocket Events | camelCase            | snake_case  | `rssiUpdate` / `rssi_update` |
