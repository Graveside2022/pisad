# Project Tasks

*This file is synced with Clode Studio and Claude's native TodoWrite system.*  
*Last updated: 2025-08-16T17:32:10.162Z*

## Backlog (6)

- [ ] **Create comprehensive developer setup guide**
  - Assignee: claude
  - Type: documentation
  - Priority: low
- [ ] **Optimize service startup times**
  - Assignee: claude
  - Type: feature
  - Priority: low
- [ ] **Add advanced performance monitoring**
  - Assignee: claude
  - Type: feature
  - Priority: low
- [ ] **Create Grafana dashboard config (Story 4.4 Sprint 3)**
  - Assignee: claude
  - Type: feature
  - Priority: low
  - Description: Optional - Create deployment/grafana-dashboard.json with panels for MAVLink latency, RSSI processing histograms, uptime and request rate
- [ ] **Add version info to UI footer (Story 4.4 Sprint 3)**
  - Assignee: claude
  - Type: feature
  - Priority: low
  - Description: Nice-to-have - Create version.json during build, add Footer component to App.tsx, fetch and display version from /api/version endpoint
- [ ] **Story 4.8: DuckDB Migration**
  - Assignee: claude
  - Type: feature
  - Priority: medium
  - Description: Full database architecture transformation from SQLite to DuckDB for advanced analytics

## To Do (4)

- [ ] **Story 4.6: Safety-Critical Coverage Compliance**
  - Assignee: claude
  - Type: feature
  - Priority: high
  - Description: BLOCKED by 4.5 - Create HAL mocks, achieve 85%+ coverage after APIs exist
- [ ] **Story 4.2: Test Coverage Maintenance**
  - Assignee: claude
  - Type: feature
  - Priority: medium
  - Description: BLOCKED by 4.5 & 4.6 - Maintain 90% coverage (ongoing after APIs complete)
- [ ] **Story 2.5: Ground Testing & Safety Validation**
  - Assignee: claude
  - Type: feature
  - Priority: high
  - Description: READY - Test HackRF & Cube Orange+ safety interlocks with real hardware
- [ ] **Story 3.4: Field Testing Campaign**
  - Assignee: claude
  - Type: feature
  - Priority: high
  - Description: READY - Real-world validation with HackRF/Cube Orange+, FAA COA needed

## In Progress (1)

- [ ] **Story 4.9: Code Optimization and Refactoring** ⏳
  - Assignee: claude
  - Type: feature
  - Priority: high
  - Description: Sprint 8 Day 3-4 complete - PRD test contracts written (1,312 lines)

## Completed (7)

- [x] ~~Story 4.5: API Implementation & Documentation 🎯 NEXT~~
  - ~~Assignee: claude~~
  - ~~Type: feature~~
  - ~~Priority: medium~~
  - ~~Description: IMMEDIATE - Implement SignalProcessor, MAVLink, and StateMachine APIs to match Story 4.9 Sprint 8 test contracts~~
- [x] ~~Story 4.7: Hardware Integration Testing~~
  - ~~Assignee: claude~~
  - ~~Type: feature~~
  - ~~Priority: medium~~
  - ~~Description: Sprint 2 Active - HAL implemented for HackRF (pyhackrf) & Cube Orange+ (pymavlink). Next: beacon generator, auto-detection, performance metrics~~
- [x] ~~Story 4.4: CI/CD Pipeline & Deployment~~
  - ~~Assignee: claude~~
  - ~~Type: feature~~
  - ~~Priority: high~~
  - ~~Description: Set up GitHub Actions CI/CD, pre-commit hooks, and deployment~~
- [x] ~~Story 4.3: Hardware Service Integration~~
  - ~~Assignee: claude~~
  - ~~Type: feature~~
  - ~~Priority: high~~
  - ~~Description: Initialize and integrate SDR, MAVLink, state machine~~
- [x] ~~Story 4.2: Comprehensive Test Coverage~~
  - ~~Assignee: claude~~
  - ~~Type: feature~~
  - ~~Priority: high~~
  - ~~Description: Fix test execution issues, increase coverage to 60% backend, add frontend/E2E tests~~
- [x] ~~Story 4.1: Frontend Application Stability~~
  - ~~Assignee: claude~~
  - ~~Type: feature~~
  - ~~Priority: high~~
  - ~~Description: Fix all TypeScript, WebSocket, and frontend-backend connection issues to get the web interface running~~
- [x] ~~Install missing backend dependencies: matplotlib, fastapi, uvicorn, httpx, pydantic-settings ✅~~
  - ~~Assignee: claude~~
  - ~~Type: feature~~
  - ~~Priority: medium~~

---
*To update tasks, use the Kanban board in Clode Studio, ask Claude to modify this file, or use Claude's native TodoWrite system.*
