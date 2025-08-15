# Tests to Keep - High-Value Test Inventory

## Summary
**Total Tests to Keep:** 1,140 high-value tests
**Categories:** Safety-critical, Core functionality, Integration, Performance
**Coverage Target:** >80% for critical paths, 100% for safety hazards

## Tier 1: Safety-Critical Tests (MUST KEEP - 184 tests)

### Safety System Tests
| File | Tests | Purpose | Traces To |
|------|-------|---------|-----------|
| test_safety.py | 28 | Safety interlock validation | HARA-PWR-001, HARA-NAV-002 |
| test_utils_safety.py | 36 | Safety utility functions | FR10, NFR12 |
| test_safety_integration.py | 15 | End-to-end safety | Story 2.2 |
| test_safety_system.py | 12 | System-wide safety | NFR12 |
| test_safety_interlock_scenario.py | 8 | SITL safety scenarios | HARA-* |
| test_safety_scenarios.py | 10 | Edge case safety | HARA-* |

### Critical State Management
| File | Tests | Purpose | Traces To |
|------|-------|---------|-----------|
| test_state_machine_enhanced.py | 32 | State transitions | FR3, FR7 |
| test_state_integration.py | 31 | State system integration | Story 3.3 |
| test_state_persistence.py | 12 | State recovery | NFR9 |

## Tier 2: Core Functionality Tests (MUST KEEP - 412 tests)

### Signal Processing Pipeline
| File | Tests | Purpose | Traces To |
|------|-------|---------|-----------|
| test_signal_processor.py | 39 | RSSI computation | FR1, FR6 |
| test_signal_processor_integration.py | 18 | Signal chain validation | Story 1.3 |
| test_noise_estimator.py | 8 | Noise floor calculation | FR6 |

### MAVLink Communication
| File | Tests | Purpose | Traces To |
|------|-------|---------|-----------|
| test_mavlink_service.py | 41 | Flight control interface | FR15, Story 2.1 |
| test_mavlink_integration.py | 22 | MAVLink protocol compliance | NFR1 |
| test_gcs_integration.py | 15 | Ground station integration | FR11 |

### Homing System
| File | Tests | Purpose | Traces To |
|------|-------|---------|-----------|
| test_homing_algorithm_comprehensive.py | 22 | Gradient climbing | FR4, Story 3.2 |
| test_homing_controller_comprehensive.py | 21 | Velocity control | FR14, FR15 |
| test_homing_integration.py | 18 | End-to-end homing | Story 3.2 |

### Search Patterns
| File | Tests | Purpose | Traces To |
|------|-------|---------|-----------|
| test_search_pattern_generator.py | 47 | Pattern generation | FR2, Story 3.1 |
| test_waypoint_exporter_enhanced.py | 26 | Waypoint creation | Story 3.1 |

### Configuration Management
| File | Tests | Purpose | Traces To |
|------|-------|---------|-----------|
| test_config_service.py | 24 | Configuration handling | Story 1.5 |
| test_database_models.py | 37 | Data persistence | Story 1.5 |
| test_models_schemas.py | 30 | Data validation | All stories |

### Field Testing Support
| File | Tests | Purpose | Traces To |
|------|-------|---------|-----------|
| test_field_test_service.py | 48 | Field test execution | Story 3.4 |
| test_telemetry_recorder.py | 26 | Data recording | FR12 |
| test_mission_replay_service.py | 14 | Mission playback | Story 3.5 |

## Tier 3: Integration Tests (KEEP - 304 tests)

### API Integration
| File | Tests | Purpose | Traces To |
|------|-------|---------|-----------|
| test_app.py | 28 | FastAPI application | Story 1.4 |
| test_api_state.py | 44 | State API endpoints | Story 2.3 |
| test_analytics_api.py | 22 | Analytics endpoints | Story 3.5 |
| test_websocket.py | 35 | Real-time updates | FR9, Story 1.4 |

### System Integration
| File | Tests | Purpose | Traces To |
|------|-------|---------|-----------|
| test_phase1_integration.py | 27 | Phase 1 validation | Epic 1 |
| test_system_integration.py | 45 | Full system test | All epics |
| test_api_system.py | 38 | System control API | FR11 |

### Hardware Integration
| File | Tests | Purpose | Traces To |
|------|-------|---------|-----------|
| test_ardupilot_sitl_integration.py | 12 | SITL validation | NFR7 |
| test_beacon_detection_scenario.py | 8 | Detection scenarios | FR1 |
| test_homing_approach_scenario.py | 9 | Approach validation | FR4 |

## Tier 4: Specialized Tests (KEEP - 240 tests)

### Performance Tests
| File | Tests | Purpose | Traces To |
|------|-------|---------|-----------|
| test_performance_analytics.py | 18 | Performance metrics | NFR2 |
| test_homing_performance.py | 7 | Homing optimization | NFR8 |
| test_recommendations_engine.py | 25 | AI recommendations | Story 3.5 |

### Hardware Mocks
| File | Tests | Purpose | Traces To |
|------|-------|---------|-----------|
| test_mock_mavlink_commands.py | 25 | MAVLink simulation | Development |
| test_mock_mavlink_safety.py | 20 | Safety simulation | Development |
| test_hackrf_mock.py | 30 | SDR simulation | Development |

### Deployment & Infrastructure
| File | Tests | Purpose | Traces To |
|------|-------|---------|-----------|
| test_deployment_config.py | 24 | Deployment validation | Operations |
| test_ci_cd_pipeline.py | 23 | CI/CD validation | Development |
| test_docker_environment.py | 15 | Container validation | Operations |

### Reporting & Analytics
| File | Tests | Purpose | Traces To |
|------|-------|---------|-----------|
| test_report_generator.py | 19 | Report generation | Story 3.5 |
| test_analytics_export.py | 14 | Data export | Story 3.5 |
| test_phase3_validation.py | 20 | Phase 3 metrics | Epic 3 |

## Test Quality Metrics

### Coverage by Component
| Component | Current | Target | Priority |
|-----------|---------|--------|----------|
| Safety System | 95% | 100% | CRITICAL |
| Signal Processing | 82% | 90% | HIGH |
| MAVLink Service | 78% | 85% | HIGH |
| State Machine | 71% | 85% | HIGH |
| Homing Algorithm | 68% | 90% | CRITICAL |
| API Endpoints | 88% | 90% | MEDIUM |
| WebSocket | 75% | 80% | MEDIUM |

### Test Value Scoring
| Score | Criteria | Example |
|-------|----------|---------|
| 10 | Prevents safety hazard | Battery failsafe test |
| 9 | Validates PRD requirement | Signal detection test |
| 8 | Ensures integration works | MAVLink communication |
| 7 | Catches common bugs | State transition validation |
| 6 | Improves reliability | Error recovery test |
| 5 | Documents behavior | API contract test |
| 4 | Aids debugging | Logging validation |
| 3 | Nice to have | UI formatting test |
| 2 | Redundant but harmless | Duplicate null check |
| 1 | Should remove | Testing language features |

## Test Maintenance Guidelines

### When to Add Tests
1. New user story implementation
2. Bug fix (regression test)
3. Safety hazard mitigation
4. Performance optimization validation
5. API contract changes

### When to Remove Tests
1. Duplicate functionality (keep best one)
2. Testing implementation not behavior
3. Testing language/framework features
4. Unmaintainable brittle tests
5. Tests that never catch real bugs

### Test Quality Standards
- Each test must trace to requirement
- Runtime <100ms for unit tests
- Clear test names describing scenario
- Minimal setup/teardown
- No external dependencies for unit tests
- Deterministic (no random/time-based)

## Validation Checklist
- [ ] All safety hazards have tests
- [ ] All PRD requirements have tests
- [ ] No duplicate test coverage
- [ ] All tests have requirement traces
- [ ] Test names clearly describe scenario
- [ ] Tests run in <5 minutes total
