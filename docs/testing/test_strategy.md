# Test Strategy Overview

## Executive Summary

This document provides a comprehensive overview of the PiSAD testing strategy, encompassing all levels of testing from unit to flight validation.

## Test Environment Setup

### Development Environment

- **Local Testing**: Raspberry Pi 5 or development machine
- **Dependencies**: Python 3.11+, Node.js 18+, pytest, Jest
- **Mocking**: pymavlink for MAVLink, RTL-SDR mocks for signal processing

### CI/CD Environment

- **Platform**: GitHub Actions
- **Test Triggers**: Push to main, pull requests, nightly builds
- **Coverage Requirements**: >80% for safety-critical components

### Hardware-in-Loop Environment

- **Equipment**: Raspberry Pi 5, Pixhawk/ArduPilot FC, RTL-SDR
- **Connection**: Serial/USB for MAVLink, USB 3.0 for SDR
- **Power**: Bench power supply or battery

## Test Execution Guide

### Running Unit Tests

```bash
# Backend Python tests
uv run pytest tests/backend/unit/ -v --cov=src/backend

# Frontend React tests
npm test

# Safety-specific tests
python scripts/test_safety_interlocks.py
```

### Running Integration Tests

```bash
# Backend integration
uv run pytest tests/backend/integration/ -v

# SITL tests (requires ArduPilot SITL)
uv run pytest tests/backend/sitl/ -v

# E2E tests
npm run test:e2e
```

### Running Hardware Tests

```bash
# HIL timing validation
python scripts/test_hil_timing.py --connection /dev/ttyACM0

# Ground vehicle tests
# Follow docs/testing/ground_vehicle_test_plan.md
```

## Known Issues and Workarounds

### Issue: USB Overcurrent on Pi

**Symptom**: RTL-SDR disconnects randomly
**Workaround**: Use powered USB hub or reduce SDR gain

### Issue: MAVLink Connection Timeout

**Symptom**: No heartbeat received
**Workaround**: Check baud rate (115200), verify SYSID_THISMAV

### Issue: Signal Processing CPU Spike

**Symptom**: High CPU usage >80%
**Workaround**: Reduce FFT size or lower update rate

## Troubleshooting Guide

### Test Failures

1. **Import Errors**
   - Verify PYTHONPATH includes src/
   - Check uv sync completed successfully

2. **Async Test Timeouts**
   - Increase timeout in pytest.ini
   - Check for blocking I/O operations

3. **Hardware Not Found**
   - Verify device permissions (dialout group)
   - Check USB connections and power

4. **Coverage Below Threshold**
   - Add missing test cases
   - Mock external dependencies properly

## Test Coverage Report

### Current Coverage Status

| Component        | Coverage | Target | Status |
| ---------------- | -------- | ------ | ------ |
| Safety System    | 70%      | 100%   | ⚠️     |
| MAVLink Service  | 14%      | 80%    | ❌     |
| Signal Processor | 0%       | 80%    | ❌     |
| State Machine    | 0%       | 90%    | ❌     |
| Config Service   | 0%       | 60%    | ❌     |

### Coverage Gaps

- MAVLink command sending paths
- Signal processing edge cases
- Emergency recovery procedures
- Configuration validation

## Lessons Learned

### What Worked Well

- Mocked SITL tests for CI/CD
- Comprehensive safety interlock testing
- Timing validation approach
- Test result archival system

### Areas for Improvement

- Need more realistic signal simulations
- Better integration test fixtures
- Automated performance regression testing
- Frontend E2E test coverage

## Testing Checklist

### Before Each Test Session

- [ ] Review test environment setup
- [ ] Check all dependencies installed
- [ ] Verify hardware connections (if applicable)
- [ ] Review recent code changes
- [ ] Clear previous test artifacts

### During Testing

- [ ] Monitor system resources
- [ ] Document any anomalies
- [ ] Save all test outputs
- [ ] Note environmental conditions
- [ ] Track timing metrics

### After Testing

- [ ] Archive test results
- [ ] Update test documentation
- [ ] File bug reports for failures
- [ ] Review coverage reports
- [ ] Plan remediation for gaps

## Continuous Improvement

### Planned Enhancements

1. Automated regression test suite
2. Performance benchmarking framework
3. Chaos engineering tests
4. Load testing for concurrent operations
5. Security penetration testing

### Metrics to Track

- Test execution time trends
- Failure rate by component
- Time to fix test failures
- Coverage trend over time
- Performance regression detection

## Compliance and Audit

### Test Evidence Requirements

- All safety tests must have timestamped results
- HIL tests require video documentation
- Flight tests need signed checklists
- Retention period: 2 years minimum

### Audit Trail

- Test run IDs for traceability
- System configuration snapshots
- Environmental conditions logged
- Operator identification required

## References

- [Testing Strategy](../architecture/testing-strategy.md)
- [Safety Requirements](../prd/epic-2-flight-integration-safety-systems.md)
- [Emergency Procedures](emergency_procedures.md)
- [Pre-flight Checklist](preflight_safety_checklist.md)
