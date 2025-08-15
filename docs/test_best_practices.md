# PISAD Test Best Practices Guide

## Executive Summary

This guide documents the best practices for writing, organizing, and maintaining tests in the PISAD project. Following these practices ensures:
- **100% Test Value Ratio**: Every test traces to user value or safety requirements
- **100% Hazard Coverage**: All HARA hazards have mitigation tests
- **<4 Minute Execution**: Entire test suite runs in under 4 minutes
- **Zero Flaky Tests**: Deterministic execution every time

## Core Principles

### 1. Backwards Analysis First
Every test MUST start with backwards analysis from user impact:

```python
def test_battery_monitoring_prevents_crash():
    """
    BACKWARDS ANALYSIS:
    - User Action: Operating drone with depleting battery
    - Expected Result: Safe RTL before critical voltage
    - Failure Impact: Uncontrolled descent, vehicle loss, mission failure

    REQUIREMENT TRACE:
    - Hazard: HARA-PWR-001 (Low battery crash)
    - User Story: #2.2 (Safety monitoring)

    TEST VALUE: Prevents $50K vehicle loss and potential injuries
    """
    # Test implementation
```

### 2. Test Categories and Organization

#### Directory Structure
```
tests/
├── unit/              # <100ms per test, isolated functions
├── integration/       # <1s per test, component interactions
├── e2e/              # <5s per test, full workflows
├── sitl/             # <10s per test, hardware simulation
├── property/         # Invariant testing with Hypothesis
├── contract/         # API schema validation
└── performance/      # Benchmark and regression tests
```

#### Categorization Rules
- **Unit**: Single function/class, no I/O, no external dependencies
- **Integration**: 2-3 components, may use test database
- **E2E**: Complete user workflow, real services
- **SITL**: Hardware-in-loop with ArduPilot SITL
- **Property**: Mathematical invariants and edge cases
- **Contract**: API and protocol compliance
- **Performance**: Speed and resource benchmarks

### 3. Performance Requirements

#### Execution Time Limits
| Category | Per Test | Total Category | Workers |
|----------|----------|----------------|---------|
| Unit | <100ms | <30s | 8 parallel |
| Integration | <1s | <2min | 4 parallel |
| E2E | <5s | <1min | 2 parallel |
| SITL | <10s | <2.5min | Serial |
| Property | <500ms | <30s | 8 parallel |
| Contract | <200ms | <20s | 4 parallel |

#### Resource Limits
- Memory per test: <50MB
- CPU per test: <1 core
- No test should create files >1MB
- Clean up all resources in teardown

## Writing Effective Tests

### 1. Test Naming Convention
```python
def test_<component>_<scenario>_<expected_outcome>():
    """Clear, specific test names that describe behavior"""
    pass

# Good examples:
def test_signal_processor_weak_signal_returns_noise_floor():
def test_battery_monitor_critical_voltage_triggers_emergency_rtl():
def test_homing_controller_signal_loss_disables_autonomous_mode():

# Bad examples:
def test_1():  # Meaningless
def test_signal():  # Too vague
def test_that_it_works():  # What works?
```

### 2. Arrange-Act-Assert Pattern
```python
def test_circuit_breaker_opens_after_threshold_failures():
    # ARRANGE - Set up test conditions
    circuit_breaker = CircuitBreaker(failure_threshold=3)
    failing_callback = Mock(side_effect=Exception("Test error"))

    # ACT - Execute the behavior
    for _ in range(3):
        with pytest.raises(Exception):
            circuit_breaker.call(failing_callback, value=42)

    # ASSERT - Verify the outcome
    assert circuit_breaker.is_open
    assert circuit_breaker.failure_count == 3
```

### 3. Fixture Best Practices
```python
# conftest.py - Shared fixtures
@pytest.fixture
def mock_sdr():
    """Reusable mock SDR for unit tests"""
    sdr = Mock(spec=SDRService)
    sdr.get_rssi.return_value = -75.0
    sdr.get_noise_floor.return_value = -95.0
    return sdr

@pytest.fixture
async def test_database():
    """Isolated database for integration tests"""
    db = await create_test_database()
    yield db
    await db.cleanup()

# Use fixtures for DRY tests
def test_signal_detection(mock_sdr, test_database):
    # Both fixtures automatically injected
    pass
```

### 4. Mocking Guidelines
```python
# Mock at the boundary, not internals
def test_mavlink_command_sending():
    # Good: Mock the serial interface
    with patch('serial.Serial') as mock_serial:
        mavlink = MAVLinkService(port='/dev/ttyACM0')
        mavlink.send_velocity(vx=5.0)
        mock_serial.return_value.write.assert_called()

    # Bad: Mock internal methods
    with patch.object(MAVLinkService, '_format_message'):
        # This tests implementation, not behavior
        pass
```

### 5. Async Test Patterns
```python
@pytest.mark.asyncio
async def test_async_signal_processing():
    """Use pytest-asyncio for async tests"""
    processor = SignalProcessor()

    # Use asyncio primitives properly
    async with processor:
        result = await processor.process_samples(samples)
        assert result.rssi > -100

    # Ensure cleanup
    assert processor.closed
```

## Eliminating Flaky Tests

### 1. Time-Independent Testing
```python
# Bad: Time-dependent
def test_timeout():
    start = time.time()
    time.sleep(1)
    do_something()
    assert time.time() - start > 1  # Flaky!

# Good: Use time control
def test_timeout(freezer):  # freezegun fixture
    freezer.move_to("2024-01-01")
    do_something()
    freezer.move_to("2024-01-01 00:00:01")
    assert timeout_occurred()
```

### 2. Deterministic Random Values
```python
# Bad: Random values
def test_signal_noise():
    noise = random.random() * 10
    assert process_signal(noise) < threshold  # Flaky!

# Good: Seeded random
@pytest.mark.parametrize("seed,expected", [
    (42, -75.3),
    (123, -82.1),
])
def test_signal_noise(seed, expected):
    random.seed(seed)
    assert process_signal() == pytest.approx(expected)
```

### 3. Proper Async Waiting
```python
# Bad: Fixed sleep
async def test_async_operation():
    start_operation()
    await asyncio.sleep(1)  # Hope it's done!
    assert operation_complete()  # Flaky!

# Good: Wait for condition
async def test_async_operation():
    start_operation()
    await wait_for_condition(
        lambda: operation_complete(),
        timeout=1.0
    )
    assert operation_complete()
```

## Property-Based Testing

### 1. Define Invariants
```python
from hypothesis import given, strategies as st

@given(
    voltage=st.floats(min_value=0, max_value=30),
    current=st.floats(min_value=0, max_value=50)
)
def test_power_never_exceeds_limit(voltage, current):
    """Power should never exceed 150W safety limit"""
    power = calculate_power(voltage, current)
    assert power <= 150, f"Power {power}W exceeds limit"
```

### 2. State Machine Testing
```python
from hypothesis.stateful import RuleBasedStateMachine, rule

class DroneStateMachine(RuleBasedStateMachine):
    """Test state transitions maintain invariants"""

    def __init__(self):
        super().__init__()
        self.drone = DroneController()

    @rule()
    def arm(self):
        self.drone.arm()
        assert self.drone.state in ['ARMED', 'ARMING_FAILED']

    @rule()
    def takeoff(self):
        if self.drone.state == 'ARMED':
            self.drone.takeoff()
            assert self.drone.altitude > 0

    @invariant()
    def battery_never_negative(self):
        assert self.drone.battery_percent >= 0
```

## Contract Testing

### 1. API Schema Validation
```python
import schemathesis

@schemathesis.test("/api/openapi.json")
def test_api_contract(case):
    """Validate all API endpoints against OpenAPI spec"""
    response = case.call()
    case.validate_response(response)
```

### 2. Message Protocol Testing
```python
def test_websocket_message_contract():
    """Validate WebSocket messages match schema"""
    schema = {
        "type": "object",
        "properties": {
            "type": {"enum": ["rssi", "state", "telemetry"]},
            "timestamp": {"type": "string", "format": "date-time"},
            "data": {"type": "object"}
        },
        "required": ["type", "timestamp", "data"]
    }

    message = get_websocket_message()
    validate(instance=message, schema=schema)
```

## Performance Testing

### 1. Benchmark Critical Paths
```python
def test_signal_processing_performance(benchmark):
    """Ensure signal processing meets latency requirements"""
    samples = generate_test_samples(1024)

    result = benchmark(process_signal, samples)

    # Assert performance requirements
    assert benchmark.stats['mean'] < 0.040  # <40ms requirement
    assert benchmark.stats['max'] < 0.100   # No outliers >100ms
```

### 2. Memory Profiling
```python
def test_memory_usage():
    """Ensure no memory leaks during operation"""
    import tracemalloc

    tracemalloc.start()

    # Run operation 1000 times
    for _ in range(1000):
        process_data()

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Should not grow beyond 50MB
    assert peak / 1024 / 1024 < 50
```

## Safety Testing Requirements

### 1. Hazard Mitigation Tests
Every HARA hazard MUST have tests:

```python
class TestBatteryHazardMitigation:
    """Tests for HARA-PWR-001: Low battery crash"""

    def test_low_battery_triggers_rtl(self):
        """Verify 19.2V triggers return-to-launch"""
        drone.battery_voltage = 19.2
        assert drone.get_commanded_mode() == "RTL"

    def test_critical_battery_triggers_land(self):
        """Verify 18.0V triggers immediate landing"""
        drone.battery_voltage = 18.0
        assert drone.get_commanded_mode() == "LAND"

    def test_battery_failsafe_cannot_be_overridden(self):
        """Verify operator cannot override battery failsafe"""
        drone.battery_voltage = 17.5
        drone.set_mode("GUIDED")  # Try to override
        assert drone.get_mode() == "LAND"  # Stays in LAND
```

### 2. Boundary Testing
Test exact boundary values:

```python
@pytest.mark.parametrize("voltage,expected", [
    (19.21, "NORMAL"),     # Just above threshold
    (19.20, "RTL"),        # Exact threshold
    (19.19, "RTL"),        # Just below threshold
    (18.01, "RTL"),        # Just above critical
    (18.00, "LAND"),       # Exact critical
    (17.99, "LAND"),       # Below critical
])
def test_battery_thresholds(voltage, expected):
    """Test exact threshold boundaries"""
    drone.battery_voltage = voltage
    assert drone.get_failsafe_action() == expected
```

## Test Metrics and Monitoring

### 1. Required Metrics
Every test run should report:
- Test Value Ratio (target: 100%)
- Hazard Coverage (target: 100%)
- Execution Time (target: <4 minutes)
- Flaky Test Count (target: 0)
- Coverage Percentage (target: >80%)

### 2. Continuous Monitoring
```bash
# Add to CI/CD pipeline
pytest --cov=src --cov-report=term-missing \
       --cov-fail-under=80 \
       -n 8 \
       --timeout=300 \
       --json-report --json-report-file=report.json

# Generate quality dashboard
python scripts/generate_test_dashboard.py

# Check quality gates
python -m src.backend.utils.test_metrics
```

## Common Anti-Patterns to Avoid

### 1. Testing Implementation Instead of Behavior
```python
# Bad: Tests HOW it works
def test_uses_fft():
    processor._use_fft = True
    processor.process()
    assert processor._fft_called  # Implementation detail

# Good: Tests WHAT it does
def test_detects_signal():
    result = processor.process(strong_signal)
    assert result.signal_detected
    assert result.snr > 12
```

### 2. Excessive Mocking
```python
# Bad: Mocks everything
def test_system():
    mock_sdr = Mock()
    mock_processor = Mock()
    mock_mavlink = Mock()
    mock_state = Mock()
    # Not testing anything real!

# Good: Mock boundaries only
def test_system():
    mock_hardware = Mock()  # External boundary
    system = System(hardware=mock_hardware)
    result = system.process()  # Real processing
    assert result.valid
```

### 3. Ignored Test Failures
```python
# Bad: Skipping failures
@pytest.mark.skip("Fails sometimes")
def test_important_feature():
    pass

# Good: Fix or remove
def test_important_feature():
    """Fixed flaky test with proper waiting"""
    pass
```

## Test Maintenance

### 1. Regular Test Audit
Monthly tasks:
- Run test metrics analyzer
- Review untraceable tests
- Update hazard coverage
- Remove obsolete tests
- Fix slow tests

### 2. Test Refactoring
When refactoring production code:
1. Run tests to establish baseline
2. Refactor code
3. Ensure tests still pass
4. Update test names/organization if needed
5. Add new tests for new behaviors

### 3. Documentation Updates
Keep test documentation current:
- Update this guide with new patterns
- Document test utilities and fixtures
- Maintain test coverage reports
- Track historical metrics

## Tooling and Commands

### Essential Tools (from CLAUDE.md)
- **fd**: Find test files (faster than find)
- **rg**: Search test patterns (45x faster than grep)
- **pytest-xdist**: Parallel test execution
- **pytest-benchmark**: Performance testing
- **hypothesis**: Property-based testing
- **pytest-cov**: Coverage reporting
- **freezegun**: Time control for tests

### Useful Commands
```bash
# Run all tests with coverage
pytest --cov=src --cov-report=html

# Run specific category
pytest tests/unit -n 8

# Run with performance profiling
pytest --profile --profile-svg

# Find slow tests
pytest --durations=10

# Run with maximum verbosity
pytest -vvv --tb=short

# Generate test metrics
python -m src.backend.utils.test_metrics

# Generate quality dashboard
./scripts/generate_test_dashboard.py
```

## Checklist for New Tests

Before committing any test, verify:

- [ ] Has backwards analysis documenting user impact
- [ ] Traces to user story or hazard ID
- [ ] Runs in appropriate time limit (<100ms unit, <1s integration)
- [ ] Uses proper fixtures instead of setup duplication
- [ ] Mocks only external boundaries
- [ ] No time.sleep() or fixed delays
- [ ] Cleans up all resources
- [ ] Has descriptive name following convention
- [ ] Actually tests behavior, not implementation
- [ ] Passes consistently (run 10 times)

## Conclusion

Following these best practices ensures our test suite remains:
- **Valuable**: Every test prevents real user problems
- **Fast**: <4 minute execution enables rapid development
- **Reliable**: Zero flaky tests builds confidence
- **Maintainable**: Clear organization and documentation

Remember: A test that doesn't trace to user value or safety is worse than no test at all - it's technical debt that slows development without preventing problems.

---
*Last Updated: 2025-08-16*
*Version: 1.0*
*Maintained by: Test Engineering Team*
