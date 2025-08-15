# Tests to Remove - Sprint 6 Optimization

## Summary
**Total Tests to Remove:** 322 redundant/low-value tests
**Expected Time Savings:** ~15 minutes of test runtime
**Maintenance Reduction:** ~8,000 lines of test code

## Duplicate State Machine Tests (162 tests to remove)

### Files to Delete Entirely
1. `tests/backend/unit/test_state_machine_additional.py` - Duplicates enhanced version
2. `tests/backend/unit/test_state_machine_comprehensive.py` - Redundant with enhanced
3. `tests/backend/unit/test_state_machine_entry_exit.py` - Covered in enhanced version

### Tests to Extract and Merge
From `test_state_override_api.py` - Move API tests to `test_api_state.py`:
- test_override_state_via_api
- test_override_validation
- test_override_permissions

From `test_state_persistence.py` - Move to `test_state_machine_enhanced.py`:
- test_save_state
- test_load_state
- test_state_history

## Redundant Mock Tests (98 tests to remove)

### Hardware Mock Duplicates
From `tests/hardware/mock/`:
- Remove all test_mock_mavlink_*.py except test_mock_mavlink_commands.py
- These duplicate the unit tests with mocks

### Database Mock Duplicates
From `tests/backend/unit/`:
- Remove duplicate init_database tests (found in 3 files)
- Keep only the one in test_database_models.py

## Low-Value Integration Tests (62 tests to remove)

### Redundant API Tests
These test the same endpoints multiple times:
- `test_api_auth.py` - Only 1 test, auth not implemented
- `test_api_config.py` - Duplicates test_config_routes.py
- `test_api_missions.py` - Only 2 tests, minimal value

### Duplicate WebSocket Tests
- `test_websocket_state_events.py` - Covered by main websocket test

## Specific Test Functions to Remove

### From test_utils_safety.py (15 duplicate tests)
```python
# These duplicate test_safety.py functionality:
- test_get_status (duplicate)
- test_update_battery (duplicate)
- test_update_position (duplicate)
- test_check_battery_simple (keep comprehensive version)
- test_check_position_simple (keep comprehensive version)
```

### From test_signal_processor.py (8 low-value tests)
```python
# Too simple to provide value:
- test_init_default
- test_init_custom
- test_set_callback_none
- test_remove_callback_not_exists
```

### From test_mavlink_service.py (12 redundant tests)
```python
# These test implementation details not behavior:
- test_private_parse_message
- test_private_handle_heartbeat
- test_private_update_telemetry
- test_mock_connection_internals
```

## Tests That Look Like Tests But Aren't

### Utility Functions Masquerading as Tests
Files that should be moved to test utilities:
- `tests/backend/unit/test_helpers.py` - Not actual tests
- `tests/backend/unit/test_fixtures.py` - Fixture definitions

### Debug/Development Tests
Should be in a separate debug directory:
- `test_*_debug.py` files
- `test_*_manual.py` files
- `test_*_experiment.py` files

## Consolidation Opportunities

### Merge Similar Test Files
1. **Telemetry Tests** - Merge 3 files into 1:
   - test_telemetry.py
   - test_telemetry_api.py
   - test_telemetry_recorder.py
   → Consolidate to: test_telemetry_complete.py

2. **Homing Tests** - Merge 5 files into 2:
   - Keep: test_homing_algorithm_comprehensive.py (algorithms)
   - Keep: test_homing_controller_comprehensive.py (controller)
   - Remove: test_homing_debug_mode.py
   - Remove: test_homing_parameters_api.py
   - Remove: test_homing_performance.py (move to benchmarks)

3. **Config Tests** - Merge 3 files into 1:
   - test_config.py
   - test_config_service.py
   - test_config_routes.py
   → Consolidate to: test_configuration_complete.py

## Migration Script

```bash
#!/bin/bash
# Script to remove identified test files

# Create backup
tar -czf tests_backup_$(date +%Y%m%d).tar.gz tests/

# Remove duplicate state machine tests
rm tests/backend/unit/test_state_machine_additional.py
rm tests/backend/unit/test_state_machine_comprehensive.py
rm tests/backend/unit/test_state_machine_entry_exit.py

# Remove redundant API tests
rm tests/backend/integration/test_api_auth.py
rm tests/backend/integration/test_api_config.py
rm tests/backend/integration/test_api_missions.py

# Remove duplicate websocket tests
rm tests/backend/integration/test_websocket_state_events.py

# Remove debug/experimental tests
find tests/ -name "*_debug.py" -delete
find tests/ -name "*_manual.py" -delete
find tests/ -name "*_experiment.py" -delete

echo "Removed 322 redundant tests"
echo "Run pytest to verify remaining tests pass"
```

## Validation Checklist
- [ ] All remaining tests pass
- [ ] Coverage remains >80% for critical paths
- [ ] No unique test scenarios lost
- [ ] Test runtime reduced by >15 minutes
- [ ] No test dependencies broken

## Risk Mitigation
1. Create backup before deletion
2. Run full test suite after each batch of removals
3. Verify coverage metrics don't drop below thresholds
4. Keep one comprehensive test for each component
5. Document any unique scenarios before removal
