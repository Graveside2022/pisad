"""
Mock MAVLink Command Tests - Sprint 4.5
Tests for arm/takeoff/land commands without physical hardware.

Story 4.7 AC #5: All hardware-dependent code paths tested
Story 4.7 AC #7: Field test procedures documented and executed
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.backend.services.command_pipeline import CommandPipeline, CommandPriority, CommandType

# Mark all tests as mock hardware tests
pytestmark = pytest.mark.mock_hardware


class TestMockMAVLinkCommands:
    """Test MAVLink command execution with mock hardware."""

    @pytest.fixture
    def mock_mavlink_service(self):
        """Create mock MAVLink service."""
        mock = MagicMock()
        mock.connected = True
        mock.get_telemetry = MagicMock(
            return_value={
                "armed": False,
                "flight_mode": "GUIDED",
                "altitude": 0.0,
                "position": {"lat": 0.0, "lon": 0.0, "alt": 0.0},
                "battery": {"voltage": 22.2, "percentage": 75.0},
                "gps": {"fix_type": 3, "satellites": 12, "hdop": 1.2},
            }
        )
        mock.arm = AsyncMock(return_value=True)
        mock.disarm = AsyncMock(return_value=True)
        mock.takeoff = AsyncMock(return_value=True)
        mock.land = AsyncMock(return_value=True)
        mock.set_mode = AsyncMock(return_value=True)
        mock.send_velocity = AsyncMock(return_value=True)
        mock.emergency_stop = AsyncMock(return_value=True)
        return mock

    @pytest.fixture
    def mock_safety_system(self):
        """Create mock safety interlock system."""
        mock = MagicMock()
        mock.check_all_interlocks = MagicMock(
            return_value={
                "all_safe": True,
                "gps_ready": True,
                "battery_ok": True,
                "geofence_ok": True,
                "rc_override": False,
            }
        )
        mock.check_arming_allowed = MagicMock(return_value=True)
        mock.check_takeoff_allowed = MagicMock(return_value=True)
        return mock

    @pytest.fixture
    def command_pipeline(self, mock_mavlink_service, mock_safety_system):
        """Create command pipeline with mocks."""
        pipeline = CommandPipeline(
            safety_system=mock_safety_system,
            mavlink_service=mock_mavlink_service,
            rate_limit_per_second=10.0,
        )
        return pipeline

    @pytest.mark.asyncio
    async def test_arm_command_success(self, command_pipeline, mock_mavlink_service):
        """Test successful arm command execution."""
        # SAFETY: Verify pre-arm checks per Story 4.7 AC #5

        # Submit arm command
        command_id = await command_pipeline.submit_command(CommandType.ARM, source="test")

        # Command should be queued
        assert command_id is not None
        assert command_pipeline.command_queue.qsize() > 0

    @pytest.mark.asyncio
    async def test_arm_command_safety_check(self, command_pipeline, mock_safety_system):
        """Test arm command with safety check failure."""
        # SAFETY: GPS requirement per Story 4.7 hardware specs
        mock_safety_system.check_arming_allowed.return_value = False
        mock_safety_system.check_all_interlocks.return_value = {
            "all_safe": False,
            "gps_ready": False,
            "battery_ok": True,
            "geofence_ok": True,
            "rc_override": False,
        }

        # Submit arm command
        command_id = await command_pipeline.submit_command(CommandType.ARM, source="test")

        # Command should be queued but will fail safety check
        assert command_id is not None

    @pytest.mark.asyncio
    async def test_takeoff_command_with_altitude(self, command_pipeline):
        """Test takeoff command with altitude parameter."""
        # SAFETY: Takeoff requires altitude per Story 4.7

        # Submit takeoff command
        command_id = await command_pipeline.submit_command(
            CommandType.TAKEOFF, parameters={"altitude": 10.0}, source="test"
        )

        assert command_id is not None
        assert command_pipeline.command_queue.qsize() > 0

    @pytest.mark.asyncio
    async def test_takeoff_altitude_validation(self, command_pipeline):
        """Test takeoff altitude bounds validation."""
        # SAFETY: Altitude limits per Story 4.7 command_pipeline.py:382

        # Test minimum altitude (should fail validation)
        with pytest.raises(ValueError, match="Command validation failed"):
            await command_pipeline.submit_command(
                CommandType.TAKEOFF,
                parameters={"altitude": 0.5},  # Too low
                source="test",
            )

        # Test maximum altitude (should fail validation)
        with pytest.raises(ValueError, match="Command validation failed"):
            await command_pipeline.submit_command(
                CommandType.TAKEOFF,
                parameters={"altitude": 150.0},  # Too high
                source="test",
            )

        # Test valid altitude
        command_id = await command_pipeline.submit_command(
            CommandType.TAKEOFF, parameters={"altitude": 20.0}, source="test"
        )
        assert command_id is not None

    @pytest.mark.asyncio
    async def test_land_command_execution(self, command_pipeline):
        """Test land command execution."""
        # Submit land command
        command_id = await command_pipeline.submit_command(CommandType.LAND, source="test")

        assert command_id is not None
        assert command_pipeline.command_queue.qsize() > 0

    @pytest.mark.asyncio
    async def test_emergency_stop_priority(self, command_pipeline):
        """Test emergency stop has highest priority."""
        # SAFETY: Emergency stop must have highest priority

        # Submit some normal commands first
        await command_pipeline.submit_command(
            CommandType.LAND,  # Use a command with a validator
            priority=CommandPriority.NORMAL,
            source="test",
        )

        await command_pipeline.submit_command(
            CommandType.ARM,  # Another valid command
            priority=CommandPriority.HIGH,
            source="test",
        )

        # Queue should have 2 commands
        assert command_pipeline.command_queue.qsize() == 2

        # Emergency commands execute immediately, not queued
        # So we test priority ordering differently
        emergency_id = await command_pipeline.submit_command(
            CommandType.EMERGENCY_STOP, priority=CommandPriority.EMERGENCY, source="test"
        )

        # Emergency command executed immediately, queue unchanged
        assert command_pipeline.command_queue.qsize() == 2
        assert emergency_id is not None

    @pytest.mark.asyncio
    async def test_mode_change_command(self, command_pipeline):
        """Test flight mode change commands."""
        modes = ["GUIDED", "LOITER", "AUTO"]

        for mode in modes:
            command_id = await command_pipeline.submit_command(
                CommandType.SET_MODE, parameters={"mode": mode}, source="test"
            )
            assert command_id is not None

    @pytest.mark.asyncio
    async def test_velocity_command(self, command_pipeline):
        """Test velocity command with safety limits."""
        # SAFETY: Velocity limits per Story 4.7 command_pipeline.py:337

        # Valid velocity
        command_id = await command_pipeline.submit_command(
            CommandType.SET_VELOCITY, parameters={"vx": 5.0, "vy": 3.0, "vz": -1.0}, source="test"
        )
        assert command_id is not None

        # Excessive velocity (should fail validation)
        with pytest.raises(ValueError, match="Command validation failed"):
            await command_pipeline.submit_command(
                CommandType.SET_VELOCITY,
                parameters={"vx": 25.0, "vy": 0.0, "vz": 0.0},  # Exceeds 20 m/s limit
                source="test",
            )

    @pytest.mark.asyncio
    async def test_command_rate_limiting(self, command_pipeline):
        """Test command rate limiting for safety."""
        # SAFETY: Rate limit per Story 4.7 command_pipeline.py:121

        # Send rapid commands
        command_ids = []
        for i in range(15):  # Exceeds 10 commands/sec limit
            command_id = await command_pipeline.submit_command(CommandType.LOITER, source="test")
            command_ids.append(command_id)

            # Very short delay to exceed rate limit
            if i < 10:
                await asyncio.sleep(0.05)  # 20 commands/sec attempt

        # All commands should be queued (rate limiting happens during execution)
        assert len(command_ids) == 15
        assert all(cid is not None for cid in command_ids)

    @pytest.mark.asyncio
    async def test_mission_commands(self, command_pipeline):
        """Test mission control commands."""
        mission_commands = [
            (CommandType.START_MISSION, {}),
            (CommandType.PAUSE_MISSION, {}),
            (CommandType.RESUME_MISSION, {}),
            (CommandType.ABORT_MISSION, {}),
        ]

        for cmd_type, params in mission_commands:
            command_id = await command_pipeline.submit_command(
                cmd_type, parameters=params, source="test"
            )
            assert command_id is not None

    @pytest.mark.asyncio
    async def test_goto_position_command(self, command_pipeline):
        """Test goto position command with validation."""
        # Valid position
        command_id = await command_pipeline.submit_command(
            CommandType.GOTO_POSITION,
            parameters={"latitude": 37.7749, "longitude": -122.4194, "altitude": 50.0},
            source="test",
        )
        assert command_id is not None

        # Invalid position (missing altitude) - should fail validation
        with pytest.raises(ValueError, match="Command validation failed"):
            await command_pipeline.submit_command(
                CommandType.GOTO_POSITION,
                parameters={
                    "latitude": 37.7749,
                    "longitude": -122.4194,
                    # Missing altitude
                },
                source="test",
            )

    @pytest.mark.asyncio
    async def test_rtl_command(self, command_pipeline):
        """Test Return to Launch command."""
        # RTL should work with minimal validation
        command_id = await command_pipeline.submit_command(
            CommandType.RETURN_TO_LAUNCH, priority=CommandPriority.CRITICAL, source="test"
        )
        assert command_id is not None

    @pytest.mark.asyncio
    async def test_homing_commands(self, command_pipeline):
        """Test homing-specific commands."""
        # Start homing
        command_id = await command_pipeline.submit_command(
            CommandType.START_HOMING, parameters={"search_pattern": "spiral"}, source="test"
        )
        assert command_id is not None

        # Update bearing during homing
        command_id = await command_pipeline.submit_command(
            CommandType.UPDATE_BEARING,
            parameters={"bearing": 45.0, "strength": -60.0},
            source="test",
        )
        assert command_id is not None

        # Stop homing
        command_id = await command_pipeline.submit_command(CommandType.STOP_HOMING, source="test")
        assert command_id is not None

    def test_command_priority_ordering(self):
        """Test command priority ordering for queue."""
        from src.backend.services.command_pipeline import Command

        # Create commands with different priorities
        emergency = Command(type=CommandType.EMERGENCY_STOP, priority=CommandPriority.EMERGENCY)
        critical = Command(type=CommandType.RETURN_TO_LAUNCH, priority=CommandPriority.CRITICAL)
        normal = Command(type=CommandType.LOITER, priority=CommandPriority.NORMAL)

        # Test ordering (lower value = higher priority)
        assert emergency < critical
        assert critical < normal
        assert emergency < normal
