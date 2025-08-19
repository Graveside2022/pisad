"""
Test suite for SDR Priority Manager

Tests priority decision making, conflict resolution, and safety override
functionality with authentic integration points.

PRD References:
- FR11: Operator override capability
- FR15: Immediate command cessation on mode change
- NFR2: <100ms latency requirement
- NFR12: Deterministic timing for safety-critical functions
"""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.backend.services.sdr_priority_manager import (
    SDRPriorityManager,
    SDRPriorityMatrix,
    SignalQuality,
)


class TestSDRPriorityMatrix:
    """Test priority matrix signal quality scoring."""

    def test_signal_quality_scoring_basic(self):
        """Test basic signal quality scoring algorithm."""
        # RED PHASE: Write failing test for signal quality scoring
        matrix = SDRPriorityMatrix()

        # Test signal quality calculation with known values
        ground_quality = matrix.calculate_signal_quality(rssi=-45.0, snr=15.0, stability=0.9)
        drone_quality = matrix.calculate_signal_quality(rssi=-52.0, snr=12.0, stability=0.8)

        # Ground should score higher (better RSSI and SNR)
        assert ground_quality.score > drone_quality.score
        assert ground_quality.confidence > 0.7
        assert drone_quality.confidence > 0.6

    def test_priority_decision_hysteresis(self):
        """Test hysteresis prevents oscillation between sources."""
        # RED PHASE: Test hysteresis logic prevents rapid switching
        matrix = SDRPriorityMatrix(hysteresis_threshold=3.0)

        # Current source is ground with slight advantage
        current_source = "ground"
        ground_quality = SignalQuality(score=75.0, confidence=0.8, rssi=-45.0)
        drone_quality = SignalQuality(score=73.0, confidence=0.8, rssi=-47.0)

        decision = matrix.make_priority_decision(ground_quality, drone_quality, current_source)

        # Should stay with ground due to hysteresis (difference < threshold)
        assert decision.selected_source == "ground"
        assert decision.reason == "hysteresis_maintained"

    def test_priority_decision_emergency_override(self):
        """Test emergency override always selects drone for safety."""
        # RED PHASE: Test safety override functionality
        matrix = SDRPriorityMatrix()

        ground_quality = SignalQuality(score=90.0, confidence=0.9, rssi=-40.0)
        drone_quality = SignalQuality(score=60.0, confidence=0.7, rssi=-55.0)

        # Emergency override should force drone selection
        decision = matrix.make_priority_decision(
            ground_quality, drone_quality, "ground", emergency_override=True
        )

        assert decision.selected_source == "drone"
        assert decision.reason == "emergency_override"
        assert decision.safety_critical is True


class TestSDRPriorityManager:
    """Test SDR priority manager functionality."""

    @pytest.fixture
    def mock_coordinator(self):
        """Mock coordination service."""
        coordinator = MagicMock()
        coordinator.active_source = "drone"
        coordinator.get_best_rssi = AsyncMock(return_value=-50.0)
        coordinator.select_best_source = AsyncMock(return_value="ground")
        return coordinator

    @pytest.fixture
    def mock_safety_manager(self):
        """Mock safety manager."""
        safety = MagicMock()
        safety.trigger_emergency_stop = MagicMock(
            return_value={"success": True, "response_time_ms": 250.0}
        )
        return safety

    @pytest.fixture
    def priority_manager(self, mock_coordinator, mock_safety_manager):
        """Create priority manager with mocked dependencies."""
        return SDRPriorityManager(coordinator=mock_coordinator, safety_manager=mock_safety_manager)

    @pytest.mark.asyncio
    async def test_automatic_source_switching(self, priority_manager):
        """Test automatic switching based on signal strength."""
        # RED PHASE: Test automatic switching logic

        # Mock ground signal becoming stronger
        priority_manager._coordinator.get_ground_rssi = MagicMock(return_value=-40.0)
        priority_manager._coordinator.get_drone_rssi = MagicMock(return_value=-55.0)

        # Execute switching decision
        decision = await priority_manager.evaluate_source_switch()

        assert decision.selected_source == "ground"
        assert decision.switch_recommended is True
        assert decision.latency_ms < 100.0  # PRD-NFR2 requirement

    @pytest.mark.asyncio
    async def test_conflict_resolution_frequency_commands(self, priority_manager):
        """Test conflict resolution for conflicting frequency commands."""
        # RED PHASE: Test frequency command conflict resolution

        # Simulate conflicting commands
        ground_command = {"frequency": 2.4e9, "source": "ground", "timestamp": time.time()}
        drone_command = {"frequency": 2.45e9, "source": "drone", "timestamp": time.time() + 0.1}

        resolution = await priority_manager.resolve_frequency_conflict(
            ground_command, drone_command
        )

        # Drone command should win for safety (newer timestamp, safety authority)
        assert resolution.selected_command == drone_command
        assert resolution.conflict_type == "drone_command_newer"
        assert resolution.resolution_time_ms < 50.0  # Fast conflict resolution

    @pytest.mark.asyncio
    async def test_safety_override_emergency_control(self, priority_manager):
        """Test safety override ensures drone maintains emergency control."""
        # RED PHASE: Test emergency safety override

        # Trigger emergency scenario
        emergency_result = await priority_manager.trigger_emergency_override()

        # Verify drone gains control and safety systems activate
        assert emergency_result["source_switched_to"] == "drone"
        assert emergency_result["safety_activated"] is True
        assert emergency_result["response_time_ms"] < 500.0  # Safety requirement

    @pytest.mark.asyncio
    async def test_graceful_degradation_communication_loss(self, priority_manager):
        """Test graceful degradation on ground communication loss."""
        # RED PHASE: Test communication loss fallback

        # Simulate ground communication loss
        priority_manager._coordinator.ground_connection_active = False

        degradation = await priority_manager.handle_communication_loss()

        assert degradation["fallback_source"] == "drone"
        assert degradation["degradation_time_s"] < 10.0  # PRD requirement
        assert degradation["safety_maintained"] is True

    @pytest.mark.asyncio
    async def test_priority_status_reporting(self, priority_manager):
        """Test priority status reporting for operator awareness."""
        # RED PHASE: Test status reporting functionality

        status = await priority_manager.get_priority_status()

        assert "active_source" in status
        assert "priority_score" in status
        assert "conflict_history" in status
        assert "emergency_override_active" in status
        assert "last_decision_timestamp" in status

    def test_performance_latency_requirements(self, priority_manager):
        """Test priority decisions meet <100ms latency requirements."""
        # RED PHASE: Test performance requirements

        start_time = time.perf_counter()

        # Execute priority decision
        ground_quality = SignalQuality(score=80.0, confidence=0.8, rssi=-45.0)
        drone_quality = SignalQuality(score=75.0, confidence=0.8, rssi=-50.0)

        decision = priority_manager._matrix.make_priority_decision(
            ground_quality, drone_quality, "drone"
        )

        latency_ms = (time.perf_counter() - start_time) * 1000

        assert latency_ms < 100.0  # PRD-NFR2 requirement
        assert decision.latency_ms < 100.0
