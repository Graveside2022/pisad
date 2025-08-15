"""
Mock MAVLink interface for testing.
"""

import asyncio
from dataclasses import dataclass
from typing import Any

from tests.hardware.mock.base import MockHardwareConfig, MockHardwareInterface


@dataclass
class MockMAVLinkMessage:
    """Mock MAVLink message structure."""

    msg_type: str
    data: dict[str, Any]
    timestamp: float = 0.0


class MockMAVLinkInterface(MockHardwareInterface):
    """Mock implementation of MAVLink communication interface."""

    def __init__(self, config: MockHardwareConfig | None = None):
        """Initialize mock MAVLink with optional configuration."""
        super().__init__(config)
        self.message_queue: list[MockMAVLinkMessage] = []
        self.telemetry = {
            "battery_voltage": 22.2,
            "battery_percent": 85.0,
            "flight_mode": "GUIDED",
            "armed": False,
            "gps_fix": 3,
            "satellites": 12,
            "altitude": 0.0,
            "latitude": 0.0,
            "longitude": 0.0,
            "heading": 0.0,
            "groundspeed": 0.0,
            "airspeed": 0.0,
        }

    async def connect(self) -> bool:
        """Connect to the mock flight controller."""
        self._track_call("connect")
        await self._simulate_delay()

        if self._should_fail():
            self._error_count += 1
            return False

        self._is_connected = True
        return True

    async def disconnect(self) -> bool:
        """Disconnect from the mock flight controller."""
        self._track_call("disconnect")
        await self._simulate_delay()

        if self._should_fail():
            self._error_count += 1
            return False

        self._is_connected = False
        return True

    async def configure(self, **kwargs) -> bool:
        """Configure the mock MAVLink interface."""
        self._track_call("configure")
        await self._simulate_delay()

        if self._should_fail():
            self._error_count += 1
            return False

        # Update configuration parameters
        for key, value in kwargs.items():
            if key in self.telemetry:
                self.telemetry[key] = value

        return True

    async def start_streaming(self) -> bool:
        """Start telemetry streaming."""
        self._track_call("start_streaming")
        await self._simulate_delay()

        if not self._is_connected:
            return False

        if self._should_fail():
            self._error_count += 1
            return False

        self._is_streaming = True
        return True

    async def stop_streaming(self) -> bool:
        """Stop telemetry streaming."""
        self._track_call("stop_streaming")
        await self._simulate_delay()

        if self._should_fail():
            self._error_count += 1
            return False

        self._is_streaming = False
        return True

    async def get_status(self) -> dict[str, Any]:
        """Get current MAVLink connection status."""
        self._track_call("get_status")
        await self._simulate_delay()

        return {
            "connected": self._is_connected,
            "streaming": self._is_streaming,
            "telemetry": self.telemetry.copy(),
            "message_count": len(self.message_queue),
            "error_count": self._error_count,
            "device_id": self.config.device_id,
        }

    async def send_command(self, command: str, params: dict[str, Any]) -> bool:
        """Send a mock command to the flight controller."""
        self._track_call("send_command")
        await self._simulate_delay()

        if not self._is_connected:
            return False

        if self._should_fail():
            self._error_count += 1
            return False

        # Add command to message queue
        msg = MockMAVLinkMessage(
            msg_type=f"COMMAND_{command}", data=params, timestamp=asyncio.get_event_loop().time()
        )
        self.message_queue.append(msg)

        return True

    async def get_telemetry(self) -> dict[str, Any]:
        """Get current telemetry data."""
        self._track_call("get_telemetry")
        await self._simulate_delay()

        if not self._is_connected:
            return {}

        if self._should_fail():
            self._error_count += 1
            raise RuntimeError("Mock telemetry error")

        return self.telemetry.copy()

    async def arm(self) -> bool:
        """Arm the mock vehicle."""
        self._track_call("arm")
        if await self.send_command("ARM", {"force": False}):
            self.telemetry["armed"] = True
            return True
        return False

    async def disarm(self) -> bool:
        """Disarm the mock vehicle."""
        self._track_call("disarm")
        if await self.send_command("DISARM", {"force": False}):
            self.telemetry["armed"] = False
            return True
        return False

    async def set_mode(self, mode: str) -> bool:
        """Set flight mode."""
        self._track_call("set_mode")
        if await self.send_command("SET_MODE", {"mode": mode}):
            self.telemetry["flight_mode"] = mode
            return True
        return False

    async def send_velocity_command(self, vx: float, vy: float, vz: float, yaw_rate: float) -> bool:
        """Send velocity command in NED frame."""
        self._track_call("send_velocity_command")
        return await self.send_command(
            "VELOCITY_NED", {"vx": vx, "vy": vy, "vz": vz, "yaw_rate": yaw_rate}
        )
