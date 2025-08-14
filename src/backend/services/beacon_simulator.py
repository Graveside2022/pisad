"""Beacon simulator service for HIL testing.

Simulates RF beacon transmissions for hardware-in-the-loop testing
without requiring physical beacon hardware.
"""

import asyncio
import math
import random
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from backend.models.schemas import BeaconConfiguration
from backend.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SimulatedBeacon:
    """Simulated beacon instance."""

    beacon_id: str
    config: BeaconConfiguration
    position: tuple[float, float, float]  # lat, lon, alt in meters
    active: bool = False
    start_time: datetime | None = None
    transmit_count: int = 0


class BeaconSimulator:
    """Service for simulating RF beacon transmissions."""

    def __init__(self):
        """Initialize beacon simulator."""
        self.beacons: dict[str, SimulatedBeacon] = {}
        self.rssi_callback: Callable[[float], None] | None = None
        self.simulation_task: asyncio.Task | None = None
        self.noise_floor_dbm = -110.0
        self.propagation_factor = 2.0  # Free space path loss exponent
        self.reference_distance_m = 1.0  # Reference distance for path loss
        self.reference_loss_db = 40.0  # Loss at reference distance

        # Load beacon profiles
        self.profiles = self._load_profiles()

    def _load_profiles(self) -> dict[str, Any]:
        """Load beacon configuration profiles."""
        try:
            profile_path = Path("config/profiles/field_test_beacon.yaml")
            if profile_path.exists():
                with open(profile_path) as f:
                    config = yaml.safe_load(f)
                    return config.get("profiles", {})
            return {}
        except Exception as e:
            logger.error(f"Failed to load beacon profiles: {e}")
            return {}

    def create_beacon(
        self,
        beacon_id: str,
        config: BeaconConfiguration,
        position: tuple[float, float, float],
    ) -> SimulatedBeacon:
        """Create a new simulated beacon.

        Args:
            beacon_id: Unique beacon identifier
            config: Beacon configuration
            position: Beacon position (lat, lon, alt)

        Returns:
            Created beacon instance
        """
        beacon = SimulatedBeacon(beacon_id=beacon_id, config=config, position=position)
        self.beacons[beacon_id] = beacon
        logger.info(f"Created simulated beacon {beacon_id} at {position}")
        return beacon

    def create_from_profile(
        self, beacon_id: str, profile_name: str, position: tuple[float, float, float]
    ) -> SimulatedBeacon | None:
        """Create beacon from a configuration profile.

        Args:
            beacon_id: Unique beacon identifier
            profile_name: Name of configuration profile
            position: Beacon position (lat, lon, alt)

        Returns:
            Created beacon or None if profile not found
        """
        profile = self.profiles.get(profile_name)
        if not profile:
            logger.error(f"Profile {profile_name} not found")
            return None

        config = BeaconConfiguration(
            frequency_hz=profile.get("frequency_hz", 433000000),
            power_dbm=profile.get("power_dbm", 10),
            modulation=profile.get("modulation", "LoRa"),
            spreading_factor=profile.get("spreading_factor", 7),
            bandwidth_hz=profile.get("bandwidth_hz", 125000),
            coding_rate=profile.get("coding_rate", 5),
            pulse_rate_hz=profile.get("pulse_rate_hz", 1.0),
            pulse_duration_ms=profile.get("pulse_duration_ms", 100),
        )

        return self.create_beacon(beacon_id, config, position)

    async def start_beacon(self, beacon_id: str) -> bool:
        """Start transmitting from a beacon.

        Args:
            beacon_id: Beacon identifier

        Returns:
            True if started successfully
        """
        beacon = self.beacons.get(beacon_id)
        if not beacon:
            logger.error(f"Beacon {beacon_id} not found")
            return False

        beacon.active = True
        beacon.start_time = datetime.now(UTC)
        logger.info(f"Started beacon {beacon_id} transmission")

        # Start simulation if not running
        if not self.simulation_task or self.simulation_task.done():
            self.simulation_task = asyncio.create_task(self._simulation_loop())

        return True

    async def stop_beacon(self, beacon_id: str) -> bool:
        """Stop transmitting from a beacon.

        Args:
            beacon_id: Beacon identifier

        Returns:
            True if stopped successfully
        """
        beacon = self.beacons.get(beacon_id)
        if not beacon:
            logger.error(f"Beacon {beacon_id} not found")
            return False

        beacon.active = False
        logger.info(f"Stopped beacon {beacon_id} transmission")

        # Stop simulation if no beacons active
        if (
            not any(b.active for b in self.beacons.values())
            and self.simulation_task
            and not self.simulation_task.done()
        ):
            self.simulation_task.cancel()

        return True

    def set_beacon_position(self, beacon_id: str, position: tuple[float, float, float]) -> bool:
        """Update beacon position.

        Args:
            beacon_id: Beacon identifier
            position: New position (lat, lon, alt)

        Returns:
            True if updated successfully
        """
        beacon = self.beacons.get(beacon_id)
        if not beacon:
            return False

        beacon.position = position
        logger.debug(f"Updated beacon {beacon_id} position to {position}")
        return True

    def set_beacon_power(self, beacon_id: str, power_dbm: float) -> bool:
        """Update beacon transmission power.

        Args:
            beacon_id: Beacon identifier
            power_dbm: Transmission power in dBm

        Returns:
            True if updated successfully
        """
        beacon = self.beacons.get(beacon_id)
        if not beacon:
            return False

        beacon.config.power_dbm = power_dbm
        logger.debug(f"Updated beacon {beacon_id} power to {power_dbm} dBm")
        return True

    def calculate_rssi(
        self, beacon_id: str, receiver_position: tuple[float, float, float]
    ) -> float:
        """Calculate RSSI at receiver position from beacon.

        Args:
            beacon_id: Beacon identifier
            receiver_position: Receiver position (lat, lon, alt)

        Returns:
            Calculated RSSI in dBm
        """
        beacon = self.beacons.get(beacon_id)
        if not beacon or not beacon.active:
            return self.noise_floor_dbm

        # Calculate distance
        distance_m = self._calculate_distance(beacon.position, receiver_position)

        # Path loss calculation (simplified free space model)
        if distance_m <= self.reference_distance_m:
            path_loss_db = self.reference_loss_db
        else:
            path_loss_db = self.reference_loss_db + 10 * self.propagation_factor * math.log10(
                distance_m / self.reference_distance_m
            )

        # Calculate received power
        rssi = beacon.config.power_dbm - path_loss_db

        # Add noise and fading
        rssi += random.gauss(0, 2)  # Add Gaussian noise

        # Apply pulse modulation
        pulse_period = 1.0 / beacon.config.pulse_rate_hz
        pulse_duty = beacon.config.pulse_duration_ms / 1000.0
        time_offset = (datetime.now(UTC).timestamp() % pulse_period) / pulse_period

        if time_offset > pulse_duty:
            # During off period
            rssi = self.noise_floor_dbm

        # Clamp to realistic range
        rssi = max(self.noise_floor_dbm, min(-20, rssi))

        return rssi

    def _calculate_distance(
        self, pos1: tuple[float, float, float], pos2: tuple[float, float, float]
    ) -> float:
        """Calculate distance between two positions.

        Uses simplified Euclidean distance for HIL testing.
        In real implementation, use haversine formula for GPS coordinates.

        Args:
            pos1: First position (lat, lon, alt)
            pos2: Second position (lat, lon, alt)

        Returns:
            Distance in meters
        """
        # Simplified calculation for testing
        # Assume positions are in meters for HIL testing
        dx = pos2[0] - pos1[0]
        dy = pos2[1] - pos1[1]
        dz = pos2[2] - pos1[2]
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    async def _simulation_loop(self):
        """Main simulation loop for active beacons."""
        logger.info("Started beacon simulation loop")

        try:
            while any(b.active for b in self.beacons.values()):
                # Simulate transmissions from all active beacons
                for beacon in self.beacons.values():
                    if beacon.active:
                        beacon.transmit_count += 1

                        # Calculate RSSI for current receiver position
                        # In real implementation, get receiver position from MAVLink
                        receiver_pos = (0, 0, 0)  # Placeholder
                        rssi = self.calculate_rssi(beacon.beacon_id, receiver_pos)

                        # Notify callback if registered
                        if self.rssi_callback:
                            self.rssi_callback(rssi)

                # Sleep based on fastest pulse rate
                min_period = min(
                    1.0 / b.config.pulse_rate_hz for b in self.beacons.values() if b.active
                )
                await asyncio.sleep(min_period)

        except asyncio.CancelledError:
            logger.info("Beacon simulation loop cancelled")
        except Exception as e:
            logger.error(f"Beacon simulation error: {e}")

    def register_rssi_callback(self, callback: Callable[[float], None]):
        """Register callback for RSSI updates.

        Args:
            callback: Function to call with RSSI values
        """
        self.rssi_callback = callback

    def get_beacon_status(self, beacon_id: str) -> dict[str, Any] | None:
        """Get status of a beacon.

        Args:
            beacon_id: Beacon identifier

        Returns:
            Beacon status dictionary or None if not found
        """
        beacon = self.beacons.get(beacon_id)
        if not beacon:
            return None

        return {
            "beacon_id": beacon.beacon_id,
            "active": beacon.active,
            "position": beacon.position,
            "power_dbm": beacon.config.power_dbm,
            "frequency_hz": beacon.config.frequency_hz,
            "modulation": beacon.config.modulation,
            "transmit_count": beacon.transmit_count,
            "uptime_s": (
                (datetime.now(UTC) - beacon.start_time).total_seconds() if beacon.start_time else 0
            ),
        }

    def get_all_beacons(self) -> list[dict[str, Any]]:
        """Get status of all beacons.

        Returns:
            List of beacon status dictionaries
        """
        return [self.get_beacon_status(bid) for bid in self.beacons if self.get_beacon_status(bid)]

    async def simulate_approach(
        self,
        beacon_id: str,
        start_position: tuple[float, float, float],
        approach_speed_mps: float = 5.0,
    ) -> list[tuple[float, float]]:
        """Simulate approach to beacon from starting position.

        Args:
            beacon_id: Target beacon identifier
            start_position: Starting position
            approach_speed_mps: Approach speed in m/s

        Returns:
            List of (distance, rssi) tuples during approach
        """
        beacon = self.beacons.get(beacon_id)
        if not beacon:
            return []

        measurements = []
        current_pos = list(start_position)

        # Calculate approach vector
        dx = beacon.position[0] - current_pos[0]
        dy = beacon.position[1] - current_pos[1]
        dz = beacon.position[2] - current_pos[2]

        total_distance = math.sqrt(dx * dx + dy * dy + dz * dz)

        # Normalize approach vector
        if total_distance > 0:
            dx /= total_distance
            dy /= total_distance
            dz /= total_distance

        # Simulate approach
        time_step = 0.5  # seconds
        while total_distance > 10:  # Stop at 10m from beacon
            # Update position
            step_distance = approach_speed_mps * time_step
            current_pos[0] += dx * step_distance
            current_pos[1] += dy * step_distance
            current_pos[2] += dz * step_distance

            # Calculate new distance and RSSI
            distance = self._calculate_distance(beacon.position, tuple(current_pos))
            rssi = self.calculate_rssi(beacon_id, tuple(current_pos))

            measurements.append((distance, rssi))
            total_distance = distance

            await asyncio.sleep(time_step)

        return measurements

    def clear_all_beacons(self):
        """Remove all simulated beacons."""
        # Stop all active beacons
        for beacon_id in list(self.beacons.keys()):
            task = asyncio.create_task(self.stop_beacon(beacon_id))
            # Store task reference to avoid warning
            _ = task

        self.beacons.clear()
        logger.info("Cleared all simulated beacons")
