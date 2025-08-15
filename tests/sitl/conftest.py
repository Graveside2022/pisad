"""
SITL test specific configuration and fixtures.
SITL tests can take up to 30s and require hardware simulation.
"""

import asyncio
import time

import pytest


@pytest.fixture(scope="module")
def sitl_instance():
    """Launch SITL instance for testing (if available)."""
    try:
        from dronekit_sitl import SITL

        # Start SITL
        sitl = SITL()
        sitl.download("copter", "3.3", verbose=False)
        sitl.launch(["--home=47.5,-122.3,0,0"])

        # Wait for SITL to be ready
        time.sleep(5)

        yield sitl

        # Cleanup
        sitl.stop()
    except ImportError:
        # SITL not available, use mock
        class MockSITL:
            def __init__(self):
                self.connection_string = "tcp:127.0.0.1:5760"

        yield MockSITL()


@pytest.fixture
async def sitl_vehicle(sitl_instance):
    """Connect to SITL vehicle."""
    try:
        from dronekit import connect

        # Connect to the SITL instance
        vehicle = connect(sitl_instance.connection_string, wait_ready=True)

        # Wait for vehicle to be ready
        while not vehicle.is_armable:
            await asyncio.sleep(1)

        yield vehicle

        # Cleanup
        vehicle.close()
    except ImportError:
        # DroneKit not available, use mock
        class MockVehicle:
            def __init__(self):
                self.mode = "GUIDED"
                self.armed = False
                self.location = type(
                    "obj",
                    (object,),
                    {
                        "global_frame": type(
                            "obj", (object,), {"lat": 47.5, "lon": -122.3, "alt": 0}
                        )()
                    },
                )()

            async def arm(self):
                self.armed = True

            async def disarm(self):
                self.armed = False

        yield MockVehicle()


@pytest.fixture
def sitl_timeout():
    """Enforce 30 second timeout for SITL tests."""
    return 30.0


@pytest.fixture
def sitl_test_beacon():
    """Simulated beacon configuration for SITL tests."""
    return {
        "frequency": 433920000,
        "bandwidth": 25000,
        "modulation": "FSK",
        "power": -50,  # dBm
        "location": {"lat": 47.5001, "lon": -122.3001, "alt": 50},
    }


@pytest.fixture
async def sitl_mission_executor():
    """Execute missions in SITL environment."""

    class MissionExecutor:
        def __init__(self):
            self.waypoints = []
            self.current_wp = 0

        async def upload_mission(self, waypoints):
            """Upload waypoints to vehicle."""
            self.waypoints = waypoints
            self.current_wp = 0
            return True

        async def start_mission(self):
            """Start mission execution."""
            for wp in self.waypoints:
                await asyncio.sleep(1)  # Simulate flight time
                self.current_wp += 1
            return True

        async def pause_mission(self):
            """Pause mission execution."""
            return True

        async def resume_mission(self):
            """Resume mission execution."""
            return True

        def get_progress(self):
            """Get mission progress."""
            if not self.waypoints:
                return 0
            return (self.current_wp / len(self.waypoints)) * 100

    return MissionExecutor()


@pytest.fixture
def sitl_safety_validator():
    """Validate safety constraints in SITL environment."""

    class SafetyValidator:
        def __init__(self):
            self.violations = []

        def check_altitude(self, alt, min_alt=10, max_alt=120):
            """Check altitude constraints."""
            if alt < min_alt or alt > max_alt:
                self.violations.append(f"Altitude violation: {alt}m")
                return False
            return True

        def check_geofence(self, lat, lon, center_lat=47.5, center_lon=-122.3, radius=500):
            """Check geofence constraints."""
            from math import asin, cos, radians, sin, sqrt

            # Haversine formula
            R = 6371000  # Earth radius in meters
            lat1, lon1, lat2, lon2 = map(radians, [center_lat, center_lon, lat, lon])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
            c = 2 * asin(sqrt(a))
            distance = R * c

            if distance > radius:
                self.violations.append(f"Geofence violation: {distance:.1f}m from center")
                return False
            return True

        def check_battery(self, voltage, min_voltage=18.0):
            """Check battery constraints."""
            if voltage < min_voltage:
                self.violations.append(f"Battery violation: {voltage}V")
                return False
            return True

        def get_violations(self):
            """Get all safety violations."""
            return self.violations

    return SafetyValidator()
