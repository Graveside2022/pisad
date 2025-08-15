#!/usr/bin/env python3
"""
Hardware Integration Test Script
Tests HackRF, Cube Orange+, and performance monitoring
"""

import asyncio
import logging
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.serial
# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.backend.hal.beacon_generator import BeaconConfig, BeaconGenerator
from src.backend.services.hardware_detector import HardwareDetector
from src.backend.services.performance_monitor import PerformanceMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


async def test_hardware_detection():
    """Test hardware auto-detection"""
    logger.info("=" * 60)
    logger.info("HARDWARE DETECTION TEST")
    logger.info("=" * 60)

    detector = HardwareDetector()
    await detector.detect_all()
    status = detector.status

    logger.info(f"SDR Connected: {status.sdr_connected}")
    if status.sdr_connected:
        logger.info(f"SDR Type: {status.sdr_type}")
        logger.info(f"SDR Info: {status.sdr_info}")

    logger.info(f"Flight Controller Connected: {status.flight_controller_connected}")
    if status.flight_controller_connected:
        logger.info(f"FC Type: {status.flight_controller_type}")
        logger.info(f"FC Info: {status.flight_controller_info}")

    return detector


async def test_beacon_generator():
    """Test software beacon generator"""
    logger.info("=" * 60)
    logger.info("BEACON GENERATOR TEST")
    logger.info("=" * 60)

    # Create beacon with test config
    config = BeaconConfig(frequency=3.2e9, pulse_width=0.001, pulse_period=0.1, power_dbm=-10)

    beacon = BeaconGenerator(config)

    # Generate test pulse
    pulse = beacon.generate_pulse(0.001)
    logger.info(f"Generated pulse: {len(pulse)} samples")

    # Generate pattern
    pattern = beacon.generate_pattern(1.0)
    logger.info(f"Generated pattern: {len(pattern)} samples")

    # Test frequency hopping
    beacon.config.hop_enabled = True
    beacon.config.hop_frequencies = [850e6, 3.2e9, 5.8e9]

    for i in range(5):
        freq = beacon.get_next_frequency()
        logger.info(f"Hop {i}: {freq/1e9:.3f} GHz")

    return beacon


async def test_performance_monitor():
    """Test performance monitoring"""
    logger.info("=" * 60)
    logger.info("PERFORMANCE MONITOR TEST")
    logger.info("=" * 60)

    monitor = PerformanceMonitor()
    await monitor.start()

    # Simulate some activity
    for i in range(5):
        monitor.record_rssi_update()
        monitor.record_state_update()
        monitor.record_mavlink_message(latency_ms=10 + i)

        if i % 2 == 0:
            monitor.record_sdr_sample(16384, dropped=0)

        await asyncio.sleep(0.2)

    # Wait for metrics update
    await asyncio.sleep(1.5)

    # Get metrics
    metrics = monitor.get_metrics_dict()
    logger.info("Performance Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value}")

    logger.info(f"Summary: {monitor.get_summary()}")

    await monitor.stop()
    return monitor


@pytest.mark.skip(reason="Requires detector fixture from main()")
async def test_hardware_streaming(detector):
    """Test actual hardware streaming if available"""
    logger.info("=" * 60)
    logger.info("HARDWARE STREAMING TEST")
    logger.info("=" * 60)

    # Test SDR streaming
    hackrf = detector.get_sdr()
    if hackrf:
        logger.info("Testing HackRF streaming...")

        samples_received = []

        def rx_callback(samples):
            samples_received.append(len(samples))
            logger.debug(f"Received {len(samples)} samples")

        # Start receiving
        if await hackrf.start_rx(rx_callback):
            logger.info("HackRF RX started")

            # Receive for 1 second
            await asyncio.sleep(1.0)

            # Stop receiving
            await hackrf.stop_rx()
            logger.info(f"Total batches received: {len(samples_received)}")
            if samples_received:
                logger.info(
                    f"Average batch size: {sum(samples_received)/len(samples_received):.0f}"
                )
    else:
        logger.warning("No SDR hardware available for streaming test")

    # Test MAVLink communication
    mavlink = detector.get_flight_controller()
    if mavlink:
        logger.info("Testing MAVLink communication...")

        # Get telemetry
        position = await mavlink.get_position()
        logger.info(f"Position: {position}")

        battery = await mavlink.get_battery()
        logger.info(f"Battery: {battery}")

        mode = await mavlink.get_flight_mode()
        logger.info(f"Flight mode: {mode}")

        gps = await mavlink.get_gps_status()
        logger.info(f"GPS: {gps}")

        # Send status message
        await mavlink.send_statustext("Hardware test complete", severity=6)
    else:
        logger.warning("No flight controller available for communication test")


async def main():
    """Run all hardware integration tests"""
    logger.info("Starting Hardware Integration Tests")
    logger.info("=" * 60)

    try:
        # Test hardware detection
        detector = await test_hardware_detection()

        # Test beacon generator
        beacon = await test_beacon_generator()

        # Test performance monitor
        monitor = await test_performance_monitor()

        # Test actual hardware if available
        await test_hardware_streaming(detector)

        logger.info("=" * 60)
        logger.info("Hardware Integration Tests Complete")

        # Cleanup
        await detector.shutdown()

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
