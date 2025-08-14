#!/usr/bin/env python3
"""Manual test utility for SDR hardware connectivity and streaming.

This script provides a command-line interface for testing SDR devices,
streaming IQ samples, and displaying real-time statistics.
"""

import argparse
import asyncio
import signal
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backend.models.schemas import SDRConfig
from src.backend.services.sdr_service import SDRService
from src.backend.utils.logging import setup_logging


class SDRTester:
    """SDR testing utility class."""

    def __init__(self, config: SDRConfig):
        """Initialize SDR tester.

        Args:
            config: SDR configuration parameters.
        """
        self.config = config
        self.service = SDRService()
        self.running = True
        self.samples_received = 0
        self.start_time: float | None = None
        self.last_rssi: float = 0.0

    async def list_devices(self) -> None:
        """List available SDR devices."""
        print("\n=== Available SDR Devices ===")
        devices = SDRService.enumerate_devices()

        if not devices:
            print("No SDR devices found!")
            print("\nTroubleshooting:")
            print("1. Check if SDR is connected via USB")
            print("2. Verify drivers are installed:")
            print("   - HackRF: hackrf-tools, libhackrf0")
            print("   - USRP: uhd-host, libuhd-dev")
            print("3. Check USB permissions (may need udev rules)")
            return

        for idx, device in enumerate(devices):
            print(f"\nDevice {idx}:")
            for key, value in device.items():
                print(f"  {key}: {value}")

    async def test_streaming(self, duration: float) -> None:
        """Test IQ sample streaming.

        Args:
            duration: Test duration in seconds.
        """
        print(f"\n=== Testing SDR Streaming for {duration} seconds ===")
        print("Configuration:")
        print(f"  Frequency: {self.config.frequency/1e9:.3f} GHz")
        print(f"  Sample Rate: {self.config.sampleRate/1e6:.1f} Msps")
        print(f"  Gain: {self.config.gain}")
        print(f"  Buffer Size: {self.config.buffer_size} samples")
        print("\nInitializing SDR...")

        try:
            await self.service.initialize(self.config)
            status = self.service.get_status()
            print(f"Device: {status.device_name} ({status.driver})")
            print(f"Status: {status.status}")

            if status.temperature:
                print(f"Temperature: {status.temperature:.1f}°C")

            print("\nStreaming started. Press Ctrl+C to stop.\n")
            self.start_time = time.time()

            # Create async task for streaming
            stream_task = asyncio.create_task(self._stream_samples())
            stats_task = asyncio.create_task(self._display_stats())

            # Run for specified duration
            await asyncio.sleep(duration)

            # Stop streaming
            self.running = False
            stream_task.cancel()
            stats_task.cancel()

            try:
                await stream_task
                await stats_task
            except asyncio.CancelledError:
                pass

            # Final statistics
            elapsed = time.time() - self.start_time
            print("\n=== Test Complete ===")
            print(f"Duration: {elapsed:.1f} seconds")
            print(f"Total Samples: {self.samples_received:,}")
            print(f"Average Rate: {self.samples_received/elapsed/1e6:.2f} Msps")

            final_status = self.service.get_status()
            if final_status.buffer_overflows > 0:
                print(f"Buffer Overflows: {final_status.buffer_overflows}")

        except Exception as e:
            print(f"Error during streaming test: {e}")
            raise
        finally:
            await self.service.shutdown()

    async def _stream_samples(self) -> None:
        """Stream and process IQ samples."""
        try:
            async for samples in self.service.stream_iq():
                if not self.running:
                    break

                self.samples_received += len(samples)

                # Calculate simple RSSI estimate
                rssi = 20 * np.log10(np.mean(np.abs(samples)) + 1e-10)

                # Store for statistics display
                self.last_rssi = rssi

        except asyncio.CancelledError:
            pass

    async def _display_stats(self) -> None:
        """Display real-time statistics."""
        try:
            while self.running:
                await asyncio.sleep(1)

                if self.start_time:
                    elapsed = time.time() - self.start_time
                    rate = self.samples_received / elapsed / 1e6

                    status = self.service.get_status()

                    # Clear line and print stats
                    print(
                        f"\rSamples: {self.samples_received:,} | "
                        f"Rate: {rate:.2f} Msps | "
                        f"RSSI: {getattr(self, 'last_rssi', 0):.1f} dB | "
                        f"Status: {status.status} | "
                        f"Overflows: {status.buffer_overflows}",
                        end="",
                        flush=True,
                    )

        except asyncio.CancelledError:
            pass

    async def test_capabilities(self) -> None:
        """Test and display device capabilities."""
        print("\n=== Testing Device Capabilities ===")

        try:
            await self.service.initialize(self.config)
            device = self.service.device

            if not device:
                print("Failed to initialize device!")
                return

            print(f"\nDevice: {self.service.status.device_name}")
            print(f"Driver: {self.service.status.driver}")

            # Frequency range
            print("\nFrequency Range:")
            freq_ranges = device.getFrequencyRange(0, 0)  # RX, channel 0
            for fr in freq_ranges:
                print(f"  {fr.minimum()/1e9:.3f} - {fr.maximum()/1e9:.3f} GHz")

            # Sample rates
            print("\nSupported Sample Rates:")
            sample_rates = device.listSampleRates(0, 0)  # RX, channel 0
            for rate in sample_rates[:10]:  # Show first 10
                print(f"  {rate/1e6:.1f} Msps")
            if len(sample_rates) > 10:
                print(f"  ... and {len(sample_rates)-10} more")

            # Gains
            print("\nGain Elements:")
            gain_names = device.listGains(0, 0)  # RX, channel 0
            for name in gain_names:
                gain_range = device.getGainRange(0, 0, name)
                print(f"  {name}: {gain_range.minimum():.0f} - {gain_range.maximum():.0f} dB")

            # Antennas
            print("\nAvailable Antennas:")
            antennas = device.listAntennas(0, 0)  # RX, channel 0
            for antenna in antennas:
                print(f"  {antenna}")

            # Sensors
            sensors = device.listSensors()
            if sensors:
                print("\nSensors:")
                for sensor in sensors:
                    try:
                        value = device.readSensor(sensor)
                        print(f"  {sensor}: {value}")
                    except Exception:
                        print(f"  {sensor}: N/A")

        except Exception as e:
            print(f"Error testing capabilities: {e}")
        finally:
            await self.service.shutdown()


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="SDR Hardware Test Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List devices command
    subparsers.add_parser("list", help="List available SDR devices")

    # Test streaming command
    stream_parser = subparsers.add_parser("stream", help="Test IQ sample streaming")
    stream_parser.add_argument(
        "-f",
        "--frequency",
        type=float,
        default=2.437e9,
        help="Center frequency in Hz (default: 2.437 GHz)",
    )
    stream_parser.add_argument(
        "-s", "--sample-rate", type=float, default=2e6, help="Sample rate in Hz (default: 2 Msps)"
    )
    stream_parser.add_argument(
        "-g", "--gain", default="AUTO", help="Gain in dB or AUTO (default: AUTO)"
    )
    stream_parser.add_argument(
        "-b", "--buffer-size", type=int, default=1024, help="Buffer size in samples (default: 1024)"
    )
    stream_parser.add_argument(
        "-d", "--duration", type=float, default=10.0, help="Test duration in seconds (default: 10)"
    )
    stream_parser.add_argument(
        "--device", type=str, default="", help='Device selection args (e.g., "driver=hackrf")'
    )

    # Test capabilities command
    caps_parser = subparsers.add_parser("capabilities", help="Display device capabilities")
    caps_parser.add_argument(
        "--device", type=str, default="", help='Device selection args (e.g., "driver=hackrf")'
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(log_level="INFO")

    # Create SDR config
    config = SDRConfig()

    if hasattr(args, "frequency"):
        config.frequency = args.frequency
    if hasattr(args, "sample_rate"):
        config.sampleRate = args.sample_rate
    if hasattr(args, "gain"):
        try:
            config.gain = float(args.gain)
        except ValueError:
            config.gain = args.gain  # Keep as string for 'AUTO'
    if hasattr(args, "buffer_size"):
        config.buffer_size = args.buffer_size
    if hasattr(args, "device"):
        config.device_args = args.device

    # Create tester
    tester = SDRTester(config)

    # Handle Ctrl+C gracefully
    def signal_handler(sig: Any, frame: Any) -> None:
        print("\n\nInterrupted by user")
        tester.running = False
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Execute command
    if args.command == "list":
        await tester.list_devices()
    elif args.command == "stream":
        await tester.test_streaming(args.duration)
    elif args.command == "capabilities":
        await tester.test_capabilities()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
