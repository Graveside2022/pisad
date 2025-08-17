"""SDR Hardware Interface Service using SoapySDR.

This module provides hardware abstraction for SDR devices (HackRF, USRP, etc.)
with async streaming, health monitoring, and automatic reconnection.
"""

import asyncio
import contextlib
import time
from collections.abc import AsyncGenerator
from typing import Any

import numpy as np

try:
    import SoapySDR

    SOAPY_AVAILABLE = True
except ImportError:
    SoapySDR = None
    SOAPY_AVAILABLE = False

from src.backend.core.exceptions import (
    PISADException,
    SDRError,
)
from src.backend.models.schemas import SDRConfig, SDRStatus
from src.backend.utils.logging import get_logger

logger = get_logger(__name__)


class SDRNotFoundError(Exception):
    """Raised when no SDR device is found."""

    pass


class SDRStreamError(Exception):
    """Raised when SDR streaming encounters an error."""

    pass


class SDRConfigError(Exception):
    """Raised when SDR configuration is invalid."""

    pass


class SDRService:
    """Service for managing SDR hardware interface."""

    def __init__(self) -> None:
        """Initialize SDR service."""
        if not SOAPY_AVAILABLE:
            logger.warning("SoapySDR not installed - SDR functionality will be limited")
        self.device: Any | None = None  # SoapySDR.Device when available
        self.stream: Any | None = None
        self.config: SDRConfig = SDRConfig()
        self.status: SDRStatus = SDRStatus(status="DISCONNECTED")
        self._stream_active = False
        self._reconnect_task: asyncio.Task[None] | None = None
        self._health_check_task: asyncio.Task[None] | None = None
        self._sample_counter = 0
        self._last_sample_time = time.time()
        self._buffer_overflows = 0

    async def __aenter__(self) -> "SDRService":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Async context manager exit."""
        await self.shutdown()
        # Return False to propagate any exceptions
        return False

    @staticmethod
    def enumerate_devices() -> list[dict[str, str]]:
        """Enumerate available SDR devices.

        Returns:
            List of device dictionaries with driver and label information.
        """
        if not SOAPY_AVAILABLE:
            logger.warning("SoapySDR not available - cannot enumerate devices")
            return []
        try:
            devices = SoapySDR.Device.enumerate()
            logger.info(f"Found {len(devices)} SDR device(s)")
            for idx, device in enumerate(devices):
                logger.debug(f"Device {idx}: {device}")
            return list(devices)  # Ensure it's a list
        except SDRError as e:
            logger.error(f"Failed to enumerate devices: {e}")
            return []

    def _select_device(self, device_args: str = "") -> dict[str, str]:
        """Select an SDR device.

        Args:
            device_args: Device selection arguments (e.g., 'driver=hackrf')

        Returns:
            Selected device dictionary.

        Raises:
            SDRNotFoundError: If no suitable device is found.
        """
        devices = self.enumerate_devices()

        if not devices:
            raise SDRNotFoundError("No SDR devices found")

        # If device args specified, try to find matching device
        if device_args:
            try:
                args_dict = dict(arg.split("=") for arg in device_args.split(",") if "=" in arg)
                for device in devices:
                    if all(device.get(k) == v for k, v in args_dict.items()):
                        logger.info(f"Selected device: {device}")
                        return device
            except ValueError:
                logger.warning(
                    f"Invalid device args format: {device_args}. Using first available device."
                )

        # Fall back to first available device
        logger.info(f"Using first available device: {devices[0]}")
        return devices[0]

    async def initialize(self, config: SDRConfig | None = None) -> None:
        """Initialize SDR device with configuration.

        Args:
            config: SDR configuration parameters.
        """
        if config:
            self.config = config

        if not SOAPY_AVAILABLE:
            logger.warning("SoapySDR not available - SDR initialization skipped")
            self.status.status = "UNAVAILABLE"
            return

        try:
            # Select and create device
            device_dict = self._select_device(self.config.device_args)
            self.device = SoapySDR.Device(device_dict)

            # Update status with device info
            self.status.device_name = device_dict.get("label", "Unknown")
            self.status.driver = device_dict.get("driver", "Unknown")

            # Configure device
            await self._configure_device()

            # Start health monitoring
            self._health_check_task = asyncio.create_task(self._health_monitor())

            self.status.status = "CONNECTED"
            logger.info(f"SDR initialized: {self.status.device_name} ({self.status.driver})")

        except Exception as e:
            self.status.status = "ERROR"
            self.status.last_error = str(e)
            logger.error(f"Failed to initialize SDR: {e}")

            # Start reconnection attempts
            if not self._reconnect_task:
                self._reconnect_task = asyncio.create_task(self._reconnect_loop())
            raise

    async def _configure_device(self) -> None:
        """Configure SDR device parameters."""
        if not self.device:
            raise SDRConfigError("Device not initialized")

        try:
            # Validate and set frequency
            freq_range = self.device.getFrequencyRange(SoapySDR.SOAPY_SDR_RX, 0)
            if freq_range:
                min_freq, max_freq = freq_range[0].minimum(), freq_range[0].maximum()
                if not (min_freq <= self.config.frequency <= max_freq):
                    raise SDRConfigError(
                        f"Frequency {self.config.frequency/1e9:.3f} GHz out of range "
                        f"[{min_freq/1e9:.3f}, {max_freq/1e9:.3f}] GHz"
                    )
            self.device.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, self.config.frequency)

            # Validate and set sample rate
            supported_rates = self.device.listSampleRates(SoapySDR.SOAPY_SDR_RX, 0)
            if supported_rates and self.config.sampleRate not in supported_rates:
                # Find closest supported rate
                closest_rate = min(
                    supported_rates, key=lambda x: abs(float(x) - self.config.sampleRate)
                )
                logger.warning(
                    f"Sample rate {self.config.sampleRate/1e6:.1f} Msps not supported, "
                    f"using {closest_rate/1e6:.1f} Msps"
                )
                self.config.sampleRate = closest_rate
            self.device.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, self.config.sampleRate)

            # Set bandwidth
            self.device.setBandwidth(SoapySDR.SOAPY_SDR_RX, 0, self.config.bandwidth)

            # Configure gain
            if self.config.gain == "AUTO":
                if self.device.hasGainMode(SoapySDR.SOAPY_SDR_RX, 0):
                    self.device.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, True)
                    logger.info("Gain set to AUTO mode")
            else:
                self.device.setGain(SoapySDR.SOAPY_SDR_RX, 0, float(self.config.gain))
                logger.info(f"Gain set to {self.config.gain} dB")

            logger.info(
                f"SDR configured: {self.config.frequency/1e9:.3f} GHz, "
                f"{self.config.sampleRate/1e6:.1f} Msps, "
                f"{self.config.bandwidth/1e6:.1f} MHz BW"
            )

        except Exception as e:
            raise SDRConfigError(f"Failed to configure device: {e}")

    async def stream_iq(self) -> AsyncGenerator[np.ndarray[Any, np.dtype[np.complex64]], None]:
        """Stream IQ samples from SDR device.

        Yields:
            Complex numpy arrays of IQ samples.
        """
        if not self.device:
            raise SDRStreamError("Device not initialized")

        # Setup stream
        self.stream = self.device.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
        self.device.activateStream(self.stream)
        self._stream_active = True
        self.status.stream_active = True

        # Prepare buffer - pre-allocate for better performance
        buffer = np.zeros(self.config.buffer_size, dtype=np.complex64)

        try:
            while self._stream_active:
                # Non-blocking read using asyncio.to_thread
                result = await asyncio.to_thread(
                    self.device.readStream,
                    self.stream,
                    [buffer],
                    self.config.buffer_size,
                    timeoutUs=100000,  # 100ms timeout
                )

                ret, flags, time_ns = result

                if ret > 0:
                    # Update statistics
                    self._sample_counter += ret
                    current_time = time.time()
                    time_diff = current_time - self._last_sample_time
                    if time_diff > 1.0:  # Update rate every second
                        self.status.samples_per_second = self._sample_counter / time_diff
                        self._sample_counter = 0
                        self._last_sample_time = current_time

                    # Check for overflow
                    if flags & SoapySDR.SOAPY_SDR_HAS_TIME:
                        self._buffer_overflows += 1
                        self.status.buffer_overflows = self._buffer_overflows
                        logger.warning(
                            f"Buffer overflow detected (total: {self._buffer_overflows})"
                        )

                    # Yield valid samples
                    yield buffer[:ret].copy()

                elif ret == SoapySDR.SOAPY_SDR_TIMEOUT:
                    # Timeout is normal, continue
                    await asyncio.sleep(0.001)
                else:
                    # Error occurred
                    error_msg = f"Stream error: {SoapySDR.errToStr(ret)}"
                    logger.error(error_msg)
                    self.status.last_error = error_msg
                    raise SDRStreamError(error_msg)

        finally:
            # Cleanup stream
            self._stream_active = False
            self.status.stream_active = False
            if self.stream and self.device:
                self.device.deactivateStream(self.stream)
                self.device.closeStream(self.stream)
                self.stream = None

    def set_frequency(self, frequency: float) -> None:
        """Set SDR center frequency.

        Args:
            frequency: Center frequency in Hz.
        """
        if not self.device:
            raise SDRConfigError("Device not initialized")

        self.device.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, frequency)
        self.config.frequency = frequency
        logger.info(f"Frequency set to {frequency/1e9:.3f} GHz")

    def get_status(self) -> SDRStatus:
        """Get current SDR status.

        Returns:
            Current status information.
        """
        # Check for temperature sensor if available
        if self.device:
            sensors = self.device.listSensors()
            for sensor in sensors:
                if "temp" in sensor.lower():
                    try:
                        temp = self.device.readSensor(sensor)
                        self.status.temperature = float(temp)
                    except Exception:
                        pass

        return self.status

    async def _health_monitor(self) -> None:
        """Periodic health monitoring task."""
        while True:
            try:
                await asyncio.sleep(5)  # Check every 5 seconds

                if self.device:
                    # Try to read a device property to check connection
                    try:
                        _ = self.device.getHardwareKey()
                        if self.status.status == "DISCONNECTED":
                            self.status.status = "CONNECTED"
                            logger.info("SDR reconnected")
                    except Exception:
                        if self.status.status == "CONNECTED":
                            self.status.status = "DISCONNECTED"
                            logger.warning("SDR disconnected")
                            # Start reconnection
                            if not self._reconnect_task:
                                self._reconnect_task = asyncio.create_task(self._reconnect_loop())

            except asyncio.CancelledError:
                break
            except PISADException as e:
                logger.error(f"Health monitor error: {e}")

    async def _reconnect_loop(self) -> None:
        """Automatic reconnection with exponential backoff."""
        backoff = 1.0  # Start with 1 second
        max_backoff = 30.0

        while self.status.status != "CONNECTED":
            try:
                await asyncio.sleep(backoff)
                logger.info(f"Attempting SDR reconnection (backoff: {backoff}s)")

                # Try to reinitialize
                await self.initialize()

                # Reset backoff on success
                backoff = 1.0
                self._reconnect_task = None
                break

            except asyncio.CancelledError:
                # Task was cancelled, exit gracefully
                logger.info("Reconnection task cancelled")
                raise
            except PISADException as e:
                logger.error(f"Reconnection failed: {e}")
                # Exponential backoff
                backoff = min(backoff * 2, max_backoff)

    async def update_config(self, config: SDRConfig) -> None:
        """Update SDR configuration dynamically.

        Args:
            config: New SDR configuration parameters.
        """
        logger.info("Updating SDR configuration")

        # Store old config for rollback
        old_config = self.config
        self.config = config

        try:
            if self.device:
                # Apply new configuration
                await self._configure_device()
                logger.info("SDR configuration updated successfully")
            else:
                # If device not initialized, just store config for next init
                logger.info("SDR config stored, will apply on next initialization")

        except Exception as e:
            # Rollback on error
            logger.error(f"Failed to update SDR config: {e}, rolling back")
            self.config = old_config
            if self.device:
                await self._configure_device()
            raise

    async def calibrate(self) -> dict[str, Any]:
        """Perform SDR calibration routine.

        This routine calibrates the SDR device for optimal performance by:
        1. Testing frequency accuracy
        2. Measuring noise floor
        3. Determining optimal gain settings
        4. Validating sample rate stability

        Returns:
            Calibration results and recommended settings
        """
        if not self.device:
            raise SDRConfigError("Device not initialized")

        logger.info("Starting SDR calibration routine...")
        calibration_results = {
            "status": "in_progress",
            "frequency_accuracy": {},
            "noise_floor": {},
            "gain_optimization": {},
            "sample_rate_stability": {},
            "recommendations": {},
            "timestamp": time.time(),
        }

        try:
            # Step 1: Test frequency accuracy
            logger.info("Testing frequency accuracy...")
            test_frequencies = [
                433.0e6,  # Common ISM band
                868.0e6,  # European ISM band
                915.0e6,  # US ISM band
                self.config.frequency,  # Target frequency
            ]

            freq_errors = []
            for freq in test_frequencies:
                try:
                    self.device.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, freq)
                    actual_freq = self.device.getFrequency(SoapySDR.SOAPY_SDR_RX, 0)
                    error_ppm = (actual_freq - freq) / freq * 1e6
                    freq_errors.append(error_ppm)
                    logger.debug(f"Frequency {freq/1e6:.1f} MHz: error = {error_ppm:.2f} ppm")
                except PISADException as e:
                    logger.warning(f"Failed to test frequency {freq/1e6:.1f} MHz: {e}")

            if freq_errors:
                avg_error = sum(freq_errors) / len(freq_errors)
                calibration_results["frequency_accuracy"] = {
                    "average_error_ppm": avg_error,
                    "max_error_ppm": max(abs(e) for e in freq_errors),
                    "recommended_ppm_correction": -avg_error,
                }
                logger.info(f"Frequency calibration: avg error = {avg_error:.2f} ppm")

            # Step 2: Measure noise floor
            logger.info("Measuring noise floor...")

            # Set to minimum gain for noise floor measurement
            original_gain = self.config.gain
            self.device.setGain(SoapySDR.SOAPY_SDR_RX, 0, 0.0)

            # Setup stream for noise measurement
            stream = self.device.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
            self.device.activateStream(stream)

            # Collect samples for noise analysis
            noise_samples = []
            buffer = np.zeros(self.config.buffer_size, dtype=np.complex64)

            for _ in range(10):  # Collect 10 buffers
                result = self.device.readStream(
                    stream, [buffer], self.config.buffer_size, timeoutUs=100000
                )
                ret, _, _ = result
                if ret > 0:
                    # Calculate power in dBm
                    power = np.abs(buffer[:ret]) ** 2
                    power_dbm = 10 * np.log10(power + 1e-10)
                    noise_samples.extend(power_dbm)
                await asyncio.sleep(0.01)

            self.device.deactivateStream(stream)
            self.device.closeStream(stream)

            if noise_samples:
                noise_floor = np.percentile(noise_samples, 10)  # 10th percentile
                noise_std = np.std(noise_samples)
                calibration_results["noise_floor"] = {
                    "noise_floor_dbm": float(noise_floor),
                    "noise_std_db": float(noise_std),
                    "samples_collected": len(noise_samples),
                }
                logger.info(f"Noise floor: {noise_floor:.1f} dBm (std: {noise_std:.1f} dB)")

            # Step 3: Optimize gain settings
            logger.info("Optimizing gain settings...")

            # Get available gain range
            gain_range = self.device.getGainRange(SoapySDR.SOAPY_SDR_RX, 0)
            if gain_range:
                min_gain = gain_range.minimum()
                max_gain = gain_range.maximum()

                # Test different gain levels
                test_gains = [min_gain, (min_gain + max_gain) / 2, max_gain]
                gain_metrics = []

                for gain in test_gains:
                    self.device.setGain(SoapySDR.SOAPY_SDR_RX, 0, gain)

                    # Measure dynamic range at this gain
                    stream = self.device.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
                    self.device.activateStream(stream)

                    samples = []
                    for _ in range(5):
                        result = self.device.readStream(
                            stream, [buffer], self.config.buffer_size, timeoutUs=100000
                        )
                        ret, _, _ = result
                        if ret > 0:
                            samples.extend(np.abs(buffer[:ret]))

                    self.device.deactivateStream(stream)
                    self.device.closeStream(stream)

                    if samples:
                        dynamic_range = 20 * np.log10(max(samples) / (min(samples) + 1e-10))
                        gain_metrics.append(
                            {
                                "gain_db": gain,
                                "dynamic_range_db": float(dynamic_range),
                                "max_amplitude": float(max(samples)),
                            }
                        )

                # Find optimal gain (best dynamic range without saturation)
                optimal_gain = original_gain
                for metric in gain_metrics:
                    if metric["max_amplitude"] < 0.9:  # Not saturating
                        optimal_gain = metric["gain_db"]

                calibration_results["gain_optimization"] = {
                    "tested_gains": gain_metrics,
                    "recommended_gain": optimal_gain,
                    "gain_range": {"min": min_gain, "max": max_gain},
                }
                logger.info(f"Optimal gain: {optimal_gain:.1f} dB")

            # Restore original gain
            if isinstance(original_gain, str) and original_gain == "AUTO":
                self.device.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, True)
            else:
                self.device.setGain(SoapySDR.SOAPY_SDR_RX, 0, float(original_gain))

            # Step 4: Test sample rate stability
            logger.info("Testing sample rate stability...")

            # Measure actual sample rate
            stream = self.device.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
            self.device.activateStream(stream)

            sample_counts = []
            time_intervals = []

            for _ in range(10):
                start_time = time.time()
                total_samples = 0

                # Collect samples for 100ms
                while time.time() - start_time < 0.1:
                    result = self.device.readStream(
                        stream, [buffer], self.config.buffer_size, timeoutUs=10000
                    )
                    ret, _, _ = result
                    if ret > 0:
                        total_samples += ret

                elapsed = time.time() - start_time
                if elapsed > 0:
                    actual_rate = total_samples / elapsed
                    sample_counts.append(actual_rate)
                    time_intervals.append(elapsed)

                await asyncio.sleep(0.05)

            self.device.deactivateStream(stream)
            self.device.closeStream(stream)

            if sample_counts:
                avg_rate = sum(sample_counts) / len(sample_counts)
                rate_stability = np.std(sample_counts) / avg_rate * 100  # Percentage

                calibration_results["sample_rate_stability"] = {
                    "configured_rate": self.config.sampleRate,
                    "measured_rate": avg_rate,
                    "stability_percent": float(rate_stability),
                    "rate_error_percent": (avg_rate - self.config.sampleRate)
                    / self.config.sampleRate
                    * 100,
                }
                logger.info(
                    f"Sample rate: {avg_rate/1e6:.2f} Msps (stability: {rate_stability:.2f}%)"
                )

            # Generate recommendations
            recommendations = []

            if calibration_results["frequency_accuracy"].get("recommended_ppm_correction"):
                ppm = calibration_results["frequency_accuracy"]["recommended_ppm_correction"]
                if abs(ppm) > 1:
                    recommendations.append(f"Apply PPM correction of {ppm:.1f}")

            if calibration_results["gain_optimization"].get("recommended_gain"):
                rec_gain = calibration_results["gain_optimization"]["recommended_gain"]
                if rec_gain != original_gain and original_gain != "AUTO":
                    recommendations.append(f"Consider using gain of {rec_gain:.1f} dB")

            if calibration_results["sample_rate_stability"].get("stability_percent", 0) > 5:
                recommendations.append("Sample rate instability detected - check USB connection")

            calibration_results["recommendations"] = recommendations
            calibration_results["status"] = "complete"

            logger.info(f"SDR calibration complete. Recommendations: {recommendations}")

            return calibration_results

        except PISADException as e:
            logger.error(f"Calibration failed: {e}")
            calibration_results["status"] = "failed"
            calibration_results["error"] = str(e)
            raise SDRConfigError(f"Calibration failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown SDR service and cleanup."""
        logger.info("Shutting down SDR service")

        # Stop streaming
        self._stream_active = False

        # Cancel tasks
        if self._health_check_task:
            self._health_check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._health_check_task
            self._health_check_task = None

        if self._reconnect_task:
            self._reconnect_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reconnect_task
            self._reconnect_task = None

        # Close device
        if self.device:
            self.device = None

        self.status.status = "DISCONNECTED"
        logger.info("SDR service shutdown complete")
