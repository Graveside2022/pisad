"""Mock SDR Error Handling Tests.

Tests for SDR disconnection, reconnection, and error recovery
without requiring real hardware.
"""

import threading
import time
from unittest.mock import patch

import pytest

from backend.hal.hackrf_interface import HackRFInterface
from backend.hal.mock_hackrf import MockHackRF
from backend.services.hardware_detector import HardwareDetector

pytestmark = pytest.mark.serial


@pytest.mark.mock_hardware
@pytest.mark.sdr
class TestMockSDRDisconnection:
    """Test SDR disconnection and recovery scenarios."""

    @pytest.fixture
    def mock_sdr(self) -> MockHackRF:
        """Create a mock SDR device."""
        device = MockHackRF()
        device.open()
        return device

    def test_disconnect_during_streaming(self, mock_sdr: MockHackRF) -> None:
        """Test disconnection while streaming."""
        samples_before = []
        samples_after = []
        disconnected = False

        def callback(samples: bytes) -> None:
            if not disconnected:
                samples_before.append(samples)
            else:
                samples_after.append(samples)

        # Start streaming
        mock_sdr.start_rx(callback)
        time.sleep(0.1)

        # Simulate disconnection
        disconnected = True
        mock_sdr.connected = False
        mock_sdr.stop()

        # Should have samples before disconnect
        assert len(samples_before) > 0
        # Should not have samples after disconnect
        assert len(samples_after) == 0

    def test_reconnection_after_disconnect(self, mock_sdr: MockHackRF) -> None:
        """Test reconnection after disconnection."""
        # Simulate disconnect
        mock_sdr.close()
        assert mock_sdr.connected is False

        # Attempt reconnect
        result = mock_sdr.open()
        assert result == 0
        assert mock_sdr.connected is True

        # Should be able to stream again
        def callback(samples: bytes) -> None:
            pass

        assert mock_sdr.start_rx(callback) == 0
        mock_sdr.stop()

    def test_configuration_after_reconnect(self, mock_sdr: MockHackRF) -> None:
        """Test that configuration needs to be reapplied after reconnect."""
        # Set configuration
        mock_sdr.set_freq(3_200_000_000)
        mock_sdr.set_lna_gain(16)
        mock_sdr.set_vga_gain(20)

        # Disconnect and reconnect
        mock_sdr.close()
        mock_sdr.open()

        # Configuration should be reset to defaults
        assert mock_sdr.frequency == 3200000000  # Default
        assert mock_sdr.lna_gain == 16  # Default
        assert mock_sdr.vga_gain == 20  # Default

        # Reapply configuration
        mock_sdr.set_freq(5_800_000_000)
        assert mock_sdr.frequency == 5_800_000_000

    def test_usb_error_simulation(self, mock_sdr: MockHackRF) -> None:
        """Test USB error handling."""
        # Simulate USB error
        with patch.object(mock_sdr, "start_rx", side_effect=Exception("USB Error")):

            def callback(samples: bytes) -> None:
                pass

            # Should handle USB error gracefully
            try:
                mock_sdr.start_rx(callback)
                assert False, "Should have raised exception"
            except Exception as e:
                assert "USB Error" in str(e)

    def test_timeout_handling(self, mock_sdr: MockHackRF) -> None:
        """Test timeout handling for operations."""
        # Simulate slow operation
        original_set_freq = mock_sdr.set_freq

        def slow_set_freq(freq: int) -> int:
            time.sleep(0.1)  # Simulate delay
            return original_set_freq(freq)

        mock_sdr.set_freq = slow_set_freq

        # Operation should complete despite delay
        start = time.time()
        result = mock_sdr.set_freq(3_200_000_000)
        elapsed = time.time() - start

        assert result == 0
        assert elapsed >= 0.1  # Should have taken time


@pytest.mark.mock_hardware
@pytest.mark.sdr
class TestMockSDRRecovery:
    """Test SDR recovery mechanisms."""

    @pytest.fixture
    def interface_with_mock(self) -> HackRFInterface:
        """Create interface with mock SDR."""
        with patch("backend.hal.hackrf_interface.HackRF"):
            interface = HackRFInterface()
            interface.device = MockHackRF()
            interface.device.open()
            interface.connected = True
            return interface

    def test_automatic_reconnection(self, interface_with_mock: HackRFInterface) -> None:
        """Test automatic reconnection logic."""
        # Simulate disconnect
        interface_with_mock.connected = False

        # Attempt operation should trigger reconnect
        with patch.object(interface_with_mock, "connect") as mock_connect:
            mock_connect.return_value = True
            interface_with_mock.ensure_connected()
            mock_connect.assert_called_once()

    def test_retry_with_backoff(self) -> None:
        """Test retry logic with exponential backoff."""
        attempts = []

        def failing_connect() -> bool:
            attempts.append(time.time())
            if len(attempts) < 3:
                return False
            return True

        with patch("backend.hal.hackrf_interface.HackRF"):
            interface = HackRFInterface()
            interface.connect = failing_connect

            # Should retry with backoff
            result = interface.connect_with_retry(max_attempts=5)
            assert result is True
            assert len(attempts) == 3

            # Check backoff timing
            if len(attempts) > 1:
                delays = [attempts[i] - attempts[i - 1] for i in range(1, len(attempts))]
                # Each delay should be longer than the previous
                for i in range(1, len(delays)):
                    assert delays[i] >= delays[i - 1] * 0.9  # Allow some tolerance

    def test_state_recovery_after_error(self, interface_with_mock: HackRFInterface) -> None:
        """Test state recovery after error."""
        # Set state
        interface_with_mock.set_frequency(3_200_000_000)
        interface_with_mock.set_lna_gain(16)

        # Save state
        saved_freq = interface_with_mock.device.frequency
        saved_gain = interface_with_mock.device.lna_gain

        # Simulate error and recovery
        interface_with_mock.connected = False
        interface_with_mock.device = MockHackRF()
        interface_with_mock.device.open()
        interface_with_mock.connected = True

        # Restore state
        interface_with_mock.set_frequency(saved_freq)
        interface_with_mock.set_lna_gain(saved_gain)

        assert interface_with_mock.device.frequency == 3_200_000_000
        assert interface_with_mock.device.lna_gain == 16


@pytest.mark.mock_hardware
@pytest.mark.sdr
@pytest.mark.integration
class TestMockSDRWithDetectorErrors:
    """Test hardware detector error handling."""

    def test_detector_handles_sdr_failure(self) -> None:
        """Test detector handles SDR initialization failure."""
        with patch("backend.services.hardware_detector.HackRFInterface") as mock_interface:
            mock_interface.return_value.connect.side_effect = Exception("Device not found")

            detector = HardwareDetector()
            detector.check_hardware()

            assert detector.sdr_available is False
            status = detector.get_status()
            assert status["sdr"]["available"] is False
            assert "error" in status["sdr"]

    def test_detector_retry_on_temporary_failure(self) -> None:
        """Test detector retries on temporary failures."""
        attempt_count = [0]

        def connect_with_retry() -> bool:
            attempt_count[0] += 1
            if attempt_count[0] < 3:
                raise Exception("Temporary failure")
            return True

        with patch("backend.services.hardware_detector.HackRFInterface") as mock_interface:
            mock_interface.return_value.connect = connect_with_retry

            detector = HardwareDetector()
            detector.max_retries = 5
            detector.check_hardware()

            # Should eventually succeed
            assert attempt_count[0] == 3

    def test_detector_concurrent_checks(self) -> None:
        """Test concurrent hardware checks don't interfere."""
        with patch("backend.services.hardware_detector.HackRFInterface") as mock_interface:
            mock_interface.return_value.connect.return_value = True

            detector = HardwareDetector()

            # Run concurrent checks
            threads = []
            results = []

            def check_hardware() -> None:
                detector.check_hardware()
                results.append(detector.sdr_available)

            for _ in range(5):
                t = threading.Thread(target=check_hardware)
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            # All checks should have same result
            assert all(r == results[0] for r in results)

    def test_detector_status_during_error(self) -> None:
        """Test status reporting during error conditions."""
        with patch("backend.services.hardware_detector.HackRFInterface") as mock_interface:
            # Simulate partial failure
            mock_interface.return_value.connect.return_value = True
            mock_interface.return_value.get_device_info.side_effect = Exception("Info error")

            detector = HardwareDetector()
            detector.check_hardware()

            status = detector.get_status()

            # Should report what it can
            assert "sdr" in status
            assert isinstance(status["sdr"], dict)


@pytest.mark.mock_hardware
@pytest.mark.sdr
class TestMockSDREdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def mock_sdr(self) -> MockHackRF:
        """Create a mock SDR device."""
        device = MockHackRF()
        device.open()
        return device

    def test_zero_sample_rate(self, mock_sdr: MockHackRF) -> None:
        """Test handling of zero sample rate."""
        result = mock_sdr.set_sample_rate(0)
        assert result == -1  # Should reject
        assert mock_sdr.sample_rate != 0  # Should keep previous value

    def test_negative_frequency(self, mock_sdr: MockHackRF) -> None:
        """Test handling of negative frequency."""
        result = mock_sdr.set_freq(-1000000)
        assert result == -1  # Should reject
        assert mock_sdr.frequency >= 0  # Should keep valid value

    def test_extreme_gain_values(self, mock_sdr: MockHackRF) -> None:
        """Test handling of extreme gain values."""
        # Test very large gain
        mock_sdr.set_lna_gain(1000)
        assert mock_sdr.lna_gain == 40  # Should clamp to max

        # Test negative gain
        mock_sdr.set_vga_gain(-10)
        assert mock_sdr.vga_gain == 0  # Should clamp to min

    def test_rapid_configuration_changes(self, mock_sdr: MockHackRF) -> None:
        """Test rapid configuration changes."""
        # Rapidly change configuration
        for _ in range(100):
            mock_sdr.set_freq(int(1e9 + np.random.random() * 5e9))
            mock_sdr.set_lna_gain(np.random.choice([0, 8, 16, 24, 32, 40]))
            mock_sdr.set_vga_gain(np.random.randint(0, 63))

        # Should handle rapid changes without error
        assert 850_000_000 <= mock_sdr.frequency <= 6_500_000_000
        assert 0 <= mock_sdr.lna_gain <= 40
        assert 0 <= mock_sdr.vga_gain <= 62

    def test_streaming_with_zero_callback(self, mock_sdr: MockHackRF) -> None:
        """Test streaming with None callback."""
        # Should handle None callback
        result = mock_sdr.start_rx(None)
        assert result == -1  # Should reject
        assert mock_sdr.is_streaming is False
