from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.backend.core.app import app

client = TestClient(app)


class TestSpectrumAPI:
    """Test spectrum waterfall API endpoints"""

    def test_get_spectrum_waterfall_endpoint_exists(self):
        """Test that /api/spectrum/waterfall endpoint exists"""
        response = client.get("/api/spectrum/waterfall")
        # Should not return 404 (endpoint exists)
        assert response.status_code != 404

    def test_spectrum_waterfall_returns_fft_data(self):
        """Test waterfall endpoint returns real FFT data from HackRF One"""
        with patch("src.backend.services.sdr_service.SDRService") as mock_sdr:
            # Mock real FFT data from HackRF One
            mock_fft_data = np.random.random(1024) * 80 - 100  # -100 to -20 dBm
            mock_sdr.return_value.get_fft_data.return_value = {
                "frequencies": np.linspace(2.4345e9, 2.4395e9, 1024),  # 5MHz bandwidth
                "magnitudes": mock_fft_data,
                "center_freq": 2.437e9,
                "sample_rate": 2.5e6,
                "timestamp": 1692000000,
            }

            response = client.get(
                "/api/spectrum/waterfall?center_freq=2437000000&bandwidth=5000000"
            )

            assert response.status_code == 200
            data = response.json()

            # Verify data structure matches frontend SpectrumData interface
            assert "frequencies" in data
            assert "magnitudes" in data
            assert "timestamp" in data
            assert "centerFreq" in data
            assert "sampleRate" in data

            # Verify data types and ranges
            assert len(data["frequencies"]) == 1024
            assert len(data["magnitudes"]) == 1024
            assert data["centerFreq"] == 2437000000
            assert data["sampleRate"] == 5000000  # Should match bandwidth parameter

    def test_spectrum_waterfall_validates_frequency_range(self):
        """Test frequency range validation per PRD-FR1 (850MHz - 6.5GHz)"""
        # Test invalid frequency below range
        response = client.get("/api/spectrum/waterfall?center_freq=800000000")  # 800MHz
        assert response.status_code == 422
        assert "frequency must be between 850 MHz and 6.5 GHz" in response.text.lower()

        # Test invalid frequency above range
        response = client.get("/api/spectrum/waterfall?center_freq=7000000000")  # 7GHz
        assert response.status_code == 422
        assert "frequency must be between 850 MHz and 6.5 GHz" in response.text.lower()

        # Test valid frequencies
        valid_freqs = [850000000, 915000000, 2437000000, 5800000000, 6500000000]
        for freq in valid_freqs:
            response = client.get(f"/api/spectrum/waterfall?center_freq={freq}")
            assert response.status_code in [
                200,
                500,
            ]  # 500 = no SDR hardware, but validation passed

    def test_spectrum_waterfall_bandwidth_validation(self):
        """Test bandwidth parameter validation"""
        # Test valid bandwidth (should be reasonable for HackRF One)
        response = client.get("/api/spectrum/waterfall?center_freq=2437000000&bandwidth=5000000")
        assert response.status_code in [200, 500]  # 500 = no SDR hardware

        # Test invalid bandwidth (too large)
        response = client.get(
            "/api/spectrum/waterfall?center_freq=2437000000&bandwidth=50000000"
        )  # 50MHz
        assert response.status_code == 422
        assert "bandwidth" in response.text.lower()

    def test_spectrum_waterfall_sdr_hardware_integration(self):
        """Test actual SDR hardware integration when available"""
        response = client.get("/api/spectrum/waterfall?center_freq=2437000000&bandwidth=5000000")

        if response.status_code == 200:
            # SDR hardware available - verify real data
            data = response.json()

            # Verify realistic RSSI values (-100 to -20 dBm)
            magnitudes = data["magnitudes"]
            assert all(
                -120 <= mag <= 0 for mag in magnitudes
            ), "RSSI values outside realistic range"

            # Verify frequency array matches bandwidth
            frequencies = data["frequencies"]
            freq_span = max(frequencies) - min(frequencies)
            expected_span = 5e6  # 5MHz bandwidth
            assert abs(freq_span - expected_span) < 1e5, "Frequency span doesn't match bandwidth"

        elif response.status_code == 500:
            # No SDR hardware - should return appropriate error
            assert "sdr" in response.text.lower() or "hardware" in response.text.lower()

        else:
            pytest.fail(f"Unexpected status code: {response.status_code}")

    def test_spectrum_waterfall_performance_timing(self):
        """Test that waterfall endpoint responds within 333ms for 3Hz update rate"""
        import time

        start_time = time.time()
        response = client.get("/api/spectrum/waterfall?center_freq=2437000000")
        end_time = time.time()

        response_time = (end_time - start_time) * 1000  # Convert to ms

        # Should respond within 333ms for 3Hz update rate
        if response.status_code == 200:
            assert response_time < 333, f"Response time {response_time}ms exceeds 333ms requirement"
