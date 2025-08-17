#!/usr/bin/env python3
"""
Test FFT RSSI calibration fix for FR6 requirements
"""

import sys

import numpy as np

sys.path.append("src")

from backend.services.signal_processor import SignalProcessor


def test_fft_rssi_calibration():
    """
    Test that FFT RSSI computation matches expected values
    Based on failing test showing ~14dB offset error
    """
    signal_processor = SignalProcessor()

    # Generate known signal power
    sample_rate = 2e6
    num_samples = 1024

    # Create a signal with known power
    # Using similar signal to the failing test
    t = np.arange(num_samples) / sample_rate
    signal_freq = 100e3  # 100 kHz
    signal_amplitude = 1.0  # Known amplitude

    # Pure sine wave signal
    signal = signal_amplitude * np.exp(1j * 2 * np.pi * signal_freq * t)

    # Compute FFT RSSI
    rssi_fft, _ = signal_processor.compute_rssi_fft(signal)

    # For a signal with amplitude 1.0, the expected power in dBm
    # depends on the reference impedance (typically 50 ohms)
    # Expected: around -14 dBm based on test expectation
    # Current: around -28 dBm with -30 offset

    print("\n=== FFT RSSI Calibration Test ===")
    print(f"Signal amplitude: {signal_amplitude}")
    print(f"Current calibration offset: {signal_processor.calibration_offset} dBm")
    print(f"Computed FFT RSSI: {rssi_fft:.1f} dBm")

    # The test expects around -14 dBm for this signal
    expected_rssi = -14.0  # dBm
    error = abs(rssi_fft - expected_rssi)

    print(f"Expected RSSI: {expected_rssi:.1f} dBm")
    print(f"Error: {error:.1f} dB")

    if error > 3.0:
        print(f"\n❌ FAILURE: FFT RSSI calibration is off by {error:.1f} dB")
        print(
            f"   Need to adjust calibration_offset from {signal_processor.calibration_offset} to approximately {signal_processor.calibration_offset + (expected_rssi - rssi_fft):.1f}"
        )
        return False
    else:
        print("\n✅ FFT RSSI calibration is correct (within 3dB tolerance)")
        return True


if __name__ == "__main__":
    success = test_fft_rssi_calibration()
    sys.exit(0 if success else 1)
