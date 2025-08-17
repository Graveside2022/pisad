#!/usr/bin/env python3
"""
Debug FFT RSSI calculation to understand the discrepancy
"""

import sys

import numpy as np

sys.path.append("src")

from backend.services.signal_processor import SignalProcessor

# Test parameters from the failing test
block_size = 1024
sample_rate = 2e6
freq = 250e3  # 250 kHz
amplitude = 2.0

# Generate the same signal
t = np.arange(block_size) / sample_rate
signal_samples = amplitude * np.exp(1j * 2 * np.pi * freq * t)

# Create processor and compute RSSI
processor = SignalProcessor()
rssi, fft_mags = processor.compute_rssi_fft(signal_samples)

print("=== FFT RSSI Debug ===")
print(f"Signal amplitude: {amplitude}")
print(f"Signal frequency: {freq/1e3:.0f} kHz")
print(f"FFT size: {block_size}")
print(f"\nProcessor calibration offset: {processor.calibration_offset} dB")
print(f"Computed FFT RSSI: {rssi:.2f} dBm")

# What the test expects
expected_power_raw = 20 * np.log10(amplitude) + 10
expected_with_offset = expected_power_raw + processor.calibration_offset
print("\nTest expected calculation:")
print(f"  Raw power: 20*log10({amplitude}) + 10 = {expected_power_raw:.2f} dB")
print(
    f"  With offset: {expected_power_raw:.2f} + {processor.calibration_offset} = {expected_with_offset:.2f} dBm"
)

# Manual calculation of what we actually get
# For complex signal, total power is sum of |FFT|^2 / N
fft_result = np.fft.fft(signal_samples * np.hanning(block_size), n=block_size)
fft_magnitude = np.abs(fft_result) / block_size
total_power = np.sum(fft_magnitude**2)
manual_rssi = 10 * np.log10(total_power) + processor.calibration_offset

print("\nManual calculation:")
print(f"  Total power (linear): {total_power:.6f}")
print(f"  Power in dB: 10*log10({total_power:.6f}) = {10*np.log10(total_power):.2f} dB")
print(
    f"  With offset: {10*np.log10(total_power):.2f} + {processor.calibration_offset} = {manual_rssi:.2f} dBm"
)

print(f"\nDiscrepancy: {abs(rssi - expected_with_offset):.2f} dB")

# The issue is that the test's expected calculation is wrong
# It uses 20*log10(amplitude) + 10, which doesn't match how FFT power is calculated
print("\n⚠️  The test's expected power calculation appears incorrect")
print("   It uses: 20*log10(amplitude) + 10")
print("   But FFT gives: 10*log10(sum(|FFT|^2/N))")
print("   These are different formulas!")
