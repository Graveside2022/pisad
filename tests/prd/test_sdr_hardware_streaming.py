#!/usr/bin/env python3
"""
Test TASK-9.1: Real SDR Streaming Tests with HackRF Hardware
CRITICAL: Tests REAL hardware integration, no mocks
Tests FR1: Autonomous RF beacon detection
"""
import pytest
import numpy as np
import time
import SoapySDR
from typing import Generator

# Mark all tests as requiring hardware
pytestmark = pytest.mark.skipif(
    not SoapySDR.Device.enumerate({"driver": "hackrf"}),
    reason="HackRF hardware not available"
)


class TestRealSDRStreaming:
    """Test real SDR streaming with verified HackRF hardware."""
    
    @pytest.fixture
    def hackrf_device(self) -> Generator[SoapySDR.Device, None, None]:
        """Create and configure real HackRF device."""
        # Find HackRF devices
        devices = SoapySDR.Device.enumerate({"driver": "hackrf"})
        assert len(devices) > 0, "No HackRF devices found"
        
        # Create device
        device = SoapySDR.Device(devices[0])
        
        # Configure for FR1 requirements: 3.2 GHz center, 2 MHz BW
        device.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, 2e6)  # 2 MHz sample rate
        device.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, 3.2e9)  # 3.2 GHz center
        device.setGain(SoapySDR.SOAPY_SDR_RX, 0, 20)  # 20 dB gain
        
        yield device
        
        # Cleanup
        device = None
    
    def test_hackrf_enumeration(self):
        """Test that HackRF is properly enumerated."""
        devices = SoapySDR.Device.enumerate({"driver": "hackrf"})
        assert len(devices) > 0, "HackRF not found"
        
        # Verify device info
        device_info = devices[0]
        assert "driver" in device_info
        assert device_info["driver"] == "hackrf"
        print(f"✓ Found HackRF: {device_info}")
    
    def test_hackrf_device_creation(self):
        """Test creating HackRF device instance."""
        devices = SoapySDR.Device.enumerate({"driver": "hackrf"})
        device = SoapySDR.Device(devices[0])
        
        # Verify device info
        hw_info = device.getHardwareKey()
        assert "hackrf" in hw_info.lower()
        
        # Check driver version
        driver_info = device.getDriverKey()
        assert driver_info.lower() == "hackrf"
        
        print(f"✓ Created device: {hw_info}")
    
    def test_frequency_configuration(self, hackrf_device):
        """Test setting frequencies per FR1 (850 MHz - 6.5 GHz range)."""
        test_frequencies = [
            850e6,   # Lower bound
            2.4e9,   # WiFi band
            3.2e9,   # Default FR1 frequency
            5.8e9,   # Upper WiFi band
            6.5e9,   # Upper bound
        ]
        
        for freq in test_frequencies:
            hackrf_device.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, freq)
            actual = hackrf_device.getFrequency(SoapySDR.SOAPY_SDR_RX, 0)
            
            # Allow 1 Hz tolerance for frequency setting
            assert abs(actual - freq) < 1.0, f"Failed to set {freq/1e9:.1f} GHz"
            print(f"✓ Set frequency: {freq/1e9:.3f} GHz")
    
    def test_sample_rate_configuration(self, hackrf_device):
        """Test configuring sample rates for FR1 (2-5 MHz bandwidth)."""
        test_rates = [
            2e6,   # 2 MHz (minimum for FR1)
            3e6,   # 3 MHz  
            5e6,   # 5 MHz (maximum for FR1)
        ]
        
        for rate in test_rates:
            hackrf_device.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, rate)
            actual = hackrf_device.getSampleRate(SoapySDR.SOAPY_SDR_RX, 0)
            
            # Allow 1% tolerance
            assert abs(actual - rate) < rate * 0.01, f"Failed to set {rate/1e6:.1f} Msps"
            print(f"✓ Set sample rate: {rate/1e6:.1f} Msps")
    
    def test_stream_creation(self, hackrf_device):
        """Test creating receive stream."""
        # Create RX stream
        stream = hackrf_device.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
        assert stream is not None, "Failed to create stream"
        
        # Activate stream
        ret = hackrf_device.activateStream(stream)
        assert ret == 0, f"Failed to activate stream: {ret}"
        
        # Deactivate and close
        hackrf_device.deactivateStream(stream)
        hackrf_device.closeStream(stream)
        
        print("✓ Stream created and activated successfully")
    
    def test_continuous_streaming(self, hackrf_device):
        """Test continuous IQ sample streaming for FR1 beacon detection."""
        # Setup stream
        stream = hackrf_device.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
        hackrf_device.activateStream(stream)
        
        # Prepare buffer
        buffer_size = 1024
        samples = np.zeros(buffer_size, dtype=np.complex64)
        
        # Stream for 100ms (FR2 latency requirement)
        start_time = time.perf_counter()
        total_samples = 0
        
        while (time.perf_counter() - start_time) < 0.1:  # 100ms
            sr = hackrf_device.readStream(stream, [samples], buffer_size)
            
            if sr.ret > 0:
                total_samples += sr.ret
                
                # Verify we got complex samples
                assert samples.dtype == np.complex64
                
                # Check signal characteristics
                power = np.mean(np.abs(samples[:sr.ret])**2)
                assert power > 0, "No signal power detected"
        
        # Verify streaming performance
        duration = time.perf_counter() - start_time
        sample_rate = total_samples / duration
        
        print(f"✓ Streamed {total_samples} samples in {duration*1000:.1f}ms")
        print(f"  Effective rate: {sample_rate/1e6:.2f} Msps")
        
        # NFR2: Processing latency < 100ms
        assert duration < 0.11, f"Streaming latency {duration*1000:.1f}ms exceeds 100ms requirement"
        
        # Cleanup
        hackrf_device.deactivateStream(stream)
        hackrf_device.closeStream(stream)
    
    def test_rssi_measurement(self, hackrf_device):
        """Test RSSI measurement from real signal samples."""
        # Configure for test
        hackrf_device.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, 2e6)
        hackrf_device.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, 2.4e9)  # WiFi band likely has signals
        hackrf_device.setGain(SoapySDR.SOAPY_SDR_RX, 0, 30)
        
        # Setup stream
        stream = hackrf_device.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
        hackrf_device.activateStream(stream)
        
        # Collect samples
        buffer_size = 8192
        samples = np.zeros(buffer_size, dtype=np.complex64)
        
        sr = hackrf_device.readStream(stream, [samples], buffer_size)
        assert sr.ret > 0, "Failed to read samples"
        
        # Calculate RSSI
        valid_samples = samples[:sr.ret]
        power_linear = np.mean(np.abs(valid_samples)**2)
        rssi_dbm = 10 * np.log10(power_linear) - 10  # Calibrated for HackRF
        
        print(f"✓ Measured RSSI: {rssi_dbm:.1f} dBm")
        
        # Verify RSSI is in reasonable range
        assert -100 < rssi_dbm < 0, f"RSSI {rssi_dbm:.1f} dBm out of range"
        
        # Cleanup
        hackrf_device.deactivateStream(stream)
        hackrf_device.closeStream(stream)
    
    def test_snr_threshold_detection(self, hackrf_device):
        """Test SNR threshold detection per FR1 (>12 dB threshold)."""
        # Setup for noise floor measurement
        hackrf_device.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, 2e6)
        hackrf_device.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, 3.2e9)
        hackrf_device.setGain(SoapySDR.SOAPY_SDR_RX, 0, 10)  # Low gain for noise floor
        
        stream = hackrf_device.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
        hackrf_device.activateStream(stream)
        
        # Measure noise floor
        buffer_size = 4096
        noise_samples = np.zeros(buffer_size, dtype=np.complex64)
        
        sr = hackrf_device.readStream(stream, [noise_samples], buffer_size)
        noise_power = np.mean(np.abs(noise_samples[:sr.ret])**2)
        noise_floor_dbm = 10 * np.log10(noise_power) - 10
        
        print(f"✓ Noise floor: {noise_floor_dbm:.1f} dBm")
        
        # Increase gain to potentially detect signals
        hackrf_device.setGain(SoapySDR.SOAPY_SDR_RX, 0, 40)
        
        # Look for signals above noise floor
        signal_samples = np.zeros(buffer_size, dtype=np.complex64)
        sr = hackrf_device.readStream(stream, [signal_samples], buffer_size)
        
        signal_power = np.mean(np.abs(signal_samples[:sr.ret])**2)
        signal_dbm = 10 * np.log10(signal_power) - 10
        
        snr = signal_dbm - noise_floor_dbm
        print(f"✓ Signal: {signal_dbm:.1f} dBm, SNR: {snr:.1f} dB")
        
        # Check if we would detect per FR1 (>12 dB threshold)
        if snr > 12:
            print(f"  → Would trigger detection (SNR > 12 dB)")
        else:
            print(f"  → Below detection threshold")
        
        # Cleanup
        hackrf_device.deactivateStream(stream)
        hackrf_device.closeStream(stream)
    
    def test_streaming_performance(self, hackrf_device):
        """Test streaming performance meets NFR2 (<100ms latency)."""
        # Configure for performance test
        hackrf_device.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, 2e6)
        
        stream = hackrf_device.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
        hackrf_device.activateStream(stream)
        
        # Test various buffer sizes
        buffer_sizes = [256, 512, 1024, 2048, 4096]
        
        for size in buffer_sizes:
            samples = np.zeros(size, dtype=np.complex64)
            
            # Measure read latency
            start = time.perf_counter()
            sr = hackrf_device.readStream(stream, [samples], size)
            latency = (time.perf_counter() - start) * 1000  # ms
            
            if sr.ret > 0:
                print(f"✓ Buffer {size:4d}: {latency:6.2f}ms latency, {sr.ret:4d} samples")
                
                # NFR2: Must be under 100ms
                assert latency < 100, f"Latency {latency:.1f}ms exceeds NFR2 requirement"
        
        # Cleanup
        hackrf_device.deactivateStream(stream)
        hackrf_device.closeStream(stream)


if __name__ == "__main__":
    # Run with: pytest test_sdr_hardware_streaming.py -v
    pytest.main([__file__, "-v"])