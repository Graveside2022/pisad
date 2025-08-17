#!/usr/bin/env python3
"""
Debug the hysteresis logic to understand the state transitions
"""
import sys
sys.path.append('src')

from backend.services.signal_processor import SignalProcessor

signal_processor = SignalProcessor()
signal_processor.is_detecting = False
signal_processor.detection_count = 0
signal_processor.loss_count = 0
signal_processor.detection_count_threshold = 3
signal_processor.loss_count_threshold = 3

trigger_threshold = 12.0
drop_threshold = 6.0
noise_floor = -80.0

# Focus on the transition from hysteresis zone to brief spike
test_sequence = [
    8.0,   # In hysteresis zone (sample 58)
    8.0,   # In hysteresis zone (sample 59)
    13.0,  # Start of spike (sample 60)
    13.0,  # Spike continues (sample 61)
    13.0,  # Spike continues (sample 62)
]

print("Starting state: is_detecting=True (from previous high signal)")
signal_processor.is_detecting = True  # Already detecting from previous high signal

for i, snr in enumerate(test_sequence, start=58):
    rssi = noise_floor + snr
    
    # Debug the decision logic
    print(f"\n--- Sample {i}: SNR={snr}dB ---")
    print(f"Before: is_detecting={signal_processor.is_detecting}, det_count={signal_processor.detection_count}, loss_count={signal_processor.loss_count}")
    
    # Check what threshold is being used
    if not signal_processor.is_detecting:
        threshold_used = trigger_threshold
        print(f"Using trigger threshold: {threshold_used}dB (not detecting)")
    else:
        threshold_used = drop_threshold  
        print(f"Using drop threshold: {threshold_used}dB (currently detecting)")
    
    signal_above = snr > threshold_used
    print(f"SNR ({snr}dB) > threshold ({threshold_used}dB)? {signal_above}")
    
    result = signal_processor.process_detection_with_debounce(
        rssi, noise_floor, trigger_threshold, drop_threshold
    )
    
    print(f"After: is_detecting={signal_processor.is_detecting}, det_count={signal_processor.detection_count}, loss_count={signal_processor.loss_count}")
    print(f"Result: {'DETECTED' if result else 'NO_SIGNAL'}")