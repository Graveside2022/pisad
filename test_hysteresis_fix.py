#!/usr/bin/env python3
"""
Test for validating hysteresis fix for FR7 - Debounced Transitions
CRITICAL: This tests REAL behavior, no mocks
"""

import sys

sys.path.append("src")

from backend.services.signal_processor import SignalProcessor


def test_hysteresis_implementation():
    """
    TDD Test: Verify proper hysteresis behavior per FR7 requirements:
    - Trigger at 12dB threshold
    - Drop at 6dB threshold
    - Proper state transitions with debouncing
    """
    signal_processor = SignalProcessor()

    # Reset state
    signal_processor.is_detecting = False
    signal_processor.detection_count = 0
    signal_processor.loss_count = 0
    signal_processor.detection_count_threshold = 3
    signal_processor.loss_count_threshold = 3

    trigger_threshold = 12.0  # dB
    drop_threshold = 6.0  # dB
    noise_floor = -80.0  # dBm

    # Test sequence: low -> high -> hysteresis zone -> low
    test_cases = [
        # (rssi, expected_state, description)
        (-75, False, "Below trigger - should not detect"),
        (-75, False, "Still below trigger"),
        (-67, False, "First above trigger - not yet detected (need 3)"),
        (-67, False, "Second above trigger - not yet detected"),
        (-67, True, "Third above trigger - DETECTED!"),
        (-72, True, "In hysteresis zone (8dB) - should MAINTAIN"),
        (-73, True, "Still in hysteresis (7dB) - should MAINTAIN"),
        (-75, True, "Just below drop (5dB) - first loss"),
        (-75, True, "Below drop - second loss"),
        (-75, False, "Below drop - third loss - DROPPED!"),
    ]

    print("\n=== Testing Hysteresis State Machine ===")
    for i, (rssi, expected, desc) in enumerate(test_cases):
        result = signal_processor.process_detection_with_debounce(
            rssi, noise_floor, trigger_threshold, drop_threshold
        )
        snr = rssi - noise_floor
        status = "✓" if result == expected else "✗"
        print(
            f"Step {i+1}: SNR={snr:5.1f}dB | Expected={expected:5} | Got={result:5} | {status} | {desc}"
        )

        if result != expected:
            print(f"FAILURE at step {i+1}: {desc}")
            print(f"  Current state: is_detecting={signal_processor.is_detecting}")
            print(f"  Detection count: {signal_processor.detection_count}")
            print(f"  Loss count: {signal_processor.loss_count}")
            return False

    print("\n✅ Hysteresis test PASSED!")
    return True


if __name__ == "__main__":
    success = test_hysteresis_implementation()
    sys.exit(0 if success else 1)
