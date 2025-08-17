#!/usr/bin/env python3
"""
Test specifically for the brief spike scenario that's failing in PRD tests
"""
import sys
sys.path.append('src')

from backend.services.signal_processor import SignalProcessor

def test_brief_spike_after_signal_loss():
    """Test that brief spike after signal loss doesn't trigger detection"""
    signal_processor = SignalProcessor()
    
    # Initialize with known state
    signal_processor.is_detecting = False
    signal_processor.detection_count = 0
    signal_processor.loss_count = 0
    signal_processor.detection_count_threshold = 3
    signal_processor.loss_count_threshold = 3
    
    trigger_threshold = 12.0
    drop_threshold = 6.0
    noise_floor = -80.0
    
    # Simulate the exact test pattern from PRD test
    snr_pattern = [
        # Initial low signal (samples 0-19)
        *[3.0] * 20,     # Below drop threshold
        # High signal (samples 20-39) 
        *[15.0] * 20,    # Above trigger threshold  
        # Medium signal (samples 40-59)
        *[8.0] * 20,     # Between thresholds (hysteresis zone)
        # Brief spike (samples 60-64)
        *[13.0] * 5,     # Above trigger but only 5 samples
        # Back to low (samples 65+)
        *[4.0] * 35      # Below drop threshold
    ]
    
    states = []
    print("\n=== Testing Brief Spike Scenario ===")
    print("Pattern: low -> high (trigger) -> medium (hysteresis) -> brief spike -> low")
    
    for i, snr in enumerate(snr_pattern):
        rssi = noise_floor + snr
        is_detected = signal_processor.process_detection_with_debounce(
            rssi, noise_floor, trigger_threshold, drop_threshold
        )
        
        state = "SIGNAL_DETECTED" if is_detected else "NO_SIGNAL"
        states.append(state)
        
        # Log key transitions
        if i in [19, 22, 39, 42, 59, 62, 64, 67]:
            print(f"Sample {i:3d}: SNR={snr:5.1f}dB, State={state:15s}, "
                  f"is_detecting={signal_processor.is_detecting}, "
                  f"det_count={signal_processor.detection_count}, "
                  f"loss_count={signal_processor.loss_count}")
    
    # Check critical assertions from PRD test
    print(f"\n=== Checking Assertions ===")
    
    # Should trigger after sustained signal (sample 23)
    assert states[23] == "SIGNAL_DETECTED", f"Should detect after debounced trigger, got {states[23]}"
    print(f"✓ Sample 23: Correctly detected after 3+ samples above trigger")
    
    # Should maintain in hysteresis zone (sample 45) 
    assert states[45] == "SIGNAL_DETECTED", f"Should maintain state between thresholds, got {states[45]}"
    print(f"✓ Sample 45: Correctly maintained in hysteresis zone")
    
    # CRITICAL: Brief spike at sample 62 should NOT trigger
    print(f"\n🔍 Sample 62 (brief spike): {states[62]}")
    if states[62] == "SIGNAL_DETECTED":
        print("❌ FAILURE: Brief spike incorrectly triggered detection!")
        print("   This is the bug - after signal is lost in hysteresis zone,")
        print("   the state machine incorrectly allows re-triggering on brief spikes")
        return False
    else:
        print("✓ Sample 62: Correctly ignored brief spike")
    
    # Should drop after sustained low signal (sample 67)
    assert states[67] == "NO_SIGNAL", f"Should drop after debounced drop, got {states[67]}"
    print(f"✓ Sample 67: Correctly dropped after sustained low signal")
    
    print("\n✅ All assertions passed!")
    return True

if __name__ == "__main__":
    success = test_brief_spike_after_signal_loss()
    sys.exit(0 if success else 1)