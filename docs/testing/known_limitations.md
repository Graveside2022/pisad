# Known Limitations

## Executive Summary

This document outlines the known limitations of the PiSAD (Pi-based Search and Delivery) system based on field testing results and technical analysis. Understanding these limitations is critical for safe and effective system deployment.

## Detection Range Limitations

### Maximum Effective Range

- **Open Field**: 500-750m with 20dBm beacon power
- **Urban Environment**: 200-400m due to RF interference
- **Forest/Dense Vegetation**: 100-300m due to signal attenuation
- **Mountainous Terrain**: Highly variable, 50-500m depending on line-of-sight

### Factors Affecting Range

1. **Beacon Power**
   - 5 dBm: 100-200m typical range
   - 10 dBm: 250-400m typical range
   - 15 dBm: 400-600m typical range
   - 20 dBm: 500-750m typical range

2. **Environmental Conditions**
   - Rain: -20% to -30% range reduction
   - Fog: -10% to -20% range reduction
   - High humidity: -5% to -15% range reduction
   - Temperature extremes: ±10% variation

3. **RF Interference**
   - Urban areas: -40% to -60% range reduction
   - Near power lines: -20% to -30% range reduction
   - Other 433 MHz devices: Potential false positives

## Approach Accuracy Limitations

### Position Accuracy

- **Best Case**: ±15m from beacon position
- **Typical**: ±25-35m from beacon position
- **Worst Case**: ±50m in challenging conditions

### Factors Affecting Accuracy

1. **GPS Precision**
   - Standard GPS: ±5-10m inherent error
   - HDOP > 2.0: Degraded accuracy
   - < 8 satellites: Unreliable positioning

2. **Signal Multipath**
   - Buildings/structures cause signal reflections
   - Can lead to 90-180° directional errors
   - Most severe in urban canyons

3. **Wind Conditions**
   - > 15 mph: Difficulty maintaining position
   - > 25 mph: Unsafe for operation
   - Gusts cause approach instability

## System Performance Limitations

### Processing Constraints

1. **Raspberry Pi 5 Limitations**
   - CPU saturation at > 10 Hz processing rate
   - Memory constraints with large signal buffers
   - Thermal throttling above 80°C

2. **Real-time Processing**
   - FFT processing latency: 50-100ms
   - State transition latency: 100-500ms typical
   - Total system latency: 200-800ms

### Communication Limitations

1. **MAVLink Bandwidth**
   - Limited to 57600 baud typically
   - Command latency: 20-50ms
   - Telemetry update rate: 4-10 Hz

2. **WebSocket Performance**
   - Maximum 30 concurrent connections
   - Update rate limited to 5 Hz for stability
   - Latency increases with distance from ground station

## Environmental Operating Limits

### Temperature Range

- **Operating**: -10°C to +45°C
- **Storage**: -20°C to +60°C
- **Optimal**: +10°C to +30°C

### Weather Conditions

- **Maximum Wind**: 25 mph sustained, 35 mph gusts
- **Precipitation**: Light rain only, no heavy rain/snow
- **Visibility**: Minimum 500m for manual override
- **Ceiling**: Minimum 150m AGL

### Altitude Limitations

- **Maximum AGL**: 400 ft (regulatory)
- **Minimum AGL**: 50 ft (safety)
- **Density Altitude**: Performance degrades > 6000 ft

## Signal Processing Limitations

### False Positive/Negative Scenarios

1. **False Positives**
   - Other 433 MHz transmitters
   - Harmonic interference
   - RF noise spikes
   - Rate: ~2-5% in urban environments

2. **False Negatives**
   - Beacon battery depletion
   - Antenna orientation mismatch
   - Severe signal attenuation
   - Rate: ~1-3% in tested conditions

### Detection Reliability

- **Clear conditions**: 95-98% detection rate
- **Moderate interference**: 85-92% detection rate
- **High interference**: 70-80% detection rate
- **Extreme conditions**: < 70% detection rate

## Safety System Limitations

### Emergency Stop

- **Response Time**: 0.5-2.0 seconds
- **Stopping Distance**: 5-20m depending on speed
- **Not Effective When**: Communication lost

### Geofence

- **Accuracy**: ±10m from boundary
- **Response Time**: 1-3 seconds
- **Limitation**: GPS-dependent, fails indoors

### Battery Failsafe

- **Trigger Level**: 20% remaining
- **Reserve Time**: 3-5 minutes typical
- **Warning**: No redundant battery system

## Operational Limitations

### Flight Time

- **Maximum**: 25-30 minutes
- **With Homing**: 15-20 minutes
- **In Wind**: 10-15 minutes
- **Cold Weather**: -30% reduction

### Search Pattern Coverage

- **Spiral Search**: 1 km² in 20 minutes
- **Grid Search**: 0.5 km² in 20 minutes
- **Sector Search**: 0.75 km² in 20 minutes

### Beacon Requirements

- **Battery Life**: 24-48 hours continuous
- **Frequency Stability**: ±1 kHz required
- **Minimum Power**: 5 dBm for 100m detection
- **Antenna Type**: Omnidirectional required

## Technical Debt and Known Issues

### Software Limitations

1. **State Machine**
   - No redundant state validation
   - Single-threaded state transitions
   - Limited error recovery paths

2. **Signal Processing**
   - Fixed FFT size (1024 samples)
   - No adaptive filtering
   - Limited multipath mitigation

3. **Navigation**
   - Simple vector-based homing
   - No obstacle avoidance
   - No path optimization

### Hardware Limitations

1. **SDR Performance**
   - Limited to 2 MHz bandwidth
   - 8-bit ADC resolution
   - Sensitivity: -110 dBm typical

2. **Antenna System**
   - Fixed omnidirectional pattern
   - No diversity reception
   - 3 dBi gain maximum

## Mitigation Strategies

### Operational Mitigations

1. **Pre-flight Planning**
   - Survey area for RF interference
   - Check weather conditions
   - Verify GPS satellite availability
   - Test beacon before deployment

2. **In-flight Procedures**
   - Monitor RSSI continuously
   - Maintain visual line of sight
   - Have manual override ready
   - Follow approach patterns

3. **Post-flight Analysis**
   - Review telemetry logs
   - Identify anomalies
   - Update operational procedures
   - Document lessons learned

### Technical Improvements (Future)

1. **Hardware Upgrades**
   - Directional antenna system
   - Redundant SDR receivers
   - RTK GPS for precision
   - Dedicated signal processor

2. **Software Enhancements**
   - Machine learning for signal classification
   - Adaptive filtering algorithms
   - Multi-beacon triangulation
   - Obstacle avoidance system

## Regulatory Compliance Limitations

- **Part 107 Compliance**: Daytime VFR only
- **Frequency Allocation**: ISM band shared use
- **Power Limits**: 20 dBm maximum (US)
- **Altitude Restrictions**: 400 ft AGL maximum

## Conclusion

The PiSAD system demonstrates effective beacon detection and homing capabilities within the documented limitations. Operators must understand these constraints to ensure safe and successful operations. Continuous testing and refinement will help push these boundaries while maintaining system reliability and safety.

## Revision History

| Version | Date       | Author          | Changes                                     |
| ------- | ---------- | --------------- | ------------------------------------------- |
| 1.0     | 2025-08-13 | Field Test Team | Initial documentation based on test results |
