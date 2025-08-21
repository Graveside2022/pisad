# PISAD Parameter Configuration Templates

## Overview

This document provides pre-configured parameter templates for different mission types, ensuring optimal PISAD RF performance while maintaining safety compliance. These templates provide starting configurations that operators can customize based on specific mission requirements.

**Configuration Philosophy:** Proven parameter sets optimized for common SAR scenarios with built-in safety margins and performance optimization.

**Usage Guidelines:** Select appropriate template based on mission type, then customize specific parameters as needed for operational requirements.

---

## **TEMPLATE SELECTION MATRIX**

| Mission Type | Primary Template | Backup Template | Special Considerations |
|--------------|------------------|-----------------|----------------------|
| Emergency Beacon (406MHz) | EMERGENCY_BEACON | EMERGENCY_BACKUP | International standard, highest priority |
| Aviation Emergency (121.5MHz) | AVIATION_EMERGENCY | DUAL_FREQUENCY | Legacy backup frequency |
| Marine Emergency (EPIRB) | MARINE_EMERGENCY | EMERGENCY_BEACON | Coast Guard coordination |
| Training Operations | TRAINING_STANDARD | CUSTOM_TRAINING | Safety margins increased |
| Multi-Aircraft SAR | COORDINATION_PRIMARY | COORDINATION_SECONDARY | Frequency deconfliction |
| High-Interference Environment | INTERFERENCE_ROBUST | ADAPTIVE_FILTERING | Urban or industrial areas |
| Research/Development | RESEARCH_BASE | CUSTOM_RESEARCH | Experimental parameters |

---

## **EMERGENCY BEACON TEMPLATE (406MHz)**

### **Primary Configuration: EMERGENCY_BEACON**

**RF Configuration Parameters:**
```
PISAD_RF_PROFILE = 0                    // Emergency beacon profile
PISAD_RF_FREQ = 406000000              // 406 MHz (international emergency)
PISAD_RF_BW = 25000                    // 25 kHz bandwidth
PISAD_RF_GAIN = 40                     // Moderate gain for sensitivity
PISAD_RF_SAMPLE_RATE = 2048000         // 2.048 MSps
PISAD_RF_LNA_GAIN = 32                 // LNA gain for weak signals
PISAD_RF_VGA_GAIN = 30                 // VGA gain for signal conditioning
```

**Signal Processing Parameters:**
```
PISAD_RSSI_ALPHA = 0.1                 // Smooth RSSI filtering
PISAD_RSSI_THRESH = -85                // Signal detection threshold (dBm)
PISAD_CONFIDENCE_THRESH = 50           // Minimum confidence for detection
PISAD_BEARING_ALPHA = 0.05             // Conservative bearing filtering
PISAD_NOISE_FLOOR_ALPHA = 0.01         // Noise floor adaptation rate
```

**Homing Algorithm Parameters:**
```
PISAD_HOMING_EN = 0                    // Disabled until manual activation
PISAD_APPROACH_SPEED = 5.0             // 5 m/s approach speed
PISAD_SPIRAL_RADIUS = 50.0             // 50m initial spiral radius
PISAD_SPIRAL_INCREMENT = 10.0          // 10m radius increase per turn
PISAD_MAX_SPIRAL_RADIUS = 200.0        // 200m maximum spiral radius
PISAD_BEARING_CONF_THRESH = 70         // 70% minimum bearing confidence
```

**Safety and Monitoring Parameters:**
```
PISAD_BATTERY_THRESH = 20              // 20% battery emergency threshold
PISAD_GPS_HDOP_MAX = 3.0              // Maximum acceptable HDOP
PISAD_TIMEOUT_SIGNAL_LOSS = 10         // 10 second signal loss timeout
PISAD_TIMEOUT_HOMING_MAX = 300         // 5 minute maximum homing time
PISAD_GEOFENCE_ENABLE = 1              // Geofence enforcement enabled
```

### **Backup Configuration: EMERGENCY_BACKUP**

**Dual-Frequency Emergency Configuration:**
```
PISAD_RF_FREQ = 121500000              // 121.5 MHz backup frequency
PISAD_RF_BW = 50000                    // 50 kHz bandwidth for aviation
PISAD_RF_GAIN = 45                     // Increased gain for legacy signals
PISAD_RSSI_THRESH = -90                // Lower threshold for weak signals
PISAD_CONFIDENCE_THRESH = 40           // Reduced confidence for backup frequency
```

---

## **AVIATION EMERGENCY TEMPLATE (121.5MHz)**

### **Primary Configuration: AVIATION_EMERGENCY**

**RF Configuration Parameters:**
```
PISAD_RF_PROFILE = 1                   // Aviation emergency profile
PISAD_RF_FREQ = 121500000              // 121.5 MHz aviation emergency
PISAD_RF_BW = 50000                    // 50 kHz bandwidth
PISAD_RF_GAIN = 45                     // Higher gain for aircraft ELTs
PISAD_RF_SAMPLE_RATE = 2048000         // 2.048 MSps
PISAD_RF_LNA_GAIN = 35                 // Increased LNA for range
PISAD_RF_VGA_GAIN = 32                 // Higher VGA for weak signals
```

**Signal Processing Parameters:**
```
PISAD_RSSI_ALPHA = 0.15                // Faster response for aircraft signals
PISAD_RSSI_THRESH = -90                // Lower threshold for aircraft ELT
PISAD_CONFIDENCE_THRESH = 45           // Reduced confidence for 121.5MHz
PISAD_BEARING_ALPHA = 0.08             // Moderate bearing filtering
PISAD_NOISE_FLOOR_ALPHA = 0.015        // Adaptive noise floor tracking
```

**Aviation-Specific Parameters:**
```
PISAD_AIRCRAFT_MODE = 1                // Aviation emergency mode enabled
PISAD_ALT_COMPENSATION = 1             // Altitude-based signal compensation
PISAD_DOPPLER_COMPENSATION = 1         // Doppler shift compensation
PISAD_APPROACH_SPEED = 8.0             // 8 m/s for aircraft approach
PISAD_MIN_APPROACH_ALT = 100           // 100m minimum approach altitude
```

---

## **MARINE EMERGENCY TEMPLATE (EPIRB)**

### **Primary Configuration: MARINE_EMERGENCY**

**RF Configuration Parameters:**
```
PISAD_RF_PROFILE = 2                   // Marine emergency profile
PISAD_RF_FREQ = 162025000              // 162.025 MHz marine emergency
PISAD_RF_BW = 25000                    // 25 kHz bandwidth
PISAD_RF_GAIN = 42                     // Optimized for marine environment
PISAD_RF_SAMPLE_RATE = 2048000         // 2.048 MSps
PISAD_RF_LNA_GAIN = 33                 // Marine environment optimization
PISAD_RF_VGA_GAIN = 31                 // Balanced VGA setting
```

**Marine Environment Parameters:**
```
PISAD_SALTWATER_COMPENSATION = 1       // Saltwater propagation compensation
PISAD_WAVE_MOTION_FILTER = 1          // Wave motion filtering enabled
PISAD_MARITIME_SEARCH_PATTERN = 1     // Maritime-optimized search patterns
PISAD_TIDAL_DRIFT_COMPENSATION = 1    // Drift compensation for floating beacons
PISAD_MIN_SEARCH_ALTITUDE = 50        // 50m minimum for marine search
```

**Signal Processing for Marine:**
```
PISAD_RSSI_ALPHA = 0.08               // Slower filtering for wave motion
PISAD_MULTIPATH_REJECTION = 1         // Water surface multipath rejection
PISAD_BEACON_DRIFT_TRACKING = 1       // Track beacon movement with currents
```

---

## **TRAINING TEMPLATE**

### **Primary Configuration: TRAINING_STANDARD**

**Training-Optimized RF Parameters:**
```
PISAD_RF_PROFILE = 3                   // Training profile
PISAD_RF_FREQ = 433920000              // 433.92 MHz ISM band for training
PISAD_RF_BW = 25000                    // 25 kHz bandwidth
PISAD_RF_GAIN = 35                     // Reduced gain for training safety
PISAD_RF_SAMPLE_RATE = 2048000         // Standard sample rate
PISAD_RF_LNA_GAIN = 25                 // Conservative LNA gain
PISAD_RF_VGA_GAIN = 25                 // Conservative VGA gain
```

**Training Safety Parameters:**
```
PISAD_TRAINING_MODE = 1                // Training mode enabled
PISAD_TRAINING_MAX_SPEED = 3.0         // 3 m/s maximum training speed
PISAD_TRAINING_MAX_RANGE = 500         // 500m maximum training range
PISAD_INSTRUCTOR_OVERRIDE = 1          // Instructor override enabled
PISAD_TRAINING_TIMEOUT = 600           // 10 minute training timeout
PISAD_SAFETY_MARGIN_FACTOR = 1.5       // Increased safety margins
```

**Performance Monitoring for Training:**
```
PISAD_PERFORMANCE_LOGGING = 1          // Detailed performance logging
PISAD_STUDENT_ASSESSMENT = 1           // Assessment data collection
PISAD_SCENARIO_TRACKING = 1            // Training scenario progress tracking
```

---

## **MULTI-AIRCRAFT COORDINATION TEMPLATES**

### **Primary Aircraft: COORDINATION_PRIMARY**

**Primary Aircraft RF Configuration:**
```
PISAD_RF_PROFILE = 0                   // Emergency beacon profile
PISAD_RF_FREQ = 406000000              // Primary frequency assignment
PISAD_COORDINATION_MODE = 1            // Coordination mode enabled
PISAD_PRIMARY_AIRCRAFT = 1             // Primary aircraft designation
PISAD_COORDINATION_FREQ = 433920000    // Inter-aircraft coordination frequency
PISAD_DATA_SHARING = 1                 // Data sharing enabled
```

**Coordination Parameters:**
```
PISAD_COORD_UPDATE_RATE = 2            // 2 Hz coordination data rate
PISAD_INTERFERENCE_AVOIDANCE = 1       // Automatic interference avoidance
PISAD_SPECTRUM_COORDINATION = 1        // Dynamic spectrum coordination
PISAD_PATTERN_COORDINATION = 1         // Search pattern coordination
```

### **Secondary Aircraft: COORDINATION_SECONDARY**

**Secondary Aircraft Configuration:**
```
PISAD_RF_FREQ = 406050000              // 50 kHz offset from primary
PISAD_PRIMARY_AIRCRAFT = 0             // Secondary aircraft designation
PISAD_FOLLOW_PRIMARY = 1               // Follow primary aircraft commands
PISAD_COORDINATION_SLAVE = 1           // Slave mode for coordination
```

---

## **HIGH-INTERFERENCE ENVIRONMENT TEMPLATE**

### **Primary Configuration: INTERFERENCE_ROBUST**

**Interference-Resistant RF Parameters:**
```
PISAD_RF_PROFILE = 4                   // High-interference profile
PISAD_RF_GAIN = 30                     // Reduced gain to prevent overload
PISAD_RF_BW = 12500                    // Narrower bandwidth
PISAD_INTERFERENCE_REJECTION = 1       // Advanced interference rejection
PISAD_ADAPTIVE_FILTERING = 1           // Adaptive filter enabled
PISAD_SPECTRAL_NOTCHING = 1            // Spectral notch filtering
```

**Advanced Signal Processing:**
```
PISAD_FFT_SIZE = 2048                  // Larger FFT for spectral resolution
PISAD_OVERLAP_FACTOR = 0.75            // 75% overlap for smoothing
PISAD_NOISE_BLANKING = 1               // Impulsive noise blanking
PISAD_AGC_SPEED = 2                    // Faster AGC for varying interference
PISAD_CONFIDENCE_THRESH = 60           // Higher confidence threshold
```

**Urban Environment Adaptations:**
```
PISAD_MULTIPATH_MITIGATION = 1         // Multipath mitigation enabled
PISAD_BUILDING_COMPENSATION = 1        // Building blockage compensation
PISAD_INTERFERENCE_MAPPING = 1         // Real-time interference mapping
```

---

## **RESEARCH/DEVELOPMENT TEMPLATE**

### **Base Configuration: RESEARCH_BASE**

**Flexible Research Parameters:**
```
PISAD_RF_PROFILE = 255                 // Custom research profile
PISAD_RF_FREQ = 434000000              // Configurable frequency
PISAD_EXPERIMENTAL_MODE = 1            // Experimental features enabled
PISAD_DEBUG_LOGGING = 1                // Detailed debug logging
PISAD_RAW_DATA_RECORDING = 1           // Raw IQ data recording
PISAD_ALGORITHM_TESTING = 1            // Algorithm testing mode
```

**Development Parameters:**
```
PISAD_DEV_OVERRIDE_SAFETY = 0          // Safety overrides disabled for research
PISAD_CUSTOM_ALGORITHM = 0             // Custom algorithm selection
PISAD_PARAMETER_SWEEP = 0              // Automated parameter sweeping
PISAD_PERFORMANCE_METRICS = 1          // Detailed performance measurement
```

---

## **PARAMETER CUSTOMIZATION GUIDELINES**

### **Frequency Selection Guidelines**

**Standard Emergency Frequencies:**
- **406 MHz:** International COSPAS-SARSAT standard, highest priority
- **121.5 MHz:** Aviation emergency backup, legacy systems
- **162.025 MHz:** Maritime emergency frequency (EPIRB)
- **243 MHz:** Military emergency frequency (restricted use)

**Regulatory Considerations:**
- Verify frequency legal for operational location
- Coordinate with emergency services to prevent interference
- Respect power limitations and bandwidth restrictions
- Ensure compliance with aviation frequency coordination

### **Performance Optimization Guidelines**

**Signal Strength Optimization:**
- Increase RF_GAIN for weak signals (max 49)
- Adjust LNA_GAIN for optimal sensitivity (typically 25-40)
- Set VGA_GAIN based on expected signal levels (20-40)
- Monitor for receiver overload with strong signals

**Detection Sensitivity Tuning:**
- Lower RSSI_THRESH for weak signals (-95 to -80 dBm)
- Adjust CONFIDENCE_THRESH based on environment (30-70%)
- Increase RSSI_ALPHA for smoother filtering (0.05-0.2)
- Set NOISE_FLOOR_ALPHA for environment adaptation (0.005-0.02)

**Homing Performance Optimization:**
- Adjust APPROACH_SPEED based on aircraft performance (2-10 m/s)
- Set BEARING_CONF_THRESH for reliable homing (60-80%)
- Configure spiral search parameters for search area
- Set timeout values appropriate for mission duration

### **Safety Parameter Guidelines**

**Critical Safety Settings (DO NOT MODIFY):**
- BATTERY_THRESH: Always maintain 20% minimum
- GPS_HDOP_MAX: Never exceed 3.0 for safety
- GEOFENCE_ENABLE: Always keep enabled (1)
- Emergency timeouts: Maintain conservative values

**Environmental Safety Adaptations:**
- Increase safety margins in challenging environments
- Reduce maximum speeds in turbulent conditions
- Extend timeouts for complex search areas
- Add altitude restrictions for terrain avoidance

---

**Document Version:** 1.0  
**Last Updated:** 2025-08-21  
**Parameter Compatibility:** PISAD System v2.0+, Mission Planner 1.3.80+  
**Safety Compliance:** All templates maintain PRD safety requirements and operational standards