# Homing Algorithm Configuration Guide

## Overview

The PISAD homing algorithm uses a gradient-based approach to guide drones toward beacon signals. This document describes all configuration parameters available for tuning the homing behavior.

## Configuration Parameters

All parameters are defined in `config/default.yaml` and can be overridden via environment variables or runtime API calls.

### Core Algorithm Settings

| Parameter                    | Default      | Range                | Description                                                                                                |
| ---------------------------- | ------------ | -------------------- | ---------------------------------------------------------------------------------------------------------- |
| `HOMING_ALGORITHM_MODE`      | `"GRADIENT"` | `SIMPLE`, `GRADIENT` | Selects homing algorithm type. GRADIENT uses advanced gradient climbing, SIMPLE uses direct RSSI following |
| `HOMING_SIGNAL_LOSS_TIMEOUT` | `5.0`        | 1.0-30.0             | Time in seconds before homing disables after signal loss                                                   |

### Velocity Control

| Parameter                      | Default | Range    | Description                                                               |
| ------------------------------ | ------- | -------- | ------------------------------------------------------------------------- |
| `HOMING_FORWARD_VELOCITY_MAX`  | `5.0`   | 0.5-10.0 | Maximum forward velocity in m/s during homing                             |
| `HOMING_YAW_RATE_MAX`          | `0.5`   | 0.1-2.0  | Maximum yaw rate in rad/s for turning toward signal                       |
| `HOMING_APPROACH_VELOCITY`     | `1.0`   | 0.5-3.0  | Reduced velocity in m/s when approaching target (RSSI > threshold)        |
| `HOMING_VELOCITY_SCALE_FACTOR` | `0.1`   | 0.01-1.0 | Scales velocity based on signal gradient strength (0.1 = 10% of gradient) |

### Gradient Calculation

| Parameter                     | Default | Range    | Description                                                        |
| ----------------------------- | ------- | -------- | ------------------------------------------------------------------ |
| `HOMING_GRADIENT_WINDOW_SIZE` | `10`    | 5-50     | Number of RSSI samples in history buffer for gradient calculation  |
| `HOMING_GRADIENT_MIN_SNR`     | `10.0`  | 3.0-30.0 | Minimum signal-to-noise ratio in dB required for reliable gradient |

### Sampling Maneuvers

When the gradient direction is unclear (low SNR or inconsistent readings), the algorithm performs S-turn sampling maneuvers.

| Parameter                     | Default | Range    | Description                                    |
| ----------------------------- | ------- | -------- | ---------------------------------------------- |
| `HOMING_SAMPLING_TURN_RADIUS` | `10.0`  | 5.0-30.0 | Radius in meters for S-turn sampling pattern   |
| `HOMING_SAMPLING_DURATION`    | `5.0`   | 3.0-15.0 | Duration in seconds for each sampling maneuver |

### Target Detection

| Parameter                   | Default | Range          | Description                                                         |
| --------------------------- | ------- | -------------- | ------------------------------------------------------------------- |
| `HOMING_APPROACH_THRESHOLD` | `-50.0` | -70.0 to -30.0 | RSSI threshold in dBm indicating close proximity to target          |
| `HOMING_PLATEAU_VARIANCE`   | `2.0`   | 0.5-5.0        | RSSI variance threshold for detecting signal plateau (target below) |

## Tuning Guide

### For Long-Range Searches

- Increase `HOMING_FORWARD_VELOCITY_MAX` to 8-10 m/s
- Increase `HOMING_GRADIENT_WINDOW_SIZE` to 20-30 samples
- Decrease `HOMING_GRADIENT_MIN_SNR` to 5-8 dB

### For Precision Location

- Decrease `HOMING_FORWARD_VELOCITY_MAX` to 2-3 m/s
- Decrease `HOMING_APPROACH_VELOCITY` to 0.5 m/s
- Increase `HOMING_GRADIENT_MIN_SNR` to 15-20 dB
- Decrease `HOMING_PLATEAU_VARIANCE` to 1.0

### For Noisy Environments

- Increase `HOMING_GRADIENT_WINDOW_SIZE` to 20-30 samples
- Increase `HOMING_GRADIENT_MIN_SNR` to 15-20 dB
- Increase `HOMING_SAMPLING_DURATION` to 8-10 seconds
- Increase `HOMING_PLATEAU_VARIANCE` to 3-4

## Runtime Configuration

Parameters can be updated at runtime via the API:

```bash
# Get current parameters
curl http://localhost:8000/api/homing/parameters

# Update parameters
curl -X PATCH http://localhost:8000/api/homing/parameters \
  -H "Content-Type: application/json" \
  -d '{
    "HOMING_FORWARD_VELOCITY_MAX": 3.0,
    "HOMING_GRADIENT_MIN_SNR": 15.0
  }'
```

## Safety Considerations

- The `SAFETY_VELOCITY_MAX_MPS` setting (default 2.0 m/s) overrides homing velocities
- Safety interlocks will disable homing if triggered
- Always test configuration changes in SITL before field deployment

## Monitoring

Enable debug logging to monitor algorithm performance:

```yaml
DEV_DEBUG_MODE: true
LOG_LEVEL: "DEBUG"
```

This will log:

- Gradient calculations and confidence scores
- State transitions (GRADIENT_CLIMB, SAMPLING, APPROACH, HOLDING)
- Velocity commands sent to flight controller
- RSSI history and variance calculations
