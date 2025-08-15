"""Hypothesis strategies for generating domain objects.

This module provides reusable strategies for property-based testing
of PISAD domain objects.
"""

from datetime import UTC, datetime, timedelta

import hypothesis.strategies as st

from src.backend.models.schemas import (
    HomingConfig,
    SDRConfig,
    SDRStatus,
    SignalConfig,
    SystemState,
)

# Basic value strategies
valid_frequencies = st.floats(min_value=850e6, max_value=6.5e9)  # HackRF range
valid_sample_rates = st.floats(min_value=1e6, max_value=20e6)  # HackRF range
valid_gains = st.one_of(
    st.just("AUTO"),
    st.floats(min_value=0, max_value=62),  # HackRF gain range
)
valid_rssi = st.floats(min_value=-120, max_value=0)  # dBm range
valid_snr = st.floats(min_value=-10, max_value=50)  # dB range
valid_percentages = st.floats(min_value=0, max_value=100)

# State strategies
flight_modes = st.sampled_from(["GUIDED", "MANUAL", "AUTO", "RTL", "LAND", "LOITER"])
gps_statuses = st.sampled_from(["NO_FIX", "2D_FIX", "3D_FIX", "RTK"])
system_states = st.sampled_from(["IDLE", "SEARCHING", "DETECTING", "HOMING", "HOLDING"])
sdr_statuses = st.sampled_from(["CONNECTED", "DISCONNECTED", "ERROR"])

# Time strategies
recent_timestamps = st.datetimes(
    min_value=datetime.now(UTC) - timedelta(hours=1),
    max_value=datetime.now(UTC) + timedelta(seconds=1),
)


@st.composite
def sdr_configs(draw, valid_only: bool = True, allow_invalid_nyquist: bool = False) -> SDRConfig:
    """Generate SDR configuration objects.

    Args:
        valid_only: If True, only generate valid configurations
        allow_invalid_nyquist: If True, allow bandwidth > sample_rate
    """
    frequency = draw(valid_frequencies)
    sample_rate = draw(valid_sample_rates)
    gain = draw(valid_gains)

    if allow_invalid_nyquist:
        bandwidth = draw(st.floats(min_value=0.5e6, max_value=30e6))
    else:
        # Ensure Nyquist criterion
        bandwidth = draw(st.floats(min_value=0.5e6, max_value=min(sample_rate, 20e6)))

    buffer_size = draw(st.sampled_from([256, 512, 1024, 2048, 4096, 8192]))

    return SDRConfig(
        frequency=frequency,
        sampleRate=sample_rate,
        gain=gain,
        bandwidth=bandwidth,
        buffer_size=buffer_size,
    )


@st.composite
def signal_configs(draw, valid_only: bool = True) -> SignalConfig:
    """Generate signal processing configuration objects."""
    fft_size = draw(st.sampled_from([256, 512, 1024, 2048, 4096]))
    ewma_alpha = draw(st.floats(min_value=0.01, max_value=0.99))

    if valid_only:
        # Ensure proper hysteresis
        trigger = draw(st.floats(min_value=-90, max_value=-30))
        drop = draw(st.floats(min_value=-100, max_value=trigger - 6))  # Min 6dB hysteresis
    else:
        trigger = draw(st.floats(min_value=-100, max_value=-20))
        drop = draw(st.floats(min_value=-110, max_value=-10))

    return SignalConfig(
        fftSize=fft_size,
        ewmaAlpha=ewma_alpha,
        triggerThreshold=trigger,
        dropThreshold=drop,
    )


@st.composite
def homing_configs(draw, safe_only: bool = True) -> HomingConfig:
    """Generate homing configuration objects.

    Args:
        safe_only: If True, only generate safe velocity/rate values
    """
    if safe_only:
        max_velocity = draw(st.floats(min_value=1, max_value=8))  # Safe range
        yaw_rate = draw(st.floats(min_value=0.1, max_value=1.5))  # Safe turn rate
    else:
        max_velocity = draw(st.floats(min_value=0.1, max_value=20))
        yaw_rate = draw(st.floats(min_value=0.01, max_value=5))

    approach_velocity = draw(st.floats(min_value=0.5, max_value=min(max_velocity * 0.6, 3)))
    timeout = draw(st.floats(min_value=3, max_value=20))
    window_size = draw(st.integers(min_value=5, max_value=30))
    min_snr = draw(st.floats(min_value=6, max_value=20))

    return HomingConfig(
        forwardVelocityMax=max_velocity,
        yawRateMax=yaw_rate,
        approachVelocity=approach_velocity,
        signalLossTimeout=timeout,
        gradientWindowSize=window_size,
        gradientMinSNR=min_snr,
    )


@st.composite
def system_states(
    draw, homing_possible: bool | None = None, safe_battery: bool = True
) -> SystemState:
    """Generate system state objects.

    Args:
        homing_possible: If True, generate states where homing could be enabled
        safe_battery: If True, ensure battery > 20%
    """
    sdr_status = draw(sdr_statuses)
    flight_mode = draw(flight_modes)
    gps_status = draw(gps_statuses)

    if safe_battery:
        battery = draw(st.floats(min_value=25, max_value=100))
    else:
        battery = draw(valid_percentages)

    # Determine homing enabled state
    if homing_possible is True:
        homing = draw(st.booleans())
        if homing:
            # Force conditions for valid homing
            flight_mode = "GUIDED"
            battery = max(battery, 25)
    elif homing_possible is False:
        homing = False
    else:
        homing = draw(st.booleans())

    # Generate consistent safety interlocks
    interlocks = {
        "mode_check": flight_mode == "GUIDED",
        "battery_check": battery >= 20,
        "geofence_check": draw(st.booleans()),
        "signal_check": draw(st.booleans()),
        "operator_check": draw(st.booleans()),
    }

    return SystemState(
        sdr_status=sdr_status,
        mavlink_connected=draw(st.booleans()),
        flight_mode=flight_mode,
        battery_percent=battery,
        gps_status=gps_status,
        homing_enabled=homing,
        safety_interlocks=interlocks,
        gradient_confidence=draw(valid_percentages),
        last_update=draw(recent_timestamps),
    )


@st.composite
def sdr_status_objects(draw, connected: bool | None = None) -> SDRStatus:
    """Generate SDR status objects."""
    if connected is True:
        status = "CONNECTED"
        device_name = draw(st.sampled_from(["HackRF One", "USRP B205mini", "RTL-SDR"]))
        stream_active = draw(st.booleans())
        samples_per_second = draw(st.floats(min_value=0, max_value=20e6)) if stream_active else 0
        last_error = None
    elif connected is False:
        status = draw(st.sampled_from(["DISCONNECTED", "ERROR"]))
        device_name = None
        stream_active = False
        samples_per_second = 0
        last_error = (
            draw(
                st.sampled_from(
                    [
                        "Device not found",
                        "USB error",
                        "Permission denied",
                        "Timeout",
                    ]
                )
            )
            if status == "ERROR"
            else None
        )
    else:
        status = draw(sdr_statuses)
        device_name = draw(st.one_of(st.none(), st.text(min_size=1, max_size=50)))
        stream_active = draw(st.booleans()) if status == "CONNECTED" else False
        samples_per_second = draw(st.floats(min_value=0, max_value=20e6)) if stream_active else 0
        last_error = draw(st.one_of(st.none(), st.text(min_size=1, max_size=200)))

    return SDRStatus(
        status=status,
        device_name=device_name,
        stream_active=stream_active,
        samples_per_second=samples_per_second,
        buffer_overflows=draw(st.integers(min_value=0, max_value=1000)),
        last_error=last_error,
        temperature=draw(st.one_of(st.none(), st.floats(min_value=20, max_value=80))),
    )


@st.composite
def rssi_time_series(draw, length: int | None = None, trending: str | None = None) -> list[float]:
    """Generate RSSI time series data.

    Args:
        length: Number of samples (None for random 3-100)
        trending: "up", "down", or None for random walk
    """
    if length is None:
        length = draw(st.integers(min_value=3, max_value=100))

    base_rssi = draw(st.floats(min_value=-90, max_value=-40))
    series = [base_rssi]

    for _ in range(length - 1):
        if trending == "up":
            delta = draw(st.floats(min_value=0, max_value=2))
        elif trending == "down":
            delta = draw(st.floats(min_value=-2, max_value=0))
        else:
            delta = draw(st.floats(min_value=-3, max_value=3))

        next_val = series[-1] + delta
        # Clamp to valid range
        next_val = max(-120, min(0, next_val))
        series.append(next_val)

    return series


@st.composite
def state_transitions(draw, valid_only: bool = True) -> tuple[str, str]:
    """Generate state transition pairs.

    Returns:
        Tuple of (from_state, to_state)
    """
    valid_transitions = {
        "IDLE": ["SEARCHING", "IDLE"],
        "SEARCHING": ["DETECTING", "IDLE", "SEARCHING"],
        "DETECTING": ["HOMING", "SEARCHING", "IDLE"],
        "HOMING": ["HOLDING", "SEARCHING", "IDLE"],
        "HOLDING": ["HOMING", "SEARCHING", "IDLE"],
    }

    from_state = draw(system_states)

    if valid_only and from_state in valid_transitions:
        to_state = draw(st.sampled_from(valid_transitions[from_state]))
    else:
        to_state = draw(system_states)

    return (from_state, to_state)


# Composite strategies for complex scenarios
@st.composite
def detection_scenarios(draw):
    """Generate complete detection scenario data."""
    return {
        "rssi_history": draw(rssi_time_series(length=20)),
        "noise_floor": draw(st.floats(min_value=-100, max_value=-70)),
        "signal_config": draw(signal_configs()),
        "system_state": draw(system_states(homing_possible=True)),
    }


@st.composite
def homing_scenarios(draw):
    """Generate complete homing scenario data."""
    return {
        "rssi_history": draw(rssi_time_series(length=30, trending="up")),
        "homing_config": draw(homing_configs(safe_only=True)),
        "system_state": draw(system_states(homing_possible=True, safe_battery=True)),
        "gradient_samples": draw(
            st.lists(
                st.floats(min_value=-80, max_value=-40),
                min_size=10,
                max_size=20,
            )
        ),
    }
