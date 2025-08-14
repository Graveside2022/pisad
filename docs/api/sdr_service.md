# SDR Service API Documentation

## Overview

The SDR Service provides a high-level interface for Software Defined Radio (SDR) hardware control and IQ data streaming. It supports multiple SDR devices through the SoapySDR abstraction layer and includes features for automatic device detection, configuration management, health monitoring, and calibration.

## Service Class

### `SDRService`

Main service class for SDR operations.

```python
from src.backend.services.sdr_service import SDRService
from src.backend.models.schemas import SDRConfig

# Initialize service
service = SDRService()

# Use as async context manager
async with service as sdr:
    # SDR operations
    pass
```

## Core Methods

### `initialize(config: SDRConfig | None = None) -> None`

Initialize the SDR hardware with the specified configuration.

**Parameters:**
- `config` (SDRConfig, optional): SDR configuration. If not provided, uses default configuration.

**Raises:**
- `SDRNotFoundError`: No SDR devices found
- `SDRConfigError`: Invalid configuration parameters

**Example:**
```python
config = SDRConfig(
    frequency=433.92e6,  # 433.92 MHz
    sampleRate=2.4e6,    # 2.4 Msps
    gain="AUTO",         # Automatic gain control
    bandwidth=2e6,       # 2 MHz bandwidth
    buffer_size=8192     # Buffer size
)

await service.initialize(config)
```

### `stream_iq() -> AsyncIterator[np.ndarray]`

Stream IQ samples from the SDR device.

**Returns:**
- AsyncIterator yielding numpy arrays of complex64 samples

**Raises:**
- `SDRStreamError`: Stream error or device disconnection

**Example:**
```python
async for samples in service.stream_iq():
    # Process IQ samples
    power = np.abs(samples) ** 2
    rssi = 10 * np.log10(np.mean(power))
```

### `calibrate() -> dict[str, Any]`

Perform comprehensive SDR calibration routine.

**Returns:**
- Dictionary containing calibration results:
  - `frequency_accuracy`: Frequency error measurements and PPM correction
  - `noise_floor`: Noise floor measurements
  - `gain_optimization`: Optimal gain settings
  - `sample_rate_stability`: Sample rate stability metrics
  - `recommendations`: List of recommended adjustments

**Example:**
```python
results = await service.calibrate()

if results["status"] == "complete":
    print(f"Noise floor: {results['noise_floor']['noise_floor_dbm']} dBm")
    print(f"Recommended PPM: {results['frequency_accuracy']['recommended_ppm_correction']}")
    
    for recommendation in results["recommendations"]:
        print(f"- {recommendation}")
```

### `set_frequency(frequency: float) -> None`

Set the center frequency.

**Parameters:**
- `frequency` (float): Center frequency in Hz

**Raises:**
- `SDRConfigError`: Frequency out of supported range

**Example:**
```python
service.set_frequency(868e6)  # Set to 868 MHz
```

### `get_status() -> SDRStatus`

Get current SDR status and metrics.

**Returns:**
- `SDRStatus` object containing:
  - `status`: Connection status (CONNECTED, DISCONNECTED, ERROR, etc.)
  - `device_name`: Device label
  - `driver`: Driver name (hackrf, rtlsdr, etc.)
  - `stream_active`: Whether streaming is active
  - `samples_per_second`: Current sample rate
  - `buffer_overflows`: Count of buffer overflows
  - `temperature`: Device temperature (if available)
  - `last_error`: Last error message

**Example:**
```python
status = service.get_status()
print(f"Device: {status.device_name}")
print(f"Status: {status.status}")
print(f"Temperature: {status.temperature}°C")
```

### `update_config(config: SDRConfig) -> None`

Update SDR configuration at runtime.

**Parameters:**
- `config` (SDRConfig): New configuration

**Example:**
```python
new_config = SDRConfig(
    frequency=915e6,
    sampleRate=3.2e6,
    gain=30
)
await service.update_config(new_config)
```

### `shutdown() -> None`

Shutdown the SDR service and cleanup resources.

**Example:**
```python
await service.shutdown()
```

## Static Methods

### `enumerate_devices() -> list[dict[str, Any]]`

Enumerate available SDR devices.

**Returns:**
- List of device dictionaries containing driver and label information

**Example:**
```python
devices = SDRService.enumerate_devices()
for device in devices:
    print(f"{device['driver']}: {device['label']}")
```

## Configuration

### `SDRConfig`

Configuration model for SDR parameters.

```python
from src.backend.models.schemas import SDRConfig

config = SDRConfig(
    frequency=433.92e6,      # Center frequency (Hz)
    sampleRate=2.4e6,        # Sample rate (samples/sec)
    gain="AUTO",             # Gain setting (dB or "AUTO")
    bandwidth=2e6,           # Filter bandwidth (Hz)
    buffer_size=8192,        # Buffer size (samples)
    device_args="",          # Device selection arguments
    antenna="RX",            # Antenna selection
    ppm_correction=0,        # Frequency correction (PPM)
)
```

**Fields:**
- `frequency` (float): Center frequency in Hz (default: 2.437 GHz)
- `sampleRate` (float): Sample rate in samples/sec (default: 2 Msps)
- `gain` (float | str): Gain in dB or "AUTO" for AGC (default: "AUTO")
- `bandwidth` (float): Filter bandwidth in Hz (default: 2 MHz)
- `buffer_size` (int): Buffer size in samples (default: 8192)
- `device_args` (str): Device selection arguments (default: "")
- `antenna` (str): Antenna selection (default: "RX")
- `ppm_correction` (float): Frequency correction in PPM (default: 0)

## Status Model

### `SDRStatus`

Status model for SDR state and metrics.

```python
from src.backend.models.schemas import SDRStatus

status = SDRStatus(
    status="CONNECTED",
    device_name="HackRF One",
    driver="hackrf",
    stream_active=True,
    samples_per_second=2.4e6,
    buffer_overflows=0,
    temperature=42.5,
    last_error=None
)
```

**Fields:**
- `status` (str): Connection status
- `device_name` (str): Device label
- `driver` (str): Driver name
- `stream_active` (bool): Streaming state
- `samples_per_second` (float): Current sample rate
- `buffer_overflows` (int): Overflow count
- `temperature` (float | None): Device temperature
- `last_error` (str | None): Last error message

## Exceptions

### `SDRNotFoundError`

Raised when no SDR devices are found.

```python
from src.backend.services.sdr_service import SDRNotFoundError

try:
    await service.initialize()
except SDRNotFoundError:
    print("No SDR devices found")
```

### `SDRConfigError`

Raised when configuration is invalid.

```python
from src.backend.services.sdr_service import SDRConfigError

try:
    service.set_frequency(10e9)  # 10 GHz
except SDRConfigError as e:
    print(f"Configuration error: {e}")
```

### `SDRStreamError`

Raised when streaming encounters an error.

```python
from src.backend.services.sdr_service import SDRStreamError

try:
    async for samples in service.stream_iq():
        process(samples)
except SDRStreamError as e:
    print(f"Stream error: {e}")
```

## Health Monitoring

The SDR service includes automatic health monitoring that:

1. **Periodic Health Checks**: Checks device connectivity every 5 seconds
2. **Automatic Reconnection**: Attempts to reconnect if device disconnects
3. **Temperature Monitoring**: Reads device temperature sensors
4. **Overflow Tracking**: Counts buffer overflows for performance monitoring

## Calibration Routine

The calibration routine performs four key measurements:

1. **Frequency Accuracy**: Tests multiple frequencies to determine PPM correction
2. **Noise Floor**: Measures noise floor at minimum gain
3. **Gain Optimization**: Tests gain levels to find optimal dynamic range
4. **Sample Rate Stability**: Verifies actual sample rate matches configuration

## Usage Examples

### Basic Usage

```python
import asyncio
from src.backend.services.sdr_service import SDRService
from src.backend.models.schemas import SDRConfig

async def main():
    # Initialize SDR
    service = SDRService()
    config = SDRConfig(frequency=433.92e6, sampleRate=2.4e6)
    
    await service.initialize(config)
    
    # Stream samples
    sample_count = 0
    async for samples in service.stream_iq():
        # Process samples
        print(f"Received {len(samples)} samples")
        
        sample_count += len(samples)
        if sample_count > 1000000:  # Stop after 1M samples
            break
    
    await service.shutdown()

asyncio.run(main())
```

### With Context Manager

```python
async def process_sdr():
    async with SDRService() as sdr:
        # Calibrate device
        calibration = await sdr.calibrate()
        print(f"Calibration: {calibration['recommendations']}")
        
        # Stream data
        async for samples in sdr.stream_iq():
            # Process samples
            pass
```

### Device Selection

```python
# List available devices
devices = SDRService.enumerate_devices()
for i, device in enumerate(devices):
    print(f"{i}: {device['driver']} - {device['label']}")

# Select specific device
config = SDRConfig(device_args="driver=hackrf")
await service.initialize(config)
```

### Error Handling

```python
async def robust_sdr_operation():
    service = SDRService()
    
    try:
        await service.initialize()
    except SDRNotFoundError:
        print("No SDR devices found. Please connect an SDR.")
        return
    except SDRConfigError as e:
        print(f"Configuration error: {e}")
        return
    
    try:
        async for samples in service.stream_iq():
            # Process samples
            pass
    except SDRStreamError as e:
        print(f"Stream error: {e}")
        # Service will attempt automatic reconnection
    finally:
        await service.shutdown()
```

## Integration with Health Check API

The SDR service integrates with the health check endpoint at `/health/sdr`:

```python
# GET /health/sdr
{
    "health": "healthy",
    "status": "CONNECTED",
    "device_name": "HackRF One",
    "driver": "hackrf",
    "stream_active": true,
    "samples_per_second": 2400000,
    "buffer_overflows": 0,
    "temperature": 42.5,
    "config": {
        "frequency": 433920000,
        "sample_rate": 2400000,
        "bandwidth": 2000000,
        "gain": "AUTO",
        "buffer_size": 8192
    }
}
```

## Performance Considerations

1. **Buffer Size**: Larger buffers reduce CPU overhead but increase latency
2. **Sample Rate**: Higher rates provide more bandwidth but require more processing
3. **Gain Settings**: AUTO gain works well for most cases; manual gain for specific scenarios
4. **USB Connection**: USB 3.0 recommended for high sample rates (>5 Msps)

## Supported Hardware

The SDR service supports any device compatible with SoapySDR, including:

- HackRF One
- RTL-SDR (RTL2832U)
- USRP (via UHD)
- LimeSDR
- BladeRF
- Airspy
- PlutoSDR

## Thread Safety

The SDR service is designed for use in async contexts. All methods are coroutine-safe but not thread-safe. Use asyncio synchronization primitives if accessing from multiple tasks.

## Dependencies

- `SoapySDR`: SDR hardware abstraction
- `numpy`: Signal processing
- `asyncio`: Asynchronous operations

## See Also

- [MAVLink Service API](./mavlink_service.md)
- [State Machine API](./state_machine.md)
- [Signal Processor API](./signal_processor.md)