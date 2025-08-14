# Field Test Beacon Setup Guide

## Overview

This guide provides instructions for setting up and configuring test beacons for PISAD field testing. The beacon transmitter simulates lost persons or objects by transmitting RF signals that the PISAD system can detect and home in on.

## Hardware Requirements

### Recommended Beacon Hardware

1. **ESP32 with LoRa Module**
   - ESP32 development board
   - SX1276/SX1278 LoRa module (433 MHz)
   - External antenna (433 MHz, 2-3 dBi gain)
   - Power supply (battery pack or USB power bank)

2. **Arduino with LoRa Shield**
   - Arduino Uno/Mega
   - Dragino LoRa Shield (433 MHz)
   - External antenna
   - 9V battery or power bank

3. **Standalone LoRa Module**
   - HopeRF RFM95W module
   - 433 MHz antenna
   - 3.3V power supply
   - USB-to-serial adapter for configuration

### Power Requirements

- Voltage: 3.3V - 5V (depending on module)
- Current: 100-150mA during transmission
- Battery capacity: Minimum 2000mAh for 4+ hours operation
- Recommended: USB power bank with 5000mAh+ capacity

## Beacon Firmware

### ESP32 LoRa Beacon Code

```cpp
#include <LoRa.h>

// Pin definitions for ESP32
#define SCK 5
#define MISO 19
#define MOSI 27
#define SS 18
#define RST 14
#define DIO0 26

// Beacon configuration
const long FREQUENCY = 433E6;  // 433 MHz
const int TX_POWER = 10;       // 10 dBm
const int SPREADING_FACTOR = 7;
const long BANDWIDTH = 125000; // 125 kHz
const int CODING_RATE = 5;     // 4/5

// Pulse configuration
const int PULSE_DURATION_MS = 100;
const int PULSE_INTERVAL_MS = 1000;

void setup() {
  Serial.begin(115200);

  // Initialize LoRa
  LoRa.setPins(SS, RST, DIO0);

  if (!LoRa.begin(FREQUENCY)) {
    Serial.println("LoRa init failed!");
    while (1);
  }

  // Configure LoRa parameters
  LoRa.setTxPower(TX_POWER);
  LoRa.setSpreadingFactor(SPREADING_FACTOR);
  LoRa.setSignalBandwidth(BANDWIDTH);
  LoRa.setCodingRate4(CODING_RATE);

  Serial.println("Beacon initialized");
}

void loop() {
  // Send beacon pulse
  LoRa.beginPacket();
  LoRa.print("BEACON_TEST_");
  LoRa.print(millis());
  LoRa.endPacket();

  Serial.println("Pulse sent");

  // Wait for next pulse
  delay(PULSE_INTERVAL_MS);
}
```

### Configuration Profiles

The beacon can be configured with different profiles for various test scenarios. See `config/profiles/field_test_beacon.yaml` for predefined profiles:

- **Low Power (5 dBm)**: Indoor/short range testing (100m)
- **Medium Power (10 dBm)**: Standard field testing (300m)
- **High Power (15 dBm)**: Long range testing (500m)
- **Maximum Power (20 dBm)**: Extended range testing (750m)

## Field Setup Procedures

### 1. Pre-Deployment Checklist

- [ ] Beacon hardware assembled and tested
- [ ] Battery fully charged (>90%)
- [ ] Antenna securely connected
- [ ] Firmware uploaded and verified
- [ ] Power output configured for test scenario
- [ ] Weatherproof enclosure (if outdoor testing)
- [ ] GPS coordinates of beacon placement recorded

### 2. Beacon Placement

#### Open Field Testing

1. Place beacon at predetermined GPS coordinates
2. Mount antenna vertically for omnidirectional pattern
3. Elevate beacon 1-2m above ground level
4. Clear line of sight in all directions
5. Mark location with visible flag/marker

#### Urban Environment Testing

1. Consider multipath reflections
2. Avoid metal structures within 5m
3. Document nearby obstacles
4. Test multiple beacon heights
5. Record environmental factors

### 3. Power Level Configuration

| Test Type   | Power Setting | Expected Range | Use Case           |
| ----------- | ------------- | -------------- | ------------------ |
| Close Range | 5 dBm         | 50-100m        | Initial validation |
| Standard    | 10 dBm        | 200-300m       | Typical scenario   |
| Extended    | 15 dBm        | 400-500m       | Long range test    |
| Maximum     | 20 dBm        | 600-750m       | Limit testing      |

### 4. Beacon Activation

1. Power on beacon hardware
2. Verify LED indicators (if present)
3. Check serial output for confirmation
4. Use SDR or spectrum analyzer to verify transmission
5. Record start time and initial RSSI reading

## Verification and Testing

### Using RTL-SDR to Verify Beacon

```bash
# Install rtl-sdr tools
sudo apt-get install rtl-sdr gqrx-sdr

# Monitor 433 MHz band
rtl_power -f 433M:434M:1k -g 50 -i 1 beacon_scan.csv

# View waterfall in GQRX
gqrx
# Set frequency to 433.000 MHz
# Adjust gain and bandwidth
# Look for beacon pulses
```

### Expected Signal Characteristics

- Center frequency: 433.000 MHz ± 10 kHz
- Bandwidth: 125 kHz (LoRa SF7)
- Pulse duration: 100-200 ms
- Pulse interval: 1 second
- Signal strength at 10m: -40 to -50 dBm

## Safety Considerations

### RF Exposure

- Maintain 1m minimum distance during transmission
- Use lowest power necessary for testing
- Limit continuous exposure time
- Follow local RF exposure guidelines

### Regulatory Compliance

- 433 MHz ISM band regulations
- Maximum power: 20 dBm (100mW) in most regions
- Duty cycle restrictions may apply
- Check local regulations before testing

## Troubleshooting

### No Signal Detected

1. Verify power supply voltage
2. Check antenna connection
3. Confirm frequency configuration
4. Test with SDR at close range
5. Check for interference sources

### Weak or Intermittent Signal

1. Replace/recharge battery
2. Check antenna SWR
3. Increase transmission power
4. Reduce spreading factor
5. Clear obstructions

### Range Less Than Expected

1. Verify antenna orientation
2. Check for environmental factors
3. Measure actual output power
4. Test in open field first
5. Document propagation conditions

## Test Data Recording

### Required Measurements

- Beacon GPS coordinates
- Transmission power (dBm)
- Environmental conditions
  - Temperature
  - Humidity
  - Wind speed
  - Precipitation
- Time of test
- Interference sources noted

### Performance Metrics

- Maximum detection range
- RSSI at various distances
- Signal stability over time
- Battery life at power level
- Environmental impact on range

## Maintenance

### Between Tests

1. Recharge/replace batteries
2. Inspect antenna connections
3. Check enclosure seals
4. Download test logs
5. Update firmware if needed

### Storage

1. Remove batteries
2. Store in dry location
3. Protect antenna from damage
4. Document last configuration
5. Schedule periodic testing

## Advanced Configurations

### Multiple Beacon Testing

- Use different spreading factors
- Offset frequencies slightly (±25 kHz)
- Implement time-division multiplexing
- Unique beacon IDs in packets

### Mobile Beacon Testing

- Mount on ground vehicle
- Use magnetic antenna mount
- Log GPS coordinates continuously
- Vary speed for Doppler testing

## References

- LoRa Alliance Technical Specifications
- Local RF regulations (FCC Part 15, CE RED)
- Antenna theory and propagation models
- Link budget calculations for 433 MHz
