// PISAD Test Beacon Transmitter Firmware
// Hardware: ESP32 with LoRa module (SX1278)
// Frequency: 433 MHz ISM band
// Power: Configurable 5-20 dBm

#include <SPI.h>
#include <LoRa.h>

// Pin definitions for ESP32 with LoRa module
#define SCK 5
#define MISO 19
#define MOSI 27
#define SS 18
#define RST 14
#define DIO0 26

// Configuration parameters
#define FREQUENCY 433E6  // 433 MHz
#define BANDWIDTH 125E3  // 125 kHz bandwidth
#define SPREADING_FACTOR 7  // SF7 for balance of range/data rate
#define CODING_RATE 5  // 4/5 coding rate

// Beacon parameters
struct BeaconConfig {
  uint8_t power_dbm;  // Transmit power in dBm (5-20)
  uint16_t pulse_rate_ms;  // Pulse rate in milliseconds
  uint16_t pulse_duration_ms;  // Pulse duration in milliseconds
  bool enabled;  // Beacon enable/disable
};

BeaconConfig config = {
  .power_dbm = 10,  // Default 10 dBm
  .pulse_rate_ms = 1000,  // Default 1 Hz
  .pulse_duration_ms = 100,  // Default 100ms pulse
  .enabled = true
};

// Serial command buffer
String commandBuffer = "";

void setup() {
  Serial.begin(115200);
  while (!Serial);
  Serial.println("PISAD Test Beacon Transmitter v1.0");
  
  // Initialize LoRa module
  LoRa.setPins(SS, RST, DIO0);
  
  if (!LoRa.begin(FREQUENCY)) {
    Serial.println("ERROR: LoRa initialization failed!");
    while (1);
  }
  
  // Configure LoRa parameters
  LoRa.setSpreadingFactor(SPREADING_FACTOR);
  LoRa.setSignalBandwidth(BANDWIDTH);
  LoRa.setCodingRate4(CODING_RATE);
  LoRa.setTxPower(config.power_dbm);
  
  Serial.println("LoRa initialized successfully");
  printConfig();
}

void loop() {
  // Check for serial commands
  if (Serial.available()) {
    handleSerialCommand();
  }
  
  // Transmit beacon pulse if enabled
  if (config.enabled) {
    transmitBeacon();
    delay(config.pulse_rate_ms);
  } else {
    delay(100);  // Check for commands more frequently when disabled
  }
}

void transmitBeacon() {
  unsigned long startTime = millis();
  
  // Begin packet transmission
  LoRa.beginPacket();
  
  // Beacon packet structure
  LoRa.write(0xBE);  // Beacon identifier byte 1
  LoRa.write(0xAC);  // Beacon identifier byte 2
  LoRa.write(config.power_dbm);  // Current power setting
  
  // Add timestamp (lower 4 bytes of millis)
  unsigned long timestamp = millis();
  LoRa.write((timestamp >> 24) & 0xFF);
  LoRa.write((timestamp >> 16) & 0xFF);
  LoRa.write((timestamp >> 8) & 0xFF);
  LoRa.write(timestamp & 0xFF);
  
  // Keep transmitting for pulse duration
  while (millis() - startTime < config.pulse_duration_ms) {
    LoRa.write(0xFF);  // Fill bytes
  }
  
  LoRa.endPacket();
  
  // Debug output
  Serial.print(".");
}

void handleSerialCommand() {
  char c = Serial.read();
  
  if (c == '\n') {
    // Process complete command
    processCommand(commandBuffer);
    commandBuffer = "";
  } else if (c != '\r') {
    commandBuffer += c;
  }
}

void processCommand(String cmd) {
  cmd.trim();
  cmd.toUpperCase();
  
  if (cmd.startsWith("POWER:")) {
    // Set transmit power: POWER:15
    int power = cmd.substring(6).toInt();
    if (power >= 5 && power <= 20) {
      config.power_dbm = power;
      LoRa.setTxPower(config.power_dbm);
      Serial.print("Power set to: ");
      Serial.print(config.power_dbm);
      Serial.println(" dBm");
    } else {
      Serial.println("ERROR: Power must be 5-20 dBm");
    }
  }
  else if (cmd.startsWith("RATE:")) {
    // Set pulse rate: RATE:500 (in ms)
    int rate = cmd.substring(5).toInt();
    if (rate >= 100 && rate <= 10000) {
      config.pulse_rate_ms = rate;
      Serial.print("Pulse rate set to: ");
      Serial.print(config.pulse_rate_ms);
      Serial.println(" ms");
    } else {
      Serial.println("ERROR: Rate must be 100-10000 ms");
    }
  }
  else if (cmd.startsWith("DURATION:")) {
    // Set pulse duration: DURATION:200 (in ms)
    int duration = cmd.substring(9).toInt();
    if (duration >= 10 && duration <= 1000) {
      config.pulse_duration_ms = duration;
      Serial.print("Pulse duration set to: ");
      Serial.print(config.pulse_duration_ms);
      Serial.println(" ms");
    } else {
      Serial.println("ERROR: Duration must be 10-1000 ms");
    }
  }
  else if (cmd == "START") {
    config.enabled = true;
    Serial.println("Beacon started");
  }
  else if (cmd == "STOP") {
    config.enabled = false;
    Serial.println("Beacon stopped");
  }
  else if (cmd == "STATUS") {
    printConfig();
  }
  else if (cmd == "HELP") {
    printHelp();
  }
  else {
    Serial.println("ERROR: Unknown command. Type HELP for commands.");
  }
}

void printConfig() {
  Serial.println("\n=== Beacon Configuration ===");
  Serial.print("Status: ");
  Serial.println(config.enabled ? "TRANSMITTING" : "STOPPED");
  Serial.print("Frequency: ");
  Serial.print(FREQUENCY / 1E6);
  Serial.println(" MHz");
  Serial.print("Power: ");
  Serial.print(config.power_dbm);
  Serial.println(" dBm");
  Serial.print("Pulse Rate: ");
  Serial.print(config.pulse_rate_ms);
  Serial.println(" ms");
  Serial.print("Pulse Duration: ");
  Serial.print(config.pulse_duration_ms);
  Serial.println(" ms");
  Serial.println("===========================\n");
}

void printHelp() {
  Serial.println("\n=== Available Commands ===");
  Serial.println("POWER:<dBm>     - Set transmit power (5-20 dBm)");
  Serial.println("RATE:<ms>       - Set pulse rate (100-10000 ms)");
  Serial.println("DURATION:<ms>   - Set pulse duration (10-1000 ms)");
  Serial.println("START           - Start beacon transmission");
  Serial.println("STOP            - Stop beacon transmission");
  Serial.println("STATUS          - Show current configuration");
  Serial.println("HELP            - Show this help message");
  Serial.println("==========================\n");
}