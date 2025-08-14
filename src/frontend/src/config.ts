// API Configuration
export const API_BASE_URL =
  process.env.REACT_APP_API_URL || "http://localhost:8000/api";
export const WS_BASE_URL =
  process.env.REACT_APP_WS_URL || "ws://localhost:8000";

// Application Settings
export const APP_NAME = "PiSAD";
export const APP_VERSION = "1.0.0";

// Update intervals (ms)
export const UPDATE_INTERVALS = {
  TELEMETRY: 100,
  SYSTEM_STATUS: 1000,
  SIGNAL_DATA: 50,
  FIELD_TEST: 1000,
};

// Map settings
export const MAP_CONFIG = {
  DEFAULT_CENTER: [37.7749, -122.4194], // San Francisco
  DEFAULT_ZOOM: 13,
  TILE_URL: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
};

// Field test settings
export const FIELD_TEST_CONFIG = {
  MAX_ITERATIONS: 100,
  DEFAULT_BEACON_POWER: 10, // dBm
  DEFAULT_FREQUENCY: 433000000, // Hz
  TIMEOUT_MS: 300000, // 5 minutes
};
