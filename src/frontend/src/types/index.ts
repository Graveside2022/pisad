/**
 * Shared TypeScript type definitions
 */

export interface SystemState {
  current_state: "IDLE" | "SEARCHING" | "DETECTING" | "HOMING" | "HOLDING";
  homing_enabled: boolean;
  flight_mode: string;
  battery_percent: number;
  gps_status: "NO_FIX" | "2D_FIX" | "3D_FIX" | "RTK";
  mavlink_connected: boolean;
  sdr_status: "CONNECTED" | "DISCONNECTED" | "ERROR";
  safety_interlocks: Record<string, boolean>;
  cpu_usage?: number;
  memory_usage?: number;
}

export interface RSSIReading {
  timestamp: string;
  rssi: number;
  noise_floor: number;
  snr: number;
  confidence: number;
  detection_id?: string;
}

export interface SignalDetection {
  id: string;
  timestamp: string;
  frequency: number;
  rssi: number;
  snr: number;
  confidence: number;
  location?: {
    latitude: number;
    longitude: number;
    altitude: number;
  };
  state: string;
}

export interface SDRConfig {
  frequency: number;
  sample_rate: number;
  gain: number;
  ppm_correction: number;
  device_index: number;
}

export interface SystemHealth {
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  temperature?: number;
  uptime: number;
  sdr_status: "CONNECTED" | "DISCONNECTED" | "ERROR";
}
