# Database Schema

```sql
-- Configuration profiles for different beacon types
CREATE TABLE config_profiles (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    name TEXT NOT NULL UNIQUE,
    sdr_config JSON NOT NULL,
    signal_config JSON NOT NULL,
    homing_config JSON NOT NULL,
    is_default BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Mission tracking
CREATE TABLE missions (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    name TEXT NOT NULL,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP NULL,
    search_area JSON NULL,
    profile_id TEXT REFERENCES config_profiles(id),
    total_detections INTEGER DEFAULT 0,
    notes TEXT NULL
);

-- Signal detection events
CREATE TABLE signal_detections (
    id TEXT PRIMARY KEY DEFAULT (lower(hex(randomblob(16)))),
    mission_id TEXT REFERENCES missions(id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    frequency REAL NOT NULL,
    rssi REAL NOT NULL,
    snr REAL NOT NULL,
    confidence REAL NOT NULL,
    location JSON NULL,
    state TEXT NOT NULL CHECK(state IN ('IDLE','SEARCHING','DETECTING','HOMING','HOLDING'))
);

-- Time-series RSSI data (optional persistence for analysis)
CREATE TABLE rssi_readings (
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    rssi REAL NOT NULL,
    noise_floor REAL NOT NULL,
    detection_id TEXT NULL REFERENCES signal_detections(id)
);

-- Indexes for performance
CREATE INDEX idx_detections_mission ON signal_detections(mission_id);
CREATE INDEX idx_detections_timestamp ON signal_detections(timestamp);
CREATE INDEX idx_rssi_detection ON rssi_readings(detection_id);
CREATE INDEX idx_rssi_timestamp ON rssi_readings(timestamp);

-- Triggers for updated_at
CREATE TRIGGER update_profile_timestamp
AFTER UPDATE ON config_profiles
BEGIN
    UPDATE config_profiles SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;
```
