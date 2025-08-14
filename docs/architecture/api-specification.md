# API Specification

## REST API Specification

```yaml
openapi: 3.0.0
info:
  title: RF-Homing SAR Drone API
  version: 1.0.0
  description: REST API for RF homing payload control and monitoring
servers:
  - url: http://localhost:8080/api
    description: Local Raspberry Pi server

paths:
  /system/status:
    get:
      summary: Get current system status
      responses:
        200:
          description: System status
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/SystemState"

  /system/homing:
    post:
      summary: Enable/disable homing mode
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                enabled:
                  type: boolean
                confirmationToken:
                  type: string
      responses:
        200:
          description: Homing state changed
        403:
          description: Safety interlock blocked activation

  /config/profiles:
    get:
      summary: List configuration profiles
      responses:
        200:
          description: List of profiles
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: "#/components/schemas/ConfigProfile"

    post:
      summary: Create new profile
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/ConfigProfile"
      responses:
        201:
          description: Profile created

  /config/profiles/{id}:
    put:
      summary: Update profile
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: "#/components/schemas/ConfigProfile"
      responses:
        200:
          description: Profile updated

  /config/profiles/{id}/activate:
    post:
      summary: Activate a profile
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
      responses:
        200:
          description: Profile activated

  /detections:
    get:
      summary: Get detection history
      parameters:
        - name: limit
          in: query
          schema:
            type: integer
            default: 100
        - name: since
          in: query
          schema:
            type: string
            format: date-time
      responses:
        200:
          description: Detection events
          content:
            application/json:
              schema:
                type: array
                items:
                  $ref: "#/components/schemas/SignalDetection"

  /missions:
    get:
      summary: List missions
      responses:
        200:
          description: List of missions

    post:
      summary: Start new mission
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                name:
                  type: string
                profileId:
                  type: string
      responses:
        201:
          description: Mission started

  /missions/{id}/end:
    post:
      summary: End active mission
      parameters:
        - name: id
          in: path
          required: true
          schema:
            type: string
      responses:
        200:
          description: Mission ended

  /search/pattern:
    post:
      summary: Configure search pattern
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                pattern:
                  type: string
                  enum: [expanding_square, spiral, lawnmower]
                spacing:
                  type: number
                velocity:
                  type: number
                bounds:
                  type: object
      responses:
        200:
          description: Pattern configured

components:
  schemas:
    SystemState:
      # As defined in data models
    ConfigProfile:
      # As defined in data models
    SignalDetection:
      # As defined in data models
```

## WebSocket Events

```typescript
// WebSocket message types for real-time communication
interface WSMessage {
  type: "rssi" | "state" | "detection" | "telemetry" | "error";
  timestamp: string;
  data: any;
}

// RSSI update (10Hz)
interface RSSIUpdate {
  type: "rssi";
  data: {
    rssi: number;
    noiseFloor: number;
    snr: number;
    confidence: number;
  };
}

// State change notification
interface StateUpdate {
  type: "state";
  data: SystemState;
}

// Detection event
interface DetectionEvent {
  type: "detection";
  data: SignalDetection;
}

// Telemetry update (2Hz)
interface TelemetryUpdate {
  type: "telemetry";
  data: {
    position: { lat: number; lon: number; alt: number };
    battery: number;
    flightMode: string;
    velocity: { forward: number; yaw: number };
  };
}
```
