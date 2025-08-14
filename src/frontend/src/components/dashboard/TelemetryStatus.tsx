import React, { useState, useEffect } from "react";
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  Chip,
  LinearProgress,
  IconButton,
  Slider,
  Stack,
  Alert,
} from "@mui/material";
import {
  Settings as SettingsIcon,
  Refresh as RefreshIcon,
  SignalCellularAlt as SignalIcon,
  Send as SendIcon,
} from "@mui/icons-material";

interface TelemetryConfig {
  rssi_rate_hz: number;
  health_interval_seconds: number;
  detection_throttle_ms: number;
}

interface TelemetryStatusData {
  rssi_rate_hz: number;
  health_interval_seconds: number;
  last_rssi_sent: string | null;
  last_health_sent: string | null;
  last_state_sent: string | null;
  messages_sent_count: number;
  bandwidth_usage_bps: number;
  gcs_connected: boolean;
}

const TelemetryStatus: React.FC = () => {
  const [telemetryStatus, setTelemetryStatus] = useState<TelemetryStatusData>({
    rssi_rate_hz: 2.0,
    health_interval_seconds: 10,
    last_rssi_sent: null,
    last_health_sent: null,
    last_state_sent: null,
    messages_sent_count: 0,
    bandwidth_usage_bps: 0,
    gcs_connected: false,
  });

  const [telemetryConfig, setTelemetryConfig] = useState<TelemetryConfig>({
    rssi_rate_hz: 2.0,
    health_interval_seconds: 10,
    detection_throttle_ms: 500,
  });

  const [showSettings, setShowSettings] = useState(false);
  const [isUpdating, setIsUpdating] = useState(false);

  // Fetch telemetry configuration
  const fetchTelemetryConfig = async () => {
    try {
      const response = await fetch("/api/telemetry/config");
      if (response.ok) {
        const data = await response.json();
        setTelemetryConfig(data);
      }
    } catch (error) {
      console.error("Failed to fetch telemetry config:", error);
    }
  };

  // Update telemetry configuration
  const updateTelemetryConfig = async () => {
    setIsUpdating(true);
    try {
      const response = await fetch("/api/telemetry/config", {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(telemetryConfig),
      });

      if (response.ok) {
        const data = await response.json();
        setTelemetryConfig(data.config);
      }
    } catch (error) {
      console.error("Failed to update telemetry config:", error);
    } finally {
      setIsUpdating(false);
    }
  };

  // WebSocket connection for real-time updates
  useEffect(() => {
    fetchTelemetryConfig();

    const ws = new WebSocket(`ws://${window.location.host}/ws`);

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "telemetry_status") {
          setTelemetryStatus(data.data);
        }
      } catch (error) {
        console.error("Failed to parse WebSocket message:", error);
      }
    };

    return () => {
      ws.close();
    };
  }, []);

  const formatTimestamp = (timestamp: string | null) => {
    if (!timestamp) return "Never";
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffSec = Math.floor(diffMs / 1000);

    if (diffSec < 60) return `${diffSec}s ago`;
    if (diffSec < 3600) return `${Math.floor(diffSec / 60)}m ago`;
    return `${Math.floor(diffSec / 3600)}h ago`;
  };

  const formatBandwidth = (bps: number) => {
    if (bps < 1000) return `${bps} bps`;
    if (bps < 1000000) return `${(bps / 1000).toFixed(1)} kbps`;
    return `${(bps / 1000000).toFixed(2)} Mbps`;
  };

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
          <SignalIcon sx={{ mr: 1 }} />
          <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
            Telemetry Status
          </Typography>
          <Chip
            label={
              telemetryStatus.gcs_connected
                ? "GCS Connected"
                : "GCS Disconnected"
            }
            color={telemetryStatus.gcs_connected ? "success" : "default"}
            size="small"
            sx={{ mr: 1 }}
          />
          <IconButton
            onClick={() => setShowSettings(!showSettings)}
            size="small"
          >
            <SettingsIcon />
          </IconButton>
          <IconButton onClick={fetchTelemetryConfig} size="small">
            <RefreshIcon />
          </IconButton>
        </Box>

        {showSettings && (
          <Box
            sx={{ mb: 3, p: 2, bgcolor: "background.default", borderRadius: 1 }}
          >
            <Typography variant="subtitle2" gutterBottom>
              Telemetry Configuration
            </Typography>
            <Stack spacing={2}>
              <Box>
                <Typography variant="body2" gutterBottom>
                  RSSI Rate: {telemetryConfig.rssi_rate_hz.toFixed(1)} Hz
                </Typography>
                <Slider
                  value={telemetryConfig.rssi_rate_hz}
                  onChange={(_, value) =>
                    setTelemetryConfig({
                      ...telemetryConfig,
                      rssi_rate_hz: value as number,
                    })
                  }
                  min={0.1}
                  max={10}
                  step={0.1}
                  marks={[
                    { value: 0.1, label: "0.1" },
                    { value: 2, label: "2" },
                    { value: 5, label: "5" },
                    { value: 10, label: "10" },
                  ]}
                  sx={{ mt: 1 }}
                />
              </Box>

              <Box>
                <Typography variant="body2" gutterBottom>
                  Health Interval: {telemetryConfig.health_interval_seconds}s
                </Typography>
                <Slider
                  value={telemetryConfig.health_interval_seconds}
                  onChange={(_, value) =>
                    setTelemetryConfig({
                      ...telemetryConfig,
                      health_interval_seconds: value as number,
                    })
                  }
                  min={1}
                  max={60}
                  step={1}
                  marks={[
                    { value: 1, label: "1s" },
                    { value: 10, label: "10s" },
                    { value: 30, label: "30s" },
                    { value: 60, label: "60s" },
                  ]}
                  sx={{ mt: 1 }}
                />
              </Box>

              <Box>
                <Typography variant="body2" gutterBottom>
                  Detection Throttle: {telemetryConfig.detection_throttle_ms}ms
                </Typography>
                <Slider
                  value={telemetryConfig.detection_throttle_ms}
                  onChange={(_, value) =>
                    setTelemetryConfig({
                      ...telemetryConfig,
                      detection_throttle_ms: value as number,
                    })
                  }
                  min={100}
                  max={5000}
                  step={100}
                  marks={[
                    { value: 100, label: "100ms" },
                    { value: 500, label: "500ms" },
                    { value: 2000, label: "2s" },
                    { value: 5000, label: "5s" },
                  ]}
                  sx={{ mt: 1 }}
                />
              </Box>

              <Box sx={{ display: "flex", justifyContent: "flex-end" }}>
                <IconButton
                  onClick={updateTelemetryConfig}
                  disabled={isUpdating}
                  color="primary"
                >
                  <SendIcon />
                </IconButton>
              </Box>
            </Stack>
            {isUpdating && <LinearProgress sx={{ mt: 1 }} />}
          </Box>
        )}

        <Grid container spacing={2}>
          <Grid size={6}>
            <Typography variant="body2" color="text.secondary">
              Current Rates
            </Typography>
            <Typography variant="body1">
              RSSI: {telemetryStatus.rssi_rate_hz.toFixed(1)} Hz
            </Typography>
            <Typography variant="body1">
              Health: every {telemetryStatus.health_interval_seconds}s
            </Typography>
          </Grid>

          <Grid size={6}>
            <Typography variant="body2" color="text.secondary">
              Bandwidth Usage
            </Typography>
            <Typography variant="body1">
              {formatBandwidth(telemetryStatus.bandwidth_usage_bps)}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {telemetryStatus.messages_sent_count} messages sent
            </Typography>
          </Grid>

          <Grid size={12}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Last Messages Sent
            </Typography>
            <Stack spacing={1}>
              <Box sx={{ display: "flex", justifyContent: "space-between" }}>
                <Typography variant="body2">RSSI:</Typography>
                <Typography variant="body2" color="text.secondary">
                  {formatTimestamp(telemetryStatus.last_rssi_sent)}
                </Typography>
              </Box>
              <Box sx={{ display: "flex", justifyContent: "space-between" }}>
                <Typography variant="body2">Health:</Typography>
                <Typography variant="body2" color="text.secondary">
                  {formatTimestamp(telemetryStatus.last_health_sent)}
                </Typography>
              </Box>
              <Box sx={{ display: "flex", justifyContent: "space-between" }}>
                <Typography variant="body2">State:</Typography>
                <Typography variant="body2" color="text.secondary">
                  {formatTimestamp(telemetryStatus.last_state_sent)}
                </Typography>
              </Box>
            </Stack>
          </Grid>
        </Grid>

        {telemetryStatus.bandwidth_usage_bps > 10000 && (
          <Alert severity="warning" sx={{ mt: 2 }}>
            High bandwidth usage detected. Consider reducing telemetry rates.
          </Alert>
        )}
      </CardContent>
    </Card>
  );
};

export default TelemetryStatus;
