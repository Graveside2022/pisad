import { useEffect, useState } from "react";
import { Paper, Typography, LinearProgress, Alert, Grid } from "@mui/material";

interface SystemHealthData {
  cpu_usage: number;
  memory_usage: number;
  disk_usage?: number;
  temperature?: number;
  uptime?: number;
  sdr_status: "CONNECTED" | "DISCONNECTED" | "ERROR";
}

function SystemHealth() {
  const [health, setHealth] = useState<SystemHealthData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch system health from API
    const fetchHealth = async () => {
      try {
        const response = await fetch("/api/system/status");
        if (response.ok) {
          const data = await response.json();
          setHealth({
            cpu_usage: data.cpu_usage || 0,
            memory_usage: data.memory_usage || 0,
            disk_usage: data.disk_usage,
            temperature: data.temperature,
            uptime: data.uptime,
            sdr_status: data.sdr_status || "DISCONNECTED",
          });
        }
      } catch (error) {
        console.error("Failed to fetch system health:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchHealth();
    const interval = setInterval(fetchHealth, 5000); // Update every 5 seconds

    return () => clearInterval(interval);
  }, []);

  const getHealthColor = (
    value: number,
    criticalThreshold: number,
    warningThreshold: number,
  ) => {
    if (value >= criticalThreshold) return "error";
    if (value >= warningThreshold) return "warning";
    return "success";
  };

  const formatUptime = (seconds?: number) => {
    if (!seconds) return "N/A";
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
  };

  if (loading) {
    return (
      <Paper elevation={2} sx={{ p: 2, height: "100%" }}>
        <Typography variant="h6" gutterBottom>
          System Health
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Loading...
        </Typography>
      </Paper>
    );
  }

  if (!health) {
    return (
      <Paper elevation={2} sx={{ p: 2, height: "100%" }}>
        <Typography variant="h6" gutterBottom>
          System Health
        </Typography>
        <Alert severity="error">Unable to fetch system health</Alert>
      </Paper>
    );
  }

  const cpuColor = getHealthColor(health.cpu_usage, 90, 80);
  const memoryColor = getHealthColor(health.memory_usage, 90, 80);

  return (
    <Paper elevation={2} sx={{ p: 2, height: "100%" }}>
      <Typography variant="h6" gutterBottom>
        System Health
      </Typography>

      <Grid container direction="column" spacing={2}>
        {/* CPU Usage */}
        <Grid>
          <Grid container justifyContent="space-between" sx={{ mb: 0.5 }}>
            <Grid>
              <Typography variant="body2" color="text.secondary">
                CPU Usage
              </Typography>
            </Grid>
            <Grid>
              <Typography variant="body2" color={`${cpuColor}.main`}>
                {health.cpu_usage.toFixed(1)}%
              </Typography>
            </Grid>
          </Grid>
          <LinearProgress
            variant="determinate"
            value={health.cpu_usage}
            color={cpuColor}
            sx={{ height: 8, borderRadius: 1 }}
          />
        </Grid>

        {/* Memory Usage */}
        <Grid>
          <Grid container justifyContent="space-between" sx={{ mb: 0.5 }}>
            <Grid>
              <Typography variant="body2" color="text.secondary">
                Memory Usage
              </Typography>
            </Grid>
            <Grid>
              <Typography variant="body2" color={`${memoryColor}.main`}>
                {health.memory_usage.toFixed(1)}%
              </Typography>
            </Grid>
          </Grid>
          <LinearProgress
            variant="determinate"
            value={health.memory_usage}
            color={memoryColor}
            sx={{ height: 8, borderRadius: 1 }}
          />
        </Grid>

        {/* Temperature if available */}
        {health.temperature !== undefined && (
          <Grid container justifyContent="space-between">
            <Grid>
              <Typography variant="body2" color="text.secondary">
                Temperature
              </Typography>
            </Grid>
            <Grid>
              <Typography variant="body2">
                {health.temperature.toFixed(1)}°C
              </Typography>
            </Grid>
          </Grid>
        )}

        {/* Uptime */}
        <Grid container justifyContent="space-between">
          <Grid>
            <Typography variant="body2" color="text.secondary">
              Uptime
            </Typography>
          </Grid>
          <Grid>
            <Typography variant="body2">
              {formatUptime(health.uptime)}
            </Typography>
          </Grid>
        </Grid>

        {/* Alerts for critical thresholds */}
        {health.cpu_usage > 80 && (
          <Grid>
            <Alert
              severity={health.cpu_usage > 90 ? "error" : "warning"}
              sx={{ py: 0.5 }}
            >
              High CPU usage
            </Alert>
          </Grid>
        )}
        {health.memory_usage > 80 && (
          <Grid>
            <Alert
              severity={health.memory_usage > 90 ? "error" : "warning"}
              sx={{ py: 0.5 }}
            >
              High memory usage
            </Alert>
          </Grid>
        )}
      </Grid>
    </Paper>
  );
}

export default SystemHealth;
