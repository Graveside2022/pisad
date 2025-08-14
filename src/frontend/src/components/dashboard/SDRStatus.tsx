import { useEffect, useState } from "react";
import { Typography, Paper, Chip, Divider, Grid } from "@mui/material";
import type { SDRConfig } from "../../types";

interface SDRStatusProps {
  status?: "CONNECTED" | "DISCONNECTED" | "ERROR";
}

function SDRStatus({ status = "DISCONNECTED" }: SDRStatusProps) {
  const [sdrConfig, setSDRConfig] = useState<SDRConfig | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch SDR configuration from API
    const fetchConfig = async () => {
      try {
        const response = await fetch("/api/system/status");
        if (response.ok) {
          const data = await response.json();
          setSDRConfig({
            frequency: data.sdr_frequency || 433920000,
            sample_rate: data.sdr_sample_rate || 2048000,
            gain: data.sdr_gain || 30,
            ppm_correction: data.sdr_ppm_correction || 0,
            device_index: data.sdr_device_index || 0,
          });
        }
      } catch (error) {
        console.error("Failed to fetch SDR config:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchConfig();
    const interval = setInterval(fetchConfig, 5000); // Refresh every 5 seconds

    return () => clearInterval(interval);
  }, []);

  const getStatusColor = () => {
    switch (status) {
      case "CONNECTED":
        return "success";
      case "ERROR":
        return "error";
      default:
        return "default";
    }
  };

  const formatFrequency = (freq: number) => {
    return `${(freq / 1000000).toFixed(3)} MHz`;
  };

  const formatSampleRate = (rate: number) => {
    return `${(rate / 1000000).toFixed(2)} MS/s`;
  };

  if (loading) {
    return (
      <Paper elevation={2} sx={{ p: 2, height: "100%" }}>
        <Typography variant="h6" gutterBottom>
          SDR Configuration
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Loading...
        </Typography>
      </Paper>
    );
  }

  return (
    <Paper elevation={2} sx={{ p: 2, height: "100%" }}>
      <Grid
        container
        justifyContent="space-between"
        alignItems="center"
        sx={{ mb: 2 }}
      >
        <Grid>
          <Typography variant="h6">SDR Configuration</Typography>
        </Grid>
        <Grid>
          <Chip
            label={status}
            color={getStatusColor()}
            size="small"
            variant={status === "CONNECTED" ? "filled" : "outlined"}
          />
        </Grid>
      </Grid>

      <Divider sx={{ mb: 2 }} />

      {sdrConfig && (
        <Grid container direction="column" spacing={1.5}>
          <Grid>
            <Typography variant="caption" color="text.secondary">
              Frequency
            </Typography>
            <Typography variant="body1" fontWeight="medium">
              {formatFrequency(sdrConfig.frequency)}
            </Typography>
          </Grid>

          <Grid>
            <Typography variant="caption" color="text.secondary">
              Sample Rate
            </Typography>
            <Typography variant="body1" fontWeight="medium">
              {formatSampleRate(sdrConfig.sample_rate)}
            </Typography>
          </Grid>

          <Grid>
            <Typography variant="caption" color="text.secondary">
              Gain
            </Typography>
            <Typography variant="body1" fontWeight="medium">
              {sdrConfig.gain} dB
            </Typography>
          </Grid>

          <Grid>
            <Typography variant="caption" color="text.secondary">
              PPM Correction
            </Typography>
            <Typography variant="body1" fontWeight="medium">
              {sdrConfig.ppm_correction}
            </Typography>
          </Grid>

          <Grid>
            <Typography variant="caption" color="text.secondary">
              Device Index
            </Typography>
            <Typography variant="body1" fontWeight="medium">
              {sdrConfig.device_index}
            </Typography>
          </Grid>
        </Grid>
      )}
    </Paper>
  );
}

export default SDRStatus;
