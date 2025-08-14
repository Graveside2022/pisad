import { Typography, LinearProgress, Paper, Grid } from "@mui/material";
import { useMemo } from "react";

interface SignalMeterProps {
  rssi: number;
  noiseFloor: number;
  snr: number;
  size?: "small" | "medium" | "large";
}

function SignalMeter({
  rssi,
  noiseFloor,
  snr,
  size = "medium",
}: SignalMeterProps) {
  const signalStrength = useMemo(() => {
    // Convert RSSI to percentage (assuming -100 dBm to -30 dBm range)
    const minRSSI = -100;
    const maxRSSI = -30;
    const percentage = Math.max(
      0,
      Math.min(100, ((rssi - minRSSI) / (maxRSSI - minRSSI)) * 100),
    );
    return percentage;
  }, [rssi]);

  const signalColor = useMemo(() => {
    if (snr > 12) return "success";
    if (snr > 6) return "warning";
    return "error";
  }, [snr]);

  const sizeConfig = {
    small: { height: 150, fontSize: "1.5rem" },
    medium: { height: 200, fontSize: "2rem" },
    large: { height: 250, fontSize: "2.5rem" },
  };

  const config = sizeConfig[size];

  return (
    <Paper
      elevation={2}
      sx={{
        p: 2,
        height: config.height,
      }}
    >
      <Grid
        container
        direction="column"
        justifyContent="space-between"
        sx={{ height: "100%" }}
      >
        <Grid>
          <Typography variant="h6" color="text.secondary" gutterBottom>
            Signal Strength
          </Typography>

          <Typography
            variant="h3"
            sx={{
              fontSize: config.fontSize,
              fontWeight: "bold",
              color: `${signalColor}.main`,
            }}
          >
            {rssi.toFixed(1)} dBm
          </Typography>
        </Grid>

        <Grid>
          <Grid container direction="column" spacing={1}>
            <Grid>
              <Typography variant="body2" color="text.secondary">
                Signal Level
              </Typography>
              <LinearProgress
                variant="determinate"
                value={signalStrength}
                color={signalColor}
                sx={{ height: 10, borderRadius: 1 }}
              />
            </Grid>

            <Grid container justifyContent="space-between" sx={{ mt: 2 }}>
              <Grid>
                <Typography variant="caption" color="text.secondary">
                  Noise Floor
                </Typography>
                <Typography variant="body2">
                  {noiseFloor.toFixed(1)} dBm
                </Typography>
              </Grid>

              <Grid>
                <Typography variant="caption" color="text.secondary">
                  SNR
                </Typography>
                <Typography variant="body2" color={`${signalColor}.main`}>
                  {snr.toFixed(1)} dB
                </Typography>
              </Grid>
            </Grid>
          </Grid>
        </Grid>
      </Grid>
    </Paper>
  );
}

export default SignalMeter;
