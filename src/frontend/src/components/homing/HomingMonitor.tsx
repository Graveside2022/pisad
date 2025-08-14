/**
 * Homing Algorithm Monitor Widget
 * Displays real-time status of the gradient-based homing algorithm
 */

import React from "react";
import {
  Box,
  Card,
  CardContent,
  Grid,
  LinearProgress,
  Typography,
  Chip,
  Switch,
  FormControlLabel,
  IconButton,
  Tooltip,
  Alert,
} from "@mui/material";
import {
  Navigation as NavigationIcon,
  BugReport as BugReportIcon,
  TrendingUp as TrendingUpIcon,
  Explore as ExploreIcon,
  CircleOutlined as CircleIcon,
  Pause as PauseIcon,
} from "@mui/icons-material";
import { useSystemState } from "../../hooks/useSystemState";
import { useWebSocket } from "../../hooks/useWebSocket";

interface HomingStatus {
  substage: "IDLE" | "GRADIENT_CLIMB" | "SAMPLING" | "APPROACH" | "HOLDING";
  sample_count: number;
  gradient_confidence: number;
  gradient_magnitude: number;
  gradient_direction: number;
  latest_rssi: number;
  debug_mode: boolean;
  debug_info?: {
    rssi_min: number;
    rssi_max: number;
    rssi_mean: number;
    rssi_variance: number;
    position_spread_x: number;
    position_spread_y: number;
    time_span: number;
    sampling_active: boolean;
  };
}

const HomingMonitor: React.FC = () => {
  useSystemState(); // For future use
  const { addMessageHandler } = useWebSocket();
  const [homingStatus, setHomingStatus] = React.useState<HomingStatus | null>(
    null,
  );
  const [debugMode, setDebugMode] = React.useState(false);

  // Update homing status from WebSocket messages
  React.useEffect(() => {
    const handleMessage = (message: any) => {
      if (message?.type === "homing_status") {
        setHomingStatus(message.data);
      }
    };

    const unsubscribe = addMessageHandler(handleMessage);
    return () => unsubscribe();
  }, [addMessageHandler]);

  const handleDebugToggle = async () => {
    try {
      const response = await fetch("/api/system/debug", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: !debugMode, target: "homing" }),
      });

      if (response.ok) {
        setDebugMode(!debugMode);
      }
    } catch (error) {
      console.error("Failed to toggle debug mode:", error);
    }
  };

  const getSubstageIcon = (substage: string) => {
    switch (substage) {
      case "GRADIENT_CLIMB":
        return <TrendingUpIcon />;
      case "SAMPLING":
        return <ExploreIcon />;
      case "APPROACH":
        return <NavigationIcon />;
      case "HOLDING":
        return <CircleIcon />;
      default:
        return <PauseIcon />;
    }
  };

  const getSubstageColor = (substage: string) => {
    switch (substage) {
      case "GRADIENT_CLIMB":
        return "success";
      case "SAMPLING":
        return "warning";
      case "APPROACH":
        return "info";
      case "HOLDING":
        return "secondary";
      default:
        return "default";
    }
  };

  const formatHeading = (degrees: number) => {
    const directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
    const index = Math.round(degrees / 45) % 8;
    return `${degrees.toFixed(1)}° ${directions[index]}`;
  };

  return (
    <Card>
      <CardContent>
        <Box
          display="flex"
          justifyContent="space-between"
          alignItems="center"
          mb={2}
        >
          <Typography variant="h6">Homing Algorithm Monitor</Typography>
          <Box display="flex" gap={1}>
            <FormControlLabel
              control={
                <Switch
                  checked={debugMode}
                  onChange={handleDebugToggle}
                  size="small"
                />
              }
              label="Debug"
            />
            <Tooltip title="Debug mode enables verbose logging">
              <IconButton size="small">
                <BugReportIcon fontSize="small" />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {homingStatus ? (
          <>
            {/* Current State */}
            <Box mb={2}>
              <Chip
                icon={getSubstageIcon(homingStatus.substage)}
                label={homingStatus.substage.replace("_", " ")}
                color={getSubstageColor(homingStatus.substage)}
                variant="outlined"
              />
            </Box>

            {/* Main Metrics */}
            <Grid container spacing={2}>
              <Grid size={6}>
                <Typography variant="caption" color="textSecondary">
                  Gradient Confidence
                </Typography>
                <Box display="flex" alignItems="center" gap={1}>
                  <LinearProgress
                    variant="determinate"
                    value={homingStatus.gradient_confidence}
                    sx={{ flexGrow: 1, height: 8, borderRadius: 1 }}
                    color={
                      homingStatus.gradient_confidence > 70
                        ? "success"
                        : homingStatus.gradient_confidence > 30
                          ? "warning"
                          : "error"
                    }
                  />
                  <Typography variant="body2">
                    {homingStatus.gradient_confidence.toFixed(1)}%
                  </Typography>
                </Box>
              </Grid>

              <Grid size={6}>
                <Typography variant="caption" color="textSecondary">
                  Target Direction
                </Typography>
                <Typography variant="body1">
                  {formatHeading(homingStatus.gradient_direction)}
                </Typography>
              </Grid>

              <Grid size={6}>
                <Typography variant="caption" color="textSecondary">
                  Gradient Magnitude
                </Typography>
                <Typography variant="body1">
                  {homingStatus.gradient_magnitude.toFixed(3)} dB/m
                </Typography>
              </Grid>

              <Grid size={6}>
                <Typography variant="caption" color="textSecondary">
                  Latest RSSI
                </Typography>
                <Typography variant="body1">
                  {homingStatus.latest_rssi.toFixed(1)} dBm
                </Typography>
              </Grid>

              <Grid size={12}>
                <Typography variant="caption" color="textSecondary">
                  Sample Buffer
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={(homingStatus.sample_count / 10) * 100}
                  sx={{ height: 6, borderRadius: 1 }}
                />
                <Typography variant="caption">
                  {homingStatus.sample_count}/10 samples
                </Typography>
              </Grid>
            </Grid>

            {/* Debug Information */}
            {debugMode && homingStatus.debug_info && (
              <Box mt={2} p={1} bgcolor="grey.100" borderRadius={1}>
                <Typography variant="caption" fontWeight="bold">
                  Debug Information
                </Typography>
                <Grid container spacing={1} mt={0.5}>
                  <Grid size={6}>
                    <Typography variant="caption" display="block">
                      RSSI Range: {homingStatus.debug_info.rssi_min.toFixed(1)}{" "}
                      to {homingStatus.debug_info.rssi_max.toFixed(1)} dBm
                    </Typography>
                  </Grid>
                  <Grid size={6}>
                    <Typography variant="caption" display="block">
                      Mean: {homingStatus.debug_info.rssi_mean.toFixed(1)} dBm
                    </Typography>
                  </Grid>
                  <Grid size={6}>
                    <Typography variant="caption" display="block">
                      Variance:{" "}
                      {homingStatus.debug_info.rssi_variance.toFixed(2)}
                    </Typography>
                  </Grid>
                  <Grid size={6}>
                    <Typography variant="caption" display="block">
                      Time Span: {homingStatus.debug_info.time_span.toFixed(1)}s
                    </Typography>
                  </Grid>
                  <Grid size={12}>
                    <Typography variant="caption" display="block">
                      Position Spread: X=
                      {homingStatus.debug_info.position_spread_x.toFixed(1)}m,
                      Y={homingStatus.debug_info.position_spread_y.toFixed(1)}m
                    </Typography>
                  </Grid>
                  {homingStatus.debug_info.sampling_active && (
                    <Grid size={12}>
                      <Alert severity="info" sx={{ py: 0 }}>
                        <Typography variant="caption">
                          Sampling maneuver active
                        </Typography>
                      </Alert>
                    </Grid>
                  )}
                </Grid>
              </Box>
            )}

            {/* Algorithm State Messages */}
            {homingStatus.substage === "SAMPLING" && (
              <Alert severity="warning" sx={{ mt: 2 }}>
                Performing S-turn sampling to improve gradient confidence
              </Alert>
            )}

            {homingStatus.substage === "APPROACH" && (
              <Alert severity="info" sx={{ mt: 2 }}>
                Strong signal detected - reduced velocity approach mode
              </Alert>
            )}

            {homingStatus.substage === "HOLDING" && (
              <Alert severity="success" sx={{ mt: 2 }}>
                Signal plateau detected - beacon likely directly below
              </Alert>
            )}
          </>
        ) : (
          <Typography color="textSecondary">
            No homing data available
          </Typography>
        )}
      </CardContent>
    </Card>
  );
};

export default HomingMonitor;
