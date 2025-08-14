import React from "react";
import {
  Box,
  Typography,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Chip,
  Divider,
  useTheme,
} from "@mui/material";
import {
  CheckCircle,
  Cancel,
  FlightTakeoff,
  Battery20,
  Battery80,
  SignalCellular4Bar,
  SignalCellularOff,
  Fence,
  Person,
} from "@mui/icons-material";
import { useSystemState } from "../../hooks/useSystemState";

export const SafetyInterlocks: React.FC = () => {
  const theme = useTheme();
  const systemState = useSystemState();

  const safetyInterlocks = systemState?.safetyInterlocks || {
    modeCheck: false,
    batteryCheck: false,
    geofenceCheck: false,
    signalCheck: false,
    operatorCheck: false,
  };

  const flightMode = systemState?.flightMode || "UNKNOWN";
  const batteryPercent = systemState?.batteryPercent || 0;
  const gpsStatus = systemState?.gpsStatus || "NO_FIX";
  const mavlinkConnected = systemState?.mavlinkConnected || false;

  const getStatusIcon = (passed: boolean) => {
    return passed ? (
      <CheckCircle sx={{ color: theme.palette.success.main }} />
    ) : (
      <Cancel sx={{ color: theme.palette.error.main }} />
    );
  };

  const getStatusChip = (passed: boolean) => {
    return (
      <Chip
        label={passed ? "PASS" : "FAIL"}
        size="small"
        color={passed ? "success" : "error"}
        sx={{ fontWeight: "bold" }}
      />
    );
  };

  const getBatteryIcon = () => {
    if (batteryPercent > 50) {
      return <Battery80 sx={{ color: theme.palette.success.main }} />;
    } else if (batteryPercent > 20) {
      return <Battery20 sx={{ color: theme.palette.warning.main }} />;
    }
    return <Battery20 sx={{ color: theme.palette.error.main }} />;
  };

  const getSignalIcon = () => {
    if (safetyInterlocks.signalCheck) {
      return <SignalCellular4Bar sx={{ color: theme.palette.success.main }} />;
    }
    return <SignalCellularOff sx={{ color: theme.palette.error.main }} />;
  };

  const allChecksPassed = Object.values(safetyInterlocks).every(
    (check) => check,
  );

  return (
    <Paper elevation={3} sx={{ p: 3, width: "100%" }}>
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          mb: 2,
        }}
      >
        <Typography variant="h5" component="h3" sx={{ fontWeight: "bold" }}>
          Safety Interlocks
        </Typography>
        <Chip
          label={allChecksPassed ? "ALL SYSTEMS GO" : "CHECKS FAILED"}
          color={allChecksPassed ? "success" : "error"}
          sx={{ fontWeight: "bold", fontSize: "1rem", px: 2 }}
        />
      </Box>

      <Divider sx={{ mb: 2 }} />

      <List sx={{ width: "100%" }}>
        <ListItem>
          <ListItemIcon>
            {getStatusIcon(safetyInterlocks.modeCheck)}
          </ListItemIcon>
          <ListItemText
            primary={
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <FlightTakeoff />
                <Typography variant="body1" sx={{ fontWeight: 500 }}>
                  Flight Mode Check
                </Typography>
              </Box>
            }
            secondary={
              <Typography variant="body2" color="text.secondary">
                Current mode: <strong>{flightMode}</strong> (Required: GUIDED)
              </Typography>
            }
          />
          {getStatusChip(safetyInterlocks.modeCheck)}
        </ListItem>

        <ListItem>
          <ListItemIcon>{getStatusIcon(mavlinkConnected)}</ListItemIcon>
          <ListItemText
            primary={
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <Typography variant="body1" sx={{ fontWeight: 500 }}>
                  Armed Status Check
                </Typography>
              </Box>
            }
            secondary={
              <Typography variant="body2" color="text.secondary">
                MAVLink:{" "}
                <strong>
                  {mavlinkConnected ? "Connected" : "Disconnected"}
                </strong>
              </Typography>
            }
          />
          {getStatusChip(mavlinkConnected)}
        </ListItem>

        <ListItem>
          <ListItemIcon>
            {getStatusIcon(safetyInterlocks.batteryCheck)}
          </ListItemIcon>
          <ListItemText
            primary={
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                {getBatteryIcon()}
                <Typography variant="body1" sx={{ fontWeight: 500 }}>
                  Battery Level Check
                </Typography>
              </Box>
            }
            secondary={
              <Typography variant="body2" color="text.secondary">
                Current: <strong>{batteryPercent}%</strong> (Minimum: 20%)
              </Typography>
            }
          />
          {getStatusChip(safetyInterlocks.batteryCheck)}
        </ListItem>

        <ListItem>
          <ListItemIcon>
            {getStatusIcon(safetyInterlocks.signalCheck)}
          </ListItemIcon>
          <ListItemText
            primary={
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                {getSignalIcon()}
                <Typography variant="body1" sx={{ fontWeight: 500 }}>
                  Signal Quality Check
                </Typography>
              </Box>
            }
            secondary={
              <Typography variant="body2" color="text.secondary">
                Signal timeout: 10 seconds
              </Typography>
            }
          />
          {getStatusChip(safetyInterlocks.signalCheck)}
        </ListItem>

        <ListItem>
          <ListItemIcon>
            {getStatusIcon(safetyInterlocks.geofenceCheck)}
          </ListItemIcon>
          <ListItemText
            primary={
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <Fence />
                <Typography variant="body1" sx={{ fontWeight: 500 }}>
                  Geofence Boundary Check
                </Typography>
              </Box>
            }
            secondary={
              <Typography variant="body2" color="text.secondary">
                GPS: <strong>{gpsStatus}</strong>
              </Typography>
            }
          />
          {getStatusChip(safetyInterlocks.geofenceCheck)}
        </ListItem>

        <ListItem>
          <ListItemIcon>
            {getStatusIcon(safetyInterlocks.operatorCheck)}
          </ListItemIcon>
          <ListItemText
            primary={
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <Person />
                <Typography variant="body1" sx={{ fontWeight: 500 }}>
                  Operator Activation Status
                </Typography>
              </Box>
            }
            secondary={
              <Typography variant="body2" color="text.secondary">
                Manual override enabled
              </Typography>
            }
          />
          {getStatusChip(safetyInterlocks.operatorCheck)}
        </ListItem>
      </List>

      {!allChecksPassed && (
        <Box
          sx={{
            mt: 2,
            p: 2,
            backgroundColor: theme.palette.error.light,
            borderRadius: 1,
          }}
        >
          <Typography
            variant="body2"
            sx={{ color: theme.palette.error.contrastText }}
          >
            <strong>⚠️ Safety checks failed:</strong> Homing cannot be activated
            until all safety interlocks pass. Ensure drone is in GUIDED mode,
            battery is above 20%, and signal is strong.
          </Typography>
        </Box>
      )}
    </Paper>
  );
};
