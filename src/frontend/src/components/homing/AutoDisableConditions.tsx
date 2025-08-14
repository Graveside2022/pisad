import React from "react";
import {
  Box,
  Typography,
  Paper,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Alert,
  useTheme,
} from "@mui/material";
import {
  SignalCellularOff,
  FlightLand,
  Battery20,
  Fence,
  StopCircle,
  Info,
} from "@mui/icons-material";
import { useSystemState } from "../../hooks/useSystemState";

export const AutoDisableConditions: React.FC = () => {
  const theme = useTheme();
  const systemState = useSystemState();

  const safetyInterlocks = systemState?.safetyInterlocks || {
    modeCheck: false,
    batteryCheck: false,
    geofenceCheck: false,
    signalCheck: false,
    operatorCheck: false,
  };

  const homingEnabled = systemState?.homingEnabled || false;
  const currentState = systemState?.currentState || "IDLE";

  const conditions = [
    {
      icon: <SignalCellularOff />,
      title: "Signal Loss",
      description: "Signal loss for 10 seconds",
      active: !safetyInterlocks.signalCheck,
      severity: "high",
    },
    {
      icon: <FlightLand />,
      title: "Mode Change",
      description: "Flight mode changed from GUIDED",
      active: !safetyInterlocks.modeCheck,
      severity: "high",
    },
    {
      icon: <Battery20 />,
      title: "Low Battery",
      description: "Battery below 20%",
      active: !safetyInterlocks.batteryCheck,
      severity: "high",
    },
    {
      icon: <Fence />,
      title: "Geofence",
      description: "Geofence boundary reached",
      active: !safetyInterlocks.geofenceCheck,
      severity: "medium",
    },
    {
      icon: <StopCircle />,
      title: "Emergency Stop",
      description: "Emergency stop activated",
      active: false,
      severity: "critical",
    },
  ];

  const activeConditions = conditions.filter((c) => c.active);

  const getConditionColor = (severity: string, active: boolean) => {
    if (!active) return theme.palette.text.secondary;
    switch (severity) {
      case "critical":
        return theme.palette.error.main;
      case "high":
        return theme.palette.warning.main;
      case "medium":
        return theme.palette.info.main;
      default:
        return theme.palette.text.primary;
    }
  };

  return (
    <Paper elevation={3} sx={{ p: 3, width: "100%" }}>
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 2 }}>
        <Info color="info" />
        <Typography variant="h5" component="h3" sx={{ fontWeight: "bold" }}>
          Auto-Disable Conditions
        </Typography>
      </Box>

      {homingEnabled && activeConditions.length > 0 && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          <strong>
            {activeConditions.length} condition(s) may trigger auto-disable
          </strong>
        </Alert>
      )}

      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
        Homing will automatically disable if any of these conditions are met:
      </Typography>

      <List sx={{ width: "100%" }}>
        {conditions.map((condition, index) => (
          <ListItem
            key={index}
            sx={{
              opacity: condition.active ? 1 : 0.6,
              backgroundColor: condition.active
                ? theme.palette.action.hover
                : "transparent",
              borderRadius: 1,
              mb: 1,
            }}
          >
            <ListItemIcon>
              <Box
                sx={{
                  color: getConditionColor(
                    condition.severity,
                    condition.active,
                  ),
                  display: "flex",
                  alignItems: "center",
                }}
              >
                {condition.icon}
              </Box>
            </ListItemIcon>
            <ListItemText
              primary={
                <Typography
                  variant="body1"
                  sx={{
                    fontWeight: condition.active ? "bold" : "normal",
                    color: condition.active
                      ? theme.palette.text.primary
                      : theme.palette.text.secondary,
                  }}
                >
                  {condition.title}
                </Typography>
              }
              secondary={
                <Typography
                  variant="body2"
                  color={condition.active ? "text.primary" : "text.secondary"}
                >
                  {condition.description}
                </Typography>
              }
            />
            {condition.active && homingEnabled && (
              <Typography
                variant="caption"
                sx={{
                  color: getConditionColor(condition.severity, true),
                  fontWeight: "bold",
                  textTransform: "uppercase",
                }}
              >
                Active
              </Typography>
            )}
          </ListItem>
        ))}
      </List>

      {currentState === "HOMING" && (
        <Alert severity="info" sx={{ mt: 2 }}>
          Homing is currently active. Any triggered condition will immediately
          disable homing and return control to manual operation.
        </Alert>
      )}
    </Paper>
  );
};
