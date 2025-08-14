import React from "react";
import {
  Card,
  CardContent,
  Typography,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  IconButton,
  Box,
  Alert,
} from "@mui/material";
import {
  CheckCircle,
  Cancel,
  Refresh,
  FlightTakeoff,
  Battery90,
  GpsFixed,
  Security,
  Fence,
  EmergencyShare,
  Sensors,
  AccountTree,
} from "@mui/icons-material";

interface ChecklistItem {
  name: string;
  status: boolean;
  icon: React.ReactElement;
  description: string;
}

interface PreflightChecklistProps {
  checklistStatus: {
    mavlink_connected: boolean;
    gps_fix_valid: boolean;
    battery_sufficient: boolean;
    safety_interlocks_passed: boolean;
    geofence_configured: boolean;
    emergency_stop_ready: boolean;
    signal_processor_active: boolean;
    state_machine_ready: boolean;
    all_passed?: boolean;
  } | null;
  onRefresh: () => void;
}

export const PreflightChecklist: React.FC<PreflightChecklistProps> = ({
  checklistStatus,
  onRefresh,
}) => {
  const getChecklistItems = (): ChecklistItem[] => {
    if (!checklistStatus) {
      return [];
    }

    return [
      {
        name: "MAVLink Connection",
        status: checklistStatus.mavlink_connected,
        icon: <FlightTakeoff />,
        description: "Communication with flight controller established",
      },
      {
        name: "GPS Fix Valid",
        status: checklistStatus.gps_fix_valid,
        icon: <GpsFixed />,
        description: "3D GPS fix with sufficient satellites",
      },
      {
        name: "Battery Sufficient",
        status: checklistStatus.battery_sufficient,
        icon: <Battery90 />,
        description: "Battery level above 30% for safe testing",
      },
      {
        name: "Safety Interlocks",
        status: checklistStatus.safety_interlocks_passed,
        icon: <Security />,
        description: "All safety checks passed",
      },
      {
        name: "Geofence Configured",
        status: checklistStatus.geofence_configured,
        icon: <Fence />,
        description: "Virtual boundary configured for safe operation",
      },
      {
        name: "Emergency Stop Ready",
        status: checklistStatus.emergency_stop_ready,
        icon: <EmergencyShare />,
        description: "Emergency stop system available and armed",
      },
      {
        name: "Signal Processor",
        status: checklistStatus.signal_processor_active,
        icon: <Sensors />,
        description: "SDR and signal processing active",
      },
      {
        name: "State Machine Ready",
        status: checklistStatus.state_machine_ready,
        icon: <AccountTree />,
        description: "System state machine initialized",
      },
    ];
  };

  const checklistItems = getChecklistItems();
  const allPassed = checklistStatus?.all_passed || false;

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
          <Typography variant="h6" sx={{ flexGrow: 1 }}>
            Preflight Checklist
          </Typography>
          <IconButton onClick={onRefresh} color="primary">
            <Refresh />
          </IconButton>
        </Box>

        {!checklistStatus ? (
          <Alert severity="info">Loading preflight checklist...</Alert>
        ) : (
          <>
            {allPassed ? (
              <Alert severity="success" sx={{ mb: 2 }}>
                All preflight checks passed - Ready for testing
              </Alert>
            ) : (
              <Alert severity="warning" sx={{ mb: 2 }}>
                Some preflight checks failed - Review items below
              </Alert>
            )}

            <List dense>
              {checklistItems.map((item) => (
                <ListItem key={item.name}>
                  <ListItemIcon>
                    {item.status ? (
                      <CheckCircle color="success" />
                    ) : (
                      <Cancel color="error" />
                    )}
                  </ListItemIcon>
                  <ListItemText
                    primary={item.name}
                    secondary={item.description}
                    primaryTypographyProps={{
                      color: item.status ? "text.primary" : "text.secondary",
                    }}
                  />
                  <ListItemIcon>{item.icon}</ListItemIcon>
                </ListItem>
              ))}
            </List>
          </>
        )}
      </CardContent>
    </Card>
  );
};
