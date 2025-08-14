import React, { useState } from "react";
import {
  Box,
  Typography,
  ToggleButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Alert,
  Snackbar,
  Paper,
  useTheme,
} from "@mui/material";
import {
  FlightTakeoff,
  FlightLand,
  Warning,
  CheckCircle,
  Cancel,
} from "@mui/icons-material";
import { useSystemState } from "../../hooks/useSystemState";
import { api } from "../../services/api";
import { AxiosError } from "axios";

interface HomingControlProps {
  onStateChange?: (enabled: boolean) => void;
}

interface ApiErrorResponse {
  error?: string;
  blockedBy?: string[];
}

export const HomingControl: React.FC<HomingControlProps> = ({
  onStateChange,
}) => {
  const theme = useTheme();
  const systemState = useSystemState();
  const [confirmDialogOpen, setConfirmDialogOpen] = useState(false);
  const [confirmationToken, setConfirmationToken] = useState("");
  const [isActivating, setIsActivating] = useState(false);
  const [snackbar, setSnackbar] = useState<{
    open: boolean;
    message: string;
    severity: "success" | "error" | "warning" | "info";
  }>({
    open: false,
    message: "",
    severity: "info",
  });

  const homingEnabled = systemState?.homingEnabled || false;
  const currentState = systemState?.currentState || "IDLE";
  const isHoming = currentState === "HOMING";

  const getStateColor = () => {
    if (isHoming) return theme.palette.error.main;
    if (homingEnabled) return theme.palette.success.main;
    return theme.palette.grey[500];
  };

  const getStateText = () => {
    if (isHoming) return "HOMING ACTIVE";
    if (homingEnabled) return "HOMING ENABLED";
    return "HOMING DISABLED";
  };

  const getStateIcon = () => {
    if (isHoming) return <FlightLand sx={{ fontSize: 40 }} />;
    if (homingEnabled) return <CheckCircle sx={{ fontSize: 40 }} />;
    return <Cancel sx={{ fontSize: 40 }} />;
  };

  const handleToggleClick = () => {
    if (!homingEnabled) {
      // Check if all safety interlocks pass before showing confirmation
      const safetyInterlocks = systemState?.safetyInterlocks;
      const allSafetyChecksPassed =
        safetyInterlocks &&
        Object.values(safetyInterlocks).every((check) => check === true);

      if (!allSafetyChecksPassed) {
        setSnackbar({
          open: true,
          message: "Cannot enable homing: Safety interlocks not satisfied",
          severity: "warning",
        });
        return;
      }

      const token = `confirm-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
      setConfirmationToken(token);
      setConfirmDialogOpen(true);
    } else {
      disableHoming();
    }
  };

  const handleConfirmActivation = async () => {
    setIsActivating(true);
    setConfirmDialogOpen(false);

    try {
      const response = await api.setHomingState(true, confirmationToken);
      setSnackbar({
        open: true,
        message: response.message || "Homing enabled successfully",
        severity: "success",
      });
      onStateChange?.(true);
    } catch (error) {
      const axiosError = error as AxiosError<ApiErrorResponse>;
      const message =
        axiosError.response?.data?.error || "Failed to enable homing";
      const blockedBy = axiosError.response?.data?.blockedBy;

      setSnackbar({
        open: true,
        message: blockedBy ? `${message}: ${blockedBy.join(", ")}` : message,
        severity: "error",
      });
    } finally {
      setIsActivating(false);
      setConfirmationToken("");
    }
  };

  const disableHoming = async () => {
    setIsActivating(true);

    try {
      const response = await api.setHomingState(false, "");
      setSnackbar({
        open: true,
        message: response.message || "Homing disabled successfully",
        severity: "info",
      });
      onStateChange?.(false);
    } catch (error) {
      const axiosError = error as AxiosError<ApiErrorResponse>;
      setSnackbar({
        open: true,
        message: axiosError.response?.data?.error || "Failed to disable homing",
        severity: "error",
      });
    } finally {
      setIsActivating(false);
    }
  };

  const handleCancelActivation = () => {
    setConfirmDialogOpen(false);
    setConfirmationToken("");
  };

  const handleSnackbarClose = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  return (
    <Paper
      elevation={3}
      sx={{ p: 3, width: "100%" }}
      data-testid="homing-control"
    >
      <Box
        sx={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 3,
        }}
      >
        <Typography variant="h4" component="h2" sx={{ fontWeight: "bold" }}>
          Homing Control
        </Typography>

        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 2,
            p: 2,
            borderRadius: 2,
            backgroundColor: getStateColor(),
            color: "white",
            minWidth: 300,
            justifyContent: "center",
          }}
        >
          {getStateIcon()}
          <Typography variant="h5" sx={{ fontWeight: "bold" }}>
            {getStateText()}
          </Typography>
        </Box>

        <ToggleButton
          value="homing"
          selected={homingEnabled}
          onChange={handleToggleClick}
          disabled={isActivating}
          data-testid="homing-toggle"
          sx={{
            width: 250,
            height: 80,
            fontSize: "1.5rem",
            fontWeight: "bold",
            backgroundColor: homingEnabled
              ? theme.palette.success.main
              : theme.palette.grey[300],
            color: homingEnabled ? "white" : "black",
            "&:hover": {
              backgroundColor: homingEnabled
                ? theme.palette.success.dark
                : theme.palette.grey[400],
            },
            "&.Mui-selected": {
              backgroundColor: theme.palette.success.main,
              color: "white",
              "&:hover": {
                backgroundColor: theme.palette.success.dark,
              },
            },
          }}
        >
          {homingEnabled ? (
            <>
              <FlightLand sx={{ mr: 1, fontSize: 30 }} />
              DISABLE
            </>
          ) : (
            <>
              <FlightTakeoff sx={{ mr: 1, fontSize: 30 }} />
              ENABLE
            </>
          )}
        </ToggleButton>

        {homingEnabled && (
          <Alert
            severity="info"
            sx={{ mt: 2, width: "100%", fontSize: "1.1rem" }}
          >
            <strong>To regain control:</strong> Switch flight mode in Mission
            Planner
          </Alert>
        )}
      </Box>

      <Dialog
        open={confirmDialogOpen}
        onClose={handleCancelActivation}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <Warning color="warning" />
          Confirm Homing Activation
        </DialogTitle>
        <DialogContent>
          <Alert severity="warning" sx={{ mb: 2 }}>
            <strong>WARNING:</strong> Enabling homing mode will allow the drone
            to autonomously navigate towards detected signals. Ensure you
            understand the implications and maintain visual contact with the
            drone.
          </Alert>
          <Typography variant="body1" paragraph>
            The drone will:
          </Typography>
          <ul>
            <li>Automatically navigate towards the strongest signal</li>
            <li>Maintain current altitude unless specified otherwise</li>
            <li>Continue homing until disabled or safety limits reached</li>
          </ul>
          <Typography variant="body1" sx={{ mt: 2, fontWeight: "bold" }}>
            Are you sure you want to enable homing mode?
          </Typography>
        </DialogContent>
        <DialogActions sx={{ p: 2 }}>
          <Button
            onClick={handleCancelActivation}
            color="inherit"
            size="large"
            sx={{ minWidth: 100 }}
          >
            Cancel
          </Button>
          <Button
            onClick={handleConfirmActivation}
            variant="contained"
            color="warning"
            size="large"
            sx={{ minWidth: 150, fontWeight: "bold" }}
            disabled={isActivating}
          >
            Confirm & Enable
          </Button>
        </DialogActions>
      </Dialog>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleSnackbarClose}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
      >
        <Alert
          onClose={handleSnackbarClose}
          severity={snackbar.severity}
          sx={{ width: "100%" }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Paper>
  );
};
