import React, { useState } from "react";
import {
  Box,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Typography,
  Alert,
  Snackbar,
  useTheme,
} from "@mui/material";
import { StopCircle, Warning, Emergency } from "@mui/icons-material";
import { api } from "../../services/api";
import { AxiosError } from "axios";

interface EmergencyStopProps {
  onEmergencyStop?: () => void;
}

interface ApiErrorResponse {
  error?: string;
}

export const EmergencyStop: React.FC<EmergencyStopProps> = ({
  onEmergencyStop,
}) => {
  const theme = useTheme();
  const [confirmDialogOpen, setConfirmDialogOpen] = useState(false);
  const [isTriggering, setIsTriggering] = useState(false);
  const [snackbar, setSnackbar] = useState<{
    open: boolean;
    message: string;
    severity: "success" | "error" | "warning" | "info";
  }>({
    open: false,
    message: "",
    severity: "info",
  });

  const handleEmergencyStopClick = () => {
    setConfirmDialogOpen(true);
  };

  const handleConfirmEmergencyStop = async () => {
    setIsTriggering(true);
    setConfirmDialogOpen(false);

    try {
      const response = await api.triggerEmergencyStop();
      setSnackbar({
        open: true,
        message: response.message || "Emergency stop activated successfully",
        severity: "success",
      });
      onEmergencyStop?.();
    } catch (error) {
      const axiosError = error as AxiosError<ApiErrorResponse>;
      setSnackbar({
        open: true,
        message:
          axiosError.response?.data?.error ||
          "Failed to trigger emergency stop",
        severity: "error",
      });
    } finally {
      setIsTriggering(false);
    }
  };

  const handleCancelEmergencyStop = () => {
    setConfirmDialogOpen(false);
  };

  const handleSnackbarClose = () => {
    setSnackbar({ ...snackbar, open: false });
  };

  return (
    <>
      <Box
        sx={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          p: 2,
        }}
      >
        <Button
          variant="contained"
          size="large"
          onClick={handleEmergencyStopClick}
          disabled={isTriggering}
          sx={{
            backgroundColor: theme.palette.error.main,
            color: "white",
            width: 250,
            height: 80,
            fontSize: "1.5rem",
            fontWeight: "bold",
            display: "flex",
            alignItems: "center",
            gap: 1,
            border: `3px solid ${theme.palette.error.dark}`,
            boxShadow: theme.shadows[4],
            "&:hover": {
              backgroundColor: theme.palette.error.dark,
              boxShadow: theme.shadows[8],
            },
            "&:disabled": {
              backgroundColor: theme.palette.grey[400],
            },
          }}
        >
          <StopCircle sx={{ fontSize: 40 }} />
          EMERGENCY STOP
        </Button>
      </Box>

      <Dialog
        open={confirmDialogOpen}
        onClose={handleCancelEmergencyStop}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 1,
            backgroundColor: theme.palette.error.main,
            color: "white",
          }}
        >
          <Emergency />
          <Typography variant="h6" sx={{ fontWeight: "bold" }}>
            CONFIRM EMERGENCY STOP
          </Typography>
        </DialogTitle>
        <DialogContent sx={{ mt: 2 }}>
          <Alert severity="error" sx={{ mb: 2 }}>
            <strong>⚠️ CRITICAL ACTION ⚠️</strong>
          </Alert>
          <Typography variant="body1" paragraph>
            This will immediately:
          </Typography>
          <ul>
            <li>Disable all homing operations</li>
            <li>Stop all autonomous navigation</li>
            <li>Return control to manual operation</li>
            <li>Log the emergency stop event</li>
          </ul>
          <Typography
            variant="body1"
            sx={{ mt: 2, fontWeight: "bold", color: theme.palette.error.main }}
          >
            Are you sure you want to trigger an EMERGENCY STOP?
          </Typography>
        </DialogContent>
        <DialogActions sx={{ p: 2, gap: 2 }}>
          <Button
            onClick={handleCancelEmergencyStop}
            variant="outlined"
            size="large"
            sx={{ minWidth: 100 }}
          >
            Cancel
          </Button>
          <Button
            onClick={handleConfirmEmergencyStop}
            variant="contained"
            color="error"
            size="large"
            sx={{
              minWidth: 150,
              fontWeight: "bold",
              backgroundColor: theme.palette.error.main,
              "&:hover": {
                backgroundColor: theme.palette.error.dark,
              },
            }}
            disabled={isTriggering}
          >
            <Warning sx={{ mr: 1 }} />
            CONFIRM STOP
          </Button>
        </DialogActions>
      </Dialog>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={handleSnackbarClose}
        anchorOrigin={{ vertical: "top", horizontal: "center" }}
      >
        <Alert
          onClose={handleSnackbarClose}
          severity={snackbar.severity}
          sx={{
            width: "100%",
            fontSize: "1.1rem",
            fontWeight: snackbar.severity === "success" ? "bold" : "normal",
          }}
        >
          {snackbar.message}
        </Alert>
      </Snackbar>
    </>
  );
};
