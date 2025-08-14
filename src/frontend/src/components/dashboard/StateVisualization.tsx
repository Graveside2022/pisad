import React, { useState, useEffect } from "react";
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Alert,
  Chip,
  Stack,
  Paper,
  Grid,
  List,
  ListItem,
  ListItemText,
} from "@mui/material";
import {
  Warning,
  RadioButtonChecked,
  ArrowForward,
} from "@mui/icons-material";
import { useWebSocket } from "../../hooks/useWebSocket";
import { useStateOverride } from "../../hooks/useStateOverride";

interface StateTransition {
  from_state: string;
  to_state: string;
  timestamp: string;
  reason: string;
}

interface StateData {
  current_state: string;
  previous_state: string;
  timestamp: string;
  reason: string;
  allowed_transitions: string[];
  state_duration_ms: number;
  history: StateTransition[];
}

const STATE_COLORS: Record<string, string> = {
  IDLE: "#9e9e9e",
  SEARCHING: "#2196f3",
  DETECTING: "#ff9800",
  HOMING: "#4caf50",
  HOLDING: "#f44336",
};

const StateVisualization: React.FC = () => {
  const [stateData, setStateData] = useState<StateData | null>(null);
  const [overrideDialogOpen, setOverrideDialogOpen] = useState(false);
  const [targetState, setTargetState] = useState("");
  const [overrideReason, setOverrideReason] = useState("");
  const [confirmToken, setConfirmToken] = useState("");
  const [operatorId, setOperatorId] = useState("");
  const [error, setError] = useState<string | null>(null);

  const { addMessageHandler } = useWebSocket();
  const { overrideState, isLoading } = useStateOverride();

  useEffect(() => {
    const handleStateUpdate = (message: any) => {
      if (message.type === "state") {
        setStateData(message.data as StateData);
      }
    };

    const unsubscribe = addMessageHandler(handleStateUpdate);

    return () => {
      unsubscribe();
    };
  }, [addMessageHandler]);

  const handleOverrideSubmit = async () => {
    if (!targetState || !confirmToken || !operatorId) {
      setError("Please fill in all required fields");
      return;
    }

    try {
      await overrideState({
        target_state: targetState,
        reason: overrideReason || "Manual override",
        confirmation_token: confirmToken,
        operator_id: operatorId,
      });
      setOverrideDialogOpen(false);
      resetOverrideForm();
    } catch (err) {
      setError("Failed to override state: " + (err as Error).message);
    }
  };

  const resetOverrideForm = () => {
    setTargetState("");
    setOverrideReason("");
    setConfirmToken("");
    setOperatorId("");
    setError(null);
  };

  const formatDuration = (ms: number): string => {
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${Math.floor(ms / 60000)}m ${Math.floor((ms % 60000) / 1000)}s`;
  };

  const formatTimestamp = (timestamp: string): string => {
    return new Date(timestamp).toLocaleTimeString();
  };

  if (!stateData) {
    return (
      <Card>
        <CardContent>
          <Typography>Waiting for state data...</Typography>
        </CardContent>
      </Card>
    );
  }

  return (
    <Grid container spacing={2}>
      {/* Current State Card */}
      <Grid size={{ xs: 12, md: 6 }}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Current State
            </Typography>
            <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
              <RadioButtonChecked
                sx={{
                  color: STATE_COLORS[stateData.current_state],
                  mr: 1,
                  fontSize: 40,
                }}
              />
              <Typography variant="h4">{stateData.current_state}</Typography>
            </Box>
            <Typography variant="body2" color="text.secondary">
              Duration: {formatDuration(stateData.state_duration_ms)}
            </Typography>
            {stateData.reason && (
              <Typography variant="body2" sx={{ mt: 1 }}>
                Reason: {stateData.reason}
              </Typography>
            )}

            <Box sx={{ mt: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Allowed Transitions:
              </Typography>
              <Stack direction="row" spacing={1} flexWrap="wrap">
                {stateData.allowed_transitions.map((state) => (
                  <Chip
                    key={state}
                    label={state}
                    size="small"
                    sx={{ backgroundColor: STATE_COLORS[state] + "30" }}
                  />
                ))}
              </Stack>
            </Box>

            <Button
              variant="outlined"
              startIcon={<Warning />}
              onClick={() => setOverrideDialogOpen(true)}
              sx={{ mt: 2 }}
              disabled={isLoading}
            >
              Manual Override
            </Button>
          </CardContent>
        </Card>
      </Grid>

      {/* State Diagram */}
      <Grid size={{ xs: 12, md: 6 }}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              State Flow Diagram
            </Typography>
            <Box sx={{ position: "relative", height: 200 }}>
              {/* Simple state flow visualization */}
              <Stack
                direction="row"
                spacing={2}
                justifyContent="center"
                alignItems="center"
              >
                {["IDLE", "SEARCHING", "DETECTING", "HOMING", "HOLDING"].map(
                  (state, index) => (
                    <React.Fragment key={state}>
                      <Paper
                        elevation={stateData.current_state === state ? 4 : 1}
                        sx={{
                          p: 1,
                          backgroundColor:
                            stateData.current_state === state
                              ? STATE_COLORS[state]
                              : "#f5f5f5",
                          color:
                            stateData.current_state === state
                              ? "white"
                              : "black",
                          minWidth: 80,
                          textAlign: "center",
                        }}
                      >
                        <Typography variant="caption">{state}</Typography>
                      </Paper>
                      {index < 4 && <ArrowForward />}
                    </React.Fragment>
                  ),
                )}
              </Stack>
            </Box>
          </CardContent>
        </Card>
      </Grid>

      {/* State History */}
      <Grid size={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              State History
            </Typography>
            <List dense>
              {stateData.history.slice(0, 10).map((transition, index) => (
                <ListItem key={index}>
                  <ListItemText
                    primary={
                      <Box
                        sx={{ display: "flex", alignItems: "center", gap: 1 }}
                      >
                        <Chip
                          label={transition.from_state}
                          size="small"
                          sx={{
                            backgroundColor:
                              STATE_COLORS[transition.from_state] + "30",
                          }}
                        />
                        <ArrowForward fontSize="small" />
                        <Chip
                          label={transition.to_state}
                          size="small"
                          sx={{
                            backgroundColor:
                              STATE_COLORS[transition.to_state] + "30",
                          }}
                        />
                        <Typography variant="body2" sx={{ ml: 2 }}>
                          {transition.reason}
                        </Typography>
                      </Box>
                    }
                    secondary={formatTimestamp(transition.timestamp)}
                  />
                </ListItem>
              ))}
            </List>
          </CardContent>
        </Card>
      </Grid>

      {/* Override Dialog */}
      <Dialog
        open={overrideDialogOpen}
        onClose={() => {
          setOverrideDialogOpen(false);
          resetOverrideForm();
        }}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>Manual State Override</DialogTitle>
        <DialogContent>
          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}
          <Alert severity="warning" sx={{ mb: 2 }}>
            Manual state override should only be used for testing or emergency
            situations.
          </Alert>

          <FormControl fullWidth sx={{ mb: 2 }}>
            <InputLabel>Target State</InputLabel>
            <Select
              value={targetState}
              onChange={(e) => setTargetState(e.target.value)}
              label="Target State"
            >
              {stateData.allowed_transitions.map((state) => (
                <MenuItem key={state} value={state}>
                  {state}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <TextField
            fullWidth
            label="Reason"
            value={overrideReason}
            onChange={(e) => setOverrideReason(e.target.value)}
            sx={{ mb: 2 }}
            helperText="Optional: Describe why this override is necessary"
          />

          <TextField
            fullWidth
            label="Confirmation Token"
            value={confirmToken}
            onChange={(e) => setConfirmToken(e.target.value)}
            sx={{ mb: 2 }}
            required
            helperText="Enter 'OVERRIDE' to confirm"
          />

          <TextField
            fullWidth
            label="Operator ID"
            value={operatorId}
            onChange={(e) => setOperatorId(e.target.value)}
            required
            helperText="Your operator identification"
          />
        </DialogContent>
        <DialogActions>
          <Button
            onClick={() => {
              setOverrideDialogOpen(false);
              resetOverrideForm();
            }}
          >
            Cancel
          </Button>
          <Button
            onClick={handleOverrideSubmit}
            variant="contained"
            color="warning"
            disabled={isLoading}
          >
            Override State
          </Button>
        </DialogActions>
      </Dialog>
    </Grid>
  );
};

export default StateVisualization;
