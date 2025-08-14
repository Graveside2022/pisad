import React, { useState, useEffect } from "react";
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Stack,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions,
  IconButton,
  Tooltip,
  Alert,
} from "@mui/material";
import {
  PlayArrow,
  Pause,
  Stop,
  RestartAlt,
  GetApp,
  Upload,
  Keyboard,
} from "@mui/icons-material";
import { type PatternControlAction, type PatternStatus } from "../../types/search";
import searchService from "../../services/search";

interface PatternControlsProps {
  patternId?: string;
  patternStatus?: PatternStatus;
  onActionComplete?: (action: PatternControlAction, success: boolean) => void;
}

const PatternControls: React.FC<PatternControlsProps> = ({
  patternId,
  patternStatus,
  onActionComplete,
}) => {
  const [confirmDialog, setConfirmDialog] =
    useState<PatternControlAction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [shortcuts, setShortcuts] = useState(false);

  useEffect(() => {
    const handleKeyPress = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey) {
        switch (e.key) {
          case "p":
            e.preventDefault();
            if (patternStatus?.state === "EXECUTING") {
              handleAction("pause");
            } else if (patternStatus?.state === "PAUSED") {
              handleAction("resume");
            }
            break;
          case "s":
            e.preventDefault();
            if (patternStatus?.state !== "IDLE") {
              setConfirmDialog("stop");
            }
            break;
          case "r":
            e.preventDefault();
            if (patternStatus?.state === "PAUSED") {
              handleAction("resume");
            }
            break;
        }
      }
    };

    if (shortcuts) {
      window.addEventListener("keydown", handleKeyPress);
      return () => window.removeEventListener("keydown", handleKeyPress);
    }
  }, [shortcuts, patternStatus]);

  const handleAction = async (action: PatternControlAction) => {
    if (!patternId && !patternStatus) {
      setError("No active pattern to control");
      return;
    }

    setLoading(true);
    setError(null);
    setConfirmDialog(null);

    try {
      const response = await searchService.controlPattern(action, patternId);
      if (response.success) {
        onActionComplete?.(action, true);
      } else {
        setError(`Failed to ${action} pattern`);
        onActionComplete?.(action, false);
      }
    } catch (err) {
      const errorMsg =
        err instanceof Error ? err.message : `Failed to ${action} pattern`;
      setError(errorMsg);
      onActionComplete?.(action, false);
    } finally {
      setLoading(false);
    }
  };

  const handleExport = async () => {
    if (!patternId && !patternStatus) {
      setError("No active pattern to export");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const content = await searchService.exportPattern(patternId, "qgc");
      const blob = new Blob([content], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `search_pattern_${patternId || "latest"}.wpl`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to export pattern");
    } finally {
      setLoading(false);
    }
  };

  const getButtonState = () => {
    if (!patternStatus) {
      return {
        canStart: false,
        canPause: false,
        canResume: false,
        canStop: false,
      };
    }

    return {
      canStart: patternStatus.state === "IDLE",
      canPause: patternStatus.state === "EXECUTING",
      canResume: patternStatus.state === "PAUSED",
      canStop:
        patternStatus.state !== "IDLE" && patternStatus.state !== "COMPLETED",
    };
  };

  const buttonState = getButtonState();

  return (
    <>
      <Card>
        <CardContent>
          <Stack spacing={3}>
            <Box
              sx={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
              }}
            >
              <Typography variant="h6">Pattern Controls</Typography>
              <Tooltip title="Toggle keyboard shortcuts">
                <IconButton
                  size="small"
                  onClick={() => setShortcuts(!shortcuts)}
                  color={shortcuts ? "primary" : "default"}
                >
                  <Keyboard />
                </IconButton>
              </Tooltip>
            </Box>

            {error && (
              <Alert severity="error" onClose={() => setError(null)}>
                {error}
              </Alert>
            )}

            <Stack direction="row" spacing={2}>
              {buttonState.canStart && (
                <Button
                  variant="contained"
                  color="primary"
                  startIcon={<PlayArrow />}
                  onClick={() => handleAction("resume")}
                  disabled={loading}
                  fullWidth
                >
                  Start Pattern
                </Button>
              )}

              {buttonState.canPause && (
                <Button
                  variant="contained"
                  color="warning"
                  startIcon={<Pause />}
                  onClick={() => handleAction("pause")}
                  disabled={loading}
                  fullWidth
                >
                  Pause
                </Button>
              )}

              {buttonState.canResume && (
                <Button
                  variant="contained"
                  color="success"
                  startIcon={<RestartAlt />}
                  onClick={() => handleAction("resume")}
                  disabled={loading}
                  fullWidth
                >
                  Resume
                </Button>
              )}

              {buttonState.canStop && (
                <Button
                  variant="outlined"
                  color="error"
                  startIcon={<Stop />}
                  onClick={() => setConfirmDialog("stop")}
                  disabled={loading}
                  fullWidth
                >
                  Stop
                </Button>
              )}
            </Stack>

            <Stack direction="row" spacing={2}>
              <Button
                variant="outlined"
                startIcon={<GetApp />}
                onClick={handleExport}
                disabled={loading || !patternStatus}
                fullWidth
              >
                Export Pattern
              </Button>

              <Button
                variant="outlined"
                startIcon={<Upload />}
                disabled
                fullWidth
              >
                Import Pattern
              </Button>
            </Stack>

            {shortcuts && (
              <Box
                sx={{
                  p: 2,
                  bgcolor: "grey.100",
                  borderRadius: 1,
                  border: "1px solid",
                  borderColor: "grey.300",
                }}
              >
                <Typography variant="subtitle2" gutterBottom>
                  Keyboard Shortcuts Active
                </Typography>
                <Typography variant="caption" component="div">
                  • Ctrl+P: Pause/Resume
                </Typography>
                <Typography variant="caption" component="div">
                  • Ctrl+S: Stop Pattern
                </Typography>
                <Typography variant="caption" component="div">
                  • Ctrl+R: Resume Pattern
                </Typography>
              </Box>
            )}

            {patternStatus && (
              <Box
                sx={{
                  p: 1,
                  bgcolor: "grey.50",
                  borderRadius: 1,
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                }}
              >
                <Typography variant="caption" color="text.secondary">
                  Pattern ID: {patternStatus.pattern_id}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  State: {patternStatus.state}
                </Typography>
              </Box>
            )}
          </Stack>
        </CardContent>
      </Card>

      <Dialog
        open={confirmDialog !== null}
        onClose={() => setConfirmDialog(null)}
      >
        <DialogTitle>
          {confirmDialog === "stop" ? "Stop Search Pattern?" : "Confirm Action"}
        </DialogTitle>
        <DialogContent>
          <DialogContentText>
            {confirmDialog === "stop" && (
              <>
                Are you sure you want to stop the search pattern? This will
                reset the pattern progress and return the drone to idle state.
                <br />
                <br />
                Current progress: {patternStatus?.progress_percent?.toFixed(1)}%
                ({patternStatus?.completed_waypoints} of{" "}
                {patternStatus?.total_waypoints} waypoints)
              </>
            )}
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmDialog(null)} color="primary">
            Cancel
          </Button>
          <Button
            onClick={() => confirmDialog && handleAction(confirmDialog)}
            color="error"
            variant="contained"
          >
            {confirmDialog === "stop" ? "Stop Pattern" : "Confirm"}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};

export default PatternControls;
