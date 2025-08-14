import React, { useEffect, useState } from "react";
import {
  Box,
  Card,
  CardContent,
  Typography,
  LinearProgress,
  Stack,
  Chip,
  Grid,
} from "@mui/material";
import { FlightTakeoff, Speed, Timer, LocationOn } from "@mui/icons-material";
import { type PatternStatus } from "../../types/search";
import searchService from "../../services/search";
import { useWebSocket } from "../../hooks/useWebSocket";

interface SearchProgressProps {
  patternId?: string;
  onStatusChange?: (status: PatternStatus) => void;
}

const SearchProgress: React.FC<SearchProgressProps> = ({
  patternId,
  onStatusChange,
}) => {
  const [status, setStatus] = useState<PatternStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { lastMessage } = useWebSocket();

  useEffect(() => {
    loadStatus();
    const interval = setInterval(loadStatus, 5000);
    return () => clearInterval(interval);
  }, [patternId]);

  useEffect(() => {
    if (lastMessage && lastMessage.type === "pattern_update") {
      const data = lastMessage.data as any;
      if (!patternId || data.pattern_id === patternId) {
        const newStatus = {
          pattern_id: data.pattern_id,
          state: data.state,
          progress_percent: data.progress_percent,
          completed_waypoints: data.completed_waypoints,
          total_waypoints: data.total_waypoints,
          current_waypoint: data.completed_waypoints + 1,
          estimated_time_remaining: data.estimated_time_remaining,
        };
        setStatus(newStatus);
        onStatusChange?.(newStatus);
      }
    }
  }, [lastMessage, patternId, onStatusChange]);

  const loadStatus = async () => {
    if (loading) return;

    setLoading(true);
    setError(null);
    try {
      const data = await searchService.getPatternStatus(patternId);
      setStatus(data);
      onStatusChange?.(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load status");
    } finally {
      setLoading(false);
    }
  };

  const formatTime = (seconds: number): string => {
    if (seconds < 60) {
      return `${Math.round(seconds)}s`;
    }
    const minutes = Math.floor(seconds / 60);
    const secs = Math.round(seconds % 60);
    if (minutes < 60) {
      return `${minutes}m ${secs}s`;
    }
    const hours = Math.floor(minutes / 60);
    const mins = minutes % 60;
    return `${hours}h ${mins}m`;
  };

  const getStateColor = (
    state: string,
  ): "default" | "primary" | "secondary" | "success" | "warning" | "error" => {
    switch (state) {
      case "EXECUTING":
        return "primary";
      case "PAUSED":
        return "warning";
      case "COMPLETED":
        return "success";
      case "IDLE":
        return "default";
      default:
        return "default";
    }
  };

  if (error && !status) {
    return (
      <Card>
        <CardContent>
          <Typography color="error">{error}</Typography>
        </CardContent>
      </Card>
    );
  }

  if (!status) {
    return (
      <Card>
        <CardContent>
          <Typography>No active search pattern</Typography>
        </CardContent>
      </Card>
    );
  }

  const progressValue = status.progress_percent || 0;
  const isActive = status.state === "EXECUTING";

  return (
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
            <Typography variant="h6">Search Progress</Typography>
            <Chip
              label={status.state}
              color={getStateColor(status.state)}
              size="small"
            />
          </Box>

          <Box>
            <Box
              sx={{ display: "flex", justifyContent: "space-between", mb: 1 }}
            >
              <Typography variant="body2" color="text.secondary">
                Waypoint {status.current_waypoint} of {status.total_waypoints}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {progressValue.toFixed(1)}%
              </Typography>
            </Box>
            <LinearProgress
              variant={isActive ? "determinate" : "determinate"}
              value={progressValue}
              sx={{ height: 8, borderRadius: 1 }}
            />
          </Box>

          <Grid container spacing={2}>
            <Grid size={6}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <LocationOn fontSize="small" color="action" />
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Current Waypoint
                  </Typography>
                  <Typography variant="body2">
                    #{status.current_waypoint}
                  </Typography>
                </Box>
              </Box>
            </Grid>

            <Grid size={6}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <FlightTakeoff fontSize="small" color="action" />
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Completed
                  </Typography>
                  <Typography variant="body2">
                    {status.completed_waypoints} waypoints
                  </Typography>
                </Box>
              </Box>
            </Grid>

            <Grid size={6}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <Timer fontSize="small" color="action" />
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Time Remaining
                  </Typography>
                  <Typography variant="body2">
                    {status.estimated_time_remaining > 0
                      ? formatTime(status.estimated_time_remaining)
                      : "Calculating..."}
                  </Typography>
                </Box>
              </Box>
            </Grid>

            <Grid size={6}>
              <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                <Speed fontSize="small" color="action" />
                <Box>
                  <Typography variant="caption" color="text.secondary">
                    Progress Rate
                  </Typography>
                  <Typography variant="body2">
                    {status.total_waypoints > 0 &&
                    status.estimated_time_remaining > 0
                      ? `${((status.total_waypoints - status.completed_waypoints) / (status.estimated_time_remaining / 60)).toFixed(1)} wp/min`
                      : "N/A"}
                  </Typography>
                </Box>
              </Box>
            </Grid>
          </Grid>

          {isActive && (
            <Box
              sx={{
                p: 2,
                bgcolor: "primary.main",
                color: "primary.contrastText",
                borderRadius: 1,
                display: "flex",
                alignItems: "center",
                gap: 1,
              }}
            >
              <Box
                sx={{
                  width: 8,
                  height: 8,
                  bgcolor: "primary.contrastText",
                  borderRadius: "50%",
                  animation: "pulse 2s infinite",
                }}
              />
              <Typography variant="body2">
                Pattern actively executing...
              </Typography>
            </Box>
          )}

          {status.state === "PAUSED" && (
            <Box
              sx={{
                p: 2,
                bgcolor: "warning.light",
                borderRadius: 1,
              }}
            >
              <Typography variant="body2">
                Pattern paused at waypoint {status.current_waypoint}
              </Typography>
            </Box>
          )}

          {status.state === "COMPLETED" && (
            <Box
              sx={{
                p: 2,
                bgcolor: "success.light",
                borderRadius: 1,
              }}
            >
              <Typography variant="body2">
                Pattern completed successfully!
              </Typography>
            </Box>
          )}
        </Stack>
      </CardContent>

      <style>{`
        @keyframes pulse {
          0% { opacity: 1; }
          50% { opacity: 0.3; }
          100% { opacity: 1; }
        }
      `}</style>
    </Card>
  );
};

export default SearchProgress;
