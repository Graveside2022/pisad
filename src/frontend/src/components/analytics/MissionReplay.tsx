import React, { useEffect, useState, useRef, useCallback } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  IconButton,
  Slider,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Paper,
  Chip,
  Alert,
  CircularProgress,
} from '@mui/material';
import Grid from '@mui/material/Grid';
import {
  PlayArrow,
  Pause,
  Stop,
  SkipPrevious,
  SkipNext,
  Speed,
  FlightTakeoff,
  Sensors,
  Timeline as TimelineIcon,
} from '@mui/icons-material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as ChartTooltip, ResponsiveContainer } from 'recharts';

interface ReplayEvent {
  timestamp: string;
  telemetry: {
    latitude: number;
    longitude: number;
    altitude: number;
    groundspeed: number;
    heading: number;
    rssi_dbm: number;
    snr_db: number;
    beacon_detected: boolean;
    system_state: string;
    battery_percent: number;
  };
  signal_detections: Array<{
    frequency: number;
    rssi: number;
    confidence: number;
  }>;
  state_changes: Array<{
    from_state: string;
    to_state: string;
    trigger: string;
  }>;
}

interface ReplayStatus {
  mission_id: string;
  state: 'stopped' | 'playing' | 'paused';
  speed: number;
  position: number;
  total: number;
  progress: number;
}

interface MissionReplayProps {
  missionId: string;
  onEventUpdate?: (event: ReplayEvent) => void;
}

const playbackSpeeds = [
  { value: 0.25, label: '0.25x' },
  { value: 0.5, label: '0.5x' },
  { value: 1, label: '1x' },
  { value: 2, label: '2x' },
  { value: 4, label: '4x' },
  { value: 10, label: '10x' },
];

const stateColors: Record<string, string> = {
  IDLE: '#9e9e9e',
  SEARCHING: '#2196f3',
  APPROACHING: '#ff9800',
  HOVERING: '#4caf50',
  RETURNING: '#9c27b0',
  LANDED: '#607d8b',
};

export const MissionReplay: React.FC<MissionReplayProps> = ({ missionId, onEventUpdate }) => {
  const [status, setStatus] = useState<ReplayStatus | null>(null);
  const [currentEvent, setCurrentEvent] = useState<ReplayEvent | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [rssiHistory, setRssiHistory] = useState<Array<{ time: number; rssi: number }>>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const rssiHistoryRef = useRef<Array<{ time: number; rssi: number }>>([]);

  // Initialize WebSocket connection
  useEffect(() => {
    const connectWebSocket = () => {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const ws = new WebSocket(`${protocol}//${window.location.host}/ws/replay`);

      ws.onopen = () => {
        console.log('WebSocket connected for replay');
        ws.send(JSON.stringify({ type: 'subscribe', mission_id: missionId }));
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'replay_event') {
            handleReplayEvent(data);
          } else if (data.type === 'status_update') {
            setStatus(data.status);
          }
        } catch (err) {
          console.error('Error parsing WebSocket message:', err);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        setError('Connection error');
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
      };

      wsRef.current = ws;
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [missionId]);

  // Load initial replay data
  useEffect(() => {
    const loadReplayData = async () => {
      try {
        const response = await fetch(`/api/analytics/replay/${missionId}`);
        if (!response.ok) {
          throw new Error('Failed to load replay data');
        }
        const data = await response.json();
        setStatus(data.status);
        setLoading(false);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load replay');
        setLoading(false);
      }
    };

    loadReplayData();
  }, [missionId]);

  const handleReplayEvent = useCallback((event: any) => {
    const replayEvent: ReplayEvent = {
      timestamp: event.timestamp,
      telemetry: event.telemetry,
      signal_detections: event.signal_detections || [],
      state_changes: event.state_changes || [],
    };

    setCurrentEvent(replayEvent);

    // Update RSSI history
    if (replayEvent.telemetry.rssi_dbm) {
      const time = new Date(replayEvent.timestamp).getTime() / 1000;
      const newPoint = { time, rssi: replayEvent.telemetry.rssi_dbm };

      rssiHistoryRef.current = [...rssiHistoryRef.current.slice(-99), newPoint];
      setRssiHistory(rssiHistoryRef.current);
    }

    // Update status
    if (event.position !== undefined && event.total !== undefined) {
      setStatus(prev => prev ? {
        ...prev,
        position: event.position,
        total: event.total,
        progress: (event.position / event.total) * 100,
      } : null);
    }

    // Notify parent component
    if (onEventUpdate) {
      onEventUpdate(replayEvent);
    }
  }, [onEventUpdate]);

  const sendControl = async (action: string, params?: any) => {
    try {
      const response = await fetch(`/api/analytics/replay/${missionId}/control`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, ...params }),
      });

      if (!response.ok) {
        throw new Error('Control command failed');
      }

      const updatedStatus = await response.json();
      setStatus(updatedStatus);
    } catch (err) {
      console.error('Control error:', err);
    }
  };

  const handlePlay = () => sendControl('play');
  const handlePause = () => sendControl('pause');
  const handleStop = () => {
    sendControl('stop');
    rssiHistoryRef.current = [];
    setRssiHistory([]);
    setCurrentEvent(null);
  };

  const handleSeek = (_: Event, value: number | number[]) => {
    const position = Array.isArray(value) ? value[0] : value;
    sendControl('seek', { position });
  };

  const handleSpeedChange = (event: any) => {
    sendControl('play', { speed: event.target.value });
  };

  const handleSkipBackward = () => {
    if (status && status.position > 0) {
      sendControl('seek', { position: Math.max(0, status.position - 10) });
    }
  };

  const handleSkipForward = () => {
    if (status && status.position < status.total - 1) {
      sendControl('seek', { position: Math.min(status.total - 1, status.position + 10) });
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ m: 2 }}>
        {error}
      </Alert>
    );
  }

  if (!status) {
    return (
      <Alert severity="info" sx={{ m: 2 }}>
        No replay data available
      </Alert>
    );
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Mission Replay
      </Typography>

      {/* Playback Controls */}
      <Paper sx={{ p: 2, mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
          <IconButton onClick={handleSkipBackward} disabled={status.state === 'playing'}>
            <SkipPrevious />
          </IconButton>

          {status.state === 'playing' ? (
            <IconButton onClick={handlePause} color="primary">
              <Pause />
            </IconButton>
          ) : (
            <IconButton onClick={handlePlay} color="primary">
              <PlayArrow />
            </IconButton>
          )}

          <IconButton onClick={handleStop} color="error">
            <Stop />
          </IconButton>

          <IconButton onClick={handleSkipForward} disabled={status.state === 'playing'}>
            <SkipNext />
          </IconButton>

          <FormControl sx={{ minWidth: 100 }}>
            <InputLabel>Speed</InputLabel>
            <Select
              value={status.speed}
              onChange={handleSpeedChange}
              label="Speed"
              size="small"
            >
              {playbackSpeeds.map(speed => (
                <MenuItem key={speed.value} value={speed.value}>
                  {speed.label}
                </MenuItem>
              ))}
            </Select>
          </FormControl>

          <Chip
            icon={<Speed />}
            label={status.state.toUpperCase()}
            color={status.state === 'playing' ? 'success' : status.state === 'paused' ? 'warning' : 'default'}
          />
        </Box>

        {/* Timeline Scrubber */}
        <Box sx={{ px: 2 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Timeline
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Typography variant="body2">
              {formatTime((status.position / status.total) * 3600)}
            </Typography>
            <Slider
              value={status.position}
              min={0}
              max={status.total - 1}
              onChange={handleSeek}
              disabled={status.state === 'playing'}
              sx={{ flex: 1 }}
            />
            <Typography variant="body2">
              {formatTime(3600)}
            </Typography>
          </Box>
          <Typography variant="body2" align="center" color="text.secondary">
            Frame {status.position} / {status.total} ({status.progress.toFixed(1)}%)
          </Typography>
        </Box>
      </Paper>

      {/* Current State and Telemetry */}
      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid size={{ xs: 12, md: 4 }}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <FlightTakeoff sx={{ mr: 1 }} />
                <Typography variant="h6">Flight Data</Typography>
              </Box>
              {currentEvent && (
                <>
                  <Typography variant="body2" color="text.secondary">
                    Position
                  </Typography>
                  <Typography variant="body1" gutterBottom>
                    {currentEvent.telemetry.latitude.toFixed(6)}, {currentEvent.telemetry.longitude.toFixed(6)}
                  </Typography>

                  <Typography variant="body2" color="text.secondary">
                    Altitude
                  </Typography>
                  <Typography variant="body1" gutterBottom>
                    {currentEvent.telemetry.altitude.toFixed(1)} m
                  </Typography>

                  <Typography variant="body2" color="text.secondary">
                    Speed / Heading
                  </Typography>
                  <Typography variant="body1" gutterBottom>
                    {currentEvent.telemetry.groundspeed.toFixed(1)} m/s @ {currentEvent.telemetry.heading.toFixed(0)}°
                  </Typography>

                  <Typography variant="body2" color="text.secondary">
                    Battery
                  </Typography>
                  <Typography variant="body1">
                    {currentEvent.telemetry.battery_percent.toFixed(0)}%
                  </Typography>
                </>
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid size={{ xs: 12, md: 4 }}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <Sensors sx={{ mr: 1 }} />
                <Typography variant="h6">Signal Data</Typography>
              </Box>
              {currentEvent && (
                <>
                  <Typography variant="body2" color="text.secondary">
                    RSSI
                  </Typography>
                  <Typography variant="body1" gutterBottom>
                    {currentEvent.telemetry.rssi_dbm.toFixed(1)} dBm
                  </Typography>

                  <Typography variant="body2" color="text.secondary">
                    SNR
                  </Typography>
                  <Typography variant="body1" gutterBottom>
                    {currentEvent.telemetry.snr_db.toFixed(1)} dB
                  </Typography>

                  <Typography variant="body2" color="text.secondary">
                    Beacon Status
                  </Typography>
                  <Chip
                    label={currentEvent.telemetry.beacon_detected ? 'DETECTED' : 'NOT DETECTED'}
                    color={currentEvent.telemetry.beacon_detected ? 'success' : 'default'}
                    size="small"
                    sx={{ mt: 0.5 }}
                  />

                  {currentEvent.signal_detections.length > 0 && (
                    <>
                      <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                        Detections
                      </Typography>
                      {currentEvent.signal_detections.map((det, idx) => (
                        <Typography key={idx} variant="body2">
                          {(det.frequency / 1e6).toFixed(3)} MHz @ {det.rssi.toFixed(1)} dBm
                        </Typography>
                      ))}
                    </>
                  )}
                </>
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid size={{ xs: 12, md: 4 }}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
                <TimelineIcon sx={{ mr: 1 }} />
                <Typography variant="h6">System State</Typography>
              </Box>
              {currentEvent && (
                <>
                  <Chip
                    label={currentEvent.telemetry.system_state}
                    sx={{
                      backgroundColor: stateColors[currentEvent.telemetry.system_state] || '#grey',
                      color: 'white',
                      fontWeight: 'bold',
                      mb: 2,
                    }}
                  />

                  {currentEvent.state_changes.length > 0 && (
                    <>
                      <Typography variant="body2" color="text.secondary" gutterBottom>
                        State Transition
                      </Typography>
                      {currentEvent.state_changes.map((change, idx) => (
                        <Box key={idx} sx={{ mb: 1 }}>
                          <Typography variant="body2">
                            {change.from_state} → {change.to_state}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            Trigger: {change.trigger}
                          </Typography>
                        </Box>
                      ))}
                    </>
                  )}

                  <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
                    Timestamp
                  </Typography>
                  <Typography variant="body2">
                    {new Date(currentEvent.timestamp).toLocaleTimeString()}
                  </Typography>
                </>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* RSSI Graph */}
      <Paper sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          RSSI History
        </Typography>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={rssiHistory}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="time"
              domain={['dataMin', 'dataMax']}
              type="number"
              tickFormatter={(value) => new Date(value * 1000).toLocaleTimeString()}
            />
            <YAxis domain={[-100, -40]} />
            <ChartTooltip
              labelFormatter={(value) => new Date(value * 1000).toLocaleTimeString()}
              formatter={(value: any) => `${value} dBm`}
            />
            <Line
              type="monotone"
              dataKey="rssi"
              stroke="#1976d2"
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </Paper>
    </Box>
  );
};
