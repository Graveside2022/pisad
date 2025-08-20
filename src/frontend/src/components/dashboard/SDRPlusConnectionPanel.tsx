import { Card, CardContent, Typography, Box, Button, Chip, Grid, LinearProgress, Tooltip, IconButton } from '@mui/material';
import { useState, useEffect } from 'react';
import { useWebSocket } from '../../hooks/useWebSocket';
import SyncIcon from '@mui/icons-material/Sync';
import InfoIcon from '@mui/icons-material/Info';
import SignalCellularAltIcon from '@mui/icons-material/SignalCellularAlt';
import RadioIcon from '@mui/icons-material/Radio';

interface SDRPlusConnectionData {
  status: 'connected' | 'disconnected' | 'connecting';
  latency?: number;
  lastSeen?: string;
  groundRssi?: number;
  droneRssi?: number;
  frequency?: number;
  signalConfidence?: number;
  processingMode?: 'basic' | 'professional';
  asvAnalyzerStatus?: {
    gpActive: boolean;
    vorActive: boolean;
    llzActive: boolean;
  };
}

interface SDRPlusConnectionPanelProps {
  // Future props for configuration
  onFrequencySync?: (frequency: number) => void;
  onModeToggle?: (mode: 'basic' | 'professional') => void;
}

function SDRPlusConnectionPanel({ onFrequencySync, onModeToggle }: SDRPlusConnectionPanelProps) {
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting'>('disconnected');
  const [healthMetrics, setHealthMetrics] = useState<{ latency?: number; lastSeen?: string }>({});
  const [signalData, setSignalData] = useState<{
    groundRssi?: number;
    droneRssi?: number;
    frequency?: number;
    signalConfidence?: number;
    processingMode?: 'basic' | 'professional';
    asvAnalyzerStatus?: {
      gpActive: boolean;
      vorActive: boolean;
      llzActive: boolean;
    };
  }>({});
  const { addMessageHandler } = useWebSocket();

  // WebSocket message handling for SDR++ connection events
  useEffect(() => {
    const cleanup = addMessageHandler((message) => {
      if (message.type === 'sdrpp_connection') {
        const data = message.data as SDRPlusConnectionData;
        setConnectionStatus(data.status);
        setHealthMetrics({
          latency: data.latency,
          lastSeen: data.lastSeen
        });
        setSignalData({
          groundRssi: data.groundRssi,
          droneRssi: data.droneRssi,
          frequency: data.frequency,
          signalConfidence: data.signalConfidence,
          processingMode: data.processingMode,
          asvAnalyzerStatus: data.asvAnalyzerStatus
        });
      }
    });
    return cleanup;
  }, [addMessageHandler]);

  const handleReconnect = () => {
    setConnectionStatus('connecting');
    // TODO: Send reconnection command to backend
    console.log('Reconnect requested');
  };

  const handleFrequencySync = () => {
    if (signalData.frequency && onFrequencySync) {
      onFrequencySync(signalData.frequency);
    }
  };

  const handleModeToggle = () => {
    const newMode = signalData.processingMode === 'professional' ? 'basic' : 'professional';
    if (onModeToggle) {
      onModeToggle(newMode);
    }
  };

  const getSignalStrengthColor = (rssi?: number): 'success' | 'warning' | 'error' => {
    if (!rssi) return 'error';
    if (rssi > -60) return 'success';
    if (rssi > -80) return 'warning';
    return 'error';
  };

  const getConfidenceColor = (confidence?: number): 'success' | 'warning' | 'error' => {
    if (!confidence) return 'error';
    if (confidence > 70) return 'success';
    if (confidence > 30) return 'warning';
    return 'error';
  };

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <RadioIcon sx={{ mr: 1 }} />
          <Typography variant="h6">
            Professional SAR Platform
          </Typography>
          <Tooltip title="SDR++ Plugin + ASV .NET Integration">
            <IconButton size="small" sx={{ ml: 1 }}>
              <InfoIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Box>

        {/* Connection Status */}
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={12} md={6}>
            <Card variant="outlined" sx={{ p: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                SDR++ Plugin Status
              </Typography>
              <Chip
                label={connectionStatus === 'connected' ? '🟢 Connected' :
                       connectionStatus === 'connecting' ? '🟡 Connecting' : '🔴 Disconnected'}
                color={connectionStatus === 'connected' ? 'success' : 'error'}
                variant="outlined"
                sx={{ mb: 1 }}
              />
              {healthMetrics.latency && (
                <Typography variant="body2" color="text.secondary">
                  Latency: {healthMetrics.latency}ms
                </Typography>
              )}
            </Card>
          </Grid>
          <Grid item xs={12} md={6}>
            <Card variant="outlined" sx={{ p: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                ASV .NET Processing
              </Typography>
              <Chip
                label={signalData.processingMode === 'professional' ? '🟢 Professional ±2°' : '🟡 Basic ±10°'}
                color={signalData.processingMode === 'professional' ? 'success' : 'warning'}
                variant="outlined"
                sx={{ mb: 1 }}
              />
              <Button size="small" onClick={handleModeToggle} sx={{ ml: 1 }}>
                Toggle Mode
              </Button>
            </Card>
          </Grid>
        </Grid>

        {/* TCP Bridge Signal Monitoring */}
        <Typography variant="subtitle2" gutterBottom>
          TCP Bridge Signal Monitoring
        </Typography>
        <Grid container spacing={2} sx={{ mb: 3 }}>
          <Grid item xs={6}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="body2" color="text.secondary">Ground Station Monitor</Typography>
              <Chip
                label={signalData.groundRssi ? `${signalData.groundRssi} dBm` : 'No Signal'}
                color={getSignalStrengthColor(signalData.groundRssi)}
                size="small"
              />
            </Box>
          </Grid>
          <Grid item xs={6}>
            <Box sx={{ textAlign: 'center' }}>
              <Typography variant="body2" color="text.secondary">HackRF Signal (Drone)</Typography>
              <Chip
                label={signalData.droneRssi ? `${signalData.droneRssi} dBm` : 'No Signal'}
                color={getSignalStrengthColor(signalData.droneRssi)}
                size="small"
              />
            </Box>
          </Grid>
        </Grid>

        {/* ASV Analyzer Status */}
        {signalData.asvAnalyzerStatus && (
          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle2" gutterBottom>
              ASV Multi-Analyzer Status
            </Typography>
            <Grid container spacing={1}>
              <Grid item>
                <Chip
                  label="GP (406 MHz)"
                  color={signalData.asvAnalyzerStatus.gpActive ? 'success' : 'default'}
                  size="small"
                />
              </Grid>
              <Grid item>
                <Chip
                  label="VOR (108-118 MHz)"
                  color={signalData.asvAnalyzerStatus.vorActive ? 'success' : 'default'}
                  size="small"
                />
              </Grid>
              <Grid item>
                <Chip
                  label="LLZ (108-112 MHz)"
                  color={signalData.asvAnalyzerStatus.llzActive ? 'success' : 'default'}
                  size="small"
                />
              </Grid>
            </Grid>
          </Box>
        )}

        {/* Signal Confidence */}
        {signalData.signalConfidence !== undefined && (
          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle2" gutterBottom>
              ASV Signal Confidence: {signalData.signalConfidence}%
            </Typography>
            <LinearProgress
              variant="determinate"
              value={signalData.signalConfidence}
              color={getConfidenceColor(signalData.signalConfidence)}
              sx={{ height: 8, borderRadius: 4 }}
            />
          </Box>
        )}

        {/* Control Buttons */}
        <Grid container spacing={1}>
          <Grid item>
            <Button
              variant="outlined"
              onClick={handleReconnect}
              disabled={connectionStatus === 'connecting'}
              startIcon={<SyncIcon />}
            >
              Reconnect
            </Button>
          </Grid>
          <Grid item>
            <Button
              variant="outlined"
              onClick={handleFrequencySync}
              disabled={!signalData.frequency}
              startIcon={<SignalCellularAltIcon />}
            >
              Sync Frequency
            </Button>
          </Grid>
        </Grid>

        {signalData.frequency && (
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
            Current Frequency: {(signalData.frequency / 1e6).toFixed(3)} MHz
          </Typography>
        )}
      </CardContent>
    </Card>
  );
}

export default SDRPlusConnectionPanel;
