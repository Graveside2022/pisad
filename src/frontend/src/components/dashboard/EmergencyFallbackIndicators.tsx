import { Card, CardContent, Typography, Box, Chip, Alert, LinearProgress } from '@mui/material';
import { useState, useEffect } from 'react';
import { useWebSocket } from '../../hooks/useWebSocket';

interface FallbackData {
  communicationStatus: 'connected' | 'degraded' | 'lost';
  fallbackMode: 'coordinated' | 'drone_only' | 'recovery';
  recoveryStatus: 'none' | 'attempting' | 'succeeded' | 'failed';
  timeInFallback: number; // seconds
  reconnectionAttempts: number;
  lastCommunication: string;
  emergencyActive: boolean;
}

interface EmergencyFallbackIndicatorsProps {
  // Future props for configuration
}

function EmergencyFallbackIndicators(_props: EmergencyFallbackIndicatorsProps) {
  const [fallbackData, setFallbackData] = useState<FallbackData>({
    communicationStatus: 'connected',
    fallbackMode: 'coordinated',
    recoveryStatus: 'none',
    timeInFallback: 0,
    reconnectionAttempts: 0,
    lastCommunication: new Date().toISOString(),
    emergencyActive: false
  });
  const { addMessageHandler } = useWebSocket();

  // WebSocket message handling for fallback status updates
  useEffect(() => {
    const cleanup = addMessageHandler((message) => {
      if (message.type === 'emergency_fallback') {
        const data = message.data as FallbackData;
        setFallbackData(data);
      }
    });
    return cleanup;
  }, [addMessageHandler]);

  const getCommStatusColor = (status: string) => {
    switch (status) {
      case 'connected': return 'success';
      case 'degraded': return 'warning';
      case 'lost': return 'error';
      default: return 'default';
    }
  };

  const getFallbackModeColor = (mode: string) => {
    switch (mode) {
      case 'coordinated': return 'success';
      case 'drone_only': return 'warning';
      case 'recovery': return 'info';
      default: return 'default';
    }
  };

  const getRecoveryStatusColor = (status: string) => {
    switch (status) {
      case 'succeeded': return 'success';
      case 'attempting': return 'info';
      case 'failed': return 'error';
      default: return 'default';
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Emergency Fallback Status
        </Typography>

        {fallbackData.emergencyActive && (
          <Alert severity="error" sx={{ mb: 2 }}>
            Emergency fallback active - Drone operating autonomously
          </Alert>
        )}

        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Communication Status
          </Typography>
          <Chip
            label={fallbackData.communicationStatus.toUpperCase()}
            color={getCommStatusColor(fallbackData.communicationStatus)}
            variant="filled"
          />
        </Box>

        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Fallback Mode
          </Typography>
          <Chip
            label={fallbackData.fallbackMode.replace('_', ' ').toUpperCase()}
            color={getFallbackModeColor(fallbackData.fallbackMode)}
            variant="outlined"
          />
          {fallbackData.timeInFallback > 0 && (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              Time in fallback: {formatTime(fallbackData.timeInFallback)}
            </Typography>
          )}
        </Box>

        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Recovery Status
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Chip
              label={fallbackData.recoveryStatus.toUpperCase()}
              color={getRecoveryStatusColor(fallbackData.recoveryStatus)}
              size="small"
            />
            {fallbackData.recoveryStatus === 'attempting' && (
              <LinearProgress sx={{ flexGrow: 1, height: 4 }} />
            )}
          </Box>
          {fallbackData.reconnectionAttempts > 0 && (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              Reconnection attempts: {fallbackData.reconnectionAttempts}
            </Typography>
          )}
        </Box>

        <Typography variant="caption" color="text.secondary">
          Last communication: {new Date(fallbackData.lastCommunication).toLocaleTimeString()}
        </Typography>
      </CardContent>
    </Card>
  );
}

export default EmergencyFallbackIndicators;
