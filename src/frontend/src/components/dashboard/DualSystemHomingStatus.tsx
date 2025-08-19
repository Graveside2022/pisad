import { Card, CardContent, Typography, Box, Chip, Grid, LinearProgress } from '@mui/material';
import { useState, useEffect } from 'react';
import { useWebSocket } from '../../hooks/useWebSocket';

interface HomingStatusData {
  activeSource: 'ground' | 'drone' | 'coordinated';
  homingAuthority: 'ground_override' | 'drone_authority' | 'coordinated';
  latencyMs: number;
  reliability: number; // 0-100 percentage
  fallbackActive: boolean;
  lastUpdate: string;
}

interface DualSystemHomingStatusProps {
  // Future props for configuration
}

function DualSystemHomingStatus(_props: DualSystemHomingStatusProps) {
  const [homingStatus, setHomingStatus] = useState<HomingStatusData>({
    activeSource: 'drone',
    homingAuthority: 'drone_authority',
    latencyMs: 0,
    reliability: 0,
    fallbackActive: false,
    lastUpdate: new Date().toISOString()
  });
  const { addMessageHandler } = useWebSocket();

  // WebSocket message handling for homing status updates
  useEffect(() => {
    const cleanup = addMessageHandler((message) => {
      if (message.type === 'dual_homing_status') {
        const data = message.data as HomingStatusData;
        setHomingStatus(data);
      }
    });
    return cleanup;
  }, [addMessageHandler]);

  const getActiveSourceColor = (source: string) => {
    switch (source) {
      case 'ground': return 'primary';
      case 'drone': return 'success';
      case 'coordinated': return 'info';
      default: return 'default';
    }
  };

  const getAuthorityColor = (authority: string) => {
    switch (authority) {
      case 'ground_override': return 'warning';
      case 'drone_authority': return 'success';
      case 'coordinated': return 'info';
      default: return 'default';
    }
  };

  const getReliabilityColor = (reliability: number) => {
    if (reliability >= 90) return 'success';
    if (reliability >= 70) return 'warning';
    return 'error';
  };

  const formatLatency = (latency: number) => `${latency.toFixed(1)}ms`;

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Dual-System Homing Status
        </Typography>

        <Grid container spacing={2}>
          <Grid item xs={6}>
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Active Source
              </Typography>
              <Chip
                label={homingStatus.activeSource.charAt(0).toUpperCase() + homingStatus.activeSource.slice(1)}
                color={getActiveSourceColor(homingStatus.activeSource)}
                variant="filled"
              />
              {homingStatus.fallbackActive && (
                <Chip
                  label="Fallback Active"
                  color="error"
                  size="small"
                  sx={{ ml: 1 }}
                />
              )}
            </Box>
          </Grid>

          <Grid item xs={6}>
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Homing Authority
              </Typography>
              <Chip
                label={homingStatus.homingAuthority.replace(/_/g, ' ').toUpperCase()}
                color={getAuthorityColor(homingStatus.homingAuthority)}
                variant="outlined"
              />
            </Box>
          </Grid>
        </Grid>

        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Performance
          </Typography>
          <Typography variant="body2" gutterBottom>
            Latency: {formatLatency(homingStatus.latencyMs)}
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography variant="body2">
              Reliability: {homingStatus.reliability.toFixed(0)}%
            </Typography>
            <LinearProgress
              variant="determinate"
              value={homingStatus.reliability}
              color={getReliabilityColor(homingStatus.reliability)}
              sx={{ flexGrow: 1, height: 8, borderRadius: 4 }}
            />
          </Box>
        </Box>

        <Typography variant="caption" color="text.secondary">
          Last update: {new Date(homingStatus.lastUpdate).toLocaleTimeString()}
        </Typography>
      </CardContent>
    </Card>
  );
}

export default DualSystemHomingStatus;
