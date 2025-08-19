import { Card, CardContent, Typography, Box, Grid, LinearProgress, Chip } from '@mui/material';
import { useState, useEffect } from 'react';
import { useWebSocket } from '../../hooks/useWebSocket';

interface SignalData {
  rssi: number;
  snr: number;
  source: 'ground' | 'drone';
  timestamp: string;
}

interface GroundSignalQualityProps {
  // Future props for configuration
}

function GroundSignalQuality(_props: GroundSignalQualityProps) {
  const [groundSignal, setGroundSignal] = useState<SignalData | null>(null);
  const [droneSignal, setDroneSignal] = useState<SignalData | null>(null);
  const { addMessageHandler } = useWebSocket();

  // WebSocket message handling for signal quality updates
  useEffect(() => {
    const cleanup = addMessageHandler((message) => {
      if (message.type === 'ground_signal_quality') {
        const data = message.data as SignalData;
        if (data.source === 'ground') {
          setGroundSignal(data);
        } else if (data.source === 'drone') {
          setDroneSignal(data);
        }
      }
    });
    return cleanup;
  }, [addMessageHandler]);

  // Calculate signal differential
  const getSignalDifferential = () => {
    if (groundSignal && droneSignal) {
      return groundSignal.rssi - droneSignal.rssi;
    }
    return 0;
  };

  const formatRSSI = (rssi: number) => `${rssi.toFixed(1)} dBm`;
  const formatSNR = (snr: number) => `${snr.toFixed(1)} dB`;

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Signal Quality Comparison
        </Typography>

        <Grid container spacing={2}>
          <Grid size={6}>
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Ground SDR++
              </Typography>
              {groundSignal ? (
                <>
                  <Typography variant="body2">
                    RSSI: {formatRSSI(groundSignal.rssi)}
                  </Typography>
                  <Typography variant="body2">
                    SNR: {formatSNR(groundSignal.snr)}
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={Math.max(0, Math.min(100, (groundSignal.rssi + 100) * 2))}
                    sx={{ mt: 1 }}
                  />
                </>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No ground signal data
                </Typography>
              )}
            </Box>
          </Grid>

          <Grid size={6}>
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" gutterBottom>
                Drone HackRF
              </Typography>
              {droneSignal ? (
                <>
                  <Typography variant="body2">
                    RSSI: {formatRSSI(droneSignal.rssi)}
                  </Typography>
                  <Typography variant="body2">
                    SNR: {formatSNR(droneSignal.snr)}
                  </Typography>
                  <LinearProgress
                    variant="determinate"
                    value={Math.max(0, Math.min(100, (droneSignal.rssi + 100) * 2))}
                    sx={{ mt: 1 }}
                  />
                </>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No drone signal data
                </Typography>
              )}
            </Box>
          </Grid>
        </Grid>

        <Box sx={{ mt: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Signal Differential
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Chip
              label={`${getSignalDifferential() > 0 ? '+' : ''}${getSignalDifferential().toFixed(1)} dB`}
              color={getSignalDifferential() > 0 ? 'success' : getSignalDifferential() < 0 ? 'warning' : 'default'}
              size="small"
            />
            <Typography variant="body2" color="text.secondary">
              {getSignalDifferential() > 0 ? 'Ground stronger' :
               getSignalDifferential() < 0 ? 'Drone stronger' : 'Equal strength'}
            </Typography>
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
}

export default GroundSignalQuality;
