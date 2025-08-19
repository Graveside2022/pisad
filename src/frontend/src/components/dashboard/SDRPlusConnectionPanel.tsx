import { Card, CardContent, Typography, Box, Button, Chip } from '@mui/material';
import { useState, useEffect } from 'react';
import { useWebSocket } from '../../hooks/useWebSocket';

interface SDRPlusConnectionData {
  status: 'connected' | 'disconnected' | 'connecting';
  latency?: number;
  lastSeen?: string;
}

interface SDRPlusConnectionPanelProps {
  // Future props for configuration
}

function SDRPlusConnectionPanel(_props: SDRPlusConnectionPanelProps) {
  const [connectionStatus, setConnectionStatus] = useState<'connected' | 'disconnected' | 'connecting'>('disconnected');
  const [healthMetrics, setHealthMetrics] = useState<{ latency?: number; lastSeen?: string }>({});
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
      }
    });
    return cleanup;
  }, [addMessageHandler]);

  const handleReconnect = () => {
    setConnectionStatus('connecting');
    // TODO: Send reconnection command to backend
    console.log('Reconnect requested');
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          SDR++ Connection
        </Typography>

        <Box sx={{ mb: 2 }}>
          <Chip
            label={connectionStatus === 'disconnected' ? 'Disconnected' : connectionStatus}
            color={connectionStatus === 'connected' ? 'success' : 'error'}
            variant="outlined"
          />
        </Box>

        <Typography variant="subtitle2" gutterBottom>
          Health Metrics
        </Typography>
        <Box sx={{ mb: 2 }}>
          {healthMetrics.latency ? (
            <Typography variant="body2" color="text.secondary">
              Latency: {healthMetrics.latency}ms
            </Typography>
          ) : null}
          {healthMetrics.lastSeen ? (
            <Typography variant="body2" color="text.secondary">
              Last seen: {new Date(healthMetrics.lastSeen).toLocaleTimeString()}
            </Typography>
          ) : null}
          {!healthMetrics.latency && !healthMetrics.lastSeen ? (
            <Typography variant="body2" color="text.secondary">
              No metrics available
            </Typography>
          ) : null}
        </Box>

        <Button
          variant="outlined"
          onClick={handleReconnect}
          disabled={connectionStatus === 'connecting'}
        >
          Reconnect
        </Button>
      </CardContent>
    </Card>
  );
}

export default SDRPlusConnectionPanel;
