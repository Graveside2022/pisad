import { Card, CardContent, Typography, Box, Button, Chip, Alert } from '@mui/material';
import { useState, useEffect } from 'react';
import { useWebSocket } from '../../hooks/useWebSocket';

interface FrequencySyncData {
  groundFrequency: number;
  droneFrequency: number;
  syncStatus: 'synchronized' | 'mismatch' | 'unknown';
  lastSyncTime?: string;
}

interface FrequencySyncIndicatorsProps {
  // Future props for configuration
}

function FrequencySyncIndicators(_props: FrequencySyncIndicatorsProps) {
  const [syncData, setSyncData] = useState<FrequencySyncData>({
    groundFrequency: 0,
    droneFrequency: 0,
    syncStatus: 'unknown'
  });
  const [isSyncing, setIsSyncing] = useState(false);
  const { addMessageHandler, sendMessage } = useWebSocket();

  // WebSocket message handling for frequency sync updates
  useEffect(() => {
    const cleanup = addMessageHandler((message) => {
      if (message.type === 'frequency_sync') {
        const data = message.data as FrequencySyncData;
        setSyncData(data);
        setIsSyncing(false);
      }
    });
    return cleanup;
  }, [addMessageHandler]);

  const handleSynchronize = () => {
    setIsSyncing(true);
    try {
      sendMessage({
        type: 'frequency_sync_command',
        data: { action: 'synchronize' }
      });
    } catch (error) {
      console.error('Failed to send synchronization message:', error);
      setIsSyncing(false);
    }
  };

  const formatFrequency = (freq: number) => {
    if (freq >= 1e9) {
      return `${(freq / 1e9).toFixed(3)} GHz`;
    } else if (freq >= 1e6) {
      return `${(freq / 1e6).toFixed(1)} MHz`;
    } else {
      return `${freq.toFixed(0)} Hz`;
    }
  };

  const getFrequencyDifference = () => {
    return Math.abs(syncData.groundFrequency - syncData.droneFrequency);
  };

  const getSyncStatusColor = () => {
    switch (syncData.syncStatus) {
      case 'synchronized': return 'success';
      case 'mismatch': return 'error';
      default: return 'default';
    }
  };

  const getSyncStatusText = () => {
    switch (syncData.syncStatus) {
      case 'synchronized': return 'Synchronized';
      case 'mismatch': return 'Frequency Mismatch';
      default: return 'Unknown';
    }
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Frequency Synchronization
        </Typography>

        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Sync Status
          </Typography>
          <Chip
            label={getSyncStatusText()}
            color={getSyncStatusColor()}
            variant="outlined"
            sx={{ mb: 1 }}
          />
        </Box>

        <Box sx={{ mb: 2 }}>
          <Typography variant="body2" gutterBottom>
            Ground: {formatFrequency(syncData.groundFrequency)}
          </Typography>
          <Typography variant="body2" gutterBottom>
            Drone: {formatFrequency(syncData.droneFrequency)}
          </Typography>
          {getFrequencyDifference() > 0 && (
            <Typography variant="body2" color="text.secondary">
              Difference: {formatFrequency(getFrequencyDifference())}
            </Typography>
          )}
        </Box>

        {syncData.syncStatus === 'mismatch' && (
          <Alert severity="warning" sx={{ mb: 2 }}>
            Frequency mismatch detected. Manual synchronization may be required.
          </Alert>
        )}

        <Button
          variant="outlined"
          onClick={handleSynchronize}
          disabled={isSyncing || syncData.syncStatus === 'synchronized'}
          fullWidth
        >
          {isSyncing ? 'Synchronizing...' : 'Synchronize'}
        </Button>

        {syncData.lastSyncTime && (
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
            Last sync: {new Date(syncData.lastSyncTime).toLocaleTimeString()}
          </Typography>
        )}
      </CardContent>
    </Card>
  );
}

export default FrequencySyncIndicators;
