import {
  Card,
  CardContent,
  Typography,
  Box,
  Button,
  Chip,
  Alert,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  FormControl,
  FormControlLabel,
  RadioGroup,
  Radio,
  Slider,
  Divider,
  IconButton,
  Collapse,
} from '@mui/material';
import { useState, useEffect } from 'react';
import { useWebSocket } from '../../hooks/useWebSocket';
import { Warning, Error, CheckCircle, ExpandMore, ExpandLess } from '@mui/icons-material';

interface ConflictData {
  conflictType: 'frequency' | 'priority' | 'none';
  severity: 'warning' | 'critical' | 'blocking';
  groundFrequency: number;
  droneFrequency: number;
  currentPriority: 'ground' | 'drone' | 'coordinated';
  recommendedPriority: 'ground' | 'drone' | 'coordinated';
  lastConflictTime?: string;
  conflictActive: boolean;
}

interface ResolutionHistoryEntry {
  timestamp: string;
  conflictType: string;
  resolution: string;
  operator: string;
  duration?: number;
}

interface ConflictResolutionProps {
  // Future props for configuration
}

function ConflictResolution(_props: ConflictResolutionProps) {
  const [conflictData, setConflictData] = useState<ConflictData>({
    conflictType: 'none',
    severity: 'warning',
    groundFrequency: 2437000000, // 2.437 GHz
    droneFrequency: 2437000000,
    currentPriority: 'coordinated',
    recommendedPriority: 'coordinated',
    conflictActive: false
  });

  const [selectedPriority, setSelectedPriority] = useState<'ground' | 'drone' | 'coordinated'>('coordinated');
  const [overrideDuration, setOverrideDuration] = useState(30); // minutes
  const [isResolving, setIsResolving] = useState(false);
  const [showHistory, setShowHistory] = useState(false);
  const [resolutionHistory, setResolutionHistory] = useState<ResolutionHistoryEntry[]>([]);

  const { addMessageHandler, sendMessage } = useWebSocket();

  // WebSocket message handling for conflict updates
  useEffect(() => {
    const cleanup = addMessageHandler((message) => {
      if (message.type === 'conflict_resolution') {
        const data = message.data as ConflictData;
        setConflictData(data);
        setSelectedPriority(data.currentPriority);
        setIsResolving(false);
      } else if (message.type === 'resolution_history') {
        const history = message.data as ResolutionHistoryEntry[];
        setResolutionHistory(history);
      }
    });
    return cleanup;
  }, [addMessageHandler]);

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
    return Math.abs(conflictData.groundFrequency - conflictData.droneFrequency);
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'warning': return <Warning color="warning" />;
      case 'critical': return <Error color="error" />;
      case 'blocking': return <Error color="error" />;
      default: return <CheckCircle color="success" />;
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'warning': return 'warning';
      case 'critical': return 'error';
      case 'blocking': return 'error';
      default: return 'success';
    }
  };

  const handleApplyResolution = () => {
    setIsResolving(true);
    sendMessage({
      type: 'apply_conflict_resolution',
      data: {
        priority: selectedPriority,
        duration: selectedPriority !== 'coordinated' ? overrideDuration : undefined,
        timestamp: new Date().toISOString()
      }
    });
  };

  const handleEmergencyRevert = () => {
    setIsResolving(true);
    sendMessage({
      type: 'emergency_revert',
      data: {
        action: 'revert_to_drone_authority',
        timestamp: new Date().toISOString()
      }
    });
  };

  const getPriorityColor = (priority: string): 'primary' | 'secondary' | 'success' | 'error' | 'info' | 'warning' | 'inherit' => {
    switch (priority) {
      case 'ground': return 'primary';
      case 'drone': return 'success';
      case 'coordinated': return 'info';
      default: return 'inherit';
    }
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Conflict Resolution
        </Typography>

        {conflictData.conflictActive && (
          <Alert
            severity={getSeverityColor(conflictData.severity)}
            icon={getSeverityIcon(conflictData.severity)}
            sx={{ mb: 2 }}
          >
            {conflictData.conflictType === 'frequency' && 'Frequency mismatch detected between ground and drone SDR'}
            {conflictData.conflictType === 'priority' && 'Priority conflict requires operator intervention'}
          </Alert>
        )}

        {/* Frequency Comparison */}
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle2" gutterBottom>
            Frequency Comparison
          </Typography>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Source</TableCell>
                <TableCell>Frequency</TableCell>
                <TableCell>Status</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              <TableRow>
                <TableCell>Ground SDR++</TableCell>
                <TableCell>{formatFrequency(conflictData.groundFrequency)}</TableCell>
                <TableCell>
                  <Chip
                    label={conflictData.currentPriority === 'ground' ? 'Active' : 'Standby'}
                    color={conflictData.currentPriority === 'ground' ? 'primary' : 'default'}
                    size="small"
                  />
                </TableCell>
              </TableRow>
              <TableRow>
                <TableCell>Drone SDR</TableCell>
                <TableCell>{formatFrequency(conflictData.droneFrequency)}</TableCell>
                <TableCell>
                  <Chip
                    label={conflictData.currentPriority === 'drone' ? 'Active' : 'Standby'}
                    color={conflictData.currentPriority === 'drone' ? 'success' : 'default'}
                    size="small"
                  />
                </TableCell>
              </TableRow>
            </TableBody>
          </Table>

          {getFrequencyDifference() > 0 && (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              Frequency difference: {formatFrequency(getFrequencyDifference())}
            </Typography>
          )}
        </Box>

        {/* Conflict Severity */}
        <Box sx={{ mb: 3 }} data-testid="conflict-severity">
          <Typography variant="subtitle2" gutterBottom>
            Conflict Severity
          </Typography>
          <Chip
            label={conflictData.severity.toUpperCase()}
            color={getSeverityColor(conflictData.severity)}
            icon={getSeverityIcon(conflictData.severity)}
            variant="outlined"
          />
          {conflictData.recommendedPriority !== conflictData.currentPriority && (
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              Recommended: Switch to {conflictData.recommendedPriority} priority
            </Typography>
          )}
        </Box>

        <Divider sx={{ my: 2 }} />

        {/* Source Priority Selection */}
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle2" gutterBottom>
            Source Priority
          </Typography>
          <FormControl component="fieldset">
            <RadioGroup
              value={selectedPriority}
              onChange={(e) => setSelectedPriority(e.target.value as any)}
              row
            >
              <FormControlLabel
                value="ground"
                control={<Radio />}
                label="Ground Priority"
              />
              <FormControlLabel
                value="drone"
                control={<Radio />}
                label="Drone Priority"
              />
              <FormControlLabel
                value="coordinated"
                control={<Radio />}
                label="Coordinated Mode"
              />
            </RadioGroup>
          </FormControl>
        </Box>

        {/* Override Duration */}
        {selectedPriority !== 'coordinated' && (
          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle2" gutterBottom>
              Override Duration
            </Typography>
            <Box sx={{ px: 2 }}>
              <Slider
                value={overrideDuration}
                onChange={(_, value) => setOverrideDuration(value as number)}
                min={5}
                max={120}
                step={5}
                marks
                valueLabelDisplay="on"
                valueLabelFormat={(value) => `${value} min`}
              />
            </Box>
            <Typography variant="caption" color="text.secondary">
              Temporary override will revert to coordinated mode after {overrideDuration} minutes
            </Typography>
          </Box>
        )}

        {/* Action Buttons */}
        <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
          <Button
            variant="contained"
            onClick={handleApplyResolution}
            disabled={isResolving || !conflictData.conflictActive}
            color={getPriorityColor(selectedPriority)}
          >
            {isResolving ? 'Applying...' : 'Apply Resolution'}
          </Button>
          <Button
            variant="outlined"
            color="error"
            onClick={handleEmergencyRevert}
            disabled={isResolving}
          >
            Emergency Revert
          </Button>
        </Box>

        {/* Resolution History */}
        <Box>
          <Box sx={{ display: 'flex', alignItems: 'center', cursor: 'pointer' }} onClick={() => setShowHistory(!showHistory)}>
            <Typography variant="subtitle2">
              Resolution History
            </Typography>
            <IconButton size="small">
              {showHistory ? <ExpandLess /> : <ExpandMore />}
            </IconButton>
          </Box>
          <Collapse in={showHistory}>
            <Box sx={{ mt: 1 }}>
              {resolutionHistory.length > 0 ? (
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Time</TableCell>
                      <TableCell>Conflict</TableCell>
                      <TableCell>Resolution</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {resolutionHistory.slice(0, 5).map((entry, index) => (
                      <TableRow key={index}>
                        <TableCell>{new Date(entry.timestamp).toLocaleTimeString()}</TableCell>
                        <TableCell>{entry.conflictType}</TableCell>
                        <TableCell>{entry.resolution}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              ) : (
                <Typography variant="body2" color="text.secondary">
                  No conflicts resolved recently
                </Typography>
              )}
            </Box>
          </Collapse>
        </Box>

        {conflictData.lastConflictTime && (
          <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
            Last conflict: {new Date(conflictData.lastConflictTime).toLocaleTimeString()}
          </Typography>
        )}
      </CardContent>
    </Card>
  );
}

export default ConflictResolution;
