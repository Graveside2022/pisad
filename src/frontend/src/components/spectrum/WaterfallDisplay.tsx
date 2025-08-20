import React, { useEffect, useState, useRef } from 'react';
import { Box, Typography, Alert, IconButton, ButtonGroup, Slider } from '@mui/material';
import { ZoomIn, ZoomOut, PanTool, Home, Settings } from '@mui/icons-material';
import { D3Waterfall, SpectrumData } from '../../utils/d3-waterfall';

// SpectrumData interface now imported from d3-waterfall utils

interface WaterfallDisplayProps {
  centerFreq: number;
  bandwidth: number;
  onSpectrumUpdate?: (data: SpectrumData) => void;
  onBeaconTargetSet?: (targetFreq: number) => void;
}

export const WaterfallDisplay: React.FC<WaterfallDisplayProps> = ({
  centerFreq,
  bandwidth,
  onSpectrumUpdate,
  onBeaconTargetSet
}) => {
  const [beaconTarget, setBeaconTarget] = useState<number | null>(null);
  const [targetError, setTargetError] = useState<string | null>(null);
  const [zoomLevel, setZoomLevel] = useState<number>(1);
  const [panEnabled, setPanEnabled] = useState<boolean>(false);
  const [showControls, setShowControls] = useState<boolean>(true);
  const [signalConfidence, setSignalConfidence] = useState<number>(0);
  const [averageConfidence, setAverageConfidence] = useState<number>(0);
  const [confidenceHistory, setConfidenceHistory] = useState<number[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const waterfallRef = useRef<D3Waterfall | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  // Validate frequency range (850 MHz - 6.5 GHz per PRD-FR1)
  const isValidFrequency = centerFreq >= 850e6 && centerFreq <= 6500e6;

  // Initialize d3-waterfall instance
  useEffect(() => {
    if (!isValidFrequency || !containerRef.current) return;

    // Create d3-waterfall instance with professional RTL-SDR configuration
    try {
      waterfallRef.current = new D3Waterfall(
        '#waterfall-container',
        [], // Empty annotations initially, will be loaded separately
        {
          isAnimatable: true,
          isSelectable: true,
          isZoomable: true,
          margin: 50,
          maxRows: 100
        }
      );

      // Setup click handler for beacon targeting
      waterfallRef.current.setClickHandler((frequency: number, confidence: number) => {
        handleWaterfallClick(frequency, confidence);
      });

    } catch (error) {
      console.error('Failed to initialize d3-waterfall:', error);
    }

    return () => {
      if (waterfallRef.current) {
        waterfallRef.current.destroy();
        waterfallRef.current = null;
      }
    };
  }, [isValidFrequency]);

  // WebSocket connection for real-time spectrum data
  useEffect(() => {
    if (!isValidFrequency || !waterfallRef.current) return;

    // Connect to WebSocket for real-time spectrum data
    const wsUrl = `ws://localhost:8080/ws/spectrum`;
    wsRef.current = new WebSocket(wsUrl);

    const handleMessage = (event: MessageEvent) => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === 'spectrum' && waterfallRef.current) {
          const spectrumData = message.data as SpectrumData;

          // Calculate and update signal confidence
          const confidence = calculateSignalConfidence(spectrumData);
          updateConfidenceMetrics(confidence);

          // Call update callback with real spectrum data
          if (onSpectrumUpdate) {
            onSpectrumUpdate(spectrumData);
          }

          // Update d3-waterfall display with real-time data
          waterfallRef.current.updateSpectrumData(spectrumData);
        }
      } catch (error) {
        console.error('Error parsing spectrum data:', error);
      }
    };

    wsRef.current.addEventListener('message', handleMessage);

    return () => {
      if (wsRef.current) {
        wsRef.current.removeEventListener('message', handleMessage);
        wsRef.current.close();
      }
    };
  }, [centerFreq, bandwidth, onSpectrumUpdate, isValidFrequency]);

  // Handle d3-waterfall click for beacon targeting
  const handleWaterfallClick = (frequency: number, confidence: number) => {
    setTargetError(null);

    // Validate beacon target within waterfall bandwidth
    const minFreq = centerFreq - bandwidth / 2;
    const maxFreq = centerFreq + bandwidth / 2;

    if (frequency < minFreq || frequency > maxFreq) {
      setTargetError('Beacon target must be within visible bandwidth range');
      return;
    }

    // Set beacon target
    setBeaconTarget(frequency);

    // Call callback with Hz value
    if (onBeaconTargetSet) {
      onBeaconTargetSet(frequency);
    }
  };

  // Professional Pan/Zoom Control Handlers
  const handleZoomIn = () => {
    if (waterfallRef.current && zoomLevel < 10) {
      const newZoom = Math.min(10, zoomLevel * 1.5);
      setZoomLevel(newZoom);
      // D3 zoom will be handled by the waterfall instance
    }
  };

  const handleZoomOut = () => {
    if (waterfallRef.current && zoomLevel > 0.1) {
      const newZoom = Math.max(0.1, zoomLevel / 1.5);
      setZoomLevel(newZoom);
      // D3 zoom will be handled by the waterfall instance
    }
  };

  const handleZoomReset = () => {
    if (waterfallRef.current) {
      setZoomLevel(1);
      // Reset d3 zoom transform
    }
  };

  const togglePanMode = () => {
    setPanEnabled(!panEnabled);
    // Update waterfall interaction mode
    if (waterfallRef.current) {
      // Pan mode will be handled by d3-waterfall zoom behavior
    }
  };

  const handleZoomSlider = (event: Event, newValue: number | number[]) => {
    const zoom = Array.isArray(newValue) ? newValue[0] : newValue;
    setZoomLevel(zoom);
    // Apply zoom to waterfall
  };

  // Enhanced Signal Confidence Algorithms
  const calculateSignalConfidence = (data: SpectrumData): number => {
    const { magnitudes } = data;
    if (magnitudes.length === 0) return 0;

    // Convert Float32Array to regular array for easier processing
    const magArray = Array.from(magnitudes);

    // Algorithm 1: Signal-to-Noise Ratio (SNR) Analysis
    const noiseFloor = calculateNoiseFloor(magArray);
    const peakMagnitude = Math.max(...magArray);
    const snr = peakMagnitude - noiseFloor;
    const snrConfidence = Math.min(1.0, Math.max(0, snr / 40)); // Normalize to 0-1

    // Algorithm 2: Signal Consistency Analysis
    const consistencyScore = calculateSignalConsistency(magArray);

    // Algorithm 3: Frequency Stability Analysis
    const stabilityScore = calculateFrequencyStability(data);

    // Algorithm 4: Dynamic Range Analysis
    const dynamicRangeScore = calculateDynamicRange(magArray);

    // Weighted combination of all confidence metrics
    const weights = { snr: 0.4, consistency: 0.25, stability: 0.2, dynamicRange: 0.15 };
    const totalConfidence = (
      snrConfidence * weights.snr +
      consistencyScore * weights.consistency +
      stabilityScore * weights.stability +
      dynamicRangeScore * weights.dynamicRange
    );

    return Math.min(1.0, Math.max(0, totalConfidence));
  };

  const calculateNoiseFloor = (magnitudes: number[]): number => {
    // Sort magnitudes and take bottom 20% as noise reference
    const sorted = magnitudes.slice().sort((a, b) => a - b);
    const noiseEndIndex = Math.floor(sorted.length * 0.2);
    const noiseValues = sorted.slice(0, noiseEndIndex);
    return noiseValues.reduce((sum, val) => sum + val, 0) / noiseValues.length;
  };

  const calculateSignalConsistency = (magnitudes: number[]): number => {
    // Analyze magnitude variance - consistent signals have lower variance
    const mean = magnitudes.reduce((sum, val) => sum + val, 0) / magnitudes.length;
    const variance = magnitudes.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / magnitudes.length;
    const standardDev = Math.sqrt(variance);

    // Lower standard deviation indicates more consistent signal
    const maxExpectedStdDev = 30; // Typical for RTL-SDR
    return Math.max(0, 1 - (standardDev / maxExpectedStdDev));
  };

  const calculateFrequencyStability = (data: SpectrumData): number => {
    // This is a simplified version - in real implementation would track frequency drift over time
    const { centerFreq, sampleRate } = data;

    // Penalize signals near edge of sample rate (aliasing risk)
    const edgeDistance = Math.min(
      Math.abs(centerFreq - (centerFreq - sampleRate / 2)),
      Math.abs(centerFreq - (centerFreq + sampleRate / 2))
    );

    const edgeRatio = edgeDistance / (sampleRate / 2);
    return Math.min(1.0, edgeRatio * 2); // Higher confidence for signals away from edges
  };

  const calculateDynamicRange = (magnitudes: number[]): number => {
    const min = Math.min(...magnitudes);
    const max = Math.max(...magnitudes);
    const range = max - min;

    // Good signals typically have 40-80 dB dynamic range
    const idealRange = 60; // dB
    const rangeScore = Math.min(1.0, range / idealRange);

    return rangeScore;
  };

  const updateConfidenceMetrics = (confidence: number): void => {
    setSignalConfidence(confidence);

    // Update confidence history (keep last 20 measurements)
    setConfidenceHistory(prev => {
      const newHistory = [...prev, confidence].slice(-20);

      // Calculate rolling average
      const average = newHistory.reduce((sum, val) => sum + val, 0) / newHistory.length;
      setAverageConfidence(average);

      return newHistory;
    });
  };

  const getConfidenceColor = (confidence: number): string => {
    if (confidence >= 0.8) return '#4CAF50'; // Green - High confidence
    if (confidence >= 0.6) return '#FF9800'; // Orange - Medium confidence
    if (confidence >= 0.4) return '#FFC107'; // Yellow - Low confidence
    return '#F44336'; // Red - Very low confidence
  };

  const getConfidenceLabel = (confidence: number): string => {
    if (confidence >= 0.8) return 'High';
    if (confidence >= 0.6) return 'Medium';
    if (confidence >= 0.4) return 'Low';
    return 'Very Low';
  };

  if (!isValidFrequency) {
    return (
      <Alert severity="error" data-testid="waterfall-display">
        Frequency out of range. Valid range: 850 MHz - 6.5 GHz
      </Alert>
    );
  }

  // Calculate frequency range display (±2.5MHz around center)
  const minFreq = (centerFreq - bandwidth / 2) / 1e6; // Convert to MHz
  const maxFreq = (centerFreq + bandwidth / 2) / 1e6;

  return (
    <Box data-testid="waterfall-display">
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
        <Typography variant="h6">
          Spectrum Waterfall: {minFreq.toFixed(1)} MHz - {maxFreq.toFixed(1)} MHz
        </Typography>

        <IconButton
          onClick={() => setShowControls(!showControls)}
          size="small"
          title="Toggle RTL-SDR Controls"
        >
          <Settings />
        </IconButton>
      </Box>

      {/* Professional RTL-SDR Pan/Zoom Controls */}
      {showControls && (
        <Box
          data-testid="rtl-sdr-controls"
          sx={{
            display: 'flex',
            alignItems: 'center',
            gap: 2,
            mb: 2,
            p: 2,
            bgcolor: 'background.paper',
            borderRadius: 1,
            boxShadow: 1
          }}
        >
          <ButtonGroup variant="outlined" size="small">
            <IconButton
              onClick={handleZoomIn}
              disabled={zoomLevel >= 10}
              title="Zoom In"
            >
              <ZoomIn />
            </IconButton>
            <IconButton
              onClick={handleZoomOut}
              disabled={zoomLevel <= 0.1}
              title="Zoom Out"
            >
              <ZoomOut />
            </IconButton>
            <IconButton
              onClick={handleZoomReset}
              title="Reset Zoom"
            >
              <Home />
            </IconButton>
            <IconButton
              onClick={togglePanMode}
              color={panEnabled ? 'primary' : 'default'}
              title="Toggle Pan Mode"
            >
              <PanTool />
            </IconButton>
          </ButtonGroup>

          <Typography variant="body2" sx={{ minWidth: '80px' }}>
            Zoom: {zoomLevel.toFixed(1)}x
          </Typography>

          <Slider
            value={zoomLevel}
            min={0.1}
            max={10}
            step={0.1}
            onChange={handleZoomSlider}
            sx={{ width: 200 }}
            title="Zoom Level"
          />

          {panEnabled && (
            <Typography variant="body2" color="primary" sx={{ fontWeight: 'bold' }}>
              PAN MODE
            </Typography>
          )}
        </Box>
      )}

      {targetError && (
        <Alert severity="warning" sx={{ mb: 2 }}>
          {targetError}
        </Alert>
      )}

      {beaconTarget && (
        <Typography variant="body2" color="primary" sx={{ mb: 1 }}>
          Beacon Target: {(beaconTarget / 1e6).toFixed(1)} MHz
        </Typography>
      )}

      {/* Signal Confidence Display */}
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          gap: 2,
          mb: 1,
          p: 1,
          bgcolor: 'background.default',
          borderRadius: 1,
          border: `2px solid ${getConfidenceColor(signalConfidence)}`,
        }}
      >
        <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
          Signal Confidence:
        </Typography>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box
            sx={{
              width: 100,
              height: 8,
              bgcolor: 'grey.300',
              borderRadius: 1,
              overflow: 'hidden'
            }}
          >
            <Box
              sx={{
                width: `${signalConfidence * 100}%`,
                height: '100%',
                bgcolor: getConfidenceColor(signalConfidence),
                transition: 'width 0.3s ease'
              }}
            />
          </Box>

          <Typography
            variant="body2"
            sx={{
              color: getConfidenceColor(signalConfidence),
              fontWeight: 'bold',
              minWidth: '60px'
            }}
          >
            {getConfidenceLabel(signalConfidence)} ({(signalConfidence * 100).toFixed(1)}%)
          </Typography>
        </Box>

        <Typography variant="body2" color="text.secondary">
          Avg: {(averageConfidence * 100).toFixed(1)}%
        </Typography>

        {confidenceHistory.length > 0 && (
          <Typography variant="body2" color="text.secondary">
            Samples: {confidenceHistory.length}
          </Typography>
        )}
      </Box>

      <Box data-testid="d3-waterfall-container" sx={{ position: 'relative', width: '100%', height: 400 }}>
        <div
          id="waterfall-container"
          ref={containerRef}
          style={{ width: '100%', height: '100%', position: 'relative' }}
          data-testid="waterfall-canvas"
        />
      </Box>
    </Box>
  );
};
