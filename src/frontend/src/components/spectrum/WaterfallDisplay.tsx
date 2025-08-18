import React, { useEffect, useState, useRef } from 'react';
import Plot from 'react-plotly.js';
import { Box, Typography, Alert } from '@mui/material';

interface SpectrumData {
  frequencies: Float32Array;
  magnitudes: Float32Array;
  timestamp: number;
  centerFreq: number;
  sampleRate: number;
}

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
  const [waterfallData, setWaterfallData] = useState<number[][]>([]);
  const [frequencies, setFrequencies] = useState<number[]>([]);
  const [beaconTarget, setBeaconTarget] = useState<number | null>(null);
  const [targetError, setTargetError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Validate frequency range (850 MHz - 6.5 GHz per PRD-FR1)
  const isValidFrequency = centerFreq >= 850e6 && centerFreq <= 6500e6;

  useEffect(() => {
    if (!isValidFrequency) return;

    // Connect to WebSocket for real-time spectrum data
    const wsUrl = `ws://localhost:8080/ws/spectrum`;
    wsRef.current = new WebSocket(wsUrl);

    const handleMessage = (event: MessageEvent) => {
      try {
        const message = JSON.parse(event.data);
        if (message.type === 'spectrum') {
          const spectrumData = message.data as SpectrumData;

          // Call update callback with real spectrum data
          if (onSpectrumUpdate) {
            onSpectrumUpdate(spectrumData);
          }

          // Update waterfall display with FFT magnitude data
          const magnitudes = Array.from(spectrumData.magnitudes);
          const freqs = Array.from(spectrumData.frequencies);

          setFrequencies(freqs);
          setWaterfallData(prev => {
            const updated = [...prev, magnitudes];
            // Keep last 100 rows for waterfall display
            return updated.slice(-100);
          });
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

  // Handle waterfall plot click for beacon targeting
  const handlePlotClick = (data: any) => {
    setTargetError(null);

    if (data.points && data.points.length > 0) {
      const clickedFreqMHz = data.points[0].x;
      const clickedFreqHz = clickedFreqMHz * 1e6;

      // Validate beacon target within waterfall bandwidth
      const minFreq = (centerFreq - bandwidth / 2) / 1e6;
      const maxFreq = (centerFreq + bandwidth / 2) / 1e6;

      if (clickedFreqMHz < minFreq || clickedFreqMHz > maxFreq) {
        setTargetError('Beacon target must be within visible bandwidth range');
        return;
      }

      // Set beacon target
      setBeaconTarget(clickedFreqHz);

      // Call callback with Hz value
      if (onBeaconTargetSet) {
        onBeaconTargetSet(clickedFreqHz);
      }
    }
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
      <Typography variant="h6" gutterBottom>
        Spectrum Waterfall: {minFreq.toFixed(1)} MHz - {maxFreq.toFixed(1)} MHz
      </Typography>

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

      <Box data-testid="waterfall-plot-container">
        <Plot
          data={[
            {
              z: waterfallData,
              type: 'heatmap' as const,
              colorscale: 'Viridis',
              showscale: true,
              colorbar: {
                title: 'Power (dBm)',
                titleside: 'right'
              }
            }
          ]}
          layout={{
            title: 'RF Spectrum Waterfall (5MHz Bandwidth)',
            xaxis: {
              title: 'Frequency (MHz)',
              range: [minFreq, maxFreq]
            },
            yaxis: {
              title: 'Time',
              autorange: true
            },
            height: 400,
            margin: { t: 50, r: 50, b: 50, l: 50 }
          }}
          config={{
            displayModeBar: true,
            responsive: true
          }}
          style={{ width: '100%', height: '400px' }}
          onClick={handlePlotClick}
        />
      </Box>
    </Box>
  );
};
