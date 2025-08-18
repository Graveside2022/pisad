import React, { useState } from 'react';
import {
  Box,
  TextField,
  Button,
  Typography,
  Alert,
  Chip,
  Stack,
  InputAdornment
} from '@mui/material';

interface FrequencyControlProps {
  currentFreq: number; // Frequency in Hz
  onChange: (freq: number) => void;
  disabled?: boolean;
}

// Common frequency presets (in MHz)
const FREQUENCY_PRESETS = [
  { label: '915 MHz', value: 915e6, description: 'LoRa ISM' },
  { label: '2437 MHz', value: 2437e6, description: 'WiFi Ch.6' },
  { label: '5800 MHz', value: 5800e6, description: '5.8GHz ISM' }
];

export const FrequencyControl: React.FC<FrequencyControlProps> = ({
  currentFreq,
  onChange,
  disabled = false
}) => {
  const [inputValue, setInputValue] = useState((currentFreq / 1e6).toString());
  const [error, setError] = useState<string | null>(null);
  const [isUpdating, setIsUpdating] = useState(false);

  // Validate frequency range per PRD-FR1 (850 MHz - 6.5 GHz)
  const validateFrequency = (freqMHz: number): boolean => {
    return freqMHz >= 850 && freqMHz <= 6500;
  };

  const handleFrequencyChange = async (freqMHz: number) => {
    setError(null);

    if (!validateFrequency(freqMHz)) {
      setError('Frequency must be between 850 MHz and 6500 MHz');
      return;
    }

    const freqHz = freqMHz * 1e6;
    setIsUpdating(true);

    try {
      // Apply frequency change immediately via API call
      const response = await fetch('/api/config/sdr', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          centerFreq: freqHz,
          immediate: true
        })
      });

      if (!response.ok) {
        throw new Error('API request failed');
      }

      // Call onChange callback with Hz value
      onChange(freqHz);

    } catch (error) {
      console.error('Failed to update frequency:', error);
      setError('Failed to update frequency. Please try again.');
    } finally {
      setIsUpdating(false);
    }
  };

  const handleInputBlur = () => {
    const freqMHz = parseFloat(inputValue);
    if (!isNaN(freqMHz)) {
      handleFrequencyChange(freqMHz);
    }
  };

  const handlePresetClick = async (freqHz: number) => {
    const freqMHz = freqHz / 1e6;
    setInputValue(freqMHz.toString());
    await handleFrequencyChange(freqMHz);
  };

  return (
    <Box>
      <Typography variant="h6" gutterBottom>
        Center Frequency Control
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Stack spacing={2}>
        {/* Manual frequency input */}
        <TextField
          label="Center Frequency"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onBlur={handleInputBlur}
          disabled={disabled || isUpdating}
          type="number"
          InputProps={{
            endAdornment: <InputAdornment position="end">MHz</InputAdornment>,
          }}
          helperText="Valid range: 850 - 6500 MHz (per PRD-FR1)"
          fullWidth
        />

        {/* Frequency presets */}
        <Box>
          <Typography variant="subtitle2" gutterBottom>
            Quick Select:
          </Typography>
          <Stack direction="row" spacing={1} flexWrap="wrap">
            {FREQUENCY_PRESETS.map((preset) => (
              <Chip
                key={preset.value}
                label={preset.label}
                onClick={() => handlePresetClick(preset.value)}
                disabled={disabled || isUpdating}
                color={Math.abs(currentFreq - preset.value) < 1e6 ? 'primary' : 'default'}
                variant={Math.abs(currentFreq - preset.value) < 1e6 ? 'filled' : 'outlined'}
              />
            ))}
          </Stack>
        </Box>

        {/* Current frequency display */}
        <Typography variant="body2" color="text.secondary">
          Current: {(currentFreq / 1e6).toFixed(1)} MHz
          {isUpdating && ' (Updating...)'}
        </Typography>
      </Stack>
    </Box>
  );
};
