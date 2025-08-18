import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { FrequencyControl } from '../../../src/frontend/src/components/spectrum/FrequencyControl';

// Mock API calls for frequency control
global.fetch = jest.fn();

describe('FrequencyControl Component', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({ success: true })
    });
  });

  test('renders frequency control with valid range (850MHz - 6.5GHz)', () => {
    const mockOnChange = jest.fn();
    render(
      <FrequencyControl
        currentFreq={2437e6}
        onChange={mockOnChange}
        disabled={false}
      />
    );

    // Verify frequency input exists
    expect(screen.getByLabelText(/center frequency/i)).toBeInTheDocument();

    // Verify current frequency displayed in MHz
    expect(screen.getByDisplayValue('2437')).toBeInTheDocument();

    // Verify unit label
    expect(screen.getByText('MHz')).toBeInTheDocument();
  });

  test('validates frequency range per PRD-FR1 (850MHz - 6.5GHz)', async () => {
    const mockOnChange = jest.fn();
    render(<FrequencyControl currentFreq={2437e6} onChange={mockOnChange} />);

    const input = screen.getByLabelText(/center frequency/i);

    // Test valid frequencies
    fireEvent.change(input, { target: { value: '915' } }); // Valid LoRa frequency
    fireEvent.blur(input);

    await waitFor(() => {
      expect(mockOnChange).toHaveBeenCalledWith(915e6);
    });

    // Test invalid frequency below range
    fireEvent.change(input, { target: { value: '800' } }); // Below 850MHz
    fireEvent.blur(input);

    expect(screen.getByText(/frequency must be between 850 MHz and 6500 MHz/i)).toBeInTheDocument();
    expect(mockOnChange).not.toHaveBeenCalledWith(800e6);

    // Test invalid frequency above range
    fireEvent.change(input, { target: { value: '7000' } }); // Above 6.5GHz
    fireEvent.blur(input);

    expect(screen.getByText(/frequency must be between 850 MHz and 6500 MHz/i)).toBeInTheDocument();
    expect(mockOnChange).not.toHaveBeenCalledWith(7000e6);
  });

  test('applies frequency change immediately via API call', async () => {
    const mockOnChange = jest.fn();
    render(<FrequencyControl currentFreq={2437e6} onChange={mockOnChange} />);

    const input = screen.getByLabelText(/center frequency/i);

    // Change to WiFi frequency (2.4GHz)
    fireEvent.change(input, { target: { value: '2400' } });
    fireEvent.blur(input);

    // Verify API call made to update SDR configuration
    await waitFor(() => {
      expect(fetch).toHaveBeenCalledWith('/api/config/sdr', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          centerFreq: 2400e6,
          immediate: true
        })
      });
    });

    // Verify onChange callback called with Hz value
    expect(mockOnChange).toHaveBeenCalledWith(2400e6);
  });

  test('handles API errors gracefully', async () => {
    (fetch as jest.Mock).mockRejectedValue(new Error('Network error'));

    const mockOnChange = jest.fn();
    render(<FrequencyControl currentFreq={2437e6} onChange={mockOnChange} />);

    const input = screen.getByLabelText(/center frequency/i);

    fireEvent.change(input, { target: { value: '915' } });
    fireEvent.blur(input);

    await waitFor(() => {
      expect(screen.getByText(/failed to update frequency/i)).toBeInTheDocument();
    });

    // Verify onChange not called on API failure
    expect(mockOnChange).not.toHaveBeenCalled();
  });

  test('shows common frequency presets for quick selection', async () => {
    const mockOnChange = jest.fn();
    render(<FrequencyControl currentFreq={2437e6} onChange={mockOnChange} />);

    // Verify common frequency buttons exist
    expect(screen.getByText('915 MHz')).toBeInTheDocument(); // LoRa
    expect(screen.getByText('2437 MHz')).toBeInTheDocument(); // WiFi
    expect(screen.getByText('5800 MHz')).toBeInTheDocument(); // 5.8GHz ISM

    // Test preset button click
    fireEvent.click(screen.getByText('915 MHz'));

    await waitFor(() => {
      expect(mockOnChange).toHaveBeenCalledWith(915e6);
    });
  });
});
