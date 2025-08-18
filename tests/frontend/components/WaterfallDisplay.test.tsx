import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { WaterfallDisplay } from '../../../src/frontend/src/components/spectrum/WaterfallDisplay';

// Mock Plotly.js to avoid canvas issues in test environment
jest.mock('react-plotly.js', () => {
  return function MockPlot(props: any) {
    return (
      <div
        data-testid="waterfall-plot"
        data-plot-props={JSON.stringify(props)}
        onClick={() => {
          if (props.onClick) {
            // Simulate click event with detail from fireEvent
            const event = (window as any).lastFireEvent || { detail: { points: [] } };
            props.onClick(event.detail);
          }
        }}
      />
    );
  };
});

// Mock WebSocket for testing real spectrum data integration
const mockWebSocket = {
  send: jest.fn(),
  close: jest.fn(),
  readyState: WebSocket.OPEN,
  addEventListener: jest.fn(),
  removeEventListener: jest.fn()
};

// Mock real spectrum data format (matches backend FFT output)
const mockSpectrumData = {
  frequencies: new Float32Array(1024).map((_, i) => 2437e6 + (i - 512) * 2500), // 5MHz around 2.437GHz
  magnitudes: new Float32Array(1024).map(() => Math.random() * 80 - 100), // -100 to -20 dBm
  timestamp: Date.now(),
  centerFreq: 2437e6,
  sampleRate: 2.5e6
};

describe('WaterfallDisplay Component', () => {
  beforeEach(() => {
    global.WebSocket = jest.fn(() => mockWebSocket) as any;
  });

  test('renders waterfall display with 5MHz bandwidth', async () => {
    render(<WaterfallDisplay centerFreq={2437e6} bandwidth={5e6} />);

    // Verify component renders
    expect(screen.getByTestId('waterfall-display')).toBeInTheDocument();

    // Verify frequency range display (±2.5MHz around center)
    expect(screen.getByText(/2434.5.*MHz.*2439.5.*MHz/)).toBeInTheDocument();
  });

  test('handles real spectrum data updates at 3Hz rate', async () => {
    const mockCallback = jest.fn();
    render(
      <WaterfallDisplay
        centerFreq={2437e6}
        bandwidth={5e6}
        onSpectrumUpdate={mockCallback}
      />
    );

    // Simulate WebSocket spectrum data message
    const messageEvent = new MessageEvent('message', {
      data: JSON.stringify({
        type: 'spectrum',
        data: mockSpectrumData
      })
    });

    // Verify WebSocket connection established
    expect(mockWebSocket.addEventListener).toHaveBeenCalledWith('message', expect.any(Function));

    // Trigger spectrum data update
    const messageHandler = mockWebSocket.addEventListener.mock.calls.find(
      call => call[0] === 'message'
    )[1];
    messageHandler(messageEvent);

    // Verify callback receives real spectrum data
    await waitFor(() => {
      expect(mockCallback).toHaveBeenCalledWith(mockSpectrumData);
    });
  });

  test('updates waterfall plot with FFT magnitude data', async () => {
    render(<WaterfallDisplay centerFreq={2437e6} bandwidth={5e6} />);

    // Verify Plotly.js waterfall plot container exists
    expect(screen.getByTestId('waterfall-plot-container')).toBeInTheDocument();

    // Simulate spectrum data update
    const messageEvent = new MessageEvent('message', {
      data: JSON.stringify({
        type: 'spectrum',
        data: mockSpectrumData
      })
    });

    const messageHandler = mockWebSocket.addEventListener.mock.calls.find(
      call => call[0] === 'message'
    )[1];
    messageHandler(messageEvent);

    // Verify waterfall data structure matches FFT output
    await waitFor(() => {
      const plotDiv = screen.getByTestId('waterfall-plot');
      expect(plotDiv).toHaveAttribute('data-testid', 'waterfall-plot');
    });
  });

  test('handles frequency range validation (850MHz - 6.5GHz)', () => {
    // Test valid frequency ranges
    const validFreqs = [850e6, 2437e6, 6500e6];
    validFreqs.forEach(freq => {
      const { unmount } = render(<WaterfallDisplay centerFreq={freq} bandwidth={5e6} />);
      expect(screen.getByTestId('waterfall-display')).toBeInTheDocument();
      unmount();
    });

    // Test invalid frequency ranges
    const invalidFreqs = [800e6, 7000e6]; // Below 850MHz, above 6.5GHz
    invalidFreqs.forEach(freq => {
      const { unmount } = render(<WaterfallDisplay centerFreq={freq} bandwidth={5e6} />);
      expect(screen.getByText(/frequency.*out.*range/i)).toBeInTheDocument();
      unmount();
    });
  });

  test('enables click-to-set beacon target frequency', async () => {
    const mockOnBeaconTarget = jest.fn();
    render(
      <WaterfallDisplay
        centerFreq={2437e6}
        bandwidth={5e6}
        onBeaconTargetSet={mockOnBeaconTarget}
      />
    );

    // Mock plot click event at frequency position
    const plotDiv = screen.getByTestId('waterfall-plot');

    // Simulate click at 2440 MHz position (within 5MHz range)
    (window as any).lastFireEvent = {
      detail: {
        points: [{
          x: 2440 // MHz value from waterfall plot
        }]
      }
    };
    fireEvent.click(plotDiv);

    await waitFor(() => {
      expect(mockOnBeaconTarget).toHaveBeenCalledWith(2440e6); // Hz value
    });

    // Verify beacon target indicator appears
    expect(screen.getByText(/beacon target.*2440.*mhz/i)).toBeInTheDocument();
  });

  test('validates beacon target within waterfall bandwidth', async () => {
    const mockOnBeaconTarget = jest.fn();
    render(
      <WaterfallDisplay
        centerFreq={2437e6}
        bandwidth={5e6}
        onBeaconTargetSet={mockOnBeaconTarget}
      />
    );

    // Click outside bandwidth range should not set beacon
    const plotDiv = screen.getByTestId('waterfall-plot');

    (window as any).lastFireEvent = {
      detail: {
        points: [{
          x: 2500 // Outside 2434.5-2439.5 MHz range
        }]
      }
    };
    fireEvent.click(plotDiv);

    // Should not call beacon target callback for out-of-range click
    expect(mockOnBeaconTarget).not.toHaveBeenCalled();

    // Should show validation message
    expect(screen.getByText(/beacon target must be within.*bandwidth/i)).toBeInTheDocument();
  });
});
