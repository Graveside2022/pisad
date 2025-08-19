import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import Dashboard from './Dashboard';

const theme = createTheme();

// Mock all the dashboard components
vi.mock('./SignalMeter', () => ({
  default: () => <div data-testid="signal-meter">Signal Meter</div>
}));

vi.mock('./RSSIGraph', () => ({
  default: () => <div data-testid="rssi-graph">RSSI Graph</div>
}));

vi.mock('./SDRStatus', () => ({
  default: () => <div data-testid="sdr-status">SDR Status</div>
}));

vi.mock('./SystemHealth', () => ({
  default: () => <div data-testid="system-health">System Health</div>
}));

vi.mock('./DetectionLog', () => ({
  default: () => <div data-testid="detection-log">Detection Log</div>
}));

vi.mock('./SDRPlusConnectionPanel', () => ({
  default: () => <div data-testid="sdrplus-connection">SDR++ Connection</div>
}));

vi.mock('./GroundSignalQuality', () => ({
  default: () => <div data-testid="ground-signal-quality">Ground Signal Quality</div>
}));

vi.mock('./FrequencySyncIndicators', () => ({
  default: () => <div data-testid="frequency-sync">Frequency Sync</div>
}));

vi.mock('./DualSystemHomingStatus', () => ({
  default: () => <div data-testid="dual-homing-status">Dual Homing Status</div>
}));

vi.mock('./EmergencyFallbackIndicators', () => ({
  default: () => <div data-testid="emergency-fallback">Emergency Fallback</div>
}));

const renderWithTheme = (component: React.ReactElement) => {
  return render(
    <ThemeProvider theme={theme}>
      {component}
    </ThemeProvider>
  );
};

describe('Dashboard Integration', () => {
  it('should render all existing components', () => {
    renderWithTheme(<Dashboard />);

    expect(screen.getByTestId('signal-meter')).toBeInTheDocument();
    expect(screen.getByTestId('rssi-graph')).toBeInTheDocument();
    expect(screen.getByTestId('sdr-status')).toBeInTheDocument();
    expect(screen.getByTestId('system-health')).toBeInTheDocument();
    expect(screen.getByTestId('detection-log')).toBeInTheDocument();
    expect(screen.getByTestId('sdrplus-connection')).toBeInTheDocument();
    expect(screen.getByTestId('ground-signal-quality')).toBeInTheDocument();
  });

  it('should render new SDR++ coordination components', () => {
    renderWithTheme(<Dashboard />);

    // These should be present after integration
    expect(screen.getByTestId('frequency-sync')).toBeInTheDocument();
    expect(screen.getByTestId('dual-homing-status')).toBeInTheDocument();
    expect(screen.getByTestId('emergency-fallback')).toBeInTheDocument();
  });

  it('should display dashboard title', () => {
    renderWithTheme(<Dashboard />);

    expect(screen.getByText('Signal Monitoring Dashboard')).toBeInTheDocument();
  });

  it('should arrange components in responsive grid layout', () => {
    renderWithTheme(<Dashboard />);

    // Check that the main container exists
    expect(screen.getByText('Signal Monitoring Dashboard').closest('.MuiGrid-container')).toBeInTheDocument();
  });
});
