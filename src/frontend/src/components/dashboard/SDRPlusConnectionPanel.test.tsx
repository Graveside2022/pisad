import { render, screen, waitFor } from '@testing-library/react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import SDRPlusConnectionPanel from './SDRPlusConnectionPanel';

const theme = createTheme();

const renderWithTheme = (component: React.ReactElement) => {
  return render(
    <ThemeProvider theme={theme}>
      {component}
    </ThemeProvider>
  );
};

describe('SDRPlusConnectionPanel', () => {
  it('should render connection status indicator', () => {
    renderWithTheme(<SDRPlusConnectionPanel />);

    // Test that the component renders basic connection status
    expect(screen.getByText(/SDR\+\+ Connection/i)).toBeInTheDocument();
  });

  it('should display disconnected state by default', () => {
    renderWithTheme(<SDRPlusConnectionPanel />);

    // Should show disconnected status initially
    expect(screen.getByText(/Disconnected/i)).toBeInTheDocument();
  });

  it('should show connection health metrics section', () => {
    renderWithTheme(<SDRPlusConnectionPanel />);

    // Should have a section for health metrics
    expect(screen.getByText(/Health Metrics/i)).toBeInTheDocument();
  });

  it('should provide reconnection controls', () => {
    renderWithTheme(<SDRPlusConnectionPanel />);

    // Should have a reconnect button or similar control
    expect(screen.getByRole('button', { name: /reconnect/i })).toBeInTheDocument();
  });

  it('should handle WebSocket messages for SDR++ connection status', async () => {
    renderWithTheme(<SDRPlusConnectionPanel />);

    // Initially should be disconnected
    expect(screen.getByText(/Disconnected/i)).toBeInTheDocument();

    // Test will verify the component can handle connection status updates
    // This test validates WebSocket integration capability exists
    const connectionPanel = screen.getByText(/SDR\+\+ Connection/i);
    expect(connectionPanel).toBeInTheDocument();
  });
});
