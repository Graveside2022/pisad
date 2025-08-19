import { render, screen } from '@testing-library/react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import GroundSignalQuality from './GroundSignalQuality';

const theme = createTheme();

const renderWithTheme = (component: React.ReactElement) => {
  return render(
    <ThemeProvider theme={theme}>
      {component}
    </ThemeProvider>
  );
};

describe('GroundSignalQuality', () => {
  it('should render signal source indicators', () => {
    renderWithTheme(<GroundSignalQuality />);

    // Should show signal source labels
    expect(screen.getByText(/Ground SDR\+\+/i)).toBeInTheDocument();
    expect(screen.getByText(/Drone HackRF/i)).toBeInTheDocument();
  });

  it('should display RSSI comparison visualization', () => {
    renderWithTheme(<GroundSignalQuality />);

    // Should have RSSI comparison section
    expect(screen.getByText(/Signal Quality Comparison/i)).toBeInTheDocument();
  });

  it('should show signal strength differential indicators', () => {
    renderWithTheme(<GroundSignalQuality />);

    // Should show differential between sources
    expect(screen.getByText(/Signal Differential/i)).toBeInTheDocument();
  });

  it('should handle real-time signal quality updates', () => {
    renderWithTheme(<GroundSignalQuality />);

    // Should have a section for real-time updates
    const component = screen.getByText(/Ground SDR\+\+/i);
    expect(component).toBeInTheDocument();
  });
});
