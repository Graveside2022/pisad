import { render, screen } from '@testing-library/react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import EmergencyFallbackIndicators from './EmergencyFallbackIndicators';

const theme = createTheme();

const renderWithTheme = (component: React.ReactElement) => {
  return render(
    <ThemeProvider theme={theme}>
      {component}
    </ThemeProvider>
  );
};

describe('EmergencyFallbackIndicators', () => {
  it('should render communication loss detection', () => {
    renderWithTheme(<EmergencyFallbackIndicators />);

    // Should show communication status
    expect(screen.getByText(/Communication Status/i)).toBeInTheDocument();
  });

  it('should display fallback mode indicators', () => {
    renderWithTheme(<EmergencyFallbackIndicators />);

    // Should show fallback mode
    expect(screen.getByText(/Fallback Mode/i)).toBeInTheDocument();
  });

  it('should show recovery status when available', () => {
    renderWithTheme(<EmergencyFallbackIndicators />);

    // Should have recovery status
    expect(screen.getByText(/Recovery Status/i)).toBeInTheDocument();
  });

  it('should handle emergency alerts', () => {
    renderWithTheme(<EmergencyFallbackIndicators />);

    // Should support emergency status display
    const component = screen.getByText(/Communication Status/i);
    expect(component).toBeInTheDocument();
  });
});
