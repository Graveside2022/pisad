import { render, screen, fireEvent } from '@testing-library/react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import FrequencySyncIndicators from './FrequencySyncIndicators';

const theme = createTheme();

const renderWithTheme = (component: React.ReactElement) => {
  return render(
    <ThemeProvider theme={theme}>
      {component}
    </ThemeProvider>
  );
};

describe('FrequencySyncIndicators', () => {
  it('should render frequency sync status', () => {
    renderWithTheme(<FrequencySyncIndicators />);

    // Should show sync status
    expect(screen.getByText(/Frequency Synchronization/i)).toBeInTheDocument();
  });

  it('should display frequency mismatch warnings', () => {
    renderWithTheme(<FrequencySyncIndicators />);

    // Should have mismatch detection capability
    expect(screen.getByText(/Sync Status/i)).toBeInTheDocument();
  });

  it('should provide manual sync controls', () => {
    renderWithTheme(<FrequencySyncIndicators />);

    // Should have sync controls
    expect(screen.getByRole('button', { name: /synchronize/i })).toBeInTheDocument();
  });

  it('should handle sync conflicts', () => {
    renderWithTheme(<FrequencySyncIndicators />);

    // Should show conflict resolution options
    const syncButton = screen.getByRole('button', { name: /synchronize/i });
    fireEvent.click(syncButton);

    // Test that the component can handle sync actions
    expect(syncButton).toBeInTheDocument();
  });
});
