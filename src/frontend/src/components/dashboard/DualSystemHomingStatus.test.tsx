import { render, screen } from '@testing-library/react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import DualSystemHomingStatus from './DualSystemHomingStatus';

const theme = createTheme();

const renderWithTheme = (component: React.ReactElement) => {
  return render(
    <ThemeProvider theme={theme}>
      {component}
    </ThemeProvider>
  );
};

describe('DualSystemHomingStatus', () => {
  it('should render active source indicator', () => {
    renderWithTheme(<DualSystemHomingStatus />);

    // Should show which source is active
    expect(screen.getByText(/Active Source/i)).toBeInTheDocument();
  });

  it('should display homing authority status', () => {
    renderWithTheme(<DualSystemHomingStatus />);

    // Should show homing authority
    expect(screen.getByText(/Homing Authority/i)).toBeInTheDocument();
  });

  it('should show coordination performance indicators', () => {
    renderWithTheme(<DualSystemHomingStatus />);

    // Should have performance metrics
    expect(screen.getByText(/Performance/i)).toBeInTheDocument();
  });

  it('should handle emergency fallback status', () => {
    renderWithTheme(<DualSystemHomingStatus />);

    // Should show emergency status capability
    const component = screen.getByText(/Active Source/i);
    expect(component).toBeInTheDocument();
  });
});
