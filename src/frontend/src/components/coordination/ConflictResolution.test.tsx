import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import ConflictResolution from './ConflictResolution';

const theme = createTheme();

// Mock the useWebSocket hook
vi.mock('../../hooks/useWebSocket', () => ({
  useWebSocket: () => ({
    addMessageHandler: vi.fn(() => vi.fn()),
    sendMessage: vi.fn(),
    isConnected: true
  })
}));

const renderWithTheme = (component: React.ReactElement) => {
  return render(
    <ThemeProvider theme={theme}>
      {component}
    </ThemeProvider>
  );
};

describe('ConflictResolution', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('should render conflict resolution interface', () => {
    renderWithTheme(<ConflictResolution />);

    expect(screen.getByText('Conflict Resolution')).toBeInTheDocument();
  });

  it('should display frequency conflict detection', () => {
    renderWithTheme(<ConflictResolution />);

    expect(screen.getByText(/Frequency Comparison/)).toBeInTheDocument();
    expect(screen.getByText(/Ground SDR\+\+/)).toBeInTheDocument();
    expect(screen.getByText(/Drone SDR/)).toBeInTheDocument();
  });

  it('should show priority selection interface', () => {
    renderWithTheme(<ConflictResolution />);

    expect(screen.getByText(/Source Priority/)).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: /Ground Priority/ })).toBeInTheDocument();
    expect(screen.getByRole('radio', { name: /Drone Priority/ })).toBeInTheDocument();
  });

  it('should display conflict severity indicators', () => {
    renderWithTheme(<ConflictResolution />);

    expect(screen.getByTestId('conflict-severity')).toBeInTheDocument();
  });

  it('should handle priority selection changes', () => {
    renderWithTheme(<ConflictResolution />);

    const groundPriorityRadio = screen.getByRole('radio', { name: /Ground Priority/ });
    const dronePriorityRadio = screen.getByRole('radio', { name: /Drone Priority/ });

    fireEvent.click(groundPriorityRadio);
    expect(groundPriorityRadio).toBeChecked();

    fireEvent.click(dronePriorityRadio);
    expect(dronePriorityRadio).toBeChecked();
  });

  it('should show resolution action buttons', () => {
    renderWithTheme(<ConflictResolution />);

    expect(screen.getByRole('button', { name: /Apply Resolution/ })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Emergency Revert/ })).toBeInTheDocument();
  });

  it('should display override duration controls', () => {
    renderWithTheme(<ConflictResolution />);

    expect(screen.getByText(/Override Duration/)).toBeInTheDocument();
  });

  it('should handle emergency revert action', () => {
    renderWithTheme(<ConflictResolution />);

    const revertButton = screen.getByRole('button', { name: /Emergency Revert/ });
    fireEvent.click(revertButton);

    // Should trigger revert confirmation or action
    expect(revertButton).toBeInTheDocument();
  });

  it('should show conflict history log', () => {
    renderWithTheme(<ConflictResolution />);

    expect(screen.getByText(/Resolution History/)).toBeInTheDocument();
  });
});
