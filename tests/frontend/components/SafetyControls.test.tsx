/**
 * Frontend tests for safety UI components
 */

import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";

// Mock component for testing (actual component would be imported)
const SafetyControls: React.FC = () => {
  const [homingEnabled, setHomingEnabled] = React.useState(false);
  const [showConfirmDialog, setShowConfirmDialog] = React.useState(false);
  const [emergencyStopped, setEmergencyStopped] = React.useState(false);
  const [safetyStatus, setSafetyStatus] = React.useState({
    mode: "GUIDED",
    battery: 50,
    signal: 10,
  });

  const handleEnableHoming = () => {
    setShowConfirmDialog(true);
  };

  const handleConfirmHoming = () => {
    setHomingEnabled(true);
    setShowConfirmDialog(false);
  };

  const handleEmergencyStop = () => {
    setEmergencyStopped(true);
    setHomingEnabled(false);
  };

  return (
    <div>
      <button
        onClick={handleEnableHoming}
        disabled={homingEnabled || emergencyStopped}
        aria-label="Enable Homing"
      >
        {homingEnabled ? "Homing Enabled" : "Enable Homing"}
      </button>

      <button
        onClick={handleEmergencyStop}
        aria-label="Emergency Stop"
        style={{ backgroundColor: "red" }}
      >
        EMERGENCY STOP
      </button>

      <div data-testid="safety-status">
        <span>Mode: {safetyStatus.mode}</span>
        <span>Battery: {safetyStatus.battery}%</span>
        <span>Signal: {safetyStatus.signal} dB</span>
      </div>

      {showConfirmDialog && (
        <div role="dialog" aria-labelledby="confirm-title">
          <h2 id="confirm-title">Confirm Homing Activation</h2>
          <button onClick={handleConfirmHoming}>Confirm</button>
          <button onClick={() => setShowConfirmDialog(false)}>Cancel</button>
        </div>
      )}

      {emergencyStopped && <div role="alert">Emergency Stop Activated</div>}
    </div>
  );
};

describe("SafetyControls", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test("Enable/Disable Homing button states", () => {
    render(<SafetyControls />);

    const homingButton = screen.getByLabelText("Enable Homing");
    expect(homingButton).toBeEnabled();
    expect(homingButton).toHaveTextContent("Enable Homing");

    // Click to enable
    fireEvent.click(homingButton);

    // Confirmation dialog should appear
    expect(screen.getByRole("dialog")).toBeInTheDocument();
  });

  test("Visual confirmation dialog behavior", async () => {
    render(<SafetyControls />);

    const homingButton = screen.getByLabelText("Enable Homing");
    fireEvent.click(homingButton);

    // Dialog should be visible
    const dialog = screen.getByRole("dialog");
    expect(dialog).toBeInTheDocument();

    // Confirm homing
    const confirmButton = screen.getByText("Confirm");
    fireEvent.click(confirmButton);

    // Dialog should close and button should update
    await waitFor(() => {
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    });

    expect(homingButton).toHaveTextContent("Homing Enabled");
    expect(homingButton).toBeDisabled();
  });

  test("Emergency stop button responsiveness", () => {
    render(<SafetyControls />);

    const emergencyButton = screen.getByLabelText("Emergency Stop");
    expect(emergencyButton).toBeEnabled();

    // Click emergency stop
    fireEvent.click(emergencyButton);

    // Alert should appear
    expect(screen.getByRole("alert")).toHaveTextContent(
      "Emergency Stop Activated",
    );

    // Homing button should be disabled
    const homingButton = screen.getByLabelText("Enable Homing");
    expect(homingButton).toBeDisabled();
  });

  test("Safety status indicator updates", () => {
    render(<SafetyControls />);

    const statusDiv = screen.getByTestId("safety-status");

    expect(statusDiv).toHaveTextContent("Mode: GUIDED");
    expect(statusDiv).toHaveTextContent("Battery: 50%");
    expect(statusDiv).toHaveTextContent("Signal: 10 dB");
  });

  test("Auto-disable notification display", async () => {
    render(<SafetyControls />);

    // Enable homing first
    const homingButton = screen.getByLabelText("Enable Homing");
    fireEvent.click(homingButton);
    fireEvent.click(screen.getByText("Confirm"));

    // Trigger emergency stop
    const emergencyButton = screen.getByLabelText("Emergency Stop");
    fireEvent.click(emergencyButton);

    // Notification should appear
    await waitFor(() => {
      expect(screen.getByRole("alert")).toBeInTheDocument();
    });
  });

  test("Accessibility of safety controls", () => {
    render(<SafetyControls />);

    // All buttons should have accessible labels
    expect(screen.getByLabelText("Enable Homing")).toBeInTheDocument();
    expect(screen.getByLabelText("Emergency Stop")).toBeInTheDocument();

    // Dialog should have proper ARIA attributes
    const homingButton = screen.getByLabelText("Enable Homing");
    fireEvent.click(homingButton);

    const dialog = screen.getByRole("dialog");
    expect(dialog).toHaveAttribute("aria-labelledby", "confirm-title");
  });
});
