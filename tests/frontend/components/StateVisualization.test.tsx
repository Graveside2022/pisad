import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import StateVisualization from "../../../src/frontend/src/components/dashboard/StateVisualization";
import { useWebSocket } from "../../../src/frontend/src/hooks/useWebSocket";
import { useStateOverride } from "../../../src/frontend/src/hooks/useStateOverride";

jest.mock("../../../src/frontend/src/hooks/useWebSocket");
jest.mock("../../../src/frontend/src/hooks/useStateOverride");

const mockStateData = {
  current_state: "IDLE",
  previous_state: "SEARCHING",
  timestamp: "2025-08-13T10:00:00Z",
  reason: "Signal lost",
  allowed_transitions: ["SEARCHING", "DETECTING"],
  state_duration_ms: 5000,
  history: [
    {
      from_state: "SEARCHING",
      to_state: "IDLE",
      timestamp: "2025-08-13T09:59:55Z",
      reason: "Signal lost",
    },
    {
      from_state: "IDLE",
      to_state: "SEARCHING",
      timestamp: "2025-08-13T09:59:00Z",
      reason: "Manual start",
    },
  ],
};

describe("StateVisualization", () => {
  let mockSubscribe: jest.Mock;
  let mockUnsubscribe: jest.Mock;
  let mockOverrideState: jest.Mock;

  beforeEach(() => {
    mockSubscribe = jest.fn().mockReturnValue(jest.fn());
    mockUnsubscribe = jest.fn();
    mockOverrideState = jest.fn();

    (useWebSocket as jest.Mock).mockReturnValue({
      subscribe: mockSubscribe,
      unsubscribe: mockUnsubscribe,
    });

    (useStateOverride as jest.Mock).mockReturnValue({
      overrideState: mockOverrideState,
      isLoading: false,
      error: null,
    });
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  it("renders waiting state when no data", () => {
    render(<StateVisualization />);
    expect(screen.getByText("Waiting for state data...")).toBeInTheDocument();
  });

  it("renders current state when data is available", () => {
    mockSubscribe.mockImplementation((event, handler) => {
      if (event === "state") {
        setTimeout(() => handler(mockStateData), 0);
      }
      return jest.fn();
    });

    render(<StateVisualization />);

    waitFor(() => {
      expect(screen.getByText("IDLE")).toBeInTheDocument();
      expect(screen.getByText("Duration: 5.0s")).toBeInTheDocument();
      expect(screen.getByText("Reason: Signal lost")).toBeInTheDocument();
    });
  });

  it("displays allowed transitions as chips", () => {
    mockSubscribe.mockImplementation((event, handler) => {
      if (event === "state") {
        setTimeout(() => handler(mockStateData), 0);
      }
      return jest.fn();
    });

    render(<StateVisualization />);

    waitFor(() => {
      expect(screen.getByText("SEARCHING")).toBeInTheDocument();
      expect(screen.getByText("DETECTING")).toBeInTheDocument();
    });
  });

  it("opens override dialog when button is clicked", () => {
    mockSubscribe.mockImplementation((event, handler) => {
      if (event === "state") {
        setTimeout(() => handler(mockStateData), 0);
      }
      return jest.fn();
    });

    render(<StateVisualization />);

    waitFor(() => {
      const overrideButton = screen.getByText("Manual Override");
      fireEvent.click(overrideButton);
      expect(screen.getByText("Manual State Override")).toBeInTheDocument();
    });
  });

  it("submits override request with correct data", async () => {
    mockSubscribe.mockImplementation((event, handler) => {
      if (event === "state") {
        setTimeout(() => handler(mockStateData), 0);
      }
      return jest.fn();
    });

    mockOverrideState.mockResolvedValue({
      success: true,
      previous_state: "IDLE",
      new_state: "SEARCHING",
      message: "State overridden successfully",
    });

    render(<StateVisualization />);

    await waitFor(() => {
      const overrideButton = screen.getByText("Manual Override");
      fireEvent.click(overrideButton);
    });

    const targetStateSelect = screen.getByLabelText("Target State");
    const reasonInput = screen.getByLabelText("Reason");
    const tokenInput = screen.getByLabelText("Confirmation Token");
    const operatorInput = screen.getByLabelText("Operator ID");

    fireEvent.change(targetStateSelect, { target: { value: "SEARCHING" } });
    fireEvent.change(reasonInput, { target: { value: "Test override" } });
    fireEvent.change(tokenInput, { target: { value: "OVERRIDE" } });
    fireEvent.change(operatorInput, { target: { value: "test-operator" } });

    const submitButton = screen.getByRole("button", { name: "Override State" });
    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(mockOverrideState).toHaveBeenCalledWith({
        target_state: "SEARCHING",
        reason: "Test override",
        confirmation_token: "OVERRIDE",
        operator_id: "test-operator",
      });
    });
  });

  it("displays state history correctly", () => {
    mockSubscribe.mockImplementation((event, handler) => {
      if (event === "state") {
        setTimeout(() => handler(mockStateData), 0);
      }
      return jest.fn();
    });

    render(<StateVisualization />);

    waitFor(() => {
      expect(screen.getByText("State History")).toBeInTheDocument();
      expect(screen.getByText("Signal lost")).toBeInTheDocument();
      expect(screen.getByText("Manual start")).toBeInTheDocument();
    });
  });

  it("displays error when override fails", async () => {
    mockSubscribe.mockImplementation((event, handler) => {
      if (event === "state") {
        setTimeout(() => handler(mockStateData), 0);
      }
      return jest.fn();
    });

    mockOverrideState.mockRejectedValue(new Error("Override failed"));

    render(<StateVisualization />);

    await waitFor(() => {
      const overrideButton = screen.getByText("Manual Override");
      fireEvent.click(overrideButton);
    });

    const tokenInput = screen.getByLabelText("Confirmation Token");
    const operatorInput = screen.getByLabelText("Operator ID");

    fireEvent.change(tokenInput, { target: { value: "OVERRIDE" } });
    fireEvent.change(operatorInput, { target: { value: "test-operator" } });

    const submitButton = screen.getByRole("button", { name: "Override State" });
    fireEvent.click(submitButton);

    await waitFor(() => {
      expect(screen.getByText(/Failed to override state/)).toBeInTheDocument();
    });
  });
});
