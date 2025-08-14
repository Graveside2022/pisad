import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import { HomingControl } from "../../../src/frontend/src/components/homing/HomingControl";
import { api } from "../../../src/frontend/src/services/api";
import * as useSystemStateModule from "../../../src/frontend/src/hooks/useSystemState";

jest.mock("../../../src/frontend/src/services/api");
jest.mock("../../../src/frontend/src/hooks/useSystemState");

describe("HomingControl", () => {
  const mockSystemState = {
    homingEnabled: false,
    currentState: "IDLE",
    flightMode: "GUIDED",
    batteryPercent: 85,
    gpsStatus: "3D_FIX",
    mavlinkConnected: true,
    sdrStatus: "CONNECTED",
    safetyInterlocks: {
      modeCheck: true,
      batteryCheck: true,
      geofenceCheck: true,
      signalCheck: true,
      operatorCheck: true,
    },
  };

  beforeEach(() => {
    jest.clearAllMocks();
    (useSystemStateModule.useSystemState as jest.Mock).mockReturnValue(
      mockSystemState,
    );
  });

  describe("Initial Rendering", () => {
    it("should render with disabled state initially", () => {
      render(<HomingControl />);

      expect(screen.getByText("Homing Control")).toBeInTheDocument();
      expect(screen.getByText("HOMING DISABLED")).toBeInTheDocument();
      expect(screen.getByText("ENABLE")).toBeInTheDocument();
    });

    it("should render with enabled state when homingEnabled is true", () => {
      (useSystemStateModule.useSystemState as jest.Mock).mockReturnValue({
        ...mockSystemState,
        homingEnabled: true,
      });

      render(<HomingControl />);

      expect(screen.getByText("HOMING ENABLED")).toBeInTheDocument();
      expect(screen.getByText("DISABLE")).toBeInTheDocument();
    });

    it("should render with active homing state", () => {
      (useSystemStateModule.useSystemState as jest.Mock).mockReturnValue({
        ...mockSystemState,
        homingEnabled: true,
        currentState: "HOMING",
      });

      render(<HomingControl />);

      expect(screen.getByText("HOMING ACTIVE")).toBeInTheDocument();
    });
  });

  describe("Toggle Button Interaction", () => {
    it("should open confirmation dialog when enabling homing", () => {
      render(<HomingControl />);

      const toggleButton = screen.getByText("ENABLE").closest("button");
      fireEvent.click(toggleButton!);

      expect(screen.getByText("Confirm Homing Activation")).toBeInTheDocument();
      expect(
        screen.getByText(/WARNING: Enabling homing mode/),
      ).toBeInTheDocument();
    });

    it("should close confirmation dialog on cancel", () => {
      render(<HomingControl />);

      const toggleButton = screen.getByText("ENABLE").closest("button");
      fireEvent.click(toggleButton!);

      const cancelButton = screen.getByText("Cancel");
      fireEvent.click(cancelButton);

      expect(
        screen.queryByText("Confirm Homing Activation"),
      ).not.toBeInTheDocument();
    });

    it("should call API to enable homing on confirmation", async () => {
      (api.setHomingState as jest.Mock).mockResolvedValue({
        homingEnabled: true,
        message: "Homing enabled successfully",
      });

      render(<HomingControl />);

      const toggleButton = screen.getByText("ENABLE").closest("button");
      fireEvent.click(toggleButton!);

      const confirmButton = screen.getByText("Confirm & Enable");
      fireEvent.click(confirmButton);

      await waitFor(() => {
        expect(api.setHomingState).toHaveBeenCalledWith(
          true,
          expect.stringMatching(/^confirm-\d+-[a-z0-9]+$/),
        );
      });
    });

    it("should call API to disable homing without confirmation", async () => {
      (useSystemStateModule.useSystemState as jest.Mock).mockReturnValue({
        ...mockSystemState,
        homingEnabled: true,
      });

      (api.setHomingState as jest.Mock).mockResolvedValue({
        homingEnabled: false,
        message: "Homing disabled successfully",
      });

      render(<HomingControl />);

      const toggleButton = screen.getByText("DISABLE").closest("button");
      fireEvent.click(toggleButton!);

      await waitFor(() => {
        expect(api.setHomingState).toHaveBeenCalledWith(false, "");
      });
    });
  });

  describe("Error Handling", () => {
    it("should display error when safety interlocks block activation", async () => {
      (api.setHomingState as jest.Mock).mockRejectedValue({
        response: {
          data: {
            error: "Safety interlock blocked",
            blockedBy: ["modeCheck", "batteryCheck"],
          },
        },
      });

      render(<HomingControl />);

      const toggleButton = screen.getByText("ENABLE").closest("button");
      fireEvent.click(toggleButton!);

      const confirmButton = screen.getByText("Confirm & Enable");
      fireEvent.click(confirmButton);

      await waitFor(() => {
        expect(
          screen.getByText(/Safety interlock blocked: modeCheck, batteryCheck/),
        ).toBeInTheDocument();
      });
    });

    it("should display generic error message on API failure", async () => {
      (api.setHomingState as jest.Mock).mockRejectedValue({
        response: {
          data: {
            error: "Network error",
          },
        },
      });

      render(<HomingControl />);

      const toggleButton = screen.getByText("ENABLE").closest("button");
      fireEvent.click(toggleButton!);

      const confirmButton = screen.getByText("Confirm & Enable");
      fireEvent.click(confirmButton);

      await waitFor(() => {
        expect(screen.getByText(/Network error/)).toBeInTheDocument();
      });
    });
  });

  describe("Override Instructions", () => {
    it("should display override instructions when homing is enabled", () => {
      (useSystemStateModule.useSystemState as jest.Mock).mockReturnValue({
        ...mockSystemState,
        homingEnabled: true,
      });

      render(<HomingControl />);

      expect(screen.getByText(/To regain control:/)).toBeInTheDocument();
      expect(
        screen.getByText(/Switch flight mode in Mission Planner/),
      ).toBeInTheDocument();
    });

    it("should not display override instructions when homing is disabled", () => {
      render(<HomingControl />);

      expect(screen.queryByText(/To regain control:/)).not.toBeInTheDocument();
    });
  });

  describe("State Change Callback", () => {
    it("should call onStateChange callback when enabling homing", async () => {
      const onStateChange = jest.fn();
      (api.setHomingState as jest.Mock).mockResolvedValue({
        homingEnabled: true,
        message: "Homing enabled successfully",
      });

      render(<HomingControl onStateChange={onStateChange} />);

      const toggleButton = screen.getByText("ENABLE").closest("button");
      fireEvent.click(toggleButton!);

      const confirmButton = screen.getByText("Confirm & Enable");
      fireEvent.click(confirmButton);

      await waitFor(() => {
        expect(onStateChange).toHaveBeenCalledWith(true);
      });
    });

    it("should call onStateChange callback when disabling homing", async () => {
      const onStateChange = jest.fn();
      (useSystemStateModule.useSystemState as jest.Mock).mockReturnValue({
        ...mockSystemState,
        homingEnabled: true,
      });

      (api.setHomingState as jest.Mock).mockResolvedValue({
        homingEnabled: false,
        message: "Homing disabled successfully",
      });

      render(<HomingControl onStateChange={onStateChange} />);

      const toggleButton = screen.getByText("DISABLE").closest("button");
      fireEvent.click(toggleButton!);

      await waitFor(() => {
        expect(onStateChange).toHaveBeenCalledWith(false);
      });
    });
  });
});
