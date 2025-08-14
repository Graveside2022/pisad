import React from "react";
import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import { SafetyInterlocks } from "../../../src/frontend/src/components/homing/SafetyInterlocks";
import * as useSystemStateModule from "../../../src/frontend/src/hooks/useSystemState";

jest.mock("../../../src/frontend/src/hooks/useSystemState");

describe("SafetyInterlocks", () => {
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
  });

  describe("All Checks Passing", () => {
    it("should display ALL SYSTEMS GO when all checks pass", () => {
      (useSystemStateModule.useSystemState as jest.Mock).mockReturnValue(
        mockSystemState,
      );

      render(<SafetyInterlocks />);

      expect(screen.getByText("ALL SYSTEMS GO")).toBeInTheDocument();
    });

    it("should display PASS status for all checks", () => {
      (useSystemStateModule.useSystemState as jest.Mock).mockReturnValue(
        mockSystemState,
      );

      render(<SafetyInterlocks />);

      const passChips = screen.getAllByText("PASS");
      expect(passChips).toHaveLength(6);
    });

    it("should not display warning message when all checks pass", () => {
      (useSystemStateModule.useSystemState as jest.Mock).mockReturnValue(
        mockSystemState,
      );

      render(<SafetyInterlocks />);

      expect(
        screen.queryByText(/Safety checks failed/),
      ).not.toBeInTheDocument();
    });
  });

  describe("Flight Mode Check", () => {
    it("should display correct flight mode", () => {
      (useSystemStateModule.useSystemState as jest.Mock).mockReturnValue(
        mockSystemState,
      );

      render(<SafetyInterlocks />);

      expect(screen.getByText("Flight Mode Check")).toBeInTheDocument();
      expect(screen.getByText(/Current mode:.*GUIDED/)).toBeInTheDocument();
      expect(screen.getByText(/Required: GUIDED/)).toBeInTheDocument();
    });

    it("should show FAIL status when not in GUIDED mode", () => {
      (useSystemStateModule.useSystemState as jest.Mock).mockReturnValue({
        ...mockSystemState,
        flightMode: "MANUAL",
        safetyInterlocks: {
          ...mockSystemState.safetyInterlocks,
          modeCheck: false,
        },
      });

      render(<SafetyInterlocks />);

      expect(screen.getByText(/Current mode:.*MANUAL/)).toBeInTheDocument();
      const failChips = screen.getAllByText("FAIL");
      expect(failChips.length).toBeGreaterThan(0);
    });
  });

  describe("Battery Check", () => {
    it("should display battery percentage", () => {
      (useSystemStateModule.useSystemState as jest.Mock).mockReturnValue(
        mockSystemState,
      );

      render(<SafetyInterlocks />);

      expect(screen.getByText("Battery Level Check")).toBeInTheDocument();
      expect(screen.getByText(/Current:.*85%/)).toBeInTheDocument();
      expect(screen.getByText(/Minimum: 20%/)).toBeInTheDocument();
    });

    it("should show FAIL status when battery is low", () => {
      (useSystemStateModule.useSystemState as jest.Mock).mockReturnValue({
        ...mockSystemState,
        batteryPercent: 15,
        safetyInterlocks: {
          ...mockSystemState.safetyInterlocks,
          batteryCheck: false,
        },
      });

      render(<SafetyInterlocks />);

      expect(screen.getByText(/Current:.*15%/)).toBeInTheDocument();
      const failChips = screen.getAllByText("FAIL");
      expect(failChips.length).toBeGreaterThan(0);
    });
  });

  describe("Armed Status Check", () => {
    it("should display MAVLink connection status", () => {
      (useSystemStateModule.useSystemState as jest.Mock).mockReturnValue(
        mockSystemState,
      );

      render(<SafetyInterlocks />);

      expect(screen.getByText("Armed Status Check")).toBeInTheDocument();
      expect(screen.getByText(/MAVLink:.*Connected/)).toBeInTheDocument();
    });

    it("should show FAIL when MAVLink is disconnected", () => {
      (useSystemStateModule.useSystemState as jest.Mock).mockReturnValue({
        ...mockSystemState,
        mavlinkConnected: false,
      });

      render(<SafetyInterlocks />);

      expect(screen.getByText(/MAVLink:.*Disconnected/)).toBeInTheDocument();
      const failChips = screen.getAllByText("FAIL");
      expect(failChips.length).toBeGreaterThan(0);
    });
  });

  describe("Signal Quality Check", () => {
    it("should display signal quality status", () => {
      (useSystemStateModule.useSystemState as jest.Mock).mockReturnValue(
        mockSystemState,
      );

      render(<SafetyInterlocks />);

      expect(screen.getByText("Signal Quality Check")).toBeInTheDocument();
      expect(
        screen.getByText(/Signal timeout: 10 seconds/),
      ).toBeInTheDocument();
    });

    it("should show FAIL when signal check fails", () => {
      (useSystemStateModule.useSystemState as jest.Mock).mockReturnValue({
        ...mockSystemState,
        safetyInterlocks: {
          ...mockSystemState.safetyInterlocks,
          signalCheck: false,
        },
      });

      render(<SafetyInterlocks />);

      const failChips = screen.getAllByText("FAIL");
      expect(failChips.length).toBeGreaterThan(0);
    });
  });

  describe("Geofence Check", () => {
    it("should display GPS status", () => {
      (useSystemStateModule.useSystemState as jest.Mock).mockReturnValue(
        mockSystemState,
      );

      render(<SafetyInterlocks />);

      expect(screen.getByText("Geofence Boundary Check")).toBeInTheDocument();
      expect(screen.getByText(/GPS:.*3D_FIX/)).toBeInTheDocument();
    });

    it("should show different GPS status levels", () => {
      (useSystemStateModule.useSystemState as jest.Mock).mockReturnValue({
        ...mockSystemState,
        gpsStatus: "NO_FIX",
      });

      render(<SafetyInterlocks />);

      expect(screen.getByText(/GPS:.*NO_FIX/)).toBeInTheDocument();
    });
  });

  describe("Multiple Failures", () => {
    it("should display CHECKS FAILED when any check fails", () => {
      (useSystemStateModule.useSystemState as jest.Mock).mockReturnValue({
        ...mockSystemState,
        safetyInterlocks: {
          modeCheck: false,
          batteryCheck: false,
          geofenceCheck: true,
          signalCheck: true,
          operatorCheck: true,
        },
      });

      render(<SafetyInterlocks />);

      expect(screen.getByText("CHECKS FAILED")).toBeInTheDocument();
    });

    it("should display warning message when checks fail", () => {
      (useSystemStateModule.useSystemState as jest.Mock).mockReturnValue({
        ...mockSystemState,
        safetyInterlocks: {
          ...mockSystemState.safetyInterlocks,
          modeCheck: false,
        },
      });

      render(<SafetyInterlocks />);

      expect(screen.getByText(/Safety checks failed/)).toBeInTheDocument();
      expect(
        screen.getByText(/Homing cannot be activated/),
      ).toBeInTheDocument();
    });
  });

  describe("No System State", () => {
    it("should handle null system state gracefully", () => {
      (useSystemStateModule.useSystemState as jest.Mock).mockReturnValue(null);

      render(<SafetyInterlocks />);

      expect(screen.getByText("Safety Interlocks")).toBeInTheDocument();
      expect(screen.getByText("CHECKS FAILED")).toBeInTheDocument();
    });

    it("should display default values when system state is missing", () => {
      (useSystemStateModule.useSystemState as jest.Mock).mockReturnValue(null);

      render(<SafetyInterlocks />);

      expect(screen.getByText(/Current mode:.*UNKNOWN/)).toBeInTheDocument();
      expect(screen.getByText(/Current:.*0%/)).toBeInTheDocument();
      expect(screen.getByText(/GPS:.*NO_FIX/)).toBeInTheDocument();
    });
  });
});
