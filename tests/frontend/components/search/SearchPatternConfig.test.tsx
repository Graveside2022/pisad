import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import SearchPatternConfig from "../../../../src/frontend/src/components/search/SearchPatternConfig";
import searchService from "../../../../src/frontend/src/services/search";

jest.mock("../../../../src/frontend/src/services/search");

describe("SearchPatternConfig", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test("renders pattern configuration form", () => {
    render(<SearchPatternConfig />);

    expect(
      screen.getByText("Search Pattern Configuration"),
    ).toBeInTheDocument();
    expect(screen.getByLabelText(/Pattern Type/i)).toBeInTheDocument();
    expect(screen.getByText(/Spacing:/i)).toBeInTheDocument();
    expect(screen.getByText(/Velocity:/i)).toBeInTheDocument();
    expect(screen.getByText(/Boundary Type/i)).toBeInTheDocument();
  });

  test("changes pattern type", () => {
    render(<SearchPatternConfig />);

    const select = screen.getByLabelText(/Pattern Type/i);
    fireEvent.mouseDown(select);

    const spiralOption = screen.getByText("Spiral");
    fireEvent.click(spiralOption);

    expect(select).toHaveTextContent("Spiral");
  });

  test("adjusts spacing slider", () => {
    render(<SearchPatternConfig />);

    const spacingText = screen.getByText(/Spacing:/i);
    expect(spacingText.textContent).toContain("75m");

    const slider = spacingText.parentElement?.querySelector(
      'input[type="range"]',
    );
    if (slider) {
      fireEvent.change(slider, { target: { value: "90" } });
      expect(spacingText.textContent).toContain("90m");
    }
  });

  test("switches between boundary types", () => {
    render(<SearchPatternConfig />);

    const cornersButton = screen.getByRole("button", {
      name: /Corner Coordinates/i,
    });
    fireEvent.click(cornersButton);

    expect(screen.getByText("Corner 1")).toBeInTheDocument();
    expect(screen.getByText("Corner 2")).toBeInTheDocument();
    expect(screen.getByText("Corner 3")).toBeInTheDocument();
    expect(screen.getByText("Corner 4")).toBeInTheDocument();

    const centerRadiusButton = screen.getByRole("button", {
      name: /Center \+ Radius/i,
    });
    fireEvent.click(centerRadiusButton);

    expect(screen.getByLabelText(/Center Latitude/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Center Longitude/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Radius/i)).toBeInTheDocument();
  });

  test("validates center radius inputs", async () => {
    render(<SearchPatternConfig />);

    const createButton = screen.getByRole("button", {
      name: /Create Pattern/i,
    });
    fireEvent.click(createButton);

    await waitFor(() => {
      expect(
        screen.getByText(/Please fill in all center and radius fields/i),
      ).toBeInTheDocument();
    });
  });

  test("creates pattern successfully", async () => {
    const mockCreatePattern = jest.fn().mockResolvedValue({
      pattern_id: "test-123",
      waypoint_count: 42,
      estimated_duration: 600,
      total_distance: 4200,
    });
    (searchService.createPattern as jest.Mock) = mockCreatePattern;

    const onPatternCreated = jest.fn();
    render(<SearchPatternConfig onPatternCreated={onPatternCreated} />);

    // Fill in center radius form
    fireEvent.change(screen.getByLabelText(/Center Latitude/i), {
      target: { value: "37.7749" },
    });
    fireEvent.change(screen.getByLabelText(/Center Longitude/i), {
      target: { value: "-122.4194" },
    });
    fireEvent.change(screen.getByLabelText(/Radius/i), {
      target: { value: "500" },
    });

    const createButton = screen.getByRole("button", {
      name: /Create Pattern/i,
    });
    fireEvent.click(createButton);

    await waitFor(() => {
      expect(mockCreatePattern).toHaveBeenCalledWith({
        pattern: "expanding_square",
        spacing: 75,
        velocity: 7,
        bounds: {
          type: "center_radius",
          center: { lat: 37.7749, lon: -122.4194 },
          radius: 500,
        },
      });
      expect(onPatternCreated).toHaveBeenCalledWith("test-123");
    });
  });

  test("handles create pattern error", async () => {
    const mockCreatePattern = jest
      .fn()
      .mockRejectedValue(new Error("API Error"));
    (searchService.createPattern as jest.Mock) = mockCreatePattern;

    render(<SearchPatternConfig />);

    // Fill in form
    fireEvent.change(screen.getByLabelText(/Center Latitude/i), {
      target: { value: "37.7749" },
    });
    fireEvent.change(screen.getByLabelText(/Center Longitude/i), {
      target: { value: "-122.4194" },
    });
    fireEvent.change(screen.getByLabelText(/Radius/i), {
      target: { value: "500" },
    });

    const createButton = screen.getByRole("button", {
      name: /Create Pattern/i,
    });
    fireEvent.click(createButton);

    await waitFor(() => {
      expect(screen.getByText(/API Error/i)).toBeInTheDocument();
    });
  });

  test("triggers preview callback", () => {
    const onPreviewRequested = jest.fn();
    render(<SearchPatternConfig onPreviewRequested={onPreviewRequested} />);

    // Fill in form
    fireEvent.change(screen.getByLabelText(/Center Latitude/i), {
      target: { value: "37.7749" },
    });
    fireEvent.change(screen.getByLabelText(/Center Longitude/i), {
      target: { value: "-122.4194" },
    });
    fireEvent.change(screen.getByLabelText(/Radius/i), {
      target: { value: "500" },
    });

    const previewButton = screen.getByRole("button", {
      name: /Preview Pattern/i,
    });
    fireEvent.click(previewButton);

    expect(onPreviewRequested).toHaveBeenCalledWith({
      pattern: "expanding_square",
      spacing: 75,
      velocity: 7,
      bounds: {
        type: "center_radius",
        center: { lat: 37.7749, lon: -122.4194 },
        radius: 500,
      },
    });
  });
});
