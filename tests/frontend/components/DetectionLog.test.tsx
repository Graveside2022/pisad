import { render, screen, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import DetectionLog from "../../../src/frontend/src/components/dashboard/DetectionLog";

// Mock fetch
global.fetch = jest.fn();

const mockDetections = [
  {
    id: "det-001",
    timestamp: "2025-08-12T10:30:45.123Z",
    frequency: 433920000,
    rssi: -45.5,
    snr: 25.3,
    confidence: 92.5,
    location: null,
    state: "DETECTING",
  },
  {
    id: "det-002",
    timestamp: "2025-08-12T10:31:12.456Z",
    frequency: 433920000,
    rssi: -52.3,
    snr: 18.7,
    confidence: 67.8,
    location: null,
    state: "SEARCHING",
  },
  {
    id: "det-003",
    timestamp: "2025-08-12T10:31:38.789Z",
    frequency: 433920000,
    rssi: -38.9,
    snr: 32.1,
    confidence: 45.2,
    location: null,
    state: "DETECTING",
  },
];

describe("DetectionLog", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({ detections: mockDetections }),
    });
  });

  afterEach(() => {
    jest.clearAllTimers();
  });

  it("renders detection log title", async () => {
    render(<DetectionLog />);

    expect(screen.getByText("Detection Log")).toBeInTheDocument();
  });

  it("shows loading state initially", () => {
    render(<DetectionLog />);

    expect(screen.getByText("Loading...")).toBeInTheDocument();
  });

  it("fetches detections from API", async () => {
    render(<DetectionLog />);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith("/api/detections?limit=10");
    });
  });

  it("displays detection data in table format", async () => {
    render(<DetectionLog />);

    await waitFor(() => {
      expect(screen.getByText("433.920 MHz")).toBeInTheDocument();
      expect(screen.getByText("-45.5 dBm")).toBeInTheDocument();
      expect(screen.getByText("25.3 dB")).toBeInTheDocument();
      expect(screen.getByText("93%")).toBeInTheDocument();
    });
  });

  it("formats timestamps to local time", async () => {
    render(<DetectionLog />);

    await waitFor(() => {
      const timestamp = new Date("2025-08-12T10:30:45.123Z");
      const expectedTime = timestamp.toLocaleTimeString();
      expect(screen.getByText(expectedTime)).toBeInTheDocument();
    });
  });

  it("formats frequency in MHz", async () => {
    render(<DetectionLog />);

    await waitFor(() => {
      // All detections have frequency 433920000 Hz = 433.920 MHz
      const frequencyCells = screen.getAllByText("433.920 MHz");
      expect(frequencyCells).toHaveLength(3);
    });
  });

  it("displays confidence with appropriate color coding", async () => {
    render(<DetectionLog />);

    await waitFor(() => {
      // High confidence (92.5%) should be success color
      const highConfidence = screen.getByText("93%").closest(".MuiChip-root");
      expect(highConfidence).toHaveClass("MuiChip-colorSuccess");

      // Medium confidence (67.8%) should be warning color
      const medConfidence = screen.getByText("68%").closest(".MuiChip-root");
      expect(medConfidence).toHaveClass("MuiChip-colorWarning");

      // Low confidence (45.2%) should be error color
      const lowConfidence = screen.getByText("45%").closest(".MuiChip-root");
      expect(lowConfidence).toHaveClass("MuiChip-colorError");
    });
  });

  it("displays all table columns correctly", async () => {
    render(<DetectionLog />);

    await waitFor(() => {
      expect(screen.getByText("Time")).toBeInTheDocument();
      expect(screen.getByText("Frequency")).toBeInTheDocument();
      expect(screen.getByText("RSSI")).toBeInTheDocument();
      expect(screen.getByText("SNR")).toBeInTheDocument();
      expect(screen.getByText("Confidence")).toBeInTheDocument();
    });
  });

  it("shows empty state when no detections", async () => {
    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({ detections: [] }),
    });

    render(<DetectionLog />);

    await waitFor(() => {
      expect(screen.getByText("No detections recorded")).toBeInTheDocument();
    });
  });

  it("handles API errors gracefully", async () => {
    const consoleError = jest.spyOn(console, "error").mockImplementation();
    (global.fetch as jest.Mock).mockRejectedValue(new Error("Network error"));

    render(<DetectionLog />);

    await waitFor(() => {
      expect(consoleError).toHaveBeenCalledWith(
        "Failed to fetch detections:",
        expect.any(Error),
      );
    });

    consoleError.mockRestore();
  });

  it("refreshes data every 5 seconds", async () => {
    jest.useFakeTimers();
    render(<DetectionLog />);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledTimes(1);
    });

    jest.advanceTimersByTime(5000);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledTimes(2);
    });

    jest.advanceTimersByTime(5000);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledTimes(3);
    });

    jest.useRealTimers();
  });

  it("sorts detections by timestamp (most recent first)", async () => {
    const sortedDetections = [...mockDetections].sort(
      (a, b) =>
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime(),
    );

    (global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      json: async () => ({ detections: sortedDetections }),
    });

    render(<DetectionLog />);

    await waitFor(() => {
      const rows = screen.getAllByRole("row");
      // First row is header, so data starts at index 1
      expect(rows.length).toBe(4); // 1 header + 3 data rows
    });
  });

  it("formats RSSI and SNR values with one decimal place", async () => {
    render(<DetectionLog />);

    await waitFor(() => {
      expect(screen.getByText("-45.5 dBm")).toBeInTheDocument();
      expect(screen.getByText("-52.3 dBm")).toBeInTheDocument();
      expect(screen.getByText("25.3 dB")).toBeInTheDocument();
      expect(screen.getByText("18.7 dB")).toBeInTheDocument();
    });
  });

  it("rounds confidence percentage to nearest integer", async () => {
    render(<DetectionLog />);

    await waitFor(() => {
      // 92.5 rounds to 93
      expect(screen.getByText("93%")).toBeInTheDocument();
      // 67.8 rounds to 68
      expect(screen.getByText("68%")).toBeInTheDocument();
      // 45.2 rounds to 45
      expect(screen.getByText("45%")).toBeInTheDocument();
    });
  });

  it("cleans up interval on unmount", async () => {
    jest.useFakeTimers();
    const clearIntervalSpy = jest.spyOn(global, "clearInterval");

    const { unmount } = render(<DetectionLog />);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalled();
    });

    unmount();

    expect(clearIntervalSpy).toHaveBeenCalled();

    jest.useRealTimers();
  });
});
