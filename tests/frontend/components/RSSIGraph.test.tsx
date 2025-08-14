import { render, screen, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import RSSIGraph from "../../../src/frontend/src/components/dashboard/RSSIGraph";

// Mock recharts to avoid rendering issues in tests
jest.mock("recharts", () => ({
  LineChart: ({ children, data }: any) => (
    <div data-testid="line-chart" data-points={data?.length || 0}>
      {children}
    </div>
  ),
  Line: ({ dataKey, name }: any) => (
    <div data-testid={`line-${dataKey}`}>{name}</div>
  ),
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  CartesianGrid: () => <div data-testid="grid" />,
  Tooltip: () => <div data-testid="tooltip" />,
  ResponsiveContainer: ({ children }: any) => <div>{children}</div>,
  Legend: () => <div data-testid="legend" />,
}));

describe("RSSIGraph", () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it("renders graph title correctly", () => {
    render(<RSSIGraph />);

    expect(screen.getByText("RSSI History (60 seconds)")).toBeInTheDocument();
  });

  it("renders line chart components", () => {
    render(<RSSIGraph />);

    expect(screen.getByTestId("line-chart")).toBeInTheDocument();
    expect(screen.getByTestId("line-rssi")).toBeInTheDocument();
    expect(screen.getByTestId("line-noiseFloor")).toBeInTheDocument();
    expect(screen.getByTestId("x-axis")).toBeInTheDocument();
    expect(screen.getByTestId("y-axis")).toBeInTheDocument();
  });

  it("starts with empty data", () => {
    render(<RSSIGraph />);

    const chart = screen.getByTestId("line-chart");
    expect(chart).toHaveAttribute("data-points", "0");
  });

  it("adds data point when RSSI values are provided", async () => {
    const { rerender } = render(<RSSIGraph />);

    rerender(
      <RSSIGraph currentRSSI={-60} currentNoiseFloor={-90} currentSNR={30} />,
    );

    await waitFor(() => {
      const chart = screen.getByTestId("line-chart");
      expect(chart).toHaveAttribute("data-points", "1");
    });
  });

  it("accumulates multiple data points", async () => {
    const { rerender } = render(<RSSIGraph />);

    // Add first data point
    rerender(
      <RSSIGraph currentRSSI={-60} currentNoiseFloor={-90} currentSNR={30} />,
    );

    // Add second data point
    rerender(
      <RSSIGraph currentRSSI={-58} currentNoiseFloor={-89} currentSNR={31} />,
    );

    // Add third data point
    rerender(
      <RSSIGraph currentRSSI={-55} currentNoiseFloor={-88} currentSNR={33} />,
    );

    await waitFor(() => {
      const chart = screen.getByTestId("line-chart");
      expect(chart).toHaveAttribute("data-points", "3");
    });
  });

  it("maintains 60-second window of data", async () => {
    const { rerender } = render(<RSSIGraph />);

    // Add 65 data points to exceed the 60-second window
    for (let i = 0; i < 65; i++) {
      rerender(
        <RSSIGraph
          currentRSSI={-60 + i * 0.5}
          currentNoiseFloor={-90}
          currentSNR={30}
        />,
      );
    }

    await waitFor(() => {
      const chart = screen.getByTestId("line-chart");
      // Should only keep last 60 points
      expect(chart).toHaveAttribute("data-points", "60");
    });
  });

  it("handles undefined values gracefully", () => {
    const { rerender } = render(<RSSIGraph />);

    // Should not add data point when values are undefined
    rerender(<RSSIGraph currentRSSI={undefined} />);

    const chart = screen.getByTestId("line-chart");
    expect(chart).toHaveAttribute("data-points", "0");
  });

  it("only adds data when all values are provided", async () => {
    const { rerender } = render(<RSSIGraph />);

    // Missing SNR
    rerender(<RSSIGraph currentRSSI={-60} currentNoiseFloor={-90} />);

    let chart = screen.getByTestId("line-chart");
    expect(chart).toHaveAttribute("data-points", "0");

    // All values provided
    rerender(
      <RSSIGraph currentRSSI={-60} currentNoiseFloor={-90} currentSNR={30} />,
    );

    await waitFor(() => {
      chart = screen.getByTestId("line-chart");
      expect(chart).toHaveAttribute("data-points", "1");
    });
  });

  it("updates chart when new RSSI data arrives", async () => {
    const { rerender } = render(<RSSIGraph />);

    const rssiValues = [-60, -58, -55, -52, -50];

    for (let i = 0; i < rssiValues.length; i++) {
      rerender(
        <RSSIGraph
          currentRSSI={rssiValues[i]}
          currentNoiseFloor={-90}
          currentSNR={rssiValues[i] + 90}
        />,
      );

      await waitFor(() => {
        const chart = screen.getByTestId("line-chart");
        expect(chart).toHaveAttribute("data-points", String(i + 1));
      });
    }
  });

  it("renders RSSI and Noise Floor lines", () => {
    render(<RSSIGraph />);

    const rssiLine = screen.getByTestId("line-rssi");
    expect(rssiLine).toHaveTextContent("RSSI");

    const noiseFloorLine = screen.getByTestId("line-noiseFloor");
    expect(noiseFloorLine).toHaveTextContent("Noise Floor");
  });

  it("simulates 10Hz update rate", async () => {
    jest.useFakeTimers();
    const { rerender } = render(<RSSIGraph />);

    // Simulate 10 updates per second for 3 seconds
    for (let i = 0; i < 30; i++) {
      rerender(
        <RSSIGraph
          currentRSSI={-60 + Math.random() * 10}
          currentNoiseFloor={-90}
          currentSNR={30 + Math.random() * 5}
        />,
      );

      jest.advanceTimersByTime(100); // 100ms = 10Hz
    }

    await waitFor(() => {
      const chart = screen.getByTestId("line-chart");
      expect(chart).toHaveAttribute("data-points", "30");
    });

    jest.useRealTimers();
  });
});
