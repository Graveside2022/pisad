import { render, screen } from "@testing-library/react";
import "@testing-library/jest-dom";
import SignalMeter from "../../../src/frontend/src/components/dashboard/SignalMeter";

describe("SignalMeter", () => {
  const defaultProps = {
    rssi: -60,
    noiseFloor: -90,
    snr: 15,
    size: "medium" as const,
  };

  it("renders signal strength with correct RSSI value", () => {
    render(<SignalMeter {...defaultProps} />);

    expect(screen.getByText("-60.0 dBm")).toBeInTheDocument();
    expect(screen.getByText("Signal Strength")).toBeInTheDocument();
  });

  it("displays noise floor value correctly", () => {
    render(<SignalMeter {...defaultProps} />);

    expect(screen.getByText("Noise Floor")).toBeInTheDocument();
    expect(screen.getByText("-90.0 dBm")).toBeInTheDocument();
  });

  it("displays SNR value correctly", () => {
    render(<SignalMeter {...defaultProps} />);

    expect(screen.getByText("SNR")).toBeInTheDocument();
    expect(screen.getByText("15.0 dB")).toBeInTheDocument();
  });

  it("applies correct color based on SNR thresholds", () => {
    const { rerender } = render(<SignalMeter {...defaultProps} snr={15} />);
    let snrElement = screen.getByText("15.0 dB");
    expect(snrElement).toHaveClass("MuiTypography-colorSuccess");

    rerender(<SignalMeter {...defaultProps} snr={8} />);
    snrElement = screen.getByText("8.0 dB");
    expect(snrElement).toHaveClass("MuiTypography-colorWarning");

    rerender(<SignalMeter {...defaultProps} snr={3} />);
    snrElement = screen.getByText("3.0 dB");
    expect(snrElement).toHaveClass("MuiTypography-colorError");
  });

  it("updates RSSI display when prop changes", () => {
    const { rerender } = render(<SignalMeter {...defaultProps} rssi={-60} />);
    expect(screen.getByText("-60.0 dBm")).toBeInTheDocument();

    rerender(<SignalMeter {...defaultProps} rssi={-45} />);
    expect(screen.getByText("-45.0 dBm")).toBeInTheDocument();

    rerender(<SignalMeter {...defaultProps} rssi={-75} />);
    expect(screen.getByText("-75.0 dBm")).toBeInTheDocument();
  });

  it("handles edge case RSSI values", () => {
    const { rerender } = render(<SignalMeter {...defaultProps} rssi={-100} />);
    expect(screen.getByText("-100.0 dBm")).toBeInTheDocument();

    rerender(<SignalMeter {...defaultProps} rssi={-30} />);
    expect(screen.getByText("-30.0 dBm")).toBeInTheDocument();

    rerender(<SignalMeter {...defaultProps} rssi={0} />);
    expect(screen.getByText("0.0 dBm")).toBeInTheDocument();
  });

  it("renders with different size configurations", () => {
    const { rerender, container } = render(
      <SignalMeter {...defaultProps} size="small" />,
    );
    let paper = container.querySelector(".MuiPaper-root");
    expect(paper).toHaveStyle({ height: "150px" });

    rerender(<SignalMeter {...defaultProps} size="medium" />);
    paper = container.querySelector(".MuiPaper-root");
    expect(paper).toHaveStyle({ height: "200px" });

    rerender(<SignalMeter {...defaultProps} size="large" />);
    paper = container.querySelector(".MuiPaper-root");
    expect(paper).toHaveStyle({ height: "250px" });
  });

  it("calculates signal strength percentage correctly", () => {
    const { container, rerender } = render(
      <SignalMeter {...defaultProps} rssi={-65} />,
    );
    const progressBar = container.querySelector(".MuiLinearProgress-root");

    // -65 dBm should be 50% between -100 and -30
    const expectedValue = ((-65 - -100) / (-30 - -100)) * 100;
    expect(progressBar).toHaveAttribute(
      "aria-valuenow",
      expectedValue.toString(),
    );

    rerender(<SignalMeter {...defaultProps} rssi={-100} />);
    expect(progressBar).toHaveAttribute("aria-valuenow", "0");

    rerender(<SignalMeter {...defaultProps} rssi={-30} />);
    expect(progressBar).toHaveAttribute("aria-valuenow", "100");
  });
});
