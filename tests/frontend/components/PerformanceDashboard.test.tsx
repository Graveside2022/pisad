import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { PerformanceDashboard } from '../../../src/frontend/src/components/analytics/PerformanceDashboard';

// Mock recharts to avoid rendering issues in tests
jest.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: any) => <div>{children}</div>,
  LineChart: ({ children }: any) => <div data-testid="line-chart">{children}</div>,
  BarChart: ({ children }: any) => <div data-testid="bar-chart">{children}</div>,
  PieChart: ({ children }: any) => <div data-testid="pie-chart">{children}</div>,
  Line: () => null,
  Bar: () => null,
  Pie: () => null,
  Cell: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
  Legend: () => null,
}));

const mockMetrics = {
  missionId: 'test-mission-123',
  detectionMetrics: {
    totalDetections: 15,
    detectionsPerHour: 8.5,
    detectionsPerKm2: 7.5,
    meanDetectionConfidence: 82.5,
    detectionCoverage: 78.0,
  },
  approachMetrics: {
    finalDistanceM: 12.5,
    approachTimeS: 180.0,
    approachEfficiency: 85.0,
    finalRssiDbm: -45.0,
    rssiImprovementDb: 25.0,
  },
  searchMetrics: {
    totalAreaKm2: 2.0,
    areaCoveredKm2: 1.6,
    coveragePercentage: 80.0,
    totalDistanceKm: 5.2,
    searchTimeMinutes: 45.0,
    averageSpeedKmh: 7.0,
    searchPatternEfficiency: 75.0,
  },
  baselineComparison: {
    timeImprovementPercent: 62.5,
    areaReductionPercent: 40.0,
    accuracyImprovementPercent: 75.0,
    costReductionPercent: 55.0,
    operatorWorkloadReduction: 60.0,
  },
  overallScore: 78.5,
  recommendations: [
    'Optimize search pattern for better coverage',
    'Adjust SDR gain settings for improved detection',
    'Consider higher altitude for initial search phase',
  ],
};

describe('PerformanceDashboard', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('should render loading state initially', () => {
    render(<PerformanceDashboard missionId="test-mission" />);
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
  });

  it('should render info message when no mission is selected', async () => {
    render(<PerformanceDashboard />);

    await waitFor(() => {
      expect(screen.getByText(/Select a mission to view performance metrics/i)).toBeInTheDocument();
    });
  });

  it('should fetch and display metrics', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => mockMetrics,
    } as Response);

    render(<PerformanceDashboard missionId="test-mission-123" />);

    await waitFor(() => {
      expect(screen.getByText('Performance Analytics')).toBeInTheDocument();
      expect(screen.getByText(/Mission: test-mission-123/i)).toBeInTheDocument();
      expect(screen.getByText(/Score: 78.5\/100/i)).toBeInTheDocument();
    });
  });

  it('should display key metric cards', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => mockMetrics,
    } as Response);

    render(<PerformanceDashboard missionId="test-mission-123" />);

    await waitFor(() => {
      expect(screen.getByText('Detection Rate')).toBeInTheDocument();
      expect(screen.getByText('8.5')).toBeInTheDocument();
      expect(screen.getByText('/hr')).toBeInTheDocument();

      expect(screen.getByText('Search Efficiency')).toBeInTheDocument();
      expect(screen.getByText('75')).toBeInTheDocument();

      expect(screen.getByText('Final Distance')).toBeInTheDocument();
      expect(screen.getByText('12.5')).toBeInTheDocument();
      expect(screen.getByText('m')).toBeInTheDocument();

      expect(screen.getByText('Coverage')).toBeInTheDocument();
      expect(screen.getByText('80')).toBeInTheDocument();
    });
  });

  it('should display charts', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => mockMetrics,
    } as Response);

    render(<PerformanceDashboard missionId="test-mission-123" />);

    await waitFor(() => {
      expect(screen.getByText('Search Area Coverage')).toBeInTheDocument();
      expect(screen.getByText('vs Baseline Performance')).toBeInTheDocument();
      expect(screen.getByText('Detection Timeline')).toBeInTheDocument();
      expect(screen.getByTestId('pie-chart')).toBeInTheDocument();
      expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
      expect(screen.getByTestId('line-chart')).toBeInTheDocument();
    });
  });

  it('should display detailed metrics sections', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => mockMetrics,
    } as Response);

    render(<PerformanceDashboard missionId="test-mission-123" />);

    await waitFor(() => {
      // Detection Performance section
      expect(screen.getByText('Detection Performance')).toBeInTheDocument();
      expect(screen.getByText('Total Detections')).toBeInTheDocument();
      expect(screen.getByText('15')).toBeInTheDocument();
      expect(screen.getByText('Mean Confidence')).toBeInTheDocument();
      expect(screen.getByText('82.5%')).toBeInTheDocument();

      // Approach Performance section
      expect(screen.getByText('Approach Performance')).toBeInTheDocument();
      expect(screen.getByText('Approach Efficiency')).toBeInTheDocument();
      expect(screen.getByText('85.0%')).toBeInTheDocument();
      expect(screen.getByText('Final RSSI')).toBeInTheDocument();
      expect(screen.getByText('-45.0 dBm')).toBeInTheDocument();
    });
  });

  it('should display recommendations', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => mockMetrics,
    } as Response);

    render(<PerformanceDashboard missionId="test-mission-123" />);

    await waitFor(() => {
      expect(screen.getByText('Recommendations')).toBeInTheDocument();
      expect(screen.getByText(/Optimize search pattern for better coverage/i)).toBeInTheDocument();
      expect(screen.getByText(/Adjust SDR gain settings for improved detection/i)).toBeInTheDocument();
      expect(screen.getByText(/Consider higher altitude for initial search phase/i)).toBeInTheDocument();
    });
  });

  it('should display overall performance score', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => mockMetrics,
    } as Response);

    render(<PerformanceDashboard missionId="test-mission-123" />);

    await waitFor(() => {
      expect(screen.getByText('Overall Performance Score')).toBeInTheDocument();
      expect(screen.getByText('79%')).toBeInTheDocument(); // Rounded from 78.5
    });
  });

  it('should handle fetch error', async () => {
    global.fetch = jest.fn().mockRejectedValueOnce(new Error('Failed to fetch'));

    render(<PerformanceDashboard missionId="test-mission-123" />);

    await waitFor(() => {
      expect(screen.getByText(/Failed to fetch/i)).toBeInTheDocument();
    });
  });

  it('should handle non-ok response', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: false,
      status: 404,
    } as Response);

    render(<PerformanceDashboard missionId="test-mission-123" />);

    await waitFor(() => {
      expect(screen.getByText(/Failed to fetch metrics/i)).toBeInTheDocument();
    });
  });

  it('should handle null approach metrics', async () => {
    const metricsWithNullValues = {
      ...mockMetrics,
      approachMetrics: {
        ...mockMetrics.approachMetrics,
        finalDistanceM: null,
        approachTimeS: null,
        finalRssiDbm: null,
      },
    };

    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => metricsWithNullValues,
    } as Response);

    render(<PerformanceDashboard missionId="test-mission-123" />);

    await waitFor(() => {
      expect(screen.getAllByText('N/A')).toHaveLength(3);
    });
  });

  it('should not display recommendations when empty', async () => {
    const metricsWithoutRecommendations = {
      ...mockMetrics,
      recommendations: [],
    };

    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => metricsWithoutRecommendations,
    } as Response);

    render(<PerformanceDashboard missionId="test-mission-123" />);

    await waitFor(() => {
      expect(screen.queryByText('Recommendations')).not.toBeInTheDocument();
    });
  });

  it('should display correct color for high performance score', async () => {
    const highScoreMetrics = {
      ...mockMetrics,
      overallScore: 85,
    };

    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => highScoreMetrics,
    } as Response);

    render(<PerformanceDashboard missionId="test-mission-123" />);

    await waitFor(() => {
      const chip = screen.getByText(/Score: 85.0\/100/i).closest('.MuiChip-root');
      expect(chip).toHaveClass('MuiChip-colorSuccess');
    });
  });

  it('should display correct color for low performance score', async () => {
    const lowScoreMetrics = {
      ...mockMetrics,
      overallScore: 45,
    };

    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => lowScoreMetrics,
    } as Response);

    render(<PerformanceDashboard missionId="test-mission-123" />);

    await waitFor(() => {
      const chip = screen.getByText(/Score: 45.0\/100/i).closest('.MuiChip-root');
      expect(chip).toHaveClass('MuiChip-colorWarning');
    });
  });
});
