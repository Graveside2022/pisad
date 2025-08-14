import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import '@testing-library/jest-dom';
import { MissionReplay } from '../../../src/frontend/src/components/analytics/MissionReplay';

// Mock WebSocket
class MockWebSocket {
  url: string;
  readyState: number = 1;
  onopen: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;

  constructor(url: string) {
    this.url = url;
    setTimeout(() => {
      if (this.onopen) {
        this.onopen(new Event('open'));
      }
    }, 0);
  }

  send(data: string) {
    // Mock send
  }

  close() {
    this.readyState = 3;
    if (this.onclose) {
      this.onclose(new CloseEvent('close'));
    }
  }

  simulateMessage(data: any) {
    if (this.onmessage) {
      this.onmessage(new MessageEvent('message', { data: JSON.stringify(data) }));
    }
  }
}

// Mock recharts
jest.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: any) => <div>{children}</div>,
  LineChart: ({ children }: any) => <div data-testid="line-chart">{children}</div>,
  Line: () => null,
  XAxis: () => null,
  YAxis: () => null,
  CartesianGrid: () => null,
  Tooltip: () => null,
}));

const mockReplayStatus = {
  mission_id: 'test-mission-123',
  state: 'stopped' as const,
  speed: 1,
  position: 0,
  total: 100,
  progress: 0,
};

const mockReplayEvent = {
  timestamp: '2024-01-01T12:00:00Z',
  telemetry: {
    latitude: 47.6062,
    longitude: -122.3321,
    altitude: 100,
    groundspeed: 5.0,
    heading: 90,
    rssi_dbm: -70,
    snr_db: 10,
    beacon_detected: false,
    system_state: 'SEARCHING',
    battery_percent: 85,
  },
  signal_detections: [
    {
      frequency: 121500000,
      rssi: -65,
      confidence: 85,
    },
  ],
  state_changes: [
    {
      from_state: 'IDLE',
      to_state: 'SEARCHING',
      trigger: 'start',
    },
  ],
};

describe('MissionReplay', () => {
  let mockWebSocket: MockWebSocket;

  beforeEach(() => {
    jest.clearAllMocks();
    
    // Setup WebSocket mock
    (global as any).WebSocket = jest.fn((url: string) => {
      mockWebSocket = new MockWebSocket(url);
      return mockWebSocket;
    });
  });

  afterEach(() => {
    if (mockWebSocket) {
      mockWebSocket.close();
    }
  });

  it('should render loading state initially', () => {
    render(<MissionReplay missionId="test-mission" />);
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
  });

  it('should load and display replay data', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        status: mockReplayStatus,
        timeline_range: {
          start: '2024-01-01T12:00:00Z',
          end: '2024-01-01T13:00:00Z',
        },
      }),
    } as Response);

    render(<MissionReplay missionId="test-mission-123" />);

    await waitFor(() => {
      expect(screen.getByText('Mission Replay')).toBeInTheDocument();
      expect(screen.getByText('STOPPED')).toBeInTheDocument();
      expect(screen.getByText('Frame 0 / 100 (0.0%)')).toBeInTheDocument();
    });
  });

  it('should display playback controls', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => ({ status: mockReplayStatus }),
    } as Response);

    render(<MissionReplay missionId="test-mission-123" />);

    await waitFor(() => {
      expect(screen.getByLabelText('Speed')).toBeInTheDocument();
      // Play button should be visible when stopped
      expect(screen.getByTestId('PlayArrowIcon')).toBeInTheDocument();
      expect(screen.getByTestId('StopIcon')).toBeInTheDocument();
      expect(screen.getByTestId('SkipPreviousIcon')).toBeInTheDocument();
      expect(screen.getByTestId('SkipNextIcon')).toBeInTheDocument();
    });
  });

  it('should handle play action', async () => {
    global.fetch = jest.fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: mockReplayStatus }),
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ ...mockReplayStatus, state: 'playing' }),
      } as Response);

    render(<MissionReplay missionId="test-mission-123" />);

    await waitFor(() => {
      expect(screen.getByTestId('PlayArrowIcon')).toBeInTheDocument();
    });

    const playButton = screen.getByTestId('PlayArrowIcon').closest('button')!;
    fireEvent.click(playButton);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        '/api/analytics/replay/test-mission-123/control',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ action: 'play' }),
        })
      );
    });
  });

  it('should handle pause action when playing', async () => {
    global.fetch = jest.fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: { ...mockReplayStatus, state: 'playing' } }),
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ ...mockReplayStatus, state: 'paused' }),
      } as Response);

    render(<MissionReplay missionId="test-mission-123" />);

    await waitFor(() => {
      expect(screen.getByTestId('PauseIcon')).toBeInTheDocument();
    });

    const pauseButton = screen.getByTestId('PauseIcon').closest('button')!;
    fireEvent.click(pauseButton);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        '/api/analytics/replay/test-mission-123/control',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ action: 'pause' }),
        })
      );
    });
  });

  it('should handle stop action', async () => {
    global.fetch = jest.fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: mockReplayStatus }),
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ ...mockReplayStatus, position: 0 }),
      } as Response);

    render(<MissionReplay missionId="test-mission-123" />);

    await waitFor(() => {
      expect(screen.getByTestId('StopIcon')).toBeInTheDocument();
    });

    const stopButton = screen.getByTestId('StopIcon').closest('button')!;
    fireEvent.click(stopButton);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        '/api/analytics/replay/test-mission-123/control',
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ action: 'stop' }),
        })
      );
    });
  });

  it('should display telemetry data when event is received', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => ({ status: mockReplayStatus }),
    } as Response);

    render(<MissionReplay missionId="test-mission-123" />);

    await waitFor(() => {
      expect(screen.getByText('Mission Replay')).toBeInTheDocument();
    });

    // Simulate WebSocket message
    mockWebSocket.simulateMessage({
      type: 'replay_event',
      ...mockReplayEvent,
      position: 10,
      total: 100,
    });

    await waitFor(() => {
      expect(screen.getByText('47.606200, -122.332100')).toBeInTheDocument();
      expect(screen.getByText('100.0 m')).toBeInTheDocument();
      expect(screen.getByText('5.0 m/s @ 90°')).toBeInTheDocument();
      expect(screen.getByText('85%')).toBeInTheDocument();
    });
  });

  it('should display signal data when event is received', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => ({ status: mockReplayStatus }),
    } as Response);

    render(<MissionReplay missionId="test-mission-123" />);

    await waitFor(() => {
      expect(screen.getByText('Mission Replay')).toBeInTheDocument();
    });

    mockWebSocket.simulateMessage({
      type: 'replay_event',
      ...mockReplayEvent,
    });

    await waitFor(() => {
      expect(screen.getByText('-70.0 dBm')).toBeInTheDocument();
      expect(screen.getByText('10.0 dB')).toBeInTheDocument();
      expect(screen.getByText('NOT DETECTED')).toBeInTheDocument();
      expect(screen.getByText('121.500 MHz @ -65.0 dBm')).toBeInTheDocument();
    });
  });

  it('should display system state when event is received', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => ({ status: mockReplayStatus }),
    } as Response);

    render(<MissionReplay missionId="test-mission-123" />);

    await waitFor(() => {
      expect(screen.getByText('Mission Replay')).toBeInTheDocument();
    });

    mockWebSocket.simulateMessage({
      type: 'replay_event',
      ...mockReplayEvent,
    });

    await waitFor(() => {
      expect(screen.getByText('SEARCHING')).toBeInTheDocument();
      expect(screen.getByText('IDLE → SEARCHING')).toBeInTheDocument();
      expect(screen.getByText('Trigger: start')).toBeInTheDocument();
    });
  });

  it('should update RSSI history chart', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => ({ status: mockReplayStatus }),
    } as Response);

    render(<MissionReplay missionId="test-mission-123" />);

    await waitFor(() => {
      expect(screen.getByText('RSSI History')).toBeInTheDocument();
      expect(screen.getByTestId('line-chart')).toBeInTheDocument();
    });

    // Simulate multiple events to build history
    for (let i = 0; i < 5; i++) {
      mockWebSocket.simulateMessage({
        type: 'replay_event',
        timestamp: new Date(Date.now() + i * 1000).toISOString(),
        telemetry: {
          ...mockReplayEvent.telemetry,
          rssi_dbm: -70 + i,
        },
        signal_detections: [],
        state_changes: [],
      });
    }

    // Chart should be rendered with data
    expect(screen.getByTestId('line-chart')).toBeInTheDocument();
  });

  it('should handle speed change', async () => {
    global.fetch = jest.fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: mockReplayStatus }),
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ ...mockReplayStatus, speed: 2 }),
      } as Response);

    render(<MissionReplay missionId="test-mission-123" />);

    await waitFor(() => {
      expect(screen.getByLabelText('Speed')).toBeInTheDocument();
    });

    const speedSelect = screen.getByLabelText('Speed');
    fireEvent.mouseDown(speedSelect);
    
    const option2x = await screen.findByText('2x');
    fireEvent.click(option2x);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        '/api/analytics/replay/test-mission-123/control',
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('"speed":2'),
        })
      );
    });
  });

  it('should handle skip forward', async () => {
    global.fetch = jest.fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: { ...mockReplayStatus, position: 10 } }),
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ ...mockReplayStatus, position: 20 }),
      } as Response);

    render(<MissionReplay missionId="test-mission-123" />);

    await waitFor(() => {
      expect(screen.getByTestId('SkipNextIcon')).toBeInTheDocument();
    });

    const skipButton = screen.getByTestId('SkipNextIcon').closest('button')!;
    fireEvent.click(skipButton);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        '/api/analytics/replay/test-mission-123/control',
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('"position":20'),
        })
      );
    });
  });

  it('should handle skip backward', async () => {
    global.fetch = jest.fn()
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: { ...mockReplayStatus, position: 20 } }),
      } as Response)
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ ...mockReplayStatus, position: 10 }),
      } as Response);

    render(<MissionReplay missionId="test-mission-123" />);

    await waitFor(() => {
      expect(screen.getByTestId('SkipPreviousIcon')).toBeInTheDocument();
    });

    const skipButton = screen.getByTestId('SkipPreviousIcon').closest('button')!;
    fireEvent.click(skipButton);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        '/api/analytics/replay/test-mission-123/control',
        expect.objectContaining({
          method: 'POST',
          body: expect.stringContaining('"position":10'),
        })
      );
    });
  });

  it('should call onEventUpdate when event is received', async () => {
    const onEventUpdate = jest.fn();
    
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => ({ status: mockReplayStatus }),
    } as Response);

    render(<MissionReplay missionId="test-mission-123" onEventUpdate={onEventUpdate} />);

    await waitFor(() => {
      expect(screen.getByText('Mission Replay')).toBeInTheDocument();
    });

    mockWebSocket.simulateMessage({
      type: 'replay_event',
      ...mockReplayEvent,
    });

    await waitFor(() => {
      expect(onEventUpdate).toHaveBeenCalledWith(
        expect.objectContaining({
          timestamp: mockReplayEvent.timestamp,
          telemetry: mockReplayEvent.telemetry,
        })
      );
    });
  });

  it('should handle connection error', async () => {
    global.fetch = jest.fn().mockRejectedValueOnce(new Error('Connection failed'));

    render(<MissionReplay missionId="test-mission-123" />);

    await waitFor(() => {
      expect(screen.getByText(/Failed to load replay/i)).toBeInTheDocument();
    });
  });

  it('should display beacon detected status', async () => {
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: async () => ({ status: mockReplayStatus }),
    } as Response);

    render(<MissionReplay missionId="test-mission-123" />);

    await waitFor(() => {
      expect(screen.getByText('Mission Replay')).toBeInTheDocument();
    });

    mockWebSocket.simulateMessage({
      type: 'replay_event',
      ...mockReplayEvent,
      telemetry: {
        ...mockReplayEvent.telemetry,
        beacon_detected: true,
      },
    });

    await waitFor(() => {
      expect(screen.getByText('DETECTED')).toBeInTheDocument();
    });
  });
});