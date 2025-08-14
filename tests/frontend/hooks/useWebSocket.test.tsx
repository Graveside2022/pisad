import { renderHook, act, waitFor } from "@testing-library/react";
import { useWebSocket } from "../../../src/frontend/src/hooks/useWebSocket";
import wsService from "../../../src/frontend/src/services/websocket";

// Mock the WebSocket service
jest.mock("../../../src/frontend/src/services/websocket", () => ({
  isConnected: jest.fn(),
  addMessageHandler: jest.fn(),
  removeMessageHandler: jest.fn(),
  sendMessage: jest.fn(),
}));

describe("useWebSocket", () => {
  beforeEach(() => {
    jest.clearAllMocks();
    (wsService.isConnected as jest.Mock).mockReturnValue(false);
  });

  it("initializes with connection status from service", () => {
    (wsService.isConnected as jest.Mock).mockReturnValue(true);
    const { result } = renderHook(() => useWebSocket());

    expect(result.current.isConnected).toBe(true);
  });

  it("updates connection status periodically", async () => {
    (wsService.isConnected as jest.Mock).mockReturnValue(false);
    const { result, rerender } = renderHook(() => useWebSocket());

    expect(result.current.isConnected).toBe(false);

    // Change the mock to return true
    (wsService.isConnected as jest.Mock).mockReturnValue(true);

    // Wait for the interval to update
    await waitFor(
      () => {
        expect(result.current.isConnected).toBe(true);
      },
      { timeout: 1500 },
    );
  });

  it("adds and removes message handler on mount/unmount", () => {
    const { unmount } = renderHook(() => useWebSocket());

    expect(wsService.addMessageHandler).toHaveBeenCalledTimes(1);

    unmount();

    expect(wsService.removeMessageHandler).toHaveBeenCalledTimes(1);
  });

  it("provides sendMessage function that calls service", () => {
    const { result } = renderHook(() => useWebSocket());

    const testMessage = { type: "rssi", data: { rssi: -60 } };

    act(() => {
      result.current.sendMessage(testMessage);
    });

    expect(wsService.sendMessage).toHaveBeenCalledWith(testMessage);
  });

  it("updates lastMessage when handler is called", () => {
    let messageHandler: any;
    (wsService.addMessageHandler as jest.Mock).mockImplementation((handler) => {
      messageHandler = handler;
    });

    const { result } = renderHook(() => useWebSocket());

    expect(result.current.lastMessage).toBeNull();

    const testMessage = {
      type: "rssi",
      data: {
        rssi: -55,
        noiseFloor: -90,
        snr: 35,
        confidence: 0.95,
      },
    };

    act(() => {
      messageHandler(testMessage);
    });

    expect(result.current.lastMessage).toEqual(testMessage);
  });

  it("handles RSSI update messages", () => {
    let messageHandler: any;
    (wsService.addMessageHandler as jest.Mock).mockImplementation((handler) => {
      messageHandler = handler;
    });

    const { result } = renderHook(() => useWebSocket());

    const rssiMessage = {
      type: "rssi",
      data: {
        rssi: -45,
        noiseFloor: -85,
        snr: 40,
        confidence: 0.98,
      },
    };

    act(() => {
      messageHandler(rssiMessage);
    });

    expect(result.current.lastMessage).toEqual(rssiMessage);
    expect(result.current.lastMessage?.type).toBe("rssi");
    expect(result.current.lastMessage?.data.rssi).toBe(-45);
  });

  it("handles detection event messages", () => {
    let messageHandler: any;
    (wsService.addMessageHandler as jest.Mock).mockImplementation((handler) => {
      messageHandler = handler;
    });

    const { result } = renderHook(() => useWebSocket());

    const detectionMessage = {
      type: "detection",
      data: {
        id: "det-123",
        timestamp: "2025-08-12T10:00:00Z",
        frequency: 433.92e6,
        rssi: -40,
        snr: 25,
        confidence: 0.92,
      },
    };

    act(() => {
      messageHandler(detectionMessage);
    });

    expect(result.current.lastMessage).toEqual(detectionMessage);
    expect(result.current.lastMessage?.type).toBe("detection");
  });

  it("adds custom message handlers", () => {
    const { result } = renderHook(() => useWebSocket());

    const customHandler = jest.fn();

    act(() => {
      const removeHandler = result.current.addMessageHandler(customHandler);
      expect(wsService.addMessageHandler).toHaveBeenCalledWith(customHandler);

      removeHandler();
      expect(wsService.removeMessageHandler).toHaveBeenCalledWith(
        customHandler,
      );
    });
  });

  it("handles rapid RSSI updates at 10Hz", () => {
    let messageHandler: any;
    (wsService.addMessageHandler as jest.Mock).mockImplementation((handler) => {
      messageHandler = handler;
    });

    const { result } = renderHook(() => useWebSocket());

    const rssiValues = [-60, -58, -55, -52, -50, -48, -45, -43, -41, -40];

    act(() => {
      rssiValues.forEach((rssi, index) => {
        const message = {
          type: "rssi",
          data: {
            rssi,
            noiseFloor: -90,
            snr: rssi + 90,
            confidence: 0.9 + index * 0.01,
          },
        };
        messageHandler(message);
      });
    });

    // Should have the last message
    expect(result.current.lastMessage?.data.rssi).toBe(-40);
  });

  it("cleans up interval on unmount", () => {
    jest.useFakeTimers();
    const clearIntervalSpy = jest.spyOn(global, "clearInterval");

    const { unmount } = renderHook(() => useWebSocket());

    unmount();

    expect(clearIntervalSpy).toHaveBeenCalled();

    jest.useRealTimers();
  });
});
