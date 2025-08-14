import { useEffect, useState, useRef } from "react";
import { websocketService } from "../services/websocket";

export interface SafetyInterlocks {
  modeCheck: boolean;
  batteryCheck: boolean;
  geofenceCheck: boolean;
  signalCheck: boolean;
  operatorCheck: boolean;
}

export interface SystemState {
  currentState: "IDLE" | "SEARCHING" | "DETECTING" | "HOMING" | "HOLDING";
  homingEnabled: boolean;
  flightMode: string;
  batteryPercent: number;
  gpsStatus: "NO_FIX" | "2D_FIX" | "3D_FIX" | "RTK";
  mavlinkConnected: boolean;
  sdrStatus: "CONNECTED" | "DISCONNECTED" | "ERROR";
  safetyInterlocks: SafetyInterlocks;
  timestamp?: number;
}

export const useSystemState = () => {
  const [systemState, setSystemState] = useState<SystemState | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | undefined>(undefined);

  useEffect(() => {
    const handleConnect = () => {
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };

    const handleDisconnect = () => {
      reconnectTimeoutRef.current = setTimeout(() => {
        websocketService.connect();
      }, 5000);
    };

    const handleSystemState = (data: SystemState) => {
      setSystemState({
        ...data,
        timestamp: Date.now(),
      });
    };

    const handleSafetyStatus = (data: {
      safetyInterlocks: SafetyInterlocks;
    }) => {
      setSystemState((prev) => {
        if (!prev) return null;
        return {
          ...prev,
          safetyInterlocks: data.safetyInterlocks,
          timestamp: Date.now(),
        };
      });
    };

    websocketService.on("connect", handleConnect);
    websocketService.on("disconnect", handleDisconnect);
    websocketService.on("systemState", handleSystemState);
    websocketService.on("safetyStatus", handleSafetyStatus);

    websocketService.connect();

    return () => {
      websocketService.off("connect", handleConnect);
      websocketService.off("disconnect", handleDisconnect);
      websocketService.off("systemState", handleSystemState);
      websocketService.off("safetyStatus", handleSafetyStatus);

      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, []);

  return systemState;
};

export const useWebSocketConnection = () => {
  const [isConnected, setIsConnected] = useState(false);

  useEffect(() => {
    const handleConnect = () => setIsConnected(true);
    const handleDisconnect = () => setIsConnected(false);

    websocketService.on("connect", handleConnect);
    websocketService.on("disconnect", handleDisconnect);

    return () => {
      websocketService.off("connect", handleConnect);
      websocketService.off("disconnect", handleDisconnect);
    };
  }, []);

  return isConnected;
};
