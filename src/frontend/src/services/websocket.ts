/**
 * WebSocket service for real-time communication with backend
 */

import { createLogger } from "../utils/logger";

const logger = createLogger("WebSocket");

export interface RSSIData {
  rssi: number;
  noiseFloor: number;
  snr: number;
  confidence: number;
  timestamp: string;
}

export interface WebSocketMessage {
  type: string;
  data: unknown;
  timestamp?: number;
}

export type MessageHandler = (message: WebSocketMessage) => void;

class WebSocketService {
  private ws: WebSocket | null = null;
  private messageHandlers: Set<MessageHandler> = new Set();
  private reconnectTimeout: number | null = null;
  private reconnectDelay = 1000;
  private maxReconnectDelay = 30000;
  private isIntentionallyClosed = false;
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private eventListeners: Map<string, Set<Function>> = new Map();

  constructor() {
    this.connect();
  }

  private getWebSocketUrl(): string {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.hostname;
    // Use environment variable or default to API port
    const port =
      process.env.REACT_APP_WS_PORT || process.env.REACT_APP_API_PORT || "8080";
    return `${protocol}//${host}:${port}/ws`;
  }

  connect(): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      return;
    }

    this.isIntentionallyClosed = false;
    const url = this.getWebSocketUrl();
    logger.info("Connecting to WebSocket:", url);

    try {
      this.ws = new WebSocket(url);

      this.ws.onopen = () => {
        console.log("WebSocket connected");
        this.reconnectDelay = 1000;
        this.emit("connect");
        this.startHeartbeat();
        this.sendMessage({ type: "ping", data: {} });
      };

      this.ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as WebSocketMessage;
          this.handleMessage(message);
          this.notifyHandlers(message);
        } catch (error) {
          console.error("Failed to parse WebSocket message:", error);
        }
      };

      this.ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        this.emit("error", error);
      };

      this.ws.onclose = () => {
        console.log("WebSocket disconnected");
        this.stopHeartbeat();
        this.emit("disconnect");
        if (!this.isIntentionallyClosed) {
          this.scheduleReconnect();
        }
      };
    } catch (error) {
      console.error("Failed to create WebSocket:", error);
      this.scheduleReconnect();
    }
  }

  private scheduleReconnect(): void {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
    }

    console.log(`Reconnecting in ${this.reconnectDelay}ms...`);
    this.reconnectTimeout = window.setTimeout(() => {
      this.connect();
    }, this.reconnectDelay);

    // Exponential backoff
    this.reconnectDelay = Math.min(
      this.reconnectDelay * 2,
      this.maxReconnectDelay,
    );
  }

  private handleMessage(message: WebSocketMessage) {
    switch (message.type) {
      case "systemState":
        this.emit("systemState", message.data);
        break;
      case "safetyStatus":
        this.emit("safetyStatus", message.data);
        break;
      case "rssiUpdate":
        this.emit("rssiUpdate", message.data);
        break;
      case "velocityUpdate":
        this.emit("velocityUpdate", message.data);
        break;
      case "positionUpdate":
        this.emit("positionUpdate", message.data);
        break;
      case "alert":
        this.emit("alert", message.data);
        break;
      case "pong":
        break;
      default:
        this.emit("message", message);
    }
  }

  private startHeartbeat() {
    this.stopHeartbeat();
    this.heartbeatInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.sendMessage({ type: "ping", data: {} });
      }
    }, 30000);
  }

  private stopHeartbeat() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  disconnect(): void {
    this.isIntentionallyClosed = true;
    this.stopHeartbeat();
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  sendMessage(message: WebSocketMessage): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      console.warn("WebSocket not connected, cannot send message");
    }
  }

  addMessageHandler(handler: MessageHandler): void {
    this.messageHandlers.add(handler);
  }

  removeMessageHandler(handler: MessageHandler): void {
    this.messageHandlers.delete(handler);
  }

  private notifyHandlers(message: WebSocketMessage): void {
    this.messageHandlers.forEach((handler) => {
      try {
        handler(message);
      } catch (error) {
        logger.error("Error in message handler", error);
      }
    });
  }

  private emit(event: string, data?: any): void {
    const listeners = this.eventListeners.get(event);
    if (listeners) {
      listeners.forEach((listener) => {
        try {
          listener(data);
        } catch (error) {
          logger.error(`Error in event listener for ${event}`, error);
        }
      });
    }
  }

  on(event: string, listener: Function): void {
    if (!this.eventListeners.has(event)) {
      this.eventListeners.set(event, new Set());
    }
    this.eventListeners.get(event)!.add(listener);
  }

  off(event: string, listener: Function): void {
    const listeners = this.eventListeners.get(event);
    if (listeners) {
      listeners.delete(listener);
    }
  }

  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN;
  }
}

// Singleton instance
const wsService = new WebSocketService();
export const websocketService = wsService;
export default wsService;
