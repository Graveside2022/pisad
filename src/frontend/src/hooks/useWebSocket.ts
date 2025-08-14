import { useEffect, useState, useCallback } from "react";
import wsService from "../services/websocket";
import type { WebSocketMessage, MessageHandler } from "../services/websocket";

export function useWebSocket() {
  const [isConnected, setIsConnected] = useState(wsService.isConnected());
  const [lastMessage, setLastMessage] = useState<WebSocketMessage | null>(null);

  useEffect(() => {
    const checkConnection = setInterval(() => {
      setIsConnected(wsService.isConnected());
    }, 1000);

    return () => clearInterval(checkConnection);
  }, []);

  const addMessageHandler = useCallback((handler: MessageHandler) => {
    wsService.addMessageHandler(handler);
    return () => wsService.removeMessageHandler(handler);
  }, []);

  const sendMessage = useCallback((message: WebSocketMessage) => {
    wsService.sendMessage(message);
  }, []);

  useEffect(() => {
    const handler: MessageHandler = (message) => {
      setLastMessage(message);
    };

    wsService.addMessageHandler(handler);
    return () => wsService.removeMessageHandler(handler);
  }, []);

  return {
    isConnected,
    lastMessage,
    sendMessage,
    addMessageHandler,
  };
}
