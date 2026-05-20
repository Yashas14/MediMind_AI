"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useAuth } from "@/components/providers/auth-provider";

interface ChatWSMessage {
  type: "message" | "typing" | "error" | "pong";
  role?: string;
  content?: string;
  confidence_score?: number;
  extracted_symptoms?: string[];
  triage_level?: string;
  disclaimer?: string;
  status?: boolean;
  detail?: string;
}

/**
 * Custom hook for WebSocket chat connections with auto-reconnect.
 */
export function useChatWebSocket(sessionId: string | null) {
  const { token } = useAuth();
  const [messages, setMessages] = useState<ChatWSMessage[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeout = useRef<NodeJS.Timeout>();

  const connect = useCallback(() => {
    if (!sessionId || !token) return;

    const wsBase =
      process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";
    const ws = new WebSocket(
      `${wsBase}/api/v1/chat/ws/${sessionId}?token=${encodeURIComponent(token)}`
    );

    ws.onopen = () => {
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      const data: ChatWSMessage = JSON.parse(event.data);
      if (data.type === "typing") {
        setIsTyping(data.status ?? false);
      } else if (data.type === "message") {
        setMessages((prev) => [...prev, data]);
        setIsTyping(false);
      } else if (data.type === "error") {
        setMessages((prev) => [
          ...prev,
          { type: "message", role: "assistant", content: data.detail || "An error occurred" },
        ]);
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
      // Auto-reconnect after 3 seconds
      reconnectTimeout.current = setTimeout(connect, 3000);
    };

    ws.onerror = () => {
      ws.close();
    };

    wsRef.current = ws;
  }, [sessionId, token]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimeout.current);
      wsRef.current?.close();
    };
  }, [connect]);

  const sendMessage = useCallback(
    (content: string) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        // Add user message to display immediately
        setMessages((prev) => [
          ...prev,
          { type: "message", role: "user", content },
        ]);
        wsRef.current.send(JSON.stringify({ type: "message", content }));
      }
    },
    []
  );

  return { messages, isConnected, isTyping, sendMessage, setMessages };
}
