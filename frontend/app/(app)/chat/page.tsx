"use client";

import { useEffect, useRef, useState } from "react";
import { useAuth } from "@/components/providers/auth-provider";
import { useRouter } from "next/navigation";
import { useChatWebSocket } from "@/hooks/use-chat-websocket";
import { useVoiceInput } from "@/hooks/use-voice-input";
import { chatApi, type ChatSession } from "@/lib/api-client";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";
import ReactMarkdown from "react-markdown";

export default function ChatPage() {
  const router = useRouter();
  const { token, isAuthenticated, isLoading } = useAuth();
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const {
    messages,
    isConnected,
    isTyping,
    sendMessage,
    setMessages,
  } = useChatWebSocket(activeSessionId);

  const {
    isListening,
    transcript,
    isSupported: voiceSupported,
    startListening,
    stopListening,
  } = useVoiceInput();

  // Redirect if not authenticated
  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      router.push("/login");
    }
  }, [isLoading, isAuthenticated, router]);

  // Load sessions on mount
  useEffect(() => {
    if (token) {
      chatApi.listSessions(token).then(setSessions).catch(console.error);
    }
  }, [token]);

  // Auto-scroll on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Apply voice transcript to input
  useEffect(() => {
    if (transcript) {
      setInput(transcript);
    }
  }, [transcript]);

  async function handleNewSession() {
    if (!token) return;
    try {
      const session = await chatApi.createSession(token);
      setSessions((prev) => [session, ...prev]);
      setActiveSessionId(session.id);
      setMessages([]);
    } catch (err) {
      console.error("Failed to create session:", err);
    }
  }

  async function handleSelectSession(id: string) {
    setActiveSessionId(id);
    setMessages([]);
    if (token) {
      try {
        const msgs = await chatApi.getMessages(id, token);
        setMessages(
          msgs.map((m) => ({
            type: "message" as const,
            role: m.role,
            content: m.content,
            confidence_score: m.confidence_score,
            extracted_symptoms: m.extracted_symptoms,
            triage_level: m.triage_level,
          }))
        );
      } catch (err) {
        console.error("Failed to load messages:", err);
      }
    }
  }

  function handleSend() {
    const text = input.trim();
    if (!text) return;
    if (!activeSessionId) {
      // Auto-create a session
      if (token) {
        chatApi.createSession(token).then((session) => {
          setSessions((prev) => [session, ...prev]);
          setActiveSessionId(session.id);
          // The WebSocket will connect once activeSessionId is set,
          // but we'll use REST as fallback for the first message
          chatApi
            .sendMessage(text, token, session.id)
            .then((msg) =>
              setMessages((prev) => [
                ...prev,
                { type: "message", role: "user", content: text },
                {
                  type: "message",
                  role: "assistant",
                  content: msg.content,
                  confidence_score: msg.confidence_score,
                  extracted_symptoms: msg.extracted_symptoms,
                  triage_level: msg.triage_level,
                },
              ])
            );
        });
        setInput("");
        return;
      }
    }
    sendMessage(text);
    setInput("");
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  function getTriageBadge(level?: string) {
    if (!level) return null;
    const variant = level.toLowerCase().replace("_", "-") as any;
    return <Badge variant={variant}>{level}</Badge>;
  }

  if (isLoading) {
    return (
      <div className="flex h-[calc(100vh-3.5rem)] items-center justify-center">
        <div className="animate-pulse text-muted-foreground">Loading…</div>
      </div>
    );
  }

  return (
    <div className="flex h-[calc(100vh-3.5rem)]">
      {/* Sidebar */}
      <aside
        className={cn(
          "flex-shrink-0 border-r bg-muted/30 transition-all duration-200",
          sidebarOpen ? "w-64" : "w-0 overflow-hidden"
        )}
      >
        <div className="flex h-full flex-col p-3">
          <Button onClick={handleNewSession} className="mb-3 w-full" size="sm">
            + New Chat
          </Button>
          <ScrollArea className="flex-1">
            <div className="space-y-1">
              {sessions.map((s) => (
                <button
                  key={s.id}
                  onClick={() => handleSelectSession(s.id)}
                  className={cn(
                    "w-full rounded-md px-3 py-2 text-left text-sm transition-colors hover:bg-accent",
                    activeSessionId === s.id
                      ? "bg-accent text-accent-foreground"
                      : "text-muted-foreground"
                  )}
                >
                  <div className="truncate font-medium">{s.title}</div>
                  <div className="text-xs opacity-60">
                    {new Date(s.created_at).toLocaleDateString()}
                  </div>
                </button>
              ))}
            </div>
          </ScrollArea>
        </div>
      </aside>

      {/* Chat Area */}
      <div className="flex flex-1 flex-col">
        {/* Chat header */}
        <div className="flex items-center gap-2 border-b px-4 py-2">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setSidebarOpen(!sidebarOpen)}
          >
            ☰
          </Button>
          <h2 className="text-sm font-semibold">
            {activeSessionId ? "AI Consultation" : "Start a new consultation"}
          </h2>
          {isConnected && (
            <span className="ml-auto flex items-center gap-1 text-xs text-green-600">
              <span className="h-2 w-2 rounded-full bg-green-500" />
              Connected
            </span>
          )}
        </div>

        {/* Messages */}
        <ScrollArea className="flex-1 p-4">
          <div className="mx-auto max-w-3xl space-y-4">
            {messages.length === 0 && (
              <div className="py-20 text-center text-muted-foreground">
                <div className="mb-4 text-5xl">💬</div>
                <h3 className="text-lg font-semibold">
                  How can I help you today?
                </h3>
                <p className="mt-1 text-sm">
                  Describe your symptoms in natural language and I&apos;ll
                  provide AI-powered analysis.
                </p>
              </div>
            )}

            {messages.map((msg, i) => (
              <div
                key={i}
                className={cn(
                  "flex",
                  msg.role === "user" ? "justify-end" : "justify-start"
                )}
              >
                <div
                  className={cn(
                    "max-w-[80%] rounded-2xl px-4 py-3",
                    msg.role === "user"
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted"
                  )}
                >
                  {msg.role === "assistant" ? (
                    <div className="prose prose-sm dark:prose-invert max-w-none">
                      <ReactMarkdown>{msg.content || ""}</ReactMarkdown>
                    </div>
                  ) : (
                    <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                  )}

                  {/* Triage badge */}
                  {msg.triage_level && (
                    <div className="mt-2">{getTriageBadge(msg.triage_level)}</div>
                  )}

                  {/* Extracted symptoms */}
                  {msg.extracted_symptoms && msg.extracted_symptoms.length > 0 && (
                    <div className="mt-2 flex flex-wrap gap-1">
                      {msg.extracted_symptoms.map((s, j) => (
                        <Badge key={j} variant="outline" className="text-xs">
                          {s.replace(/_/g, " ")}
                        </Badge>
                      ))}
                    </div>
                  )}

                  {/* Confidence score */}
                  {msg.confidence_score != null && msg.confidence_score > 0 && (
                    <div className="mt-1 text-xs opacity-60">
                      Confidence: {(msg.confidence_score * 100).toFixed(0)}%
                    </div>
                  )}
                </div>
              </div>
            ))}

            {/* Typing indicator */}
            {isTyping && (
              <div className="flex justify-start">
                <div className="rounded-2xl bg-muted px-4 py-3">
                  <div className="flex space-x-1">
                    <span className="h-2 w-2 animate-pulse-dot rounded-full bg-muted-foreground" />
                    <span className="h-2 w-2 animate-pulse-dot rounded-full bg-muted-foreground [animation-delay:0.3s]" />
                    <span className="h-2 w-2 animate-pulse-dot rounded-full bg-muted-foreground [animation-delay:0.6s]" />
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </ScrollArea>

        {/* Input area */}
        <div className="border-t p-4">
          <div className="mx-auto flex max-w-3xl items-end gap-2">
            {/* Voice button */}
            {voiceSupported && (
              <Button
                variant={isListening ? "destructive" : "outline"}
                size="icon"
                onClick={isListening ? stopListening : startListening}
                title={isListening ? "Stop recording" : "Start voice input"}
              >
                {isListening ? "⏹" : "🎤"}
              </Button>
            )}

            <Textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Describe your symptoms…"
              className="min-h-[44px] max-h-32 resize-none"
              rows={1}
            />

            <Button
              onClick={handleSend}
              disabled={!input.trim()}
              size="icon"
            >
              ➤
            </Button>
          </div>
          <p className="mt-2 text-center text-xs text-muted-foreground">
            ⚠️ AI responses are informational only — always consult a doctor.
          </p>
        </div>
      </div>
    </div>
  );
}
