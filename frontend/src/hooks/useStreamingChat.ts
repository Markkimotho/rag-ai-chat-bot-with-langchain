// Generic streaming-chat state machine shared by RegularChat, StudyAgent, and
// ProgrammingAssistant. Differences (endpoint, request body) are injected.

import { useCallback, useEffect, useRef, useState } from "react";
import { streamPost } from "../api/stream";
import type { ChatMessage, StreamEvent } from "../api/types";

export interface SendOptions {
  /** Endpoint to POST to, e.g. "/api/chat/stream". */
  endpoint: string;
  /** Build the request body from the user's text. */
  buildBody: (text: string) => unknown;
  /** Model label to stamp on the assistant message. */
  model?: string;
}

export interface StreamingChat {
  messages: ChatMessage[];
  streaming: boolean;
  send: (text: string, opts: SendOptions) => void;
  cancel: () => void;
  reset: () => void;
}

export function useStreamingChat(): StreamingChat {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streaming, setStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setStreaming(false);
    setMessages((prev) =>
      prev.map((m) => (m.streaming ? { ...m, streaming: false } : m)),
    );
  }, []);

  const reset = useCallback(() => {
    cancel();
    setMessages([]);
  }, [cancel]);

  const send = useCallback(
    (text: string, opts: SendOptions) => {
      const trimmed = text.trim();
      if (!trimmed || streaming) return;

      const controller = new AbortController();
      abortRef.current = controller;
      setStreaming(true);

      setMessages((prev) => [
        ...prev,
        { role: "user", content: trimmed },
        {
          role: "assistant",
          content: "",
          streaming: true,
          model: opts.model,
          tools: [],
        },
      ]);

      const patchLast = (patch: (m: ChatMessage) => ChatMessage) =>
        setMessages((prev) => {
          const next = [...prev];
          const i = next.length - 1;
          if (i >= 0 && next[i].role === "assistant") next[i] = patch(next[i]);
          return next;
        });

      const onEvent = (ev: StreamEvent) => {
        switch (ev.type) {
          case "token":
            patchLast((m) => ({ ...m, content: m.content + ev.text }));
            break;
          case "sources":
            patchLast((m) => ({ ...m, sources: ev.sources }));
            break;
          case "tool":
            patchLast((m) => ({ ...m, tools: [...(m.tools ?? []), ev.name] }));
            break;
          case "error":
            patchLast((m) => ({
              ...m,
              content: m.content || ev.message,
              error: true,
            }));
            break;
          case "done":
            patchLast((m) => ({ ...m, streaming: false }));
            setStreaming(false);
            abortRef.current = null;
            break;
        }
      };

      streamPost(opts.endpoint, opts.buildBody(trimmed), {
        onEvent,
        signal: controller.signal,
      });
    },
    [streaming],
  );

  // Cancel any in-flight stream on unmount.
  useEffect(() => () => abortRef.current?.abort(), []);

  return { messages, streaming, send, cancel, reset };
}
