// POST-based SSE client. EventSource only supports GET, but our streaming
// endpoints take JSON bodies, so we read response.body as a ReadableStream and
// parse "data: {json}\n\n" frames ourselves.

import type { StreamEvent } from "./types";

export function parseSSEChunk(
  buffer: string,
): { events: StreamEvent[]; rest: string } {
  const events: StreamEvent[] = [];
  const parts = buffer.split("\n\n");
  const rest = parts.pop() ?? "";
  for (const part of parts) {
    const line = part.split("\n").find((l) => l.startsWith("data:"));
    if (!line) continue;
    const payload = line.slice(5).trim();
    if (!payload) continue;
    try {
      events.push(JSON.parse(payload) as StreamEvent);
    } catch {
      /* ignore malformed frame */
    }
  }
  return { events, rest };
}

export interface StreamHandlers {
  onEvent: (event: StreamEvent) => void;
  signal?: AbortSignal;
}

export async function streamPost(
  path: string,
  body: unknown,
  { onEvent, signal }: StreamHandlers,
): Promise<void> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

  if (!res.ok || !res.body) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* non-JSON */
    }
    onEvent({ type: "error", message: detail });
    onEvent({ type: "done" });
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const { events, rest } = parseSSEChunk(buffer);
      buffer = rest;
      for (const ev of events) onEvent(ev);
    }
  } catch (err) {
    if ((err as Error).name !== "AbortError") {
      onEvent({ type: "error", message: (err as Error).message });
      onEvent({ type: "done" });
    }
  }
}
