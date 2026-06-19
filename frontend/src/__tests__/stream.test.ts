import { describe, expect, it, vi } from "vitest";
import { parseSSEChunk, streamPost } from "../api/stream";
import type { StreamEvent } from "../api/types";

describe("parseSSEChunk", () => {
  it("parses complete frames and keeps the trailing partial", () => {
    const buf =
      'data: {"type":"token","text":"a"}\n\n' +
      'data: {"type":"token","text":"b"}\n\n' +
      'data: {"type":"sou';
    const { events, rest } = parseSSEChunk(buf);
    expect(events).toEqual([
      { type: "token", text: "a" },
      { type: "token", text: "b" },
    ]);
    expect(rest).toBe('data: {"type":"sou');
  });

  it("ignores malformed JSON frames", () => {
    const { events } = parseSSEChunk("data: not-json\n\n");
    expect(events).toEqual([]);
  });
});

function streamResponse(frames: string[]): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const f of frames) controller.enqueue(encoder.encode(f));
      controller.close();
    },
  });
  return new Response(body, { status: 200 });
}

describe("streamPost", () => {
  it("emits each parsed SSE event in order", async () => {
    const frames = [
      'data: {"type":"token","text":"Hello"}\n\n',
      'data: {"type":"token","text":" world"}\n\ndata: {"type":"done"}\n\n',
    ];
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(streamResponse(frames)),
    );

    const got: StreamEvent[] = [];
    await streamPost("/api/x", { a: 1 }, { onEvent: (e) => got.push(e) });

    expect(got).toEqual([
      { type: "token", text: "Hello" },
      { type: "token", text: " world" },
      { type: "done" },
    ]);
    vi.unstubAllGlobals();
  });

  it("surfaces an error + done when the response is not ok", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "bad" }), { status: 500 }),
      ),
    );
    const got: StreamEvent[] = [];
    await streamPost("/api/x", {}, { onEvent: (e) => got.push(e) });
    expect(got).toEqual([
      { type: "error", message: "bad" },
      { type: "done" },
    ]);
    vi.unstubAllGlobals();
  });
});
