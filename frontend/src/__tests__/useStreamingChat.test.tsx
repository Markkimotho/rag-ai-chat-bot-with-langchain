import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { StreamEvent } from "../api/types";

// Drive the hook by capturing the onEvent callback streamPost is given.
const streamPost = vi.fn();
vi.mock("../api/stream", () => ({
  streamPost: (path: string, body: unknown, handlers: { onEvent: (e: StreamEvent) => void }) =>
    streamPost(path, body, handlers),
}));

import { useStreamingChat } from "../hooks/useStreamingChat";

afterEach(() => streamPost.mockReset());

describe("useStreamingChat", () => {
  it("accumulates tokens and attaches sources, then finalizes on done", async () => {
    let emit: (e: StreamEvent) => void = () => {};
    streamPost.mockImplementation((_p, _b, { onEvent }) => {
      emit = onEvent;
    });

    const { result } = renderHook(() => useStreamingChat());

    act(() => {
      result.current.send("hi", {
        endpoint: "/api/chat/stream",
        buildBody: (t) => ({ question: t }),
        model: "qwen2.5:7b",
      });
    });

    // user message + empty assistant placeholder
    expect(result.current.messages).toHaveLength(2);
    expect(result.current.messages[0]).toMatchObject({ role: "user", content: "hi" });
    expect(result.current.streaming).toBe(true);

    act(() => emit({ type: "token", text: "Hel" }));
    act(() => emit({ type: "token", text: "lo" }));
    act(() => emit({ type: "sources", sources: [{ source: "a.pdf", page: 1 }] }));
    act(() => emit({ type: "done" }));

    await waitFor(() => expect(result.current.streaming).toBe(false));
    const assistant = result.current.messages[1];
    expect(assistant.content).toBe("Hello");
    expect(assistant.sources).toEqual([{ source: "a.pdf", page: 1 }]);
    expect(assistant.streaming).toBe(false);
  });

  it("flags an error event on the assistant message", async () => {
    let emit: (e: StreamEvent) => void = () => {};
    streamPost.mockImplementation((_p, _b, { onEvent }) => {
      emit = onEvent;
    });
    const { result } = renderHook(() => useStreamingChat());
    act(() => {
      result.current.send("hi", {
        endpoint: "/x",
        buildBody: (t) => ({ t }),
      });
    });
    act(() => emit({ type: "error", message: "boom" }));
    act(() => emit({ type: "done" }));

    await waitFor(() => expect(result.current.streaming).toBe(false));
    expect(result.current.messages[1]).toMatchObject({
      error: true,
      content: "boom",
    });
  });

  it("ignores a send while already streaming", () => {
    streamPost.mockImplementation(() => {});
    const { result } = renderHook(() => useStreamingChat());
    act(() => {
      result.current.send("first", { endpoint: "/x", buildBody: (t) => ({ t }) });
    });
    act(() => {
      result.current.send("second", { endpoint: "/x", buildBody: (t) => ({ t }) });
    });
    expect(streamPost).toHaveBeenCalledOnce();
  });
});
