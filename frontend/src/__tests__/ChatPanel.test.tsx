import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ChatPanel } from "../components/ChatPanel";
import type { ChatMessage } from "../api/types";

describe("ChatPanel", () => {
  it("shows the empty state when there are no messages", () => {
    render(
      <ChatPanel
        messages={[]}
        streaming={false}
        placeholder="Ask…"
        onSend={vi.fn()}
        onStop={vi.fn()}
        empty={<div>Get started</div>}
      />,
    );
    expect(screen.getByText("Get started")).toBeInTheDocument();
  });

  it("renders user and assistant messages", () => {
    const messages: ChatMessage[] = [
      { role: "user", content: "hello" },
      { role: "assistant", content: "hi there", model: "qwen2.5:7b" },
    ];
    render(
      <ChatPanel
        messages={messages}
        streaming={false}
        placeholder="Ask…"
        onSend={vi.fn()}
        onStop={vi.fn()}
      />,
    );
    expect(screen.getByText("hello")).toBeInTheDocument();
    expect(screen.getByText("hi there")).toBeInTheDocument();
  });

  it("sends on submit and clears the input", async () => {
    const onSend = vi.fn();
    render(
      <ChatPanel
        messages={[]}
        streaming={false}
        placeholder="Ask…"
        onSend={onSend}
        onStop={vi.fn()}
      />,
    );
    const input = screen.getByLabelText("Message");
    await userEvent.type(input, "what is recursion?");
    await userEvent.click(screen.getByRole("button", { name: /send message/i }));
    expect(onSend).toHaveBeenCalledWith("what is recursion?");
    expect((input as HTMLTextAreaElement).value).toBe("");
  });

  it("submits on Enter (without Shift)", async () => {
    const onSend = vi.fn();
    render(
      <ChatPanel
        messages={[]}
        streaming={false}
        placeholder="Ask…"
        onSend={onSend}
        onStop={vi.fn()}
      />,
    );
    const input = screen.getByLabelText("Message");
    await userEvent.type(input, "hi{Enter}");
    expect(onSend).toHaveBeenCalledWith("hi");
  });

  it("shows a Stop button while streaming and calls onStop", async () => {
    const onStop = vi.fn();
    render(
      <ChatPanel
        messages={[{ role: "assistant", content: "...", streaming: true }]}
        streaming
        placeholder="Ask…"
        onSend={vi.fn()}
        onStop={onStop}
      />,
    );
    await userEvent.click(screen.getByRole("button", { name: /stop generating/i }));
    expect(onStop).toHaveBeenCalledOnce();
  });
});
