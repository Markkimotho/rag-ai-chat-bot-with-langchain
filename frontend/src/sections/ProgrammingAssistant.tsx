import { useRef } from "react";
import { ChatPanel } from "../components/ChatPanel";
import { EmptyState } from "../components/EmptyState";
import { useModelContext } from "../context/ModelContext";
import { useStreamingChat } from "../hooks/useStreamingChat";
import styles from "./Section.module.css";

const EXAMPLES = [
  "Write a debounce function in TypeScript",
  "Explain Python decorators with an example",
  "Why am I getting 'index out of range' here?",
  "Refactor this loop to be more idiomatic",
];

export function ProgrammingAssistant() {
  const { selected } = useModelContext();
  const { messages, streaming, send, cancel } = useStreamingChat();
  const threadId = useRef(`code-${crypto.randomUUID()}`);

  const onSend = (text: string) =>
    send(text, {
      endpoint: "/api/code-chat/stream",
      model: selected,
      buildBody: (message) => ({
        message,
        thread_id: threadId.current,
        model: selected || undefined,
      }),
    });

  return (
    <div className={styles.section}>
      <ChatPanel
        messages={messages}
        streaming={streaming}
        placeholder="Ask anything about code…"
        onSend={onSend}
        onStop={cancel}
        empty={
          <EmptyState
            title="Programming Assistant"
            description="A coding companion for any language or stack. It writes, explains, and debugs code — no documents needed."
          >
            {EXAMPLES.map((e) => (
              <button
                key={e}
                type="button"
                className={styles.chip}
                onClick={() => onSend(e)}
              >
                {e}
              </button>
            ))}
          </EmptyState>
        }
      />
    </div>
  );
}
