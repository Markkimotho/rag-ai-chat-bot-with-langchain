import { useRef, useState } from "react";
import { ChatPanel } from "../components/ChatPanel";
import { EmptyState } from "../components/EmptyState";
import { Toggle } from "../components/Toggle";
import { useKnowledgeBase } from "../context/KnowledgeBaseContext";
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
  const { count } = useKnowledgeBase();
  const { messages, streaming, send, cancel } = useStreamingChat();
  const threadId = useRef(`code-${crypto.randomUUID()}`);
  const [useKb, setUseKb] = useState(false);

  const onSend = (text: string) =>
    send(text, {
      endpoint: "/api/code-chat/stream",
      model: selected,
      buildBody: (message) => ({
        message,
        thread_id: threadId.current,
        model: selected || undefined,
        use_kb: useKb,
      }),
    });

  return (
    <div className={styles.section}>
      <ChatPanel
        messages={messages}
        streaming={streaming}
        placeholder={
          useKb ? "Ask about your code/docs…" : "Ask anything about code…"
        }
        onSend={onSend}
        onStop={cancel}
        toolbar={
          <Toggle
            checked={useKb}
            onChange={setUseKb}
            label="Use my documents"
            hint={count > 0 ? `${count} chunks` : "knowledge base empty"}
            disabled={count === 0}
          />
        }
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
