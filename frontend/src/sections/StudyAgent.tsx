import { useRef, useState } from "react";
import { ChatPanel } from "../components/ChatPanel";
import { EmptyState } from "../components/EmptyState";
import { useModelContext } from "../context/ModelContext";
import { useStreamingChat } from "../hooks/useStreamingChat";
import styles from "./Section.module.css";

const EXAMPLES = [
  "Quiz me on Python async programming",
  "Search for system design interview prep and quiz me",
  "Generate 5 behavioral interview questions",
  "What topics are in my knowledge base?",
];

export function StudyAgent() {
  const { selected } = useModelContext();
  const { messages, streaming, send, cancel, reset } = useStreamingChat();
  const threadId = useRef(`agent-${crypto.randomUUID()}`);
  const [, force] = useState(0);

  const onSend = (text: string) =>
    send(text, {
      endpoint: "/api/agent/stream",
      model: selected,
      buildBody: (message) => ({
        message,
        thread_id: threadId.current,
        model: selected || undefined,
      }),
    });

  const newSession = () => {
    reset();
    threadId.current = `agent-${crypto.randomUUID()}`;
    force((n) => n + 1);
  };

  return (
    <div className={styles.section}>
      <ChatPanel
        messages={messages}
        streaming={streaming}
        placeholder="Ask your tutor to quiz, search, or explain…"
        onSend={onSend}
        onStop={cancel}
        toolbar={
          <button
            type="button"
            className={styles.toolbarBtn}
            onClick={newSession}
            disabled={streaming}
          >
            New session
          </button>
        }
        empty={
          <EmptyState
            title="Study Agent"
            description="An AI tutor with tools. It can search the web, build a knowledge base, quiz you, grade answers, and explain concepts — all in one conversation."
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
