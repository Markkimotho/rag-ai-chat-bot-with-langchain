import { useRef, useState } from "react";
import { ChatPanel } from "../components/ChatPanel";
import { EmptyState } from "../components/EmptyState";
import type { ChatMode } from "../api/types";
import { useKnowledgeBase } from "../context/KnowledgeBaseContext";
import { useModelContext } from "../context/ModelContext";
import { useStreamingChat } from "../hooks/useStreamingChat";
import styles from "./Section.module.css";

const STARTERS = [
  "What are the key findings across all documents?",
  "Summarize the most important points.",
  "What risks or challenges are mentioned?",
  "List the main recommendations.",
];

export function RegularChat() {
  const { selected } = useModelContext();
  const { count } = useKnowledgeBase();
  const { messages, streaming, send, cancel } = useStreamingChat();
  const sessionId = useRef(`chat-${crypto.randomUUID()}`);
  const [mode, setMode] = useState<ChatMode>("langchain");

  const onSend = (text: string) =>
    send(text, {
      endpoint: "/api/chat/stream",
      model: selected,
      buildBody: (question) => ({
        question,
        session_id: sessionId.current,
        mode,
        model: selected || undefined,
      }),
    });

  return (
    <div className={styles.section}>
      <ChatPanel
        messages={messages}
        streaming={streaming}
        placeholder="Ask a question about your documents…"
        onSend={onSend}
        onStop={cancel}
        toolbar={
          <div className={styles.modeToggle} role="group" aria-label="Orchestration mode">
            {(["langchain", "langgraph"] as ChatMode[]).map((m) => (
              <button
                key={m}
                type="button"
                className={`${styles.modeBtn} ${mode === m ? styles.modeActive : ""}`}
                aria-pressed={mode === m}
                onClick={() => setMode(m)}
              >
                {m === "langchain" ? "LangChain" : "LangGraph"}
              </button>
            ))}
          </div>
        }
        empty={
          <EmptyState
            title="Regular Chat"
            description={
              count === 0
                ? "Your knowledge base is empty. Upload a PDF or scrape a topic from the top bar, then ask questions grounded in your documents."
                : "Ask questions grounded in your indexed documents. Answers include source citations."
            }
          >
            {count > 0 &&
              STARTERS.map((q) => (
                <button
                  key={q}
                  type="button"
                  className={styles.chip}
                  onClick={() => onSend(q)}
                >
                  {q}
                </button>
              ))}
          </EmptyState>
        }
      />
    </div>
  );
}
