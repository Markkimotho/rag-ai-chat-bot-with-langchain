import {
  useEffect,
  useRef,
  useState,
  type FormEvent,
  type ReactNode,
} from "react";
import type { ChatMessage } from "../api/types";
import { MessageBubble } from "./MessageBubble";
import styles from "./ChatPanel.module.css";

interface Props {
  messages: ChatMessage[];
  streaming: boolean;
  placeholder: string;
  onSend: (text: string) => void;
  onStop: () => void;
  /** Shown when there are no messages yet. */
  empty?: ReactNode;
  /** Optional controls rendered above the input (e.g. mode toggle). */
  toolbar?: ReactNode;
}

export function ChatPanel({
  messages,
  streaming,
  placeholder,
  onSend,
  onStop,
  empty,
  toolbar,
}: Props) {
  const [text, setText] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollRef.current;
    // scrollTo is unavailable in jsdom; guard so tests and SSR don't crash.
    el?.scrollTo?.({ top: el.scrollHeight });
  }, [messages]);

  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (!text.trim() || streaming) return;
    onSend(text);
    setText("");
  };

  return (
    <div className={styles.panel}>
      <div className={styles.scroll} ref={scrollRef}>
        {messages.length === 0 && empty ? (
          <div className={styles.emptyWrap}>{empty}</div>
        ) : (
          <div className={styles.messages}>
            {messages.map((m, i) => (
              <MessageBubble key={i} message={m} />
            ))}
          </div>
        )}
      </div>

      <form className={styles.composer} onSubmit={submit}>
        {toolbar && <div className={styles.toolbar}>{toolbar}</div>}
        <div className={styles.inputRow}>
          <textarea
            className={styles.input}
            value={text}
            placeholder={placeholder}
            rows={1}
            aria-label="Message"
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit(e);
              }
            }}
          />
          {streaming ? (
            <button
              type="button"
              className={styles.stop}
              onClick={onStop}
              aria-label="Stop generating"
            >
              Stop
            </button>
          ) : (
            <button
              type="submit"
              className={styles.send}
              disabled={!text.trim()}
              aria-label="Send message"
            >
              Send
            </button>
          )}
        </div>
      </form>
    </div>
  );
}
