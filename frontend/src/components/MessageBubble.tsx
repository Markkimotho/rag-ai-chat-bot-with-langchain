import type { ChatMessage } from "../api/types";
import { Markdown } from "./Markdown";
import { Spinner } from "./Spinner";
import styles from "./MessageBubble.module.css";

export function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";
  const showCursor = message.streaming && message.content.length > 0;

  return (
    <div className={`${styles.row} ${isUser ? styles.user : styles.assistant}`}>
      <div className={styles.bubble} data-error={message.error || undefined}>
        {!isUser && message.model && (
          <div className={styles.model}>{message.model}</div>
        )}

        {!isUser && message.tools && message.tools.length > 0 && (
          <div className={styles.tools}>
            {message.tools.map((t) => (
              <span key={t} className={styles.tool}>
                {t.replace(/_/g, " ")}
              </span>
            ))}
          </div>
        )}

        {isUser ? (
          <p className={styles.userText}>{message.content}</p>
        ) : message.content ? (
          <>
            <Markdown>{message.content}</Markdown>
            {showCursor && <span className={styles.cursor} aria-hidden />}
          </>
        ) : (
          <Spinner label="Thinking" />
        )}

        {!isUser && message.sources && message.sources.length > 0 && (
          <details className={styles.sources}>
            <summary>Sources ({message.sources.length})</summary>
            <ul>
              {message.sources.map((s, i) => (
                <li key={`${s.source}-${s.page}-${i}`}>
                  {s.source} — page {s.page}
                </li>
              ))}
            </ul>
          </details>
        )}
      </div>
    </div>
  );
}
