import { useState } from "react";
import { useKnowledgeBase } from "../context/KnowledgeBaseContext";
import { useModelContext } from "../context/ModelContext";
import { KnowledgeBasePanel } from "./KnowledgeBasePanel";
import { ModelPicker } from "./ModelPicker";
import styles from "./TopBar.module.css";

interface Props {
  title: string;
  isMobile: boolean;
  onOpenMenu: () => void;
}

export function TopBar({ title, isMobile, onOpenMenu }: Props) {
  const { online, modelCount } = useModelContext();
  const { count, sources, busy } = useKnowledgeBase();
  const [kbOpen, setKbOpen] = useState(false);

  return (
    <header className={styles.bar}>
      <div className={styles.left}>
        {isMobile && (
          <button
            type="button"
            className={styles.hamburger}
            onClick={onOpenMenu}
            aria-label="Open menu"
          >
            <span /> <span /> <span />
          </button>
        )}
        <h1 className={styles.title}>{title}</h1>
        <span
          className={styles.status}
          data-online={online}
          title={online ? "Ollama connected" : "Ollama offline"}
        >
          <span className={styles.dot} />
          {online ? `${modelCount} models` : "offline"}
        </span>
      </div>

      <div className={styles.right}>
        <ModelPicker />

        <div className={styles.kbWrap}>
          <button
            type="button"
            className={styles.kbButton}
            aria-haspopup="dialog"
            aria-expanded={kbOpen}
            onClick={() => setKbOpen((o) => !o)}
          >
            <svg viewBox="0 0 16 16" width="15" height="15" fill="none" aria-hidden>
              <ellipse cx="8" cy="4" rx="5.5" ry="2" stroke="currentColor" strokeWidth="1.3" />
              <path d="M2.5 4v8c0 1.1 2.46 2 5.5 2s5.5-.9 5.5-2V4" stroke="currentColor" strokeWidth="1.3" />
              <path d="M2.5 8c0 1.1 2.46 2 5.5 2s5.5-.9 5.5-2" stroke="currentColor" strokeWidth="1.3" />
            </svg>
            <span className={styles.kbLabel}>
              {sources.length > 0
                ? `${sources.length} file${sources.length !== 1 ? "s" : ""}`
                : "Knowledge base"}
            </span>
            {count > 0 && <span className={styles.kbBadge}>{count}</span>}
            {busy && <span className={styles.kbDot} aria-label="working" />}
          </button>
          {kbOpen && <KnowledgeBasePanel onClose={() => setKbOpen(false)} />}
        </div>
      </div>
    </header>
  );
}
