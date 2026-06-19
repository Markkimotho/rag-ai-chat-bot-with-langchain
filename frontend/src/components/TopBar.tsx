import { useRef, useState } from "react";
import { useKnowledgeBase } from "../context/KnowledgeBaseContext";
import { useModelContext } from "../context/ModelContext";
import { ModelPicker } from "./ModelPicker";
import styles from "./TopBar.module.css";

interface Props {
  title: string;
  isMobile: boolean;
  onOpenMenu: () => void;
}

export function TopBar({ title, isMobile, onOpenMenu }: Props) {
  const { online, modelCount } = useModelContext();
  const { count, busy, upload, scrape, clear } = useKnowledgeBase();
  const fileRef = useRef<HTMLInputElement>(null);
  const [topic, setTopic] = useState("");
  const [note, setNote] = useState<string | null>(null);

  const flash = (m: string) => {
    setNote(m);
    setTimeout(() => setNote(null), 4000);
  };

  const onFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    try {
      flash(await upload(Array.from(files)));
    } catch {
      flash("Upload failed.");
    }
    if (fileRef.current) fileRef.current.value = "";
  };

  const onScrape = async () => {
    if (!topic.trim()) return;
    try {
      flash(await scrape(topic.trim()));
      setTopic("");
    } catch {
      flash("Scrape failed.");
    }
  };

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

        <div className={styles.kb}>
          <span className={styles.kbCount} title="Indexed chunks">
            {count} chunks
          </span>
          <input
            ref={fileRef}
            type="file"
            accept="application/pdf"
            multiple
            hidden
            onChange={(e) => onFiles(e.target.files)}
          />
          <button
            type="button"
            className={styles.kbBtn}
            onClick={() => fileRef.current?.click()}
            disabled={busy}
          >
            Upload PDF
          </button>
          <div className={styles.scrapeBox}>
            <input
              className={styles.scrapeInput}
              placeholder="Scrape topic…"
              value={topic}
              aria-label="Scrape a web topic"
              onChange={(e) => setTopic(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && onScrape()}
            />
            <button
              type="button"
              className={styles.kbBtn}
              onClick={onScrape}
              disabled={busy || !topic.trim()}
            >
              {busy ? "…" : "Index"}
            </button>
          </div>
          <button
            type="button"
            className={styles.kbBtnGhost}
            onClick={() => clear()}
            disabled={busy || count === 0}
          >
            Clear
          </button>
        </div>
      </div>

      {note && <div className={styles.note}>{note}</div>}
    </header>
  );
}
