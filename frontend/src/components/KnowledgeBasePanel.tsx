import { useEffect, useRef, useState } from "react";
import { useKnowledgeBase } from "../context/KnowledgeBaseContext";
import { Spinner } from "./Spinner";
import styles from "./KnowledgeBasePanel.module.css";

type Status = { kind: "ok" | "err"; text: string } | null;

export function KnowledgeBasePanel({ onClose }: { onClose: () => void }) {
  const { sources, count, busy, upload, scrape, clear, refresh } =
    useKnowledgeBase();
  const fileRef = useRef<HTMLInputElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const [topic, setTopic] = useState("");
  const [status, setStatus] = useState<Status>(null);

  useEffect(() => {
    refresh();
  }, [refresh]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    const onClick = (e: MouseEvent) => {
      if (panelRef.current && !panelRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    document.addEventListener("keydown", onKey);
    document.addEventListener("mousedown", onClick);
    return () => {
      document.removeEventListener("keydown", onKey);
      document.removeEventListener("mousedown", onClick);
    };
  }, [onClose]);

  const onFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setStatus(null);
    try {
      const detail = await upload(Array.from(files));
      setStatus({ kind: "ok", text: detail });
    } catch {
      setStatus({ kind: "err", text: "Upload failed." });
    }
    if (fileRef.current) fileRef.current.value = "";
  };

  const onScrape = async () => {
    if (!topic.trim()) return;
    setStatus(null);
    try {
      const detail = await scrape(topic.trim());
      setStatus({ kind: "ok", text: detail });
      setTopic("");
    } catch {
      setStatus({ kind: "err", text: "Scrape failed." });
    }
  };

  const onClear = async () => {
    await clear();
    setStatus({ kind: "ok", text: "Knowledge base cleared." });
  };

  return (
    <div className={styles.panel} ref={panelRef} role="dialog" aria-label="Knowledge base">
      <div className={styles.header}>
        <span className={styles.title}>Knowledge base</span>
        <span className={styles.count}>
          {sources.length} file{sources.length !== 1 ? "s" : ""} · {count} chunks
        </span>
      </div>

      <div className={styles.actions}>
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
          className={styles.primary}
          onClick={() => fileRef.current?.click()}
          disabled={busy}
        >
          Upload PDF
        </button>
        <div className={styles.scrapeRow}>
          <input
            className={styles.scrapeInput}
            placeholder="Scrape a web topic…"
            value={topic}
            aria-label="Scrape a web topic"
            onChange={(e) => setTopic(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && onScrape()}
          />
          <button
            type="button"
            className={styles.secondary}
            onClick={onScrape}
            disabled={busy || !topic.trim()}
          >
            Index
          </button>
        </div>
      </div>

      {busy && (
        <div className={styles.working}>
          <Spinner label="Indexing…" />
        </div>
      )}

      {status && (
        <div
          className={status.kind === "ok" ? styles.ok : styles.err}
          role="status"
        >
          {status.kind === "ok" ? "✓ " : "✗ "}
          {status.text}
        </div>
      )}

      <div className={styles.listWrap}>
        {sources.length === 0 ? (
          <div className={styles.empty}>
            No files yet. Upload a PDF or scrape a topic to build your knowledge base.
          </div>
        ) : (
          <ul className={styles.list}>
            {sources.map((s) => (
              <li key={s.source} className={styles.item}>
                <span className={styles.badge} data-type={s.type}>
                  {s.type === "web" ? "web" : "pdf"}
                </span>
                <span className={styles.name} title={s.source}>
                  {s.source}
                </span>
                <span className={styles.chunks}>{s.chunks}</span>
              </li>
            ))}
          </ul>
        )}
      </div>

      {sources.length > 0 && (
        <button
          type="button"
          className={styles.clear}
          onClick={onClear}
          disabled={busy}
        >
          Clear knowledge base
        </button>
      )}
    </div>
  );
}
