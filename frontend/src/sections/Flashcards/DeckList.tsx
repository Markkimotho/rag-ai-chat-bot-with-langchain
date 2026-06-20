import { useState } from "react";
import { api, ApiError } from "../../api/client";
import type { DeckSummary } from "../../api/types";
import { EmptyState } from "../../components/EmptyState";
import { useKnowledgeBase } from "../../context/KnowledgeBaseContext";
import { useModelContext } from "../../context/ModelContext";
import { Select } from "../../components/Select";
import styles from "./Flashcards.module.css";

interface Props {
  decks: DeckSummary[];
  loading: boolean;
  onOpen: (id: string) => void;
  onChanged: () => void;
}

export function DeckList({ decks, loading, onOpen, onChanged }: Props) {
  const { selected } = useModelContext();
  const { count } = useKnowledgeBase();
  const [name, setName] = useState("");
  const [topic, setTopic] = useState("");
  const [n, setN] = useState(10);
  const [busy, setBusy] = useState<"create" | "generate" | null>(null);
  const [error, setError] = useState<string | null>(null);

  const create = async () => {
    if (!name.trim()) return;
    setBusy("create");
    setError(null);
    try {
      const deck = await api.createDeck(name.trim());
      setName("");
      onChanged();
      onOpen(deck.id);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Could not create deck.");
    } finally {
      setBusy(null);
    }
  };

  const generate = async () => {
    if (!topic.trim()) return;
    setBusy("generate");
    setError(null);
    try {
      const deck = await api.generateDeck({
        topic: topic.trim(),
        n,
        model: selected || undefined,
      });
      setTopic("");
      onChanged();
      onOpen(deck.id);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "Generation failed.");
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className={styles.list}>
      <div className={styles.makeRow}>
        <div className={styles.makeCard}>
          <h3 className={styles.makeTitle}>New deck</h3>
          <div className={styles.inlineForm}>
            <input
              className={styles.input}
              placeholder="Deck name…"
              value={name}
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && create()}
            />
            <button
              type="button"
              className={styles.primary}
              disabled={!name.trim() || busy !== null}
              onClick={create}
            >
              {busy === "create" ? "Creating…" : "Create"}
            </button>
          </div>
        </div>

        <div className={styles.makeCard}>
          <h3 className={styles.makeTitle}>Generate from your documents</h3>
          <div className={styles.inlineForm}>
            <input
              className={styles.input}
              placeholder="Topic, e.g. binary search…"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && generate()}
            />
            <div className={styles.countSelect}>
              <Select
                ariaLabel="Number of cards"
                value={String(n)}
                options={[5, 10, 15, 20].map((v) => ({
                  value: String(v),
                  label: `${v} cards`,
                }))}
                onChange={(v) => setN(Number(v))}
              />
            </div>
            <button
              type="button"
              className={styles.primary}
              disabled={!topic.trim() || busy !== null || count === 0}
              onClick={generate}
            >
              {busy === "generate" ? "Generating…" : "Generate"}
            </button>
          </div>
          {count === 0 && (
            <p className={styles.hint}>
              Upload a PDF or scrape a topic first to generate from your material.
            </p>
          )}
        </div>
      </div>

      {error && <div className={styles.error}>{error}</div>}

      {loading ? (
        <p className={styles.hint}>Loading decks…</p>
      ) : decks.length === 0 ? (
        <EmptyState
          title="No decks yet"
          description="Create a deck by hand or generate one from your uploaded documents."
        />
      ) : (
        <div className={styles.deckGrid}>
          {decks.map((d) => (
            <button
              key={d.id}
              type="button"
              className={styles.deckTile}
              onClick={() => onOpen(d.id)}
            >
              <span className={styles.deckName}>{d.name}</span>
              <span className={styles.deckCount}>
                {d.card_count} card{d.card_count !== 1 ? "s" : ""}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
