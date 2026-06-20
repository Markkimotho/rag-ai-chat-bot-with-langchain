import { useEffect, useState } from "react";
import { api } from "../../api/client";
import type { Deck } from "../../api/types";
import { Spinner } from "../../components/Spinner";
import styles from "./Flashcards.module.css";

interface Props {
  deckId: string;
  onBack: () => void;
  onStudy: (deck: Deck) => void;
  onChanged: () => void;
}

export function DeckEditor({ deckId, onBack, onStudy, onChanged }: Props) {
  const [deck, setDeck] = useState<Deck | null>(null);
  const [front, setFront] = useState("");
  const [back, setBack] = useState("");
  const [busy, setBusy] = useState(false);

  const load = async () => {
    setDeck(await api.deck(deckId));
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [deckId]);

  const addCard = async () => {
    if (!front.trim() || !back.trim()) return;
    setBusy(true);
    try {
      setDeck(await api.addCard(deckId, front.trim(), back.trim()));
      setFront("");
      setBack("");
      onChanged();
    } finally {
      setBusy(false);
    }
  };

  const removeCard = async (cardId: string) => {
    setDeck(await api.deleteCard(deckId, cardId));
    onChanged();
  };

  const removeDeck = async () => {
    await api.deleteDeck(deckId);
    onChanged();
    onBack();
  };

  if (!deck) {
    return (
      <div className={styles.editor}>
        <Spinner label="Loading deck…" />
      </div>
    );
  }

  return (
    <div className={styles.editor}>
      <div className={styles.editorTop}>
        <button type="button" className={styles.linkBtn} onClick={onBack}>
          ← All decks
        </button>
        <div className={styles.editorActions}>
          <button
            type="button"
            className={styles.danger}
            onClick={removeDeck}
          >
            Delete deck
          </button>
          <button
            type="button"
            className={styles.primary}
            disabled={deck.cards.length === 0}
            onClick={() => onStudy(deck)}
          >
            Study ({deck.cards.length})
          </button>
        </div>
      </div>

      <h2 className={styles.editorTitle}>{deck.name}</h2>

      <div className={styles.addCard}>
        <input
          className={styles.input}
          placeholder="Front (question / term)"
          value={front}
          onChange={(e) => setFront(e.target.value)}
        />
        <input
          className={styles.input}
          placeholder="Back (answer)"
          value={back}
          onChange={(e) => setBack(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && addCard()}
        />
        <button
          type="button"
          className={styles.primary}
          disabled={!front.trim() || !back.trim() || busy}
          onClick={addCard}
        >
          Add card
        </button>
      </div>

      {deck.cards.length === 0 ? (
        <p className={styles.hint}>No cards yet. Add one above.</p>
      ) : (
        <ul className={styles.cardList}>
          {deck.cards.map((c) => (
            <li key={c.id} className={styles.cardRow}>
              <span className={styles.cardFront}>{c.front}</span>
              <span className={styles.cardBack}>{c.back}</span>
              <button
                type="button"
                className={styles.cardDelete}
                aria-label="Delete card"
                onClick={() => removeCard(c.id)}
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
