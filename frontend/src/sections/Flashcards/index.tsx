import { useCallback, useEffect, useState } from "react";
import { api } from "../../api/client";
import type { Deck, DeckSummary } from "../../api/types";
import { DeckEditor } from "./DeckEditor";
import { DeckList } from "./DeckList";
import { StudyMode } from "./StudyMode";
import sectionStyles from "../Section.module.css";
import styles from "./Flashcards.module.css";

type View =
  | { kind: "list" }
  | { kind: "edit"; deckId: string }
  | { kind: "study"; deck: Deck };

export function Flashcards() {
  const [decks, setDecks] = useState<DeckSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [view, setView] = useState<View>({ kind: "list" });

  const refresh = useCallback(async () => {
    try {
      const { decks } = await api.decks();
      setDecks(decks);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  return (
    <div className={sectionStyles.section}>
      <div className={styles.scroll}>
        <div className={styles.container}>
          {view.kind === "list" && (
            <DeckList
              decks={decks}
              loading={loading}
              onOpen={(deckId) => setView({ kind: "edit", deckId })}
              onChanged={refresh}
            />
          )}
          {view.kind === "edit" && (
            <DeckEditor
              deckId={view.deckId}
              onBack={() => setView({ kind: "list" })}
              onStudy={(deck) => setView({ kind: "study", deck })}
              onChanged={refresh}
            />
          )}
          {view.kind === "study" && (
            <StudyMode
              deckName={view.deck.name}
              cards={view.deck.cards}
              onExit={() => setView({ kind: "edit", deckId: view.deck.id })}
            />
          )}
        </div>
      </div>
    </div>
  );
}
