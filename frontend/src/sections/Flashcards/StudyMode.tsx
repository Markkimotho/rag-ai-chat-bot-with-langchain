import { useCallback, useEffect, useMemo, useState } from "react";
import type { Flashcard } from "../../api/types";
import { FlipCard } from "./FlipCard";
import styles from "./Flashcards.module.css";

interface Props {
  deckName: string;
  cards: Flashcard[];
  onExit: () => void;
}

function shuffle<T>(arr: T[]): T[] {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export function StudyMode({ deckName, cards, onExit }: Props) {
  const order = useMemo(() => shuffle(cards), [cards]);
  const [idx, setIdx] = useState(0);
  const [flipped, setFlipped] = useState(false);
  const [known, setKnown] = useState(0);
  const [done, setDone] = useState(false);

  const current = order[idx];
  const total = order.length;

  const next = useCallback(
    (gotIt: boolean) => {
      if (gotIt) setKnown((k) => k + 1);
      if (idx + 1 >= total) {
        setDone(true);
      } else {
        setIdx((i) => i + 1);
        setFlipped(false);
      }
    },
    [idx, total],
  );

  const restart = () => {
    setIdx(0);
    setFlipped(false);
    setKnown(0);
    setDone(false);
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (done) return;
      if (e.code === "Space") {
        e.preventDefault();
        setFlipped((f) => !f);
      } else if (e.key === "ArrowRight" || e.key === "Enter") {
        e.preventDefault();
        next(true);
      } else if (e.key === "ArrowLeft") {
        e.preventDefault();
        next(false);
      }
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [done, next]);

  if (done) {
    const pct = total ? Math.round((known / total) * 100) : 0;
    return (
      <div className={styles.studyDone}>
        <div className={styles.doneScore}>{pct}%</div>
        <div className={styles.doneSub}>
          You knew {known} of {total} cards in “{deckName}”
        </div>
        <div className={styles.doneActions}>
          <button type="button" className={styles.secondary} onClick={onExit}>
            Back to deck
          </button>
          <button type="button" className={styles.primary} onClick={restart}>
            Study again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.study}>
      <div className={styles.studyTop}>
        <button type="button" className={styles.linkBtn} onClick={onExit}>
          ← Exit
        </button>
        <span className={styles.studyCounter}>
          {idx + 1} / {total} · known {known}
        </span>
      </div>
      <div className={styles.progressTrack}>
        <div
          className={styles.progressFill}
          style={{ width: `${(idx / total) * 100}%` }}
        />
      </div>

      <FlipCard
        front={current.front}
        back={current.back}
        flipped={flipped}
        onFlip={() => setFlipped((f) => !f)}
      />

      <div className={styles.studyControls}>
        <button
          type="button"
          className={styles.again}
          onClick={() => next(false)}
        >
          Again
        </button>
        <button
          type="button"
          className={styles.gotIt}
          onClick={() => next(true)}
        >
          Got it
        </button>
      </div>
      <p className={styles.kbdHint}>
        Space to flip · ← Again · → / Enter to advance
      </p>
    </div>
  );
}
