import styles from "./Flashcards.module.css";

interface Props {
  front: string;
  back: string;
  flipped: boolean;
  onFlip: () => void;
}

/** A 3D flip card. Click or press the card to flip between front and back. */
export function FlipCard({ front, back, flipped, onFlip }: Props) {
  return (
    <button
      type="button"
      className={styles.flipCard}
      data-flipped={flipped}
      onClick={onFlip}
      aria-label={flipped ? "Show question" : "Show answer"}
    >
      <div className={styles.flipInner}>
        <div className={styles.faceFront}>
          <span className={styles.faceTag}>Question</span>
          <p className={styles.faceText}>{front}</p>
          <span className={styles.faceHint}>click to flip</span>
        </div>
        <div className={styles.faceBack}>
          <span className={styles.faceTag}>Answer</span>
          <p className={styles.faceText}>{back}</p>
        </div>
      </div>
    </button>
  );
}
