import type { QuizQuestion, ValidationResult } from "../../api/types";
import { Markdown } from "../../components/Markdown";
import styles from "./QuizPrep.module.css";

interface Props {
  result: ValidationResult;
  question: QuizQuestion;
  isLast: boolean;
  onNext: () => void;
}

export function FeedbackCard({ result, question, isLast, onNext }: Props) {
  const explanation = question.explanation ?? question.sample_answer ?? "";

  return (
    <div className={styles.quizCard}>
      <div
        className={`${styles.verdict} ${
          result.is_correct ? styles.correct : styles.incorrect
        }`}
      >
        <span className={styles.verdictIcon}>
          {result.is_correct ? "✓" : "✗"}
        </span>
        <span>
          {result.is_correct ? "Correct" : `Incorrect — ${result.score}/100`}
        </span>
      </div>

      {result.feedback && <p className={styles.feedback}>{result.feedback}</p>}

      {explanation && (
        <details className={styles.explain} open={!result.is_correct}>
          <summary>Explanation</summary>
          <Markdown>{explanation}</Markdown>
        </details>
      )}

      {result.key_missed && result.key_missed.length > 0 && (
        <div className={styles.missed}>
          Missed: {result.key_missed.join(", ")}
        </div>
      )}
      {result.hint && <div className={styles.hint}>Hint: {result.hint}</div>}

      <button type="button" className={styles.primary} onClick={onNext}>
        {isLast ? "See results" : "Next question →"}
      </button>
    </div>
  );
}
