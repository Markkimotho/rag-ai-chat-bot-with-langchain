import { type AnswerRecord, scoreSummary } from "./quizReducer";
import styles from "./QuizPrep.module.css";

interface Props {
  answers: AnswerRecord[];
  topic: string;
  onRetake: () => void;
  onNew: () => void;
}

export function ResultsCard({ answers, topic, onRetake, onNew }: Props) {
  const { total, correct, pct } = scoreSummary(answers);
  const grade = pct >= 80 ? "good" : pct >= 60 ? "ok" : "poor";
  const wrong = answers.filter((a) => !a.result.is_correct);

  return (
    <div className={styles.results}>
      <div className={`${styles.scoreHero} ${styles[grade]}`}>
        <div className={styles.scorePct}>{pct}%</div>
        <div className={styles.scoreSub}>
          {correct} / {total} correct · {topic}
        </div>
      </div>

      <h3 className={styles.breakdownTitle}>Question breakdown</h3>
      <div className={styles.breakdown}>
        {answers.map((a, i) => (
          <details key={i} className={styles.breakItem}>
            <summary>
              <span
                className={
                  a.result.is_correct ? styles.dotOk : styles.dotBad
                }
              />
              Q{i + 1}: {a.question.slice(0, 80)}
              {a.question.length > 80 ? "…" : ""}
            </summary>
            <div className={styles.breakBody}>
              <div>
                <strong>Your answer:</strong> {a.studentAnswer || "—"}
              </div>
              {!a.result.is_correct && (
                <div>
                  <strong>Correct:</strong> {a.correctAnswer}
                </div>
              )}
              {a.result.feedback && <div>{a.result.feedback}</div>}
            </div>
          </details>
        ))}
      </div>

      {wrong.length > 0 && (
        <div className={styles.review}>
          You missed {wrong.length} question{wrong.length > 1 ? "s" : ""}. Switch
          to the <strong>Study Agent</strong> to dig into these topics.
        </div>
      )}

      <div className={styles.resultActions}>
        <button type="button" className={styles.secondary} onClick={onRetake}>
          Retake same quiz
        </button>
        <button type="button" className={styles.primary} onClick={onNew}>
          New quiz
        </button>
      </div>
    </div>
  );
}
