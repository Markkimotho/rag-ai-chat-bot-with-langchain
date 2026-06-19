import { useState } from "react";
import type { QuizQuestion } from "../../api/types";
import styles from "./QuizPrep.module.css";

interface Props {
  question: QuizQuestion;
  idx: number;
  total: number;
  correctSoFar: number;
  submitting: boolean;
  onSubmit: (studentAnswer: string) => void;
}

export function QuestionCard({
  question,
  idx,
  total,
  correctSoFar,
  submitting,
  onSubmit,
}: Props) {
  const [choice, setChoice] = useState("");
  const [text, setText] = useState("");
  const type = question.type;

  const canSubmit =
    type === "short_answer" ? text.trim().length > 0 : choice.length > 0;

  const submit = () => {
    if (!canSubmit || submitting) return;
    onSubmit(type === "short_answer" ? text.trim() : choice);
  };

  return (
    <div className={styles.quizCard}>
      <div className={styles.progressTrack}>
        <div
          className={styles.progressFill}
          style={{ width: `${(idx / total) * 100}%` }}
        />
      </div>
      <div className={styles.metaRow}>
        <span>
          Question {idx + 1} of {total}
        </span>
        <span>
          Score: {correctSoFar}/{idx}
        </span>
      </div>

      <div className={styles.questionText}>{question.question}</div>

      {type === "mcq" && question.options && (
        <div className={styles.options} role="radiogroup" aria-label="Options">
          {Object.entries(question.options)
            .sort(([a], [b]) => a.localeCompare(b))
            .map(([key, val]) => (
              <label
                key={key}
                className={`${styles.option} ${choice === key ? styles.optionSel : ""}`}
              >
                <input
                  type="radio"
                  name={`q-${idx}`}
                  value={key}
                  checked={choice === key}
                  onChange={() => setChoice(key)}
                />
                <span className={styles.optKey}>{key}</span>
                <span>{val}</span>
              </label>
            ))}
        </div>
      )}

      {type === "true_false" && (
        <div className={styles.options} role="radiogroup" aria-label="True or false">
          {["TRUE", "FALSE"].map((v) => (
            <label
              key={v}
              className={`${styles.option} ${choice === v ? styles.optionSel : ""}`}
            >
              <input
                type="radio"
                name={`q-${idx}`}
                value={v}
                checked={choice === v}
                onChange={() => setChoice(v)}
              />
              <span>{v === "TRUE" ? "True" : "False"}</span>
            </label>
          ))}
        </div>
      )}

      {type === "short_answer" && (
        <textarea
          className={styles.answerArea}
          placeholder="Write your answer (2–4 sentences)…"
          value={text}
          aria-label="Your answer"
          onChange={(e) => setText(e.target.value)}
        />
      )}

      {question.source && (
        <div className={styles.source}>Source: {question.source}</div>
      )}

      <button
        type="button"
        className={styles.primary}
        disabled={!canSubmit || submitting}
        onClick={submit}
      >
        {submitting ? "Checking…" : "Submit answer"}
      </button>
    </div>
  );
}
