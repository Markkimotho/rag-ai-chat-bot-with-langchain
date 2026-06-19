import { useState } from "react";
import type { Difficulty, QuestionType } from "../../api/types";
import { useKnowledgeBase } from "../../context/KnowledgeBaseContext";
import { useModelContext } from "../../context/ModelContext";
import { EmptyState } from "../../components/EmptyState";
import styles from "./QuizPrep.module.css";

export interface SetupValues {
  topic: string;
  questionType: QuestionType;
  n: number;
  difficulty: Difficulty;
  examType: string;
}

interface Props {
  generating: boolean;
  error: string | null;
  onGenerate: (v: SetupValues) => void;
}

const TYPES: { value: QuestionType; label: string }[] = [
  { value: "mcq", label: "Multiple Choice" },
  { value: "true_false", label: "True / False" },
  { value: "short_answer", label: "Short Answer" },
  { value: "mixed", label: "Mixed" },
];
const DIFFICULTIES: Difficulty[] = ["easy", "medium", "hard", "mixed"];
const EXAM_TYPES = [
  "general exam",
  "technical interview",
  "job interview",
  "certification",
  "university exam",
];

export function SetupPanel({ generating, error, onGenerate }: Props) {
  const { count } = useKnowledgeBase();
  const { selected } = useModelContext();
  const [topic, setTopic] = useState("");
  const [questionType, setQuestionType] = useState<QuestionType>("mcq");
  const [n, setN] = useState(10);
  const [difficulty, setDifficulty] = useState<Difficulty>("medium");
  const [examType, setExamType] = useState(EXAM_TYPES[0]);

  const kbEmpty = count === 0;
  const disabled = generating || kbEmpty || !topic.trim();

  if (kbEmpty) {
    return (
      <EmptyState
        title="No study material yet"
        description="Upload a PDF or scrape a topic from the top bar, then come back to generate a quiz."
      />
    );
  }

  return (
    <div className={styles.setup}>
      <div className={styles.field}>
        <label htmlFor="quiz-topic">Topic</label>
        <input
          id="quiz-topic"
          className={styles.input}
          value={topic}
          placeholder="e.g. Binary search trees"
          onChange={(e) => setTopic(e.target.value)}
        />
      </div>

      <div className={styles.grid}>
        <div className={styles.field}>
          <label htmlFor="quiz-type">Question type</label>
          <select
            id="quiz-type"
            className={styles.input}
            value={questionType}
            onChange={(e) => setQuestionType(e.target.value as QuestionType)}
          >
            {TYPES.map((t) => (
              <option key={t.value} value={t.value}>
                {t.label}
              </option>
            ))}
          </select>
        </div>

        <div className={styles.field}>
          <label htmlFor="quiz-n">Questions</label>
          <select
            id="quiz-n"
            className={styles.input}
            value={n}
            onChange={(e) => setN(Number(e.target.value))}
          >
            {[5, 10, 15, 20].map((v) => (
              <option key={v} value={v}>
                {v}
              </option>
            ))}
          </select>
        </div>

        <div className={styles.field}>
          <label htmlFor="quiz-diff">Difficulty</label>
          <select
            id="quiz-diff"
            className={styles.input}
            value={difficulty}
            onChange={(e) => setDifficulty(e.target.value as Difficulty)}
          >
            {DIFFICULTIES.map((d) => (
              <option key={d} value={d}>
                {d[0].toUpperCase() + d.slice(1)}
              </option>
            ))}
          </select>
        </div>

        <div className={styles.field}>
          <label htmlFor="quiz-exam">Exam type</label>
          <select
            id="quiz-exam"
            className={styles.input}
            value={examType}
            onChange={(e) => setExamType(e.target.value)}
          >
            {EXAM_TYPES.map((x) => (
              <option key={x} value={x}>
                {x[0].toUpperCase() + x.slice(1)}
              </option>
            ))}
          </select>
        </div>
      </div>

      {error && <div className={styles.error}>{error}</div>}

      <button
        type="button"
        className={styles.primary}
        disabled={disabled}
        onClick={() =>
          onGenerate({ topic: topic.trim(), questionType, n, difficulty, examType })
        }
      >
        {generating ? `Generating ${n} questions…` : "Generate quiz"}
      </button>
      <p className={styles.hint}>
        Generated with <code>{selected || "default model"}</code> from your
        indexed material.
      </p>
    </div>
  );
}
