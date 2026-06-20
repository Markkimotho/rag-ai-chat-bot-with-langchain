import { useState } from "react";
import type { Difficulty, QuestionType } from "../../api/types";
import { useKnowledgeBase } from "../../context/KnowledgeBaseContext";
import { useModelContext } from "../../context/ModelContext";
import { EmptyState } from "../../components/EmptyState";
import { Select } from "../../components/Select";
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
        <Select
          label="Question type"
          value={questionType}
          options={TYPES.map((t) => ({ value: t.value, label: t.label }))}
          onChange={(v) => setQuestionType(v as QuestionType)}
        />
        <Select
          label="Questions"
          value={String(n)}
          options={[5, 10, 15, 20].map((v) => ({ value: String(v), label: String(v) }))}
          onChange={(v) => setN(Number(v))}
        />
        <Select
          label="Difficulty"
          value={difficulty}
          options={DIFFICULTIES.map((d) => ({
            value: d,
            label: d[0].toUpperCase() + d.slice(1),
          }))}
          onChange={(v) => setDifficulty(v as Difficulty)}
        />
        <Select
          label="Exam type"
          value={examType}
          options={EXAM_TYPES.map((x) => ({
            value: x,
            label: x[0].toUpperCase() + x.slice(1),
          }))}
          onChange={setExamType}
        />
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
