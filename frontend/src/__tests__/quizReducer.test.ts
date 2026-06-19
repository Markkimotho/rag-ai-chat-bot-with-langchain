import { describe, expect, it } from "vitest";
import {
  initialState,
  quizReducer,
  scoreSummary,
  type AnswerRecord,
  type QuizConfig,
} from "../sections/QuizPrep/quizReducer";
import type { QuizQuestion, ValidationResult } from "../api/types";

const config: QuizConfig = {
  topic: "trees",
  questionType: "mcq",
  difficulty: "medium",
  examType: "general exam",
};

const questions: QuizQuestion[] = [
  { type: "mcq", question: "Q1", options: { A: "x", B: "y" }, correct: "A" },
  { type: "mcq", question: "Q2", options: { A: "x", B: "y" }, correct: "B" },
];

function record(idx: number, correct: boolean): AnswerRecord {
  const result: ValidationResult = {
    score: correct ? 100 : 0,
    is_correct: correct,
    feedback: "",
    key_missed: [],
    hint: "",
  };
  return {
    questionIdx: idx,
    question: questions[idx].question,
    questionType: "mcq",
    studentAnswer: "A",
    correctAnswer: "A",
    result,
  };
}

describe("quizReducer", () => {
  it("START moves setup -> question and stores questions", () => {
    const s = quizReducer(initialState, { type: "START", questions, config });
    expect(s.phase).toBe("question");
    expect(s.questions).toHaveLength(2);
    expect(s.idx).toBe(0);
  });

  it("SUBMIT moves to feedback and records the answer", () => {
    let s = quizReducer(initialState, { type: "START", questions, config });
    s = quizReducer(s, { type: "SUBMIT", record: record(0, true) });
    expect(s.phase).toBe("feedback");
    expect(s.answers).toHaveLength(1);
    expect(s.lastResult?.is_correct).toBe(true);
  });

  it("NEXT advances to the next question, then to results on the last", () => {
    let s = quizReducer(initialState, { type: "START", questions, config });
    s = quizReducer(s, { type: "SUBMIT", record: record(0, true) });
    s = quizReducer(s, { type: "NEXT" });
    expect(s.phase).toBe("question");
    expect(s.idx).toBe(1);

    s = quizReducer(s, { type: "SUBMIT", record: record(1, false) });
    s = quizReducer(s, { type: "NEXT" });
    expect(s.phase).toBe("results");
  });

  it("RETAKE resets progress but keeps questions", () => {
    let s = quizReducer(initialState, { type: "START", questions, config });
    s = quizReducer(s, { type: "SUBMIT", record: record(0, true) });
    s = quizReducer(s, { type: "RETAKE" });
    expect(s.phase).toBe("question");
    expect(s.idx).toBe(0);
    expect(s.answers).toHaveLength(0);
    expect(s.questions).toHaveLength(2);
  });

  it("RESET returns to the initial setup state", () => {
    let s = quizReducer(initialState, { type: "START", questions, config });
    s = quizReducer(s, { type: "RESET" });
    expect(s).toEqual(initialState);
  });
});

describe("scoreSummary", () => {
  it("computes correct count and percentage", () => {
    const { total, correct, pct } = scoreSummary([
      record(0, true),
      record(1, false),
    ]);
    expect(total).toBe(2);
    expect(correct).toBe(1);
    expect(pct).toBe(50);
  });
});
