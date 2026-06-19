// Pure quiz state machine: setup -> question -> feedback -> results.
// Kept side-effect free so it can be unit-tested without React.

import type { QuizQuestion, ValidationResult } from "../../api/types";

export type Phase = "setup" | "question" | "feedback" | "results";

export interface AnswerRecord {
  questionIdx: number;
  question: string;
  questionType: string;
  studentAnswer: string;
  correctAnswer: string;
  result: ValidationResult;
}

export interface QuizConfig {
  topic: string;
  questionType: string;
  difficulty: string;
  examType: string;
}

export interface QuizState {
  phase: Phase;
  questions: QuizQuestion[];
  idx: number;
  answers: AnswerRecord[];
  lastResult: ValidationResult | null;
  config: QuizConfig | null;
}

export type QuizAction =
  | { type: "START"; questions: QuizQuestion[]; config: QuizConfig }
  | { type: "SUBMIT"; record: AnswerRecord }
  | { type: "NEXT" }
  | { type: "RETAKE" }
  | { type: "RESET" };

export const initialState: QuizState = {
  phase: "setup",
  questions: [],
  idx: 0,
  answers: [],
  lastResult: null,
  config: null,
};

export function quizReducer(state: QuizState, action: QuizAction): QuizState {
  switch (action.type) {
    case "START":
      return {
        phase: "question",
        questions: action.questions,
        idx: 0,
        answers: [],
        lastResult: null,
        config: action.config,
      };

    case "SUBMIT":
      return {
        ...state,
        phase: "feedback",
        answers: [...state.answers, action.record],
        lastResult: action.record.result,
      };

    case "NEXT": {
      const nextIdx = state.idx + 1;
      const done = nextIdx >= state.questions.length;
      return {
        ...state,
        phase: done ? "results" : "question",
        idx: nextIdx,
        lastResult: null,
      };
    }

    case "RETAKE":
      return {
        ...state,
        phase: "question",
        idx: 0,
        answers: [],
        lastResult: null,
      };

    case "RESET":
      return initialState;

    default:
      return state;
  }
}

export function scoreSummary(answers: AnswerRecord[]) {
  const total = answers.length;
  const correct = answers.filter((a) => a.result.is_correct).length;
  const pct = total ? Math.round((correct / total) * 100) : 0;
  return { total, correct, pct };
}
