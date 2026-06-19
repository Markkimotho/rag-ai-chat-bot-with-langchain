// Typed fetch wrapper for the JSON routes. Streaming routes live in stream.ts.

import type {
  ChatMode,
  CountResponse,
  Difficulty,
  Health,
  IngestResponse,
  Models,
  PullResponse,
  QuestionType,
  QuizQuestion,
  ValidationResult,
} from "./types";

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<Health>("/api/health"),
  models: () => request<Models>("/api/models"),
  pullModel: (model: string) =>
    request<PullResponse>("/api/models/pull", {
      method: "POST",
      body: JSON.stringify({ model }),
    }),

  kbCount: () => request<CountResponse>("/api/kb/count"),
  kbScrape: (topic: string, num_results = 3) =>
    request<IngestResponse>("/api/kb/scrape", {
      method: "POST",
      body: JSON.stringify({ topic, num_results }),
    }),
  kbClear: () =>
    request<{ cleared: boolean }>("/api/kb", { method: "DELETE" }),
  kbUpload: (files: File[]) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    // Let the browser set the multipart boundary; do not send JSON header.
    return request<IngestResponse>("/api/kb/upload", {
      method: "POST",
      headers: {},
      body: form,
    });
  },

  quizGenerate: (body: {
    topic: string;
    question_type: QuestionType;
    n: number;
    difficulty: Difficulty;
    exam_type: string;
    model?: string;
  }) =>
    request<QuizQuestion[]>("/api/quiz/generate", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  quizValidate: (body: {
    question: string;
    correct_answer: unknown;
    student_answer: string;
    question_type: "mcq" | "true_false" | "short_answer";
    model?: string;
  }) =>
    request<ValidationResult>("/api/quiz/validate", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  clearChat: (sessionId: string) =>
    request<{ cleared: boolean }>(`/api/chat/${encodeURIComponent(sessionId)}`, {
      method: "DELETE",
    }),
};

export type { ChatMode };
