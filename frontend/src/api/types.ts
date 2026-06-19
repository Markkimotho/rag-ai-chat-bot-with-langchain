// HTTP contract — keep in sync with app/api_models.py.

export interface Health {
  online: boolean;
  model_count: number;
}

export interface Models {
  supported: string[];
  installed: string[];
}

export interface PullResponse {
  ok: boolean;
  message: string;
}

export interface CountResponse {
  count: number;
}

export interface IngestResponse {
  chunks_added: number;
  detail: string;
}

export type ChatMode = "langchain" | "langgraph";

export interface Source {
  source: string;
  page: number | string;
}

export type QuestionType = "mcq" | "true_false" | "short_answer" | "mixed";
export type Difficulty = "easy" | "medium" | "hard" | "mixed";

export interface QuizQuestion {
  type: QuestionType;
  question: string;
  options?: Record<string, string>;
  correct?: string;
  sample_answer?: string;
  explanation?: string;
  source?: string;
  difficulty?: string;
}

export interface ValidationResult {
  score: number;
  is_correct: boolean;
  feedback: string;
  key_missed: string[];
  hint: string;
}

// SSE event frames emitted by streaming endpoints.
export type StreamEvent =
  | { type: "token"; text: string }
  | { type: "sources"; sources: Source[] }
  | { type: "tool"; name: string }
  | { type: "error"; message: string }
  | { type: "done" };

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  tools?: string[];
  model?: string;
  streaming?: boolean;
  error?: boolean;
}
