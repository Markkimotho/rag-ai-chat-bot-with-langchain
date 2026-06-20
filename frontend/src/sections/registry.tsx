import type { ReactNode } from "react";

export type SectionId = "quiz" | "agent" | "code" | "chat" | "cards";

export interface SectionDef {
  id: SectionId;
  path: string;
  label: string;
  blurb: string;
  icon: ReactNode;
}

// Inline SVGs keep the bundle dependency-free and recolor via currentColor.
const QuizIcon = (
  <svg viewBox="0 0 24 24" width="20" height="20" fill="none" aria-hidden>
    <path
      d="M9 11l2 2 4-4m-9 9h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

const AgentIcon = (
  <svg viewBox="0 0 24 24" width="20" height="20" fill="none" aria-hidden>
    <path
      d="M12 3a4 4 0 014 4v1h1a3 3 0 013 3v2a3 3 0 01-3 3h-1l-4 4-4-4H7a3 3 0 01-3-3v-2a3 3 0 013-3h1V7a4 4 0 014-4z"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinejoin="round"
    />
    <circle cx="9.5" cy="11.5" r="1" fill="currentColor" />
    <circle cx="14.5" cy="11.5" r="1" fill="currentColor" />
  </svg>
);

const CodeIcon = (
  <svg viewBox="0 0 24 24" width="20" height="20" fill="none" aria-hidden>
    <path
      d="M8 9l-3 3 3 3m8-6l3 3-3 3m-2-9l-4 12"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

const ChatIcon = (
  <svg viewBox="0 0 24 24" width="20" height="20" fill="none" aria-hidden>
    <path
      d="M21 12a8 8 0 01-11.6 7.1L4 20l1-5A8 8 0 1121 12z"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinejoin="round"
    />
  </svg>
);

const CardsIcon = (
  <svg viewBox="0 0 24 24" width="20" height="20" fill="none" aria-hidden>
    <rect x="3" y="6" width="13" height="14" rx="2" stroke="currentColor" strokeWidth="1.8" />
    <path d="M8 3h11a2 2 0 012 2v11" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" />
  </svg>
);

export const SECTIONS: SectionDef[] = [
  {
    id: "quiz",
    path: "/quiz",
    label: "Quiz Prep",
    blurb: "Generate and take quizzes from your material.",
    icon: QuizIcon,
  },
  {
    id: "agent",
    path: "/agent",
    label: "Study Agent",
    blurb: "An AI tutor that can search, quiz, and explain.",
    icon: AgentIcon,
  },
  {
    id: "code",
    path: "/code",
    label: "Programming Assistant",
    blurb: "A coding companion for any language or stack.",
    icon: CodeIcon,
  },
  {
    id: "chat",
    path: "/chat",
    label: "Regular Chat",
    blurb: "Ask questions grounded in your documents.",
    icon: ChatIcon,
  },
  {
    id: "cards",
    path: "/flashcards",
    label: "Flashcards",
    blurb: "Create, generate, and study flashcards.",
    icon: CardsIcon,
  },
];

export function sectionForPath(pathname: string): SectionDef {
  return SECTIONS.find((s) => pathname.startsWith(s.path)) ?? SECTIONS[0];
}
