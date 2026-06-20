import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { StudyMode } from "../sections/Flashcards/StudyMode";
import type { Flashcard } from "../api/types";

const cards: Flashcard[] = [
  { id: "1", front: "Q1", back: "A1", created_at: 0 },
  { id: "2", front: "Q2", back: "A2", created_at: 0 },
];

describe("StudyMode", () => {
  it("flips the card to reveal the answer (Space)", async () => {
    render(<StudyMode deckName="Deck" cards={cards} onExit={vi.fn()} />);
    // Both faces are in the DOM (3D flip); assert the flip-state attribute toggles.
    const flip = screen.getByRole("button", { name: /show answer/i });
    expect(flip).toHaveAttribute("data-flipped", "false");
    await userEvent.keyboard(" ");
    expect(
      screen.getByRole("button", { name: /show question/i }),
    ).toHaveAttribute("data-flipped", "true");
  });

  it("advances through cards and shows a completion score", async () => {
    render(<StudyMode deckName="Deck" cards={cards} onExit={vi.fn()} />);
    expect(screen.getByText(/1 \/ 2/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Got it" }));
    expect(screen.getByText(/2 \/ 2/)).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Got it" }));
    // Both marked "Got it" → 100%
    expect(screen.getByText("100%")).toBeInTheDocument();
    expect(screen.getByText(/knew 2 of 2/i)).toBeInTheDocument();
  });

  it("counts only known cards toward the score", async () => {
    render(<StudyMode deckName="Deck" cards={cards} onExit={vi.fn()} />);
    await userEvent.click(screen.getByRole("button", { name: "Again" }));
    await userEvent.click(screen.getByRole("button", { name: "Got it" }));
    expect(screen.getByText("50%")).toBeInTheDocument();
  });

  it("exits via the back button", async () => {
    const onExit = vi.fn();
    render(<StudyMode deckName="Deck" cards={cards} onExit={onExit} />);
    await userEvent.click(screen.getByRole("button", { name: /exit/i }));
    expect(onExit).toHaveBeenCalledOnce();
  });
});
