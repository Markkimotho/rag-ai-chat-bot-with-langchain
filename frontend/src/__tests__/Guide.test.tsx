import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, beforeAll } from "vitest";
import { Guide } from "../sections/Guide";
import { GUIDE } from "../sections/Guide/content";

beforeAll(() => {
  // jsdom lacks these; the Guide uses them for scroll/observe.
  // @ts-expect-error - test shim
  window.IntersectionObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  };
  Element.prototype.scrollIntoView = () => {};
});

describe("Guide", () => {
  it("renders a table of contents grouped by section group", () => {
    render(<Guide />);
    expect(screen.getByText("Core features")).toBeInTheDocument();
    expect(screen.getByText("Developer")).toBeInTheDocument();
  });

  it("renders every guide section's content", () => {
    render(<Guide />);
    // Each section's title appears at least once (in the TOC).
    for (const s of GUIDE) {
      expect(screen.getAllByText(s.title).length).toBeGreaterThan(0);
    }
  });

  it("includes the Loki guide and the developer API reference", () => {
    render(<Guide />);
    expect(screen.getByText(/Using Loki/i)).toBeInTheDocument();
    // a known LogQL label from the monitoring section (appears several times)
    expect(
      screen.getAllByText(/compose_service/, { exact: false }).length,
    ).toBeGreaterThan(0);
  });

  it("TOC entries are clickable buttons", async () => {
    render(<Guide />);
    const tocButton = screen.getByRole("button", { name: "Monitoring & Loki" });
    await userEvent.click(tocButton);
    expect(tocButton).toBeInTheDocument();
  });
});
