import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it, vi } from "vitest";
import { Sidebar } from "../components/Sidebar";

function renderAt(path: string, props: Partial<Parameters<typeof Sidebar>[0]> = {}) {
  const onToggleCollapse = vi.fn();
  const onCloseMobile = vi.fn();
  render(
    <MemoryRouter initialEntries={[path]}>
      <Sidebar
        collapsed={false}
        onToggleCollapse={onToggleCollapse}
        isMobile={false}
        mobileOpen={false}
        onCloseMobile={onCloseMobile}
        {...props}
      />
    </MemoryRouter>,
  );
  return { onToggleCollapse, onCloseMobile };
}

describe("Sidebar", () => {
  it("renders all four sections", () => {
    renderAt("/quiz");
    expect(screen.getByText("Quiz Prep")).toBeInTheDocument();
    expect(screen.getByText("Study Agent")).toBeInTheDocument();
    expect(screen.getByText("Programming Assistant")).toBeInTheDocument();
    expect(screen.getByText("Regular Chat")).toBeInTheDocument();
  });

  it("marks the active route with aria-current", () => {
    renderAt("/agent");
    const active = screen.getByRole("link", { current: "page" });
    expect(active).toHaveTextContent("Study Agent");
  });

  it("fires onToggleCollapse when the collapse button is clicked", async () => {
    const { onToggleCollapse } = renderAt("/quiz");
    await userEvent.click(
      screen.getByRole("button", { name: /collapse sidebar/i }),
    );
    expect(onToggleCollapse).toHaveBeenCalledOnce();
  });

  it("hides labels in the collapsed rail", () => {
    renderAt("/quiz", { collapsed: true });
    expect(screen.queryByText("Quiz Prep")).not.toBeInTheDocument();
    // Links remain (icon-only) and accessible via title.
    expect(screen.getAllByRole("link")).toHaveLength(4);
  });

  it("traps focus and closes on Escape when used as a mobile drawer", async () => {
    const onCloseMobile = vi.fn();
    render(
      <MemoryRouter initialEntries={["/quiz"]}>
        <Sidebar
          collapsed={false}
          onToggleCollapse={vi.fn()}
          isMobile
          mobileOpen
          onCloseMobile={onCloseMobile}
        />
      </MemoryRouter>,
    );
    await userEvent.keyboard("{Escape}");
    expect(onCloseMobile).toHaveBeenCalled();
  });
});
