import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Toggle } from "../components/Toggle";

describe("Toggle", () => {
  it("renders as a switch reflecting checked state", () => {
    render(<Toggle checked onChange={vi.fn()} label="Use my documents" />);
    const sw = screen.getByRole("switch", { name: "Use my documents" });
    expect(sw).toHaveAttribute("aria-checked", "true");
  });

  it("calls onChange with the toggled value", async () => {
    const onChange = vi.fn();
    render(<Toggle checked={false} onChange={onChange} label="KB" />);
    await userEvent.click(screen.getByRole("switch", { name: "KB" }));
    expect(onChange).toHaveBeenCalledWith(true);
  });

  it("does not fire when disabled", async () => {
    const onChange = vi.fn();
    render(<Toggle checked={false} onChange={onChange} label="KB" disabled />);
    await userEvent.click(screen.getByRole("switch", { name: "KB" }));
    expect(onChange).not.toHaveBeenCalled();
  });
});
