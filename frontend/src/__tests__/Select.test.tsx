import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { Select, type SelectOption } from "../components/Select";

const OPTS: SelectOption[] = [
  { value: "a", label: "Alpha" },
  { value: "b", label: "Beta", hint: "not installed" },
  { value: "c", label: "Gamma" },
];

function Harness({ onChange }: { onChange?: (v: string) => void }) {
  const [value, setValue] = useState("a");
  return (
    <Select
      ariaLabel="Letter"
      value={value}
      options={OPTS}
      onChange={(v) => {
        setValue(v);
        onChange?.(v);
      }}
    />
  );
}

describe("Select", () => {
  it("renders the selected label on the trigger", () => {
    render(<Harness />);
    expect(screen.getByRole("button", { name: "Letter" })).toHaveTextContent("Alpha");
  });

  it("opens the listbox on click and shows options + hints", async () => {
    render(<Harness />);
    await userEvent.click(screen.getByRole("button", { name: "Letter" }));
    expect(screen.getByRole("listbox")).toBeInTheDocument();
    expect(screen.getByRole("option", { name: /Beta/ })).toHaveTextContent(
      "not installed",
    );
  });

  it("selects an option by click and updates the trigger", async () => {
    const onChange = vi.fn();
    render(<Harness onChange={onChange} />);
    await userEvent.click(screen.getByRole("button", { name: "Letter" }));
    await userEvent.click(screen.getByRole("option", { name: /Gamma/ }));
    expect(onChange).toHaveBeenCalledWith("c");
    expect(screen.getByRole("button", { name: "Letter" })).toHaveTextContent("Gamma");
  });

  it("is keyboard operable (open, arrow down, enter)", async () => {
    const onChange = vi.fn();
    render(<Harness onChange={onChange} />);
    const trigger = screen.getByRole("button", { name: "Letter" });
    trigger.focus();
    await userEvent.keyboard("{Enter}"); // open
    await userEvent.keyboard("{ArrowDown}"); // move a -> b
    await userEvent.keyboard("{Enter}"); // choose b
    expect(onChange).toHaveBeenCalledWith("b");
  });

  it("closes on Escape without changing value", async () => {
    const onChange = vi.fn();
    render(<Harness onChange={onChange} />);
    await userEvent.click(screen.getByRole("button", { name: "Letter" }));
    await userEvent.keyboard("{Escape}");
    expect(screen.queryByRole("listbox")).not.toBeInTheDocument();
    expect(onChange).not.toHaveBeenCalled();
  });

  it("marks the current option aria-selected", async () => {
    render(<Harness />);
    await userEvent.click(screen.getByRole("button", { name: "Letter" }));
    expect(screen.getByRole("option", { name: /Alpha/ })).toHaveAttribute(
      "aria-selected",
      "true",
    );
  });
});
