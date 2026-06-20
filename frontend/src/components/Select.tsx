import { useEffect, useId, useRef, useState } from "react";
import styles from "./Select.module.css";

export interface SelectOption {
  value: string;
  label: string;
  hint?: string; // small trailing note, e.g. "not installed"
}

interface Props {
  value: string;
  options: SelectOption[];
  onChange: (value: string) => void;
  label?: string;
  ariaLabel?: string;
  placeholder?: string;
  disabled?: boolean;
}

/**
 * Accessible custom dropdown styled to the app theme (replaces native <select>).
 * Implements the listbox pattern: button toggles a popup list; arrow keys move,
 * Enter/Space select, Escape closes, click-outside dismisses.
 */
export function Select({
  value,
  options,
  onChange,
  label,
  ariaLabel,
  placeholder = "Select…",
  disabled,
}: Props) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const rootRef = useRef<HTMLDivElement>(null);
  const listId = useId();

  const selected = options.find((o) => o.value === value);

  useEffect(() => {
    if (!open) return;
    const onDocClick = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [open]);

  useEffect(() => {
    if (open) {
      const idx = options.findIndex((o) => o.value === value);
      setActive(idx >= 0 ? idx : 0);
    }
  }, [open, options, value]);

  const choose = (idx: number) => {
    const opt = options[idx];
    if (opt) {
      onChange(opt.value);
      setOpen(false);
    }
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (disabled) return;
    if (!open) {
      if (e.key === "Enter" || e.key === " " || e.key === "ArrowDown") {
        e.preventDefault();
        setOpen(true);
      }
      return;
    }
    switch (e.key) {
      case "ArrowDown":
        e.preventDefault();
        setActive((a) => Math.min(a + 1, options.length - 1));
        break;
      case "ArrowUp":
        e.preventDefault();
        setActive((a) => Math.max(a - 1, 0));
        break;
      case "Enter":
      case " ":
        e.preventDefault();
        choose(active);
        break;
      case "Escape":
        e.preventDefault();
        setOpen(false);
        break;
      case "Home":
        e.preventDefault();
        setActive(0);
        break;
      case "End":
        e.preventDefault();
        setActive(options.length - 1);
        break;
    }
  };

  return (
    <div className={styles.wrap} ref={rootRef}>
      {label && <span className={styles.label}>{label}</span>}
      <button
        type="button"
        className={styles.trigger}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={ariaLabel ?? label}
        disabled={disabled}
        onClick={() => !disabled && setOpen((o) => !o)}
        onKeyDown={onKeyDown}
      >
        <span className={styles.value} data-placeholder={!selected}>
          {selected ? selected.label : placeholder}
        </span>
        <span className={styles.chevron} data-open={open} aria-hidden>
          ⌄
        </span>
      </button>

      {open && (
        <ul className={styles.list} role="listbox" id={listId} aria-label={ariaLabel ?? label}>
          {options.map((opt, i) => (
            <li
              key={opt.value}
              role="option"
              aria-selected={opt.value === value}
              className={`${styles.option} ${i === active ? styles.active : ""} ${
                opt.value === value ? styles.selected : ""
              }`}
              onMouseEnter={() => setActive(i)}
              onMouseDown={(e) => {
                e.preventDefault();
                choose(i);
              }}
            >
              <span className={styles.optLabel}>{opt.label}</span>
              {opt.hint && <span className={styles.optHint}>{opt.hint}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
