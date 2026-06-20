import styles from "./Toggle.module.css";

interface Props {
  checked: boolean;
  onChange: (checked: boolean) => void;
  label: string;
  hint?: string;
  disabled?: boolean;
}

/** Accessible on/off switch styled to the active section accent. */
export function Toggle({ checked, onChange, label, hint, disabled }: Props) {
  return (
    <label className={`${styles.wrap} ${disabled ? styles.disabled : ""}`}>
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-label={label}
        disabled={disabled}
        className={styles.track}
        data-on={checked}
        onClick={() => !disabled && onChange(!checked)}
      >
        <span className={styles.thumb} />
      </button>
      <span className={styles.label}>{label}</span>
      {hint && <span className={styles.hint}>{hint}</span>}
    </label>
  );
}
