import styles from "./Spinner.module.css";

export function Spinner({ label }: { label?: string }) {
  return (
    <span className={styles.wrap} role="status" aria-live="polite">
      <span className={styles.dot} />
      <span className={styles.dot} />
      <span className={styles.dot} />
      {label && <span className={styles.label}>{label}</span>}
    </span>
  );
}
