import styles from "./ErrorState.module.css";

interface Props {
  message: string;
  onRetry?: () => void;
}

export function ErrorState({ message, onRetry }: Props) {
  return (
    <div className={styles.wrap} role="alert">
      <span className={styles.msg}>{message}</span>
      {onRetry && (
        <button className={styles.retry} onClick={onRetry} type="button">
          Retry
        </button>
      )}
    </div>
  );
}
