import type { ReactNode } from "react";
import styles from "./EmptyState.module.css";

interface Props {
  icon?: ReactNode;
  title: string;
  description?: string;
  children?: ReactNode;
}

export function EmptyState({ icon, title, description, children }: Props) {
  return (
    <div className={styles.wrap}>
      {icon && <div className={styles.icon}>{icon}</div>}
      <div className={styles.title}>{title}</div>
      {description && <div className={styles.desc}>{description}</div>}
      {children && <div className={styles.actions}>{children}</div>}
    </div>
  );
}
