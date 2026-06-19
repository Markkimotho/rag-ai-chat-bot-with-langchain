import { useEffect, useRef } from "react";
import { NavLink } from "react-router-dom";
import { SECTIONS } from "../sections/registry";
import styles from "./Sidebar.module.css";

interface Props {
  collapsed: boolean;
  onToggleCollapse: () => void;
  isMobile: boolean;
  mobileOpen: boolean;
  onCloseMobile: () => void;
}

export function Sidebar({
  collapsed,
  onToggleCollapse,
  isMobile,
  mobileOpen,
  onCloseMobile,
}: Props) {
  const navRef = useRef<HTMLElement>(null);

  // Mobile drawer: close on Esc, trap focus while open.
  useEffect(() => {
    if (!isMobile || !mobileOpen) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onCloseMobile();
      if (e.key === "Tab" && navRef.current) {
        const focusable = navRef.current.querySelectorAll<HTMLElement>(
          'a, button, [tabindex]:not([tabindex="-1"])',
        );
        if (focusable.length === 0) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (e.shiftKey && document.activeElement === first) {
          e.preventDefault();
          last.focus();
        } else if (!e.shiftKey && document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };
    document.addEventListener("keydown", onKey);
    navRef.current?.querySelector<HTMLElement>("a")?.focus();
    return () => document.removeEventListener("keydown", onKey);
  }, [isMobile, mobileOpen, onCloseMobile]);

  const isRail = !isMobile && collapsed;

  const aside = (
    <aside
      ref={navRef}
      className={`${styles.sidebar} ${isRail ? styles.rail : ""} ${
        isMobile ? styles.mobile : ""
      } ${isMobile && mobileOpen ? styles.open : ""}`}
      aria-label="Sections"
    >
      <div className={styles.brand}>
        <span className={styles.logo} aria-hidden>
          ◆
        </span>
        {!isRail && <span className={styles.brandName}>Exam Prep AI</span>}
      </div>

      <nav className={styles.nav}>
        {SECTIONS.map((s) => (
          <NavLink
            key={s.id}
            to={s.path}
            data-section={s.id}
            title={isRail ? s.label : undefined}
            className={({ isActive }) =>
              `${styles.item} ${isActive ? styles.active : ""}`
            }
            onClick={() => isMobile && onCloseMobile()}
          >
            <span className={styles.icon}>{s.icon}</span>
            {!isRail && <span className={styles.itemLabel}>{s.label}</span>}
          </NavLink>
        ))}
      </nav>

      {!isMobile && (
        <button
          type="button"
          className={styles.collapse}
          onClick={onToggleCollapse}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          aria-expanded={!collapsed}
        >
          <span className={styles.collapseIcon} data-collapsed={collapsed}>
            ‹
          </span>
          {!isRail && <span>Collapse</span>}
        </button>
      )}
    </aside>
  );

  if (isMobile) {
    return (
      <>
        {mobileOpen && (
          <div
            className={styles.backdrop}
            onClick={onCloseMobile}
            aria-hidden
          />
        )}
        {aside}
      </>
    );
  }
  return aside;
}
