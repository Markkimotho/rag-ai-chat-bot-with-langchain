import { useEffect, useMemo, useRef, useState } from "react";
import { Markdown } from "../../components/Markdown";
import sectionStyles from "../Section.module.css";
import { GUIDE, type GuideSection } from "./content";
import styles from "./Guide.module.css";

export function Guide() {
  const [active, setActive] = useState(GUIDE[0].id);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Group sections for the table of contents.
  const groups = useMemo(() => {
    const map = new Map<string, GuideSection[]>();
    for (const s of GUIDE) {
      const arr = map.get(s.group) ?? [];
      arr.push(s);
      map.set(s.group, arr);
    }
    return [...map.entries()];
  }, []);

  // Highlight the section currently in view.
  useEffect(() => {
    const root = scrollRef.current;
    if (!root) return;
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible[0]) setActive(visible[0].target.id);
      },
      { root, rootMargin: "0px 0px -70% 0px", threshold: 0 },
    );
    root.querySelectorAll("[data-guide-section]").forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, []);

  const jump = (id: string) => {
    const el = scrollRef.current?.querySelector(`#${CSS.escape(id)}`);
    el?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  return (
    <div className={sectionStyles.section}>
      <div className={styles.layout}>
        <nav className={styles.toc} aria-label="Guide contents">
          {groups.map(([group, sections]) => (
            <div key={group} className={styles.tocGroup}>
              <div className={styles.tocGroupTitle}>{group}</div>
              {sections.map((s) => (
                <button
                  key={s.id}
                  type="button"
                  className={`${styles.tocLink} ${active === s.id ? styles.tocActive : ""}`}
                  aria-current={active === s.id ? "true" : undefined}
                  onClick={() => jump(s.id)}
                >
                  {s.title}
                </button>
              ))}
            </div>
          ))}
        </nav>

        <div className={styles.content} ref={scrollRef}>
          {GUIDE.map((s) => (
            <section
              key={s.id}
              id={s.id}
              data-guide-section
              className={styles.article}
            >
              <Markdown>{s.body}</Markdown>
            </section>
          ))}
          <div className={styles.spacer} />
        </div>
      </div>
    </div>
  );
}
