import { useState } from "react";
import { api } from "../api/client";
import { useModelContext } from "../context/ModelContext";
import styles from "./ModelPicker.module.css";

export function ModelPicker() {
  const { supported, installed, selected, setSelected, refresh } =
    useModelContext();
  const [pulling, setPulling] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const isInstalled = installed.includes(selected);

  const pull = async () => {
    setPulling(true);
    setMsg(`Pulling ${selected}…`);
    try {
      const res = await api.pullModel(selected);
      setMsg(res.message);
      await refresh();
    } catch {
      setMsg("Pull failed.");
    } finally {
      setPulling(false);
    }
  };

  return (
    <div className={styles.wrap}>
      <label className={styles.label} htmlFor="model-select">
        Model
      </label>
      <select
        id="model-select"
        className={styles.select}
        value={selected}
        onChange={(e) => setSelected(e.target.value)}
      >
        {supported.map((m) => (
          <option key={m} value={m}>
            {m}
            {installed.includes(m) ? "" : "  (not installed)"}
          </option>
        ))}
      </select>
      {!isInstalled && selected && (
        <button
          type="button"
          className={styles.pull}
          onClick={pull}
          disabled={pulling}
        >
          {pulling ? "Pulling…" : "Pull"}
        </button>
      )}
      {msg && <span className={styles.msg}>{msg}</span>}
    </div>
  );
}
