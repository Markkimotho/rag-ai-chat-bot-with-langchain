import { useState } from "react";
import { api } from "../api/client";
import { useModelContext } from "../context/ModelContext";
import { Select, type SelectOption } from "./Select";
import styles from "./ModelPicker.module.css";

export function ModelPicker() {
  const { supported, installed, selected, setSelected, refresh } =
    useModelContext();
  const [pulling, setPulling] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  const isInstalled = installed.includes(selected);

  const options: SelectOption[] = supported.map((m) => ({
    value: m,
    label: m,
    hint: installed.includes(m) ? undefined : "not installed",
  }));

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
      <div className={styles.selectBox}>
        <Select
          ariaLabel="Model"
          value={selected}
          options={options}
          onChange={setSelected}
          placeholder="Model"
        />
      </div>
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
