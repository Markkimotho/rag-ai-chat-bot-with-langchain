import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import { api } from "../api/client";

interface ModelState {
  supported: string[];
  installed: string[];
  selected: string;
  online: boolean;
  modelCount: number;
  loading: boolean;
  setSelected: (m: string) => void;
  refresh: () => Promise<void>;
}

const ModelContext = createContext<ModelState | null>(null);

const STORAGE_KEY = "exam-prep.selected-model";

export function ModelProvider({ children }: { children: ReactNode }) {
  const [supported, setSupported] = useState<string[]>([]);
  const [installed, setInstalled] = useState<string[]>([]);
  const [selected, setSelectedState] = useState<string>(
    () => localStorage.getItem(STORAGE_KEY) ?? "",
  );
  const [online, setOnline] = useState(false);
  const [modelCount, setModelCount] = useState(0);
  const [loading, setLoading] = useState(true);

  const setSelected = useCallback((m: string) => {
    setSelectedState(m);
    localStorage.setItem(STORAGE_KEY, m);
  }, []);

  const refresh = useCallback(async () => {
    try {
      const [models, health] = await Promise.all([api.models(), api.health()]);
      setSupported(models.supported);
      setInstalled(models.installed);
      setOnline(health.online);
      setModelCount(health.model_count);
      setSelectedState((cur) => {
        if (cur && models.supported.includes(cur)) return cur;
        const next = models.installed[0] ?? models.supported[0] ?? "";
        if (next) localStorage.setItem(STORAGE_KEY, next);
        return next;
      });
    } catch {
      setOnline(false);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(async () => {
      try {
        const health = await api.health();
        setOnline(health.online);
        setModelCount(health.model_count);
      } catch {
        setOnline(false);
      }
    }, 15000);
    return () => clearInterval(id);
  }, [refresh]);

  const value = useMemo(
    () => ({
      supported,
      installed,
      selected,
      online,
      modelCount,
      loading,
      setSelected,
      refresh,
    }),
    [supported, installed, selected, online, modelCount, loading, setSelected, refresh],
  );

  return <ModelContext.Provider value={value}>{children}</ModelContext.Provider>;
}

export function useModelContext(): ModelState {
  const ctx = useContext(ModelContext);
  if (!ctx) throw new Error("useModelContext must be used within ModelProvider");
  return ctx;
}
