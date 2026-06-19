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

interface KBState {
  count: number;
  busy: boolean;
  refresh: () => Promise<void>;
  upload: (files: File[]) => Promise<string>;
  scrape: (topic: string, numResults?: number) => Promise<string>;
  clear: () => Promise<void>;
}

const KBContext = createContext<KBState | null>(null);

export function KnowledgeBaseProvider({ children }: { children: ReactNode }) {
  const [count, setCount] = useState(0);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const { count } = await api.kbCount();
      setCount(count);
    } catch {
      /* leave previous count on transient failure */
    }
  }, []);

  const upload = useCallback(
    async (files: File[]) => {
      setBusy(true);
      try {
        const res = await api.kbUpload(files);
        await refresh();
        return res.detail || `Added ${res.chunks_added} chunks.`;
      } finally {
        setBusy(false);
      }
    },
    [refresh],
  );

  const scrape = useCallback(
    async (topic: string, numResults = 3) => {
      setBusy(true);
      try {
        const res = await api.kbScrape(topic, numResults);
        await refresh();
        return res.detail || `Added ${res.chunks_added} chunks.`;
      } finally {
        setBusy(false);
      }
    },
    [refresh],
  );

  const clear = useCallback(async () => {
    setBusy(true);
    try {
      await api.kbClear();
      await refresh();
    } finally {
      setBusy(false);
    }
  }, [refresh]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const value = useMemo(
    () => ({ count, busy, refresh, upload, scrape, clear }),
    [count, busy, refresh, upload, scrape, clear],
  );

  return <KBContext.Provider value={value}>{children}</KBContext.Provider>;
}

export function useKnowledgeBase(): KBState {
  const ctx = useContext(KBContext);
  if (!ctx) throw new Error("useKnowledgeBase must be used within KnowledgeBaseProvider");
  return ctx;
}
