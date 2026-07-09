import { createContext, useContext, useEffect, useMemo, useState } from "react";
import type { DatasetResponse, TrainResult } from "./types";

type AppState = {
  dataset: DatasetResponse | null;
  setDataset: (dataset: DatasetResponse | null) => void;
  target: string;
  setTarget: (target: string) => void;
  taskType: "classification" | "regression" | "";
  setTaskType: (taskType: "classification" | "regression" | "") => void;
  results: TrainResult[];
  setResults: (results: TrainResult[]) => void;
  darkMode: boolean;
  setDarkMode: (darkMode: boolean) => void;
};

const AppStateContext = createContext<AppState | null>(null);
const STORAGE_KEY = "datastory_ai_state";

type StoredAppState = {
  dataset: DatasetResponse | null;
  target: string;
  taskType: "classification" | "regression" | "";
  results: TrainResult[];
  darkMode: boolean;
};

function loadStoredState(): StoredAppState {
  if (typeof window === "undefined") {
    return { dataset: null, target: "", taskType: "", results: [], darkMode: false };
  }
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return { dataset: null, target: "", taskType: "", results: [], darkMode: false };
    const parsed = JSON.parse(raw) as Partial<StoredAppState>;
    return {
      dataset: parsed.dataset ?? null,
      target: parsed.target ?? "",
      taskType: parsed.taskType === "classification" || parsed.taskType === "regression" ? parsed.taskType : "",
      results: Array.isArray(parsed.results) ? parsed.results : [],
      darkMode: Boolean(parsed.darkMode),
    };
  } catch {
    return { dataset: null, target: "", taskType: "", results: [], darkMode: false };
  }
}

export function AppStateProvider({ children }: { children: React.ReactNode }) {
  const [stored] = useState(loadStoredState);
  const [dataset, setDataset] = useState<DatasetResponse | null>(stored.dataset);
  const [target, setTarget] = useState(stored.target);
  const [taskType, setTaskType] = useState<"classification" | "regression" | "">(stored.taskType);
  const [results, setResults] = useState<TrainResult[]>(stored.results);
  const [darkMode, setDarkMode] = useState(stored.darkMode);

  useEffect(() => {
    window.localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({ dataset, target, taskType, results, darkMode }),
    );
  }, [dataset, darkMode, results, target, taskType]);

  const value = useMemo(
    () => ({ dataset, setDataset, target, setTarget, taskType, setTaskType, results, setResults, darkMode, setDarkMode }),
    [dataset, target, taskType, results, darkMode],
  );
  return <AppStateContext.Provider value={value}>{children}</AppStateContext.Provider>;
}

export function useAppState() {
  const ctx = useContext(AppStateContext);
  if (!ctx) throw new Error("useAppState must be used inside AppStateProvider");
  return ctx;
}
