import { useEffect, useState } from "react";
import { currentTheme, type ThemeMode } from "../lib/chartTheme";

export type ThemePreference = "system" | "light" | "dark";
const KEY = "ledger.theme";

function readPref(): ThemePreference {
  try {
    const v = localStorage.getItem(KEY);
    return v === "light" || v === "dark" ? v : "system";
  } catch {
    return "system";
  }
}

export function applyThemePreference(pref: ThemePreference): void {
  const root = document.documentElement;
  if (pref === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", pref);
  try {
    if (pref === "system") localStorage.removeItem(KEY);
    else localStorage.setItem(KEY, pref);
  } catch {
    /* ignore */
  }
}

/** The user's preference plus the resolved mode, kept in sync with the OS setting. */
export function useTheme(): { pref: ThemePreference; mode: ThemeMode; setPref: (p: ThemePreference) => void } {
  const [pref, setPrefState] = useState<ThemePreference>(readPref);
  const [mode, setMode] = useState<ThemeMode>(() => {
    applyThemePreference(readPref());
    return currentTheme();
  });

  useEffect(() => {
    applyThemePreference(pref);
    setMode(currentTheme());
    const mq = window.matchMedia?.("(prefers-color-scheme: dark)");
    const onChange = () => setMode(currentTheme());
    mq?.addEventListener("change", onChange);
    return () => mq?.removeEventListener("change", onChange);
  }, [pref]);

  return { pref, mode, setPref: setPrefState };
}
