import { createContext } from "react";
import type { ThemeMode } from "./lib/chartTheme";

export const ThemeContext = createContext<ThemeMode>("light");

export interface SessionCtx {
  sessionId: string | null;
  notify: (message: string, tone?: "info" | "error" | "success") => void;
}

export const SessionContext = createContext<SessionCtx>({ sessionId: null, notify: () => {} });
