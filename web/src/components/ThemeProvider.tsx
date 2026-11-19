"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";

type Theme = "light" | "dark";

const ThemeContext = createContext<{
  theme: Theme;
  toggle: () => void;
  setTheme: (t: Theme) => void;
} | null>(null);

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setThemeState] = useState<Theme>("dark");
  useEffect(() => {
    const stored = localStorage.getItem("moderator:theme") as Theme | null;
    const initial = stored === "light" || stored === "dark" ? stored : "dark";
    setThemeState(initial);
    document.documentElement.classList.toggle("dark", initial === "dark");
    document.documentElement.classList.toggle("light", initial === "light");
  }, []);

  const setTheme = useCallback((t: Theme) => {
    setThemeState(t);
    localStorage.setItem("moderator:theme", t);
    document.documentElement.classList.toggle("dark", t === "dark");
    document.documentElement.classList.toggle("light", t === "light");
  }, []);

  const toggle = useCallback(() => setTheme(theme === "dark" ? "light" : "dark"), [setTheme, theme]);

  const value = useMemo(() => ({ theme, toggle, setTheme }), [theme, toggle, setTheme]);

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error("useTheme outside ThemeProvider");
  return ctx;
}