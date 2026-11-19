"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useTheme } from "./ThemeProvider";

const links = [
  { href: "/", label: "Analyze" },
  { href: "/dashboard", label: "Dashboard" },
  { href: "/summary", label: "Summary" },
  { href: "/evaluation", label: "Evaluation" },
];

export function SiteHeader() {
  const pathname = usePathname();
  const { theme, toggle } = useTheme();

  return (
    <header
      className="sticky top-0 z-30 border-b border-border backdrop-blur-xl"
      style={{ background: "var(--header)" }}
    >
      <div className="mx-auto flex max-w-[1440px] items-center gap-4 px-4 py-3.5 sm:gap-6 sm:px-6">
        <Link href="/" className="group flex items-center gap-2.5">
          <span
            className="flex h-9 w-9 items-center justify-center rounded-xl text-sm font-bold text-white shadow-lg"
            style={{
              background: "linear-gradient(135deg, var(--accent), var(--accent-2))",
              boxShadow: "0 8px 20px -6px var(--glow)",
            }}
          >
            CM
          </span>
          <span className="text-sm font-bold tracking-tight text-fg">
            Comment<span className="text-accent">Moderator</span>
          </span>
        </Link>
        <nav className="hidden items-center gap-1 sm:flex">
          {links.map((l) => {
            const active = pathname === l.href;
            return (
              <Link
                key={l.href}
                href={l.href}
                className={`rounded-lg px-3 py-1.5 text-sm font-medium transition ${
                  active
                    ? "bg-accent-soft text-accent shadow-sm"
                    : "text-muted hover:bg-surface-2 hover:text-fg"
                }`}
              >
                {l.label}
              </Link>
            );
          })}
        </nav>
        <div className="ml-auto flex items-center gap-2">
          <nav className="flex gap-0.5 sm:hidden">
            {links.slice(0, 2).map((l) => (
              <Link
                key={l.href}
                href={l.href}
                className={`rounded-lg px-2 py-1 text-xs font-medium ${
                  pathname === l.href ? "text-accent" : "text-muted"
                }`}
              >
                {l.label}
              </Link>
            ))}
          </nav>
          <button
            type="button"
            onClick={toggle}
            className="btn-ghost !text-sm"
            aria-label="Toggle theme"
          >
            {theme === "dark" ? "☀ Light" : "☾ Dark"}
          </button>
        </div>
      </div>
    </header>
  );
}