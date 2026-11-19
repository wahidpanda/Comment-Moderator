"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useTheme } from "./ThemeProvider";

const links = [
  { href: "/", label: "Analyze" },
  { href: "/dashboard", label: "Dashboard" },
];

export function SiteHeader() {
  const pathname = usePathname();
  const { theme, toggle } = useTheme();

  return (
    <header
      className="sticky top-0 z-30 border-b border-border backdrop-blur-md"
      style={{ background: "var(--header)" }}
    >
      <div className="mx-auto flex max-w-[1440px] items-center gap-6 px-4 py-3 sm:px-6">
        <Link href="/" className="text-sm font-bold tracking-tight text-fg">
          Comment<span className="text-accent">Moderator</span>
        </Link>
        <nav className="flex gap-1">
          {links.map((l) => {
            const active = pathname === l.href;
            return (
              <Link
                key={l.href}
                href={l.href}
                className={`rounded-lg px-3 py-1.5 text-sm font-medium transition ${
                  active ? "bg-accent-soft text-accent" : "text-muted hover:text-fg"
                }`}
              >
                {l.label}
              </Link>
            );
          })}
        </nav>
        <button
          type="button"
          onClick={toggle}
          className="btn-ghost ml-auto !text-sm"
          aria-label="Toggle theme"
        >
          {theme === "dark" ? "Light" : "Dark"}
        </button>
      </div>
    </header>
  );
}