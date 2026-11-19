import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class", ".dark"],
  content: ["./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        fg: "var(--fg)",
        muted: "var(--muted)",
        border: "var(--border)",
        surface: "var(--surface)",
        "surface-2": "var(--surface-2)",
        accent: "var(--accent)",
        "accent-soft": "var(--accent-soft)",
        danger: "var(--danger)",
      },
    },
  },
  plugins: [],
};

export default config;