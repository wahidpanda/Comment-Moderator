export type ModerationResult = {
  id: string;
  risk: number;
  label: string;
  reasons: string[];
};

/** Heuristic labels with ~0% exact-match accuracy on creator eval — show only as hints. */
const EXPERIMENTAL_LABELS = new Set(["critique", "sarcasm"]);

export function formatModerationLabel(label: string): string {
  if (EXPERIMENTAL_LABELS.has(label)) return `${label} (experimental)`;
  return label;
}

export function injectBadge(host: Element, result: ModerationResult): void {
  if (host.querySelector(".cm-mod-badge")) return;
  const el = document.createElement("span");
  el.className = "cm-mod-badge";
  const color = result.risk >= 70 ? "#ef4444" : result.risk >= 40 ? "#f59e0b" : "#22c55e";
  const label = formatModerationLabel(result.label);
  el.title = result.reasons.join("; ") || label;
  el.textContent = `${label} (${result.risk})`;
  Object.assign(el.style, {
    marginLeft: "8px",
    padding: "2px 6px",
    borderRadius: "4px",
    fontSize: "11px",
    background: color,
    color: "#111",
    fontWeight: "600",
  });
  host.appendChild(el);
}

export function mountAnalyzeButton(label: string, onClick: () => void): void {
  if (document.getElementById("cm-analyze-btn")) return;
  const btn = document.createElement("button");
  btn.id = "cm-analyze-btn";
  btn.textContent = label;
  Object.assign(btn.style, {
    position: "fixed",
    bottom: "24px",
    left: "24px",
    zIndex: "2147483647",
    padding: "10px 14px",
    background: "#4f46e5",
    color: "#fff",
    border: "none",
    borderRadius: "8px",
    cursor: "pointer",
    fontWeight: "600",
  });
  btn.addEventListener("click", onClick);
  document.body.appendChild(btn);
}