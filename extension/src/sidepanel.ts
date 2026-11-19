type PanelState = {
  lastSummary?: Record<string, unknown>;
  lastError?: string | null;
  lastResultCount?: number;
  lastAnalyzedAt?: string;
};

const HINT = `Nothing analyzed for this browser session yet.

1. Keep the local API running (python -m api.main on port 8000).
2. On the YouTube video tab, scroll until comments load.
3. Click the purple **Analyze comments** button (bottom-right of the page).

Opening the toolbar icon only shows this panel — it does not scrape comments.`;

const SUMMARY_KEYS = [
  "total",
  "needs_human",
  "excluded_owner_comments",
  "elapsed_ms",
  "creator_bucket_llm_requests",
  "moderation_llm_requests",
] as const;

function render(state: PanelState): void {
  const root = document.getElementById("root");
  if (!root) return;

  if (state.lastError) {
    root.innerHTML = "";
    const err = document.createElement("p");
    err.className = "error";
    err.textContent = state.lastError;
    root.appendChild(err);
    const hint = document.createElement("p");
    hint.className = "hint";
    hint.textContent = "Check that http://127.0.0.1:8000/health works in your browser.";
    root.appendChild(hint);
    return;
  }

  const summary = state.lastSummary;
  if (!summary || Object.keys(summary).length === 0) {
    root.innerHTML = "";
    const p = document.createElement("p");
    p.className = "hint";
    p.style.whiteSpace = "pre-line";
    p.textContent = HINT;
    root.appendChild(p);
    return;
  }

  root.innerHTML = "";
  if (state.lastAnalyzedAt) {
    const meta = document.createElement("p");
    meta.className = "meta";
    meta.textContent = `Last run: ${new Date(state.lastAnalyzedAt).toLocaleString()} · ${
      state.lastResultCount ?? summary.total ?? "?"
    } comments scored`;
    root.appendChild(meta);
  }

  const seen = new Set<string>();
  for (const key of SUMMARY_KEYS) {
    if (key in summary) {
      appendRow(root, key, summary[key]);
      seen.add(key);
    }
  }

  const layers = summary.layer_counts;
  if (layers && typeof layers === "object") {
    for (const [k, v] of Object.entries(layers as Record<string, number>)) {
      appendRow(root, `layer: ${k}`, v);
    }
    seen.add("layer_counts");
  }

  const buckets = summary.creator_bucket_counts;
  if (buckets && typeof buckets === "object") {
    for (const [k, v] of Object.entries(buckets as Record<string, number>)) {
      if ((v as number) > 0) appendRow(root, `bucket: ${k}`, v);
    }
    seen.add("creator_bucket_counts");
  }

  for (const [k, v] of Object.entries(summary)) {
    if (seen.has(k) || k.startsWith("latency_") || k === "efficiency") continue;
    if (typeof v === "object") continue;
    appendRow(root, k, v);
  }
}

function appendRow(root: HTMLElement, label: string, value: unknown): void {
  const row = document.createElement("div");
  row.className = "row";
  const left = document.createElement("span");
  left.textContent = label.replace(/_/g, " ");
  const right = document.createElement("strong");
  right.textContent = typeof value === "object" ? JSON.stringify(value) : String(value);
  row.append(left, right);
  root.appendChild(row);
}

function load(): void {
  chrome.storage.session.get(
    ["lastSummary", "lastError", "lastResultCount", "lastAnalyzedAt"],
    (data) => {
      render({
        lastSummary: data.lastSummary as Record<string, unknown> | undefined,
        lastError: (data.lastError as string | null) ?? null,
        lastResultCount: data.lastResultCount as number | undefined,
        lastAnalyzedAt: data.lastAnalyzedAt as string | undefined,
      });
    }
  );
}

chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== "session") return;
  chrome.storage.session.get(
    ["lastSummary", "lastError", "lastResultCount", "lastAnalyzedAt"],
    (data) => {
      render({
        lastSummary: data.lastSummary as Record<string, unknown> | undefined,
        lastError: (data.lastError as string | null) ?? null,
        lastResultCount: data.lastResultCount as number | undefined,
        lastAnalyzedAt: data.lastAnalyzedAt as string | undefined,
      });
    }
  );
  void changes;
});

load();