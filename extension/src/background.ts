const API = "http://127.0.0.1:8000";

async function handleAnalyze(comments: { id: string; text: string }[]) {
  const res = await fetch(`${API}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ comments }),
  });
  const text = await res.text();
  if (!res.ok) {
    const err = text.slice(0, 280) || `HTTP ${res.status}`;
    await chrome.storage.session.set({
      lastSummary: undefined,
      lastError: err,
      lastAnalyzedAt: new Date().toISOString(),
    });
    return { ok: false as const, error: err };
  }
  const data = JSON.parse(text) as { summary?: Record<string, unknown>; results?: unknown[] };
  await chrome.storage.session.set({
    lastSummary: data.summary ?? {},
    lastError: null,
    lastResultCount: data.results?.length ?? 0,
    lastAnalyzedAt: new Date().toISOString(),
  });
  return { ok: true as const, data };
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "OPEN_PANEL" && sender.tab?.id) {
    void chrome.sidePanel.open({ tabId: sender.tab.id });
    sendResponse({ ok: true });
    return false;
  }
  if (msg.type === "ANALYZE") {
    handleAnalyze(msg.comments as { id: string; text: string }[])
      .then((out) => sendResponse(out))
      .catch(async (err) => {
        const error = err instanceof Error ? err.message : String(err);
        await chrome.storage.session.set({
          lastError:
            error === "Failed to fetch"
              ? "Cannot reach API at http://127.0.0.1:8000 — start: python -m api.main"
              : error,
        });
        sendResponse({ ok: false, error });
      });
    return true;
  }
  return false;
});

chrome.action.onClicked.addListener(async (tab) => {
  if (tab.id) await chrome.sidePanel.open({ tabId: tab.id });
});