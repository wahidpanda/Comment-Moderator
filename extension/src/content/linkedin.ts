import { LINKEDIN_LOAD_MORE, LINKEDIN_SELECTORS } from "../config/linkedin";
import { parseCommentsFromDocument } from "../parsers/linkedin";
import { injectBadge, mountAnalyzeButton, type ModerationResult } from "./shared";

/** Only runs when user clicks — never auto-clicks load-more. */
async function loadMoreOnce(): Promise<void> {
  const btn = document.querySelector(LINKEDIN_SELECTORS.loadMoreButton);
  if (btn instanceof HTMLButtonElement && !btn.disabled) {
    btn.click();
    await new Promise((r) => setTimeout(r, LINKEDIN_LOAD_MORE.clickDelayMs));
  }
}

async function analyzePage(): Promise<void> {
  await loadMoreOnce();
  const comments = parseCommentsFromDocument(document).map((c) => ({ id: c.id, text: c.text }));
  const response = await chrome.runtime.sendMessage({ type: "ANALYZE", comments });
  if (!response?.ok) {
    alert(`Analyze failed: ${response?.error ?? "unknown"}`);
    return;
  }
  const results = response.data.results as ModerationResult[];
  const articles = document.querySelectorAll(LINKEDIN_SELECTORS.commentArticle);
  articles.forEach((node, index) => {
    const textEl = node.querySelector(LINKEDIN_SELECTORS.commentText);
    const match = results.find((r) => r.id === `li-${index}`);
    if (textEl && match) injectBadge(textEl, match);
  });
}

mountAnalyzeButton("Analyze (LinkedIn)", () => {
  void analyzePage();
});

const loadMoreFab = document.createElement("button");
loadMoreFab.textContent = "Load more comments";
Object.assign(loadMoreFab.style, {
  position: "fixed",
  bottom: "64px",
  right: "16px",
  zIndex: "99999",
  padding: "8px 12px",
  background: "#334155",
  color: "#fff",
  border: "none",
  borderRadius: "8px",
  cursor: "pointer",
});
loadMoreFab.addEventListener("click", () => {
  void loadMoreOnce();
});
document.body.appendChild(loadMoreFab);