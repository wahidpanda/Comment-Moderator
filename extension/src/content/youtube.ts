import { YOUTUBE_SCROLL, YOUTUBE_SELECTORS } from "../config/youtube";
import { parseCommentsFromDocument } from "../parsers/youtube";
import { injectBadge, mountAnalyzeButton, type ModerationResult } from "./shared";

async function autoScrollComments(): Promise<void> {
  for (let i = 0; i < YOUTUBE_SCROLL.maxRounds; i++) {
    window.scrollBy(0, YOUTUBE_SCROLL.pixelStep);
    await new Promise((r) => setTimeout(r, YOUTUBE_SCROLL.throttleMs));
    const more = document.querySelector(YOUTUBE_SELECTORS.moreReplies);
    if (more instanceof HTMLElement) more.click();
  }
}

async function analyzePage(): Promise<void> {
  const btn = document.getElementById("cm-analyze-btn");
  if (btn instanceof HTMLButtonElement) {
    btn.disabled = true;
    btn.textContent = "Analyzing…";
  }
  try {
    await autoScrollComments();
    const comments = parseCommentsFromDocument(document).map((c) => ({
      id: c.id,
      text: c.text,
    }));
    if (comments.length === 0) {
      alert(
        "No comments found on this page.\n\nScroll down until the YouTube comments section is visible, then try again."
      );
      return;
    }
    const response = await chrome.runtime.sendMessage({ type: "ANALYZE", comments });
    if (!response?.ok) {
      alert(`Analyze failed: ${response?.error ?? "unknown"}`);
      chrome.runtime.sendMessage({ type: "OPEN_PANEL" });
      return;
    }
  const results = response.data.results as ModerationResult[];
  const threads = document.querySelectorAll(YOUTUBE_SELECTORS.commentThread);
  threads.forEach((node, index) => {
    const textEl = node.querySelector(YOUTUBE_SELECTORS.commentText);
    const match = results.find((r) => r.id === (node as HTMLElement).id || r.id === `yt-${index}`);
    if (textEl && match) injectBadge(textEl, match);
  });
    chrome.runtime.sendMessage({ type: "OPEN_PANEL" });
  } finally {
    if (btn instanceof HTMLButtonElement) {
      btn.disabled = false;
      btn.textContent = "Analyze comments";
    }
  }
}

function boot(): void {
  mountAnalyzeButton("Analyze comments", () => {
    void analyzePage();
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot, { once: true });
} else {
  boot();
}