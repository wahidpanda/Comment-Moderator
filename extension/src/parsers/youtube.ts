import { YOUTUBE_SELECTORS } from "../config/youtube";

export type ScrapedComment = { id: string; text: string };

export function parseCommentsFromDocument(doc: Document): ScrapedComment[] {
  const threads = doc.querySelectorAll(YOUTUBE_SELECTORS.commentThread);
  const out: ScrapedComment[] = [];
  threads.forEach((node, index) => {
    const textEl = node.querySelector(YOUTUBE_SELECTORS.commentText);
    const text = textEl?.textContent?.trim() ?? "";
    if (!text) return;
    const id = (node as HTMLElement).id || `yt-${index}`;
    out.push({ id, text });
  });
  return out;
}