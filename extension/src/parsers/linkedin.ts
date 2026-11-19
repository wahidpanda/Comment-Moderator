import { LINKEDIN_SELECTORS } from "../config/linkedin";

export type ScrapedComment = { id: string; text: string };

export function parseCommentsFromDocument(doc: Document): ScrapedComment[] {
  const articles = doc.querySelectorAll(LINKEDIN_SELECTORS.commentArticle);
  const out: ScrapedComment[] = [];
  articles.forEach((node, index) => {
    const textEl = node.querySelector(LINKEDIN_SELECTORS.commentText);
    const text = textEl?.textContent?.trim() ?? "";
    if (!text) return;
    out.push({ id: `li-${index}`, text });
  });
  return out;
}