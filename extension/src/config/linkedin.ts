/**
 * LinkedIn DOM selectors — TODO(verify) frequently.
 * Extension-only; read-only in logged-in session.
 */
export const LINKEDIN_SELECTORS = {
  commentArticle: "article.comments-comment-entity",
  commentText: ".comments-comment-item__main-content",
  loadMoreButton:
    "button.comments-comments-list__load-more-comments-button, button.artdeco-button--muted", // TODO(verify)
  repliesButton: "button.comments-comment-social-bar__reply-action", // TODO(verify)
} as const;

export const LINKEDIN_LOAD_MORE = {
  clickDelayMs: 1200,
  maxClicks: 20,
} as const;