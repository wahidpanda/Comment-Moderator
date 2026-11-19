/**
 * YouTube DOM selectors — TODO(verify) when YouTube changes layout.
 * Update here only; keep HTML fixtures in extension/fixtures/.
 */
export const YOUTUBE_SELECTORS = {
  /** Top-level comment thread host */
  commentThread: "ytd-comment-thread-renderer",
  /** Comment text inside a thread */
  commentText: "#content-text",
  /** Comment action row (holds buttons) */
  actionRow: "#action-buttons",
  /** Replies expander */
  moreReplies: "ytd-button-renderer#more-replies",
  /** Continuation for infinite scroll */
  continuation: "ytd-continuations-item-renderer",
} as const;

export const YOUTUBE_SCROLL = {
  throttleMs: 800,
  maxRounds: 40,
  pixelStep: 1200,
} as const;