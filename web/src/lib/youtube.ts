const ELEVEN = /[A-Za-z0-9_-]{11}/;

/** Strip whitespace and stray quote/bracket wrappers from pasted text. */
export function sanitizeYouTubeInput(raw: string): string {
  return raw.trim().replace(/^[\s"'<>]+|[\s"'<>]+$/g, "");
}

/**
 * Extract a YouTube video id from watch/shorts/embed/live links, youtu.be, or a bare 11-char id.
 * Handles share URLs where `v=` is not the first query param (e.g. ?si=…&v=…).
 */
export function extractYouTubeVideoId(raw: string): string | null {
  const url = sanitizeYouTubeInput(raw);
  if (!url) return null;
  if (/^[A-Za-z0-9_-]{11}$/.test(url)) return url;

  const patterns = [
    /[?&]v=([A-Za-z0-9_-]{11})\b/,
    /youtube\.com\/shorts\/([A-Za-z0-9_-]{11})\b/,
    /youtube\.com\/embed\/([A-Za-z0-9_-]{11})\b/,
    /youtube\.com\/live\/([A-Za-z0-9_-]{11})\b/,
    /youtu\.be\/([A-Za-z0-9_-]{11})\b/,
    /youtube\.com\/watch\?v=([A-Za-z0-9_-]{11})\b/,
  ];
  for (const re of patterns) {
    const m = url.match(re);
    if (m?.[1]) return m[1];
  }

  const loose = url.match(ELEVEN);
  if (loose && /youtube|youtu\.be/i.test(url)) return loose[0];
  return null;
}

export function normalizeYouTubeUrl(url: string): string {
  const id = extractYouTubeVideoId(url);
  return id ? `https://www.youtube.com/watch?v=${id}` : sanitizeYouTubeInput(url);
}

export function youtubeEmbedUrl(videoId: string): string {
  return `https://www.youtube-nocookie.com/embed/${videoId}?rel=0&modestbranding=1`;
}

export function youtubeThumbnailUrl(videoId: string): string {
  return `https://i.ytimg.com/vi/${videoId}/hqdefault.jpg`;
}