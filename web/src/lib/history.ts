import type { AnalyzeResponse, YouTubeVideoInfo } from "./api";

export type HistoryEntry = {
  url: string;
  videoId: string;
  title: string;
  channel: string;
  thumbnail_url: string;
  lastAnalyzedAt: string;
  video: YouTubeVideoInfo | null;
  analyze: AnalyzeResponse;
};

const HISTORY_KEY = "moderator:history";
const SESSION_KEY = "moderator:session";
const MAX_HISTORY = 40;

export type AnalyzeSession = {
  url: string;
  video: YouTubeVideoInfo | null;
  results: AnalyzeResponse["results"];
  summary: AnalyzeResponse["summary"];
};

export function loadSession(): AnalyzeSession | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = localStorage.getItem(SESSION_KEY);
    return raw ? (JSON.parse(raw) as AnalyzeSession) : null;
  } catch {
    return null;
  }
}

export function saveSession(session: AnalyzeSession): void {
  localStorage.setItem(SESSION_KEY, JSON.stringify(session));
}

export function loadHistory(): HistoryEntry[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    return raw ? (JSON.parse(raw) as HistoryEntry[]) : [];
  } catch {
    return [];
  }
}

export function upsertHistory(entry: Omit<HistoryEntry, "lastAnalyzedAt">): HistoryEntry[] {
  const full: HistoryEntry = { ...entry, lastAnalyzedAt: new Date().toISOString() };
  const prev = loadHistory().filter((h) => h.videoId !== full.videoId);
  const next = [full, ...prev].slice(0, MAX_HISTORY);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(next));
  return next;
}

export function removeFromHistory(videoId: string): HistoryEntry[] {
  const next = loadHistory().filter((h) => h.videoId !== videoId);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(next));
  return next;
}