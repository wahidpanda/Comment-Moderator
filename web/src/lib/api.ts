const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8000";

export const EXPECTED_API_FEATURE_VERSION = "0.2.0";

export async function readApiError(res: Response, fallback = "Request failed"): Promise<string> {
  const text = await res.text();
  try {
    const body = JSON.parse(text) as { detail?: string };
    if (body.detail === "Not Found") {
      return (
        `API returned 404 for ${res.url}. The server on port 8000 is probably an old build — ` +
        "stop it and restart from the project root (Python 3.11): python -m api.main"
      );
    }
    if (typeof body.detail === "string") return body.detail;
  } catch {
    /* plain text */
  }
  return text || `${fallback} (HTTP ${res.status})`;
}

export type ModerationResult = {
  id: string;
  text: string;
  label: string;
  creator_bucket?: string;
  scores: Record<string, number>;
  risk: number;
  decided_by: string;
  reasons: string[];
  needs_human: boolean;
  trace?: Record<string, unknown>;
};

export type EfficiencySummary = {
  total: number;
  layer_counts: Record<string, number>;
  layer_pct: Record<string, number>;
  moderation_llm_requests: number;
  creator_bucket_llm_requests: number;
  llm_requests_analyze: number;
  llm_requests_insights?: number;
  llm_requests_ask?: number;
  llm_requests_reply?: number;
  hypothetical_llm_only_requests: number;
  hypothetical_bucket_requests?: number;
  hypothetical_llm_only_with_buckets: number;
  batch_size: number;
  bucket_batch_size?: number;
  requests_saved_vs_llm_only: number;
  transformer_ambiguous_pct?: number;
  transformer_threat_scan_pct?: number;
  openrouter_daily_used: number;
  openrouter_daily_limit: number;
};

export type AnalyzeSummary = Record<
  string,
  number | Record<string, number> | string[] | EfficiencySummary
>;

export type AnalyzeResponse = {
  results: ModerationResult[];
  summary: AnalyzeSummary;
};

export type YouTubeVideoInfo = {
  video_id: string;
  channel_id?: string | null;
  title: string;
  channel: string;
  description: string;
  summary: string;
  summary_source: "llm" | "extractive";
  thumbnail_url: string;
  published_at: string | null;
  view_count: number | null;
  comment_count: number | null;
};

export type ReplySuggestion = {
  id: string;
  suggested_reply: string;
  notes: string[];
};

export type ReplySuggestionsResponse = {
  suggestions: ReplySuggestion[];
  llm_requests: number;
  cache_hits: number;
  error?: string | null;
  budget_exhausted: boolean;
};

export type ClusteredQuestion = {
  theme: string;
  examples: string[];
  count_estimate?: number | null;
};

export type CommentInsights = {
  themes: string[];
  questions: ClusteredQuestion[];
  requests: string[];
  sentiment: { overall: string; summary: string };
  source: "llm" | "heuristic";
  llm_requests: number;
  cache_hit: boolean;
  budget_exhausted: boolean;
  error?: string | null;
};

export type CommentAskResult = {
  answer: string;
  citation_ids: string[];
  source: "llm" | "heuristic";
  llm_requests: number;
  cache_hit: boolean;
  budget_exhausted: boolean;
  error?: string | null;
};

export async function askAboutComments(
  url: string,
  question: string,
  comments: { id: string; text: string; creator_bucket?: string }[],
  opts?: { video_id?: string; title?: string; channel?: string }
): Promise<CommentAskResult> {
  const res = await fetch(`${API_BASE}/youtube/comment-ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      url,
      question,
      video_id: opts?.video_id,
      title: opts?.title,
      channel: opts?.channel,
      comments,
    }),
  });
  if (!res.ok) throw new Error(await readApiError(res, "Ask failed"));
  return res.json() as Promise<CommentAskResult>;
}

export async function fetchCommentInsights(
  url: string,
  comments: { id: string; text: string; creator_bucket?: string }[],
  opts?: {
    video_id?: string;
    title?: string;
    channel?: string;
    bucket_counts?: Record<string, number>;
  }
): Promise<CommentInsights> {
  const res = await fetch(`${API_BASE}/youtube/comment-insights`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      url,
      video_id: opts?.video_id,
      title: opts?.title,
      channel: opts?.channel,
      comments,
      bucket_counts: opts?.bucket_counts,
    }),
  });
  if (!res.ok) throw new Error(await readApiError(res, "Comment insights failed"));
  return res.json() as Promise<CommentInsights>;
}

export async function fetchReplySuggestions(
  url: string,
  items: { id: string; text: string; creator_bucket?: string }[],
  opts?: { channel?: string; owner_reply_samples?: string[] }
): Promise<ReplySuggestionsResponse> {
  const res = await fetch(`${API_BASE}/youtube/reply-suggestions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      url,
      channel: opts?.channel,
      owner_reply_samples: opts?.owner_reply_samples ?? [],
      items,
    }),
  });
  if (!res.ok) throw new Error(await readApiError(res, "Reply suggestions failed"));
  return res.json() as Promise<ReplySuggestionsResponse>;
}

export async function fetchYouTubeVideo(url: string): Promise<YouTubeVideoInfo> {
  const qs = new URLSearchParams({ url });
  const res = await fetch(`${API_BASE}/youtube/video?${qs}`);
  if (!res.ok) throw new Error(await readApiError(res, "Video fetch failed"));
  return res.json() as Promise<YouTubeVideoInfo>;
}

export async function analyzeYouTube(url: string, maxComments = 200): Promise<AnalyzeResponse> {
  const start = await fetch(`${API_BASE}/analyze/youtube`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url, max_comments: maxComments }),
  });
  if (!start.ok) throw new Error(await readApiError(start, "Analyze start failed"));
  const job = (await start.json()) as { id: string };
  if (!job.id) throw new Error("Analyze API returned no job id — restart the API (python -m api.main).");
  for (let i = 0; i < 120; i++) {
    await new Promise((r) => setTimeout(r, 1000));
    const poll = await fetch(`${API_BASE}/jobs/${job.id}`);
    if (!poll.ok) throw new Error(await readApiError(poll, "Job poll failed"));
    const data = await poll.json();
    if (data.status === "completed" && data.result) return data.result as AnalyzeResponse;
    if (data.status === "failed") throw new Error(data.error ?? "Job failed");
  }
  throw new Error("Timed out waiting for job");
}

export type HealthPayload = {
  status: string;
  api_feature_version?: string;
  routes?: string[];
  openrouter_budget?: { daily_used: number };
};

export async function fetchHealth(): Promise<HealthPayload> {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<HealthPayload>;
}

export function toCsv(results: ModerationResult[]): string {
  const header = "id,label,risk,decided_by,needs_human,text";
  const rows = results.map((r) =>
    [
      r.id,
      r.label,
      r.risk,
      r.decided_by,
      r.needs_human,
      `"${r.text.replace(/"/g, '""')}"`,
    ].join(",")
  );
  return [header, ...rows].join("\n");
}