"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { HistoryRail } from "@/components/HistoryRail";
import { VideoFrame } from "@/components/VideoFrame";
import {
  analyzeYouTube,
  askAboutComments,
  fetchCommentInsights,
  EXPECTED_API_FEATURE_VERSION,
  fetchHealth,
  fetchReplySuggestions,
  fetchYouTubeVideo,
  toCsv,
  type AnalyzeResponse,
  type CommentAskResult,
  type CommentInsights,
  type EfficiencySummary,
  type ModerationResult,
  type YouTubeVideoInfo,
} from "@/lib/api";
import {
  loadHistory,
  loadSession,
  saveSession,
  upsertHistory,
  removeFromHistory,
  type HistoryEntry,
} from "@/lib/history";
import {
  extractYouTubeVideoId,
  normalizeYouTubeUrl,
  sanitizeYouTubeInput,
  youtubeThumbnailUrl,
} from "@/lib/youtube";

const CREATOR_BUCKETS = [
  "neutral_discussion",
  "appreciation",
  "question",
  "critical",
  "request",
  "discussion",
  "sarcastic",
  "nasty",
  "spam_scam",
  "harassment",
] as const;

const LAYER_ORDER = ["rules", "spam", "heuristic", "lr", "transformer", "llm"] as const;

const LAYER_LABELS: Record<string, string> = {
  rules: "Rules",
  spam: "Spam",
  heuristic: "Heuristic",
  lr: "LR",
  transformer: "Detoxify",
  llm: "LLM",
};

const BUCKET_LABELS: Record<string, string> = {
  neutral_discussion: "Neutral / Discussion",
  appreciation: "Appreciation",
  question: "Question",
  critical: "Critical",
  request: "Request",
  sarcastic: "Sarcastic",
  discussion: "Discussion",
  nasty: "Nasty",
  spam_scam: "Spam / scam",
  harassment: "Harassment",
};

const BLOCKED_REPLY_LABELS = new Set(["toxic", "threat", "identity_attack", "spam"]);
const BLOCKED_REPLY_BUCKETS = new Set(["nasty", "spam_scam", "harassment"]);

export default function AnalyzePage() {
  const [url, setUrl] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [video, setVideo] = useState<YouTubeVideoInfo | null>(null);
  const [results, setResults] = useState<ModerationResult[]>([]);
  const [summary, setSummary] = useState<AnalyzeResponse["summary"]>({});
  const [filter, setFilter] = useState("");
  const [bucketFilter, setBucketFilter] = useState("");
  const [sort, setSort] = useState<"risk" | "label">("risk");
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [replyDrafts, setReplyDrafts] = useState<Record<string, string>>({});
  const [replyLoadingId, setReplyLoadingId] = useState<string | null>(null);
  const [insights, setInsights] = useState<CommentInsights | null>(null);
  const [insightsLoading, setInsightsLoading] = useState(false);
  const [askQuestion, setAskQuestion] = useState("");
  const [askLoading, setAskLoading] = useState(false);
  const [askHistory, setAskHistory] = useState<
    { question: string; result: CommentAskResult }[]
  >([]);
  const [focusCommentId, setFocusCommentId] = useState<string | null>(null);
  const commentRefs = useRef<Record<string, HTMLLIElement | null>>({});
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [hydrated, setHydrated] = useState(false);
  const [replyLlmExtra, setReplyLlmExtra] = useState(0);
  const [dailyLlmUsed, setDailyLlmUsed] = useState<number | null>(null);
  const [apiStale, setApiStale] = useState(false);

  const videoId = useMemo(() => extractYouTubeVideoId(url), [url]);

  useEffect(() => {
    if (!hydrated || !videoId) return;
    if (video?.video_id === videoId) return;
    setVideo({
      video_id: videoId,
      title: "Preview — run Analyze or Generate summary for details",
      channel: "",
      description: "",
      summary: "",
      summary_source: "extractive",
      thumbnail_url: youtubeThumbnailUrl(videoId),
      published_at: null,
      view_count: null,
      comment_count: null,
    });
  }, [hydrated, videoId, video?.video_id]);

  useEffect(() => {
    const session = loadSession();
    if (session?.url) {
      setUrl(session.url);
      setVideo(session.video);
      setResults(session.results);
      setSummary(session.summary);
    }
    setHistory(loadHistory());
    setHydrated(true);
    void fetchHealth()
      .then((h) => {
        setDailyLlmUsed(h.openrouter_budget?.daily_used ?? null);
        const routes = h.routes ?? [];
        const versionOk = h.api_feature_version === EXPECTED_API_FEATURE_VERSION;
        const hasInsights = routes.includes("/youtube/comment-insights");
        setApiStale(!versionOk || !hasInsights);
      })
      .catch(() => setApiStale(true));
  }, []);

  const loadVideoSummary = useCallback(async () => {
    const id = extractYouTubeVideoId(url);
    if (!id) {
      setError("Paste a valid YouTube or Shorts URL first");
      return;
    }
    setSummaryLoading(true);
    setError(null);
    try {
      const info = await fetchYouTubeVideo(normalizeYouTubeUrl(url));
      setVideo(info);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Summary failed");
    } finally {
      setSummaryLoading(false);
    }
  }, [url]);

  const persist = useCallback(
    (nextUrl: string, v: YouTubeVideoInfo | null, data: AnalyzeResponse) => {
      saveSession({
        url: nextUrl,
        video: v,
        results: data.results,
        summary: data.summary,
      });
      sessionStorage.setItem("moderator:last", JSON.stringify(data));
    },
    []
  );

  useEffect(() => {
    if (!hydrated) return;
    if (!url && results.length === 0) return;
    saveSession({ url, video, results, summary });
  }, [url, video, results, summary, hydrated]);

  const runAnalyze = useCallback(
    async (targetUrl: string) => {
      const id = extractYouTubeVideoId(targetUrl);
      if (!id) {
        setError("Paste a valid YouTube or Shorts URL");
        return;
      }
      const normalized = normalizeYouTubeUrl(targetUrl);
      setUrl(normalized);
      setLoading(true);
      setError(null);
      try {
        const data = await analyzeYouTube(normalized);
        const meta = video?.video_id === id ? video : await fetchYouTubeVideo(normalized).catch(() => null);
        setResults(data.results);
        setSummary(data.summary);
        setInsights(null);
        setAskHistory([]);
        setReplyLlmExtra(0);
        if (meta) setVideo(meta);
        void fetchHealth()
          .then((h) => {
            setDailyLlmUsed(h.openrouter_budget?.daily_used ?? null);
            const routes = h.routes ?? [];
            setApiStale(
              h.api_feature_version !== EXPECTED_API_FEATURE_VERSION ||
                !routes.includes("/youtube/comment-insights")
            );
          })
          .catch(() => setApiStale(true));
        persist(normalized, meta ?? video, data);
        setHistory(
          upsertHistory({
            url: normalized,
            videoId: meta?.video_id ?? id,
            title: meta?.title ?? video?.title ?? normalized,
            channel: meta?.channel ?? video?.channel ?? "",
            thumbnail_url: meta?.thumbnail_url ?? video?.thumbnail_url ?? `https://i.ytimg.com/vi/${id}/hqdefault.jpg`,
            video: meta ?? video,
            analyze: data,
          })
        );
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed");
      } finally {
        setLoading(false);
      }
    },
    [persist]
  );

  const filtered = useMemo(() => {
    let rows = [...results];
    if (filter) rows = rows.filter((r) => r.label === filter);
    if (bucketFilter) rows = rows.filter((r) => (r.creator_bucket ?? "") === bucketFilter);
    rows.sort((a, b) => (sort === "risk" ? b.risk - a.risk : a.label.localeCompare(b.label)));
    return rows;
  }, [results, filter, sort]);

  const bucketCounts = useMemo(() => {
    const fromSummary = summary.creator_bucket_counts as Record<string, number> | undefined;
    if (fromSummary && Object.keys(fromSummary).length > 0) return fromSummary;
    const counts: Record<string, number> = {};
    for (const r of results) {
      const b = r.creator_bucket ?? "neutral_discussion";
      counts[b] = (counts[b] ?? 0) + 1;
    }
    return counts;
  }, [results, summary]);

  const excludedOwner = Number(summary.excluded_owner_comments ?? 0);
  const ownerSamples = Array.isArray(summary.owner_reply_samples)
    ? summary.owner_reply_samples.filter((s): s is string => typeof s === "string")
    : [];
  const ownerReplyCount = Number(summary.owner_reply_count ?? ownerSamples.length);

  const needsHuman = useMemo(() => results.filter((r) => r.needs_human).length, [results]);

  const efficiency = useMemo(() => {
    const raw = summary.efficiency;
    if (raw && typeof raw === "object" && !Array.isArray(raw) && "layer_pct" in raw) {
      return raw as EfficiencySummary;
    }
    return null;
  }, [summary]);

  const sessionLlmExtra = useMemo(() => {
    let extra = replyLlmExtra;
    if (insights && !insights.cache_hit) extra += insights.llm_requests;
    for (const h of askHistory) {
      if (!h.result.cache_hit) extra += h.result.llm_requests;
    }
    return extra;
  }, [insights, askHistory, replyLlmExtra]);

  function canSuggestReply(comment: ModerationResult): boolean {
    if (BLOCKED_REPLY_LABELS.has(comment.label)) return false;
    if (comment.creator_bucket && BLOCKED_REPLY_BUCKETS.has(comment.creator_bucket)) return false;
    return true;
  }

  async function suggestRepliesBatch() {
    if (!url.trim()) return;
    const eligible = results
      .filter(canSuggestReply)
      .filter((r) => ["question", "appreciation"].includes(r.creator_bucket ?? ""))
      .slice(0, 15);
    if (eligible.length === 0) return;
    setReplyLoadingId("__batch__");
    try {
      const res = await fetchReplySuggestions(
        normalizeYouTubeUrl(url),
        eligible.map((r) => ({ id: r.id, text: r.text, creator_bucket: r.creator_bucket })),
        { channel: video?.channel, owner_reply_samples: ownerSamples }
      );
      for (const row of res.suggestions) {
        if (row.suggested_reply) {
          setReplyDrafts((d) => ({ ...d, [row.id]: row.suggested_reply }));
        }
      }
      if (res.llm_requests > 0) setReplyLlmExtra((n) => n + res.llm_requests);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Batch reply failed");
    } finally {
      setReplyLoadingId(null);
    }
  }

  async function suggestReplyFor(comment: ModerationResult) {
    if (!url.trim() || !canSuggestReply(comment)) return;
    setReplyLoadingId(comment.id);
    try {
      const res = await fetchReplySuggestions(
        normalizeYouTubeUrl(url),
        [{ id: comment.id, text: comment.text, creator_bucket: comment.creator_bucket }],
        {
          channel: video?.channel,
          owner_reply_samples: ownerSamples,
        }
      );
      const row = res.suggestions.find((s) => s.id === comment.id);
      if (row?.suggested_reply) {
        setReplyDrafts((d) => ({ ...d, [comment.id]: row.suggested_reply }));
      } else if (row?.notes?.length) {
        setReplyDrafts((d) => ({ ...d, [comment.id]: row.notes.join(" ") }));
      }
      if (res.llm_requests > 0) setReplyLlmExtra((n) => n + res.llm_requests);
      if (res.error && !row?.suggested_reply) {
        setError(res.error);
      }
      if (res.budget_exhausted) {
        setError("OpenRouter daily budget reached — try again tomorrow or raise the limit in .env.");
      }
      void fetchHealth()
        .then((h) => setDailyLlmUsed(h.openrouter_budget?.daily_used ?? null))
        .catch(() => undefined);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Reply suggestion failed";
      setError(
        msg === "Failed to fetch"
          ? "Cannot reach API at http://127.0.0.1:8000 — restart python -m api.main and refresh this page."
          : msg
      );
    } finally {
      setReplyLoadingId(null);
    }
  }

  const loadCommentInsights = useCallback(async () => {
    if (!url.trim() || results.length === 0) return;
    setInsightsLoading(true);
    setError(null);
    try {
      const data = await fetchCommentInsights(
        normalizeYouTubeUrl(url),
        results.map((r) => ({
          id: r.id,
          text: r.text,
          creator_bucket: r.creator_bucket,
        })),
        {
          video_id: video?.video_id ?? videoId ?? undefined,
          title: video?.title,
          channel: video?.channel,
          bucket_counts: bucketCounts,
        }
      );
      setInsights(data);
      if (data.budget_exhausted) {
        setError("OpenRouter daily budget reached — showing heuristic insights.");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Insights failed");
    } finally {
      setInsightsLoading(false);
    }
  }, [url, results, video, videoId, bucketCounts]);

  const submitAsk = useCallback(async () => {
    const q = askQuestion.trim();
    if (!q || !url.trim() || results.length === 0) return;
    setAskLoading(true);
    setError(null);
    try {
      const result = await askAboutComments(
        normalizeYouTubeUrl(url),
        q,
        results.map((r) => ({
          id: r.id,
          text: r.text,
          creator_bucket: r.creator_bucket,
        })),
        {
          video_id: video?.video_id ?? videoId ?? undefined,
          title: video?.title,
          channel: video?.channel,
        }
      );
      setAskHistory((h) => [{ question: q, result }, ...h].slice(0, 8));
      setAskQuestion("");
      if (result.budget_exhausted) {
        setError("OpenRouter daily budget reached — answer may use keyword fallback.");
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Ask failed");
    } finally {
      setAskLoading(false);
    }
  }, [askQuestion, url, results, video, videoId]);

  useEffect(() => {
    if (!focusCommentId) return;
    const el = commentRefs.current[focusCommentId];
    el?.scrollIntoView({ behavior: "smooth", block: "center" });
    const t = window.setTimeout(() => setFocusCommentId(null), 3000);
    return () => window.clearTimeout(t);
  }, [focusCommentId]);

  function copyReply(id: string) {
    const text = replyDrafts[id];
    if (!text) return;
    void navigator.clipboard.writeText(text);
  }

  function downloadCsv() {
    const blob = new Blob([toCsv(results)], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "moderation-results.csv";
    a.click();
  }

  return (
    <div className="space-y-6">
      <section className="panel-hero flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
        <div className="relative z-[1] min-w-0 max-w-xl">
          <p className="section-kicker">Creator triage</p>
          <h1 className="page-title mt-2">Analyze YouTube comments</h1>
          <p className="page-lead">
            Hybrid rules, LR, and Detoxify — plus optional LLM buckets, insights, and reply drafts. Nothing is posted
            or deleted automatically.
          </p>
        </div>
        <div className="relative z-[1] flex w-full flex-col gap-3 sm:max-w-lg">
          <div className="flex flex-col gap-2 sm:flex-row">
          <input
            className="input-field min-w-0 flex-1 shadow-sm"
            placeholder="youtube.com/watch?v=… or /shorts/…"
            value={url}
            onChange={(e) => setUrl(sanitizeYouTubeInput(e.target.value))}
            onPaste={(e) => {
              const pasted = e.clipboardData.getData("text");
              const id = extractYouTubeVideoId(pasted);
              if (id) {
                e.preventDefault();
                setUrl(normalizeYouTubeUrl(pasted));
                setError(null);
              }
            }}
            onKeyDown={(e) => e.key === "Enter" && void runAnalyze(url)}
          />
          <button
            type="button"
            onClick={() => void runAnalyze(url)}
            disabled={loading || !videoId}
            className="btn-primary shrink-0"
          >
            {loading ? "Analyzing…" : "Analyze video"}
          </button>
          </div>
          {results.length > 0 && (
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              <div className="stat-tile">
                <strong>{String(summary.total ?? results.length)}</strong>
                <span>Comments</span>
              </div>
              <div className="stat-tile">
                <strong>{needsHuman}</strong>
                <span>Needs review</span>
              </div>
              <div className="stat-tile">
                <strong>
                  {Number(summary.transformer_threat_scan_pct ?? 0) <= 1
                    ? Math.round(Number(summary.transformer_threat_scan_pct ?? 0) * 100)
                    : Math.round(Number(summary.transformer_threat_scan_pct ?? 0))}
                  %
                </strong>
                <span>Threat scanned</span>
              </div>
              <div className="stat-tile">
                <strong>{dailyLlmUsed ?? "—"}</strong>
                <span>LLM reqs today</span>
              </div>
            </div>
          )}
        </div>
      </section>

      {apiStale && (
        <p className="rounded-xl border border-amber-500/40 bg-amber-500/10 px-4 py-2 text-sm text-amber-900 dark:text-amber-100">
          Moderation API on port 8000 looks outdated (missing insights / ask / reply routes). Stop the old process
          and restart from <code className="text-xs">D:\Project\comment</code>:{" "}
          <code className="text-xs">D:\python311\python.exe -m api.main</code>
        </p>
      )}

      {error && (
        <p className="rounded-xl border border-danger/30 bg-danger/10 px-4 py-2 text-sm text-danger">{error}</p>
      )}

      <HistoryRail
        items={history}
        activeId={videoId}
        onSelect={(h) => {
          setUrl(h.url);
          setVideo(h.video);
          setResults(h.analyze.results);
          setSummary(h.analyze.summary);
          saveSession({
            url: h.url,
            video: h.video,
            results: h.analyze.results,
            summary: h.analyze.summary,
          });
        }}
        onReanalyze={(h) => void runAnalyze(h.url)}
        onRemove={(id) => setHistory(removeFromHistory(id))}
      />

      <div className="grid gap-6 lg:grid-cols-[minmax(300px,38%)_1fr] lg:items-start">
        <aside className="space-y-4 lg:sticky lg:top-[4.5rem]">
          {videoId ? (
            <VideoFrame videoId={videoId} title={video?.title} />
          ) : (
            <VideoPlaceholder message="Paste a YouTube or Shorts URL to preview" />
          )}

          <div className="panel space-y-3 p-4">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-muted">Video summary</h2>
              <button
                type="button"
                onClick={() => void loadVideoSummary()}
                disabled={summaryLoading || !videoId}
                className="btn-ghost !text-[11px]"
              >
                {summaryLoading ? "Summarizing…" : "Generate summary"}
              </button>
            </div>
            {video?.summary_source === "llm" && (
              <span className="inline-block rounded-full bg-accent-soft px-2 py-0.5 text-[10px] font-medium text-accent">
                AI summary
              </span>
            )}
            {video ? (
              <>
                <p className="text-base font-medium leading-snug text-fg">{video.title}</p>
                <p className="text-xs text-muted">{video.channel}</p>
                {video.view_count != null && (
                  <p className="text-xs text-muted">
                    {formatCount(video.view_count)} views
                    {video.comment_count != null ? ` · ${formatCount(video.comment_count)} on YouTube` : ""}
                  </p>
                )}
                <p className="text-sm leading-relaxed text-muted">{video.summary}</p>
              </>
            ) : !videoId ? (
              <p className="text-sm text-muted">Summary loads after a valid link.</p>
            ) : null}
          </div>

          {results.length > 0 && (
            <div className="panel space-y-3 p-4 text-sm">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-muted">Creator buckets</h2>
              <p className="text-fg">
                <span className="font-semibold">{String(summary.total ?? results.length)}</span> viewer comments
                {excludedOwner > 0 && (
                  <span className="text-muted"> · {excludedOwner} owner comments excluded</span>
                )}
                {ownerReplyCount > 0 && (
                  <span className="text-muted"> · {ownerReplyCount} owner replies for style</span>
                )}
                <button
                  type="button"
                  className="btn-ghost ml-2 !text-[11px]"
                  disabled={replyLoadingId === "__batch__" || results.length === 0}
                  onClick={() => void suggestRepliesBatch()}
                >
                  {replyLoadingId === "__batch__" ? "Drafting batch…" : "Batch reply drafts"}
                </button>
                {needsHuman > 0 && (
                  <span className="text-amber-600 dark:text-amber-400"> · {needsHuman} for review</span>
                )}
              </p>
              <ul className="flex flex-wrap gap-2">
                {CREATOR_BUCKETS.map((bucket) => {
                  const n = bucketCounts[bucket] ?? 0;
                  if (n === 0) return null;
                  return (
                    <li key={bucket}>
                      <button
                        type="button"
                        onClick={() => setBucketFilter((f) => (f === bucket ? "" : bucket))}
                        className={`rounded-full px-2.5 py-1 text-xs font-medium transition ${bucketChipClass(bucket)} ${
                          bucketFilter === bucket ? "ring-2 ring-accent ring-offset-1 ring-offset-[var(--bg)]" : ""
                        }`}
                      >
                        {BUCKET_LABELS[bucket] ?? bucket} · {n}
                      </button>
                    </li>
                  );
                })}
              </ul>
            </div>
          )}

          {results.length > 0 && efficiency && (
            <div className="panel space-y-3 p-4 text-sm">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-muted">Efficiency</h2>
              <p className="text-[11px] text-muted">
                Hybrid vs hypothetical LLM-only (batch size {efficiency.batch_size}).
              </p>
              <ul className="space-y-1.5">
                {LAYER_ORDER.filter((k) => (efficiency.layer_counts[k] ?? 0) > 0).map((k) => {
                  const n = efficiency.layer_counts[k] ?? 0;
                  const pct = efficiency.layer_pct[k] ?? 0;
                  return (
                    <li key={k} className="flex items-center gap-2 text-xs">
                      <span className="w-20 font-medium text-fg">{LAYER_LABELS[k] ?? k}</span>
                      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-2">
                        <div className="h-full rounded-full bg-accent" style={{ width: `${pct}%` }} />
                      </div>
                      <span className="w-16 text-right text-muted">
                        {pct.toFixed(0)}% ({n})
                      </span>
                    </li>
                  );
                })}
              </ul>
              <dl className="grid grid-cols-2 gap-x-2 gap-y-1 text-[11px]">
                <dt className="text-muted">Moderation LLM reqs</dt>
                <dd className="text-right font-medium text-fg">{efficiency.moderation_llm_requests}</dd>
                <dt className="text-muted">Bucket LLM reqs</dt>
                <dd className="text-right font-medium text-fg">{efficiency.creator_bucket_llm_requests}</dd>
                <dt className="text-muted">Insights / ask / reply (session)</dt>
                <dd className="text-right font-medium text-fg">{sessionLlmExtra}</dd>
                <dt className="text-muted">Hypothetical mod-only</dt>
                <dd className="text-right font-medium text-fg">
                  {efficiency.hypothetical_llm_only_requests} (ceil n / {efficiency.batch_size})
                </dd>
                <dt className="text-muted">Hypothetical + buckets</dt>
                <dd className="text-right font-medium text-fg">
                  {efficiency.hypothetical_llm_only_with_buckets}
                  {efficiency.hypothetical_bucket_requests != null
                    ? ` (buckets ${efficiency.hypothetical_bucket_requests})`
                    : ""}
                </dd>
                <dt className="text-muted">% full Detoxify (ambiguous LR)</dt>
                <dd className="text-right font-medium text-fg">
                  {efficiency.transformer_ambiguous_pct ?? Number(summary.transformer_ambiguous_pct ?? 0)}%
                </dd>
                <dt className="text-muted">% comments threat-scanned</dt>
                <dd className="text-right font-medium text-fg">
                  {efficiency.transformer_threat_scan_pct ?? Number(summary.transformer_threat_scan_pct ?? 0)}%
                </dd>
              </dl>
              <p className="text-[10px] text-muted">
                OpenRouter today: {dailyLlmUsed ?? efficiency.openrouter_daily_used} /{" "}
                {efficiency.openrouter_daily_limit} · saved ~{efficiency.requests_saved_vs_llm_only} vs LLM-only
              </p>
            </div>
          )}

          {results.length > 0 && (
            <div className="panel space-y-3 p-4 text-sm">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <h2 className="text-xs font-semibold uppercase tracking-wider text-muted">Comment insights</h2>
                <button
                  type="button"
                  onClick={() => void loadCommentInsights()}
                  disabled={insightsLoading}
                  className="btn-ghost !text-[11px]"
                >
                  {insightsLoading ? "Analyzing…" : "Generate insights"}
                </button>
              </div>
              <p className="text-[11px] text-muted">One AI pass over this comment set — themes, questions, requests, sentiment.</p>
              {!insights && !insightsLoading && (
                <p className="text-xs text-muted">Run after analyze; uses 1 OpenRouter request (cached per video).</p>
              )}
              {insights && (
                <div className="space-y-3">
                  <div className="flex flex-wrap gap-2">
                    <span
                      className={`rounded-full px-2 py-0.5 text-[10px] font-medium ${
                        insights.source === "llm" ? "bg-accent-soft text-accent" : "border border-border bg-surface-2 text-muted"
                      }`}
                    >
                      {insights.source === "llm" ? "AI" : "Heuristic"}
                    </span>
                    {insights.cache_hit && (
                      <span className="rounded-full border border-border bg-surface-2 px-2 py-0.5 text-[10px] text-muted">cached</span>
                    )}
                    {insights.llm_requests > 0 && (
                      <span className="text-[10px] text-muted">{insights.llm_requests} LLM req</span>
                    )}
                    <span
                      className={`rounded-full px-2 py-0.5 text-[10px] font-medium capitalize ${sentimentClass(
                        insights.sentiment.overall
                      )}`}
                    >
                      {insights.sentiment.overall}
                    </span>
                  </div>
                  {insights.sentiment.summary && (
                    <p className="text-xs leading-relaxed text-muted">{insights.sentiment.summary}</p>
                  )}
                  {insights.themes.length > 0 && <InsightsBulletList title="Themes" items={insights.themes} />}
                  {insights.questions.length > 0 && (
                    <div>
                      <p className="mb-1 text-[11px] font-medium text-fg">Common questions</p>
                      <ul className="space-y-2 text-xs text-muted">
                        {insights.questions.map((q, idx) => (
                          <li key={idx}>
                            <span className="font-medium text-fg">{q.theme}</span>
                            {q.count_estimate != null && (
                              <span className="text-muted"> · ~{q.count_estimate}</span>
                            )}
                            {q.examples.length > 0 && (
                              <ul className="mt-1 list-disc pl-4">
                                {q.examples.slice(0, 3).map((ex, i) => (
                                  <li key={i} className="italic">
                                    {ex}
                                  </li>
                                ))}
                              </ul>
                            )}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                  {insights.requests.length > 0 && (
                    <InsightsBulletList title="Requests & next-video ideas" items={insights.requests} />
                  )}
                </div>
              )}
            </div>
          )}

          {results.length > 0 && (
            <div className="panel space-y-3 p-4 text-sm">
              <h2 className="text-xs font-semibold uppercase tracking-wider text-muted">Ask your comments</h2>
              <p className="text-[11px] text-muted">One request per question; answers cite comment ids from this analyze run.</p>
              <textarea
                className="input-field min-h-[72px] w-full text-xs"
                value={askQuestion}
                onChange={(e) => setAskQuestion(e.target.value)}
                placeholder="e.g. What are people confused about?"
                aria-label="Question about comments"
                onKeyDown={(e) => {
                  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) void submitAsk();
                }}
              />
              <button
                type="button"
                className="btn-primary w-full !py-2 !text-xs"
                disabled={askLoading || !askQuestion.trim()}
                onClick={() => void submitAsk()}
              >
                {askLoading ? "Thinking…" : "Ask"}
              </button>
              {askHistory.length > 0 && (
                <ul className="max-h-64 space-y-3 overflow-y-auto border-t border-border pt-3">
                  {askHistory.map((item, idx) => (
                    <li key={idx} className="space-y-1.5 rounded-lg bg-surface-2 p-2.5">
                      <p className="text-[11px] font-medium text-fg">Q: {item.question}</p>
                      <p className="text-xs leading-relaxed text-muted">{item.result.answer}</p>
                      <div className="flex flex-wrap items-center gap-1.5">
                        <span
                          className={`rounded px-1.5 py-0.5 text-[10px] ${
                            item.result.source === "llm" ? "bg-accent-soft text-accent" : "border border-border bg-surface-2 text-muted"
                          }`}
                        >
                          {item.result.source === "llm" ? "AI" : "Heuristic"}
                        </span>
                        {item.result.cache_hit && (
                          <span className="text-[10px] text-muted">cached</span>
                        )}
                        {item.result.citation_ids.map((id) => (
                          <button
                            key={id}
                            type="button"
                            className="rounded-md bg-sky-500/15 px-1.5 py-0.5 font-mono text-[10px] text-sky-800 dark:text-sky-200"
                            onClick={() => {
                              setBucketFilter("");
                              setFilter("");
                              setFocusCommentId(id);
                            }}
                            title="Scroll to comment"
                          >
                            {shortCommentId(id)}
                          </button>
                        ))}
                      </div>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          )}
        </aside>

        <section className="panel flex min-h-[420px] flex-col overflow-hidden">
          <div className="flex flex-wrap items-center gap-3 border-b border-border px-4 py-3">
            <h2 className="mr-auto text-sm font-medium text-fg">Comments &amp; results</h2>
            {results.length > 0 && (
              <>
                <select
                  className="input-field !py-1.5 !text-xs"
                  value={bucketFilter}
                  onChange={(e) => setBucketFilter(e.target.value)}
                >
                  <option value="">All buckets</option>
                  {CREATOR_BUCKETS.map((b) => (
                    <option key={b} value={b}>
                      {BUCKET_LABELS[b]}
                    </option>
                  ))}
                </select>
                <select className="input-field !py-1.5 !text-xs" value={filter} onChange={(e) => setFilter(e.target.value)}>
                  <option value="">All labels</option>
                  {[...new Set(results.map((r) => r.label))].map((l) => (
                    <option key={l} value={l}>
                      {l}
                    </option>
                  ))}
                </select>
                <select
                  className="input-field !py-1.5 !text-xs"
                  value={sort}
                  onChange={(e) => setSort(e.target.value as "risk" | "label")}
                >
                  <option value="risk">Risk ↓</option>
                  <option value="label">Label</option>
                </select>
                <button type="button" onClick={downloadCsv} className="btn-ghost">
                  Export CSV
                </button>
              </>
            )}
          </div>

          <div className="flex-1 overflow-y-auto px-3 py-3 sm:px-4" style={{ maxHeight: "calc(100vh - 12rem)" }}>
            {loading && results.length === 0 && (
              <p className="py-12 text-center text-sm text-muted">Fetching latest comments…</p>
            )}
            {!loading && results.length === 0 && (
              <p className="py-12 text-center text-sm text-muted">Analyze to score comments.</p>
            )}
            <ul className="space-y-2">
              {filtered.map((r) => (
                <li
                  key={r.id}
                  ref={(el) => {
                    commentRefs.current[r.id] = el;
                  }}
                  className={`comment-card ${
                    focusCommentId === r.id
                      ? "!border-sky-500/60 bg-sky-500/10 ring-2 ring-sky-500/30"
                      : ""
                  }`}
                >
                  <div className="mb-2 flex flex-wrap items-center gap-2">
                    <RiskPill risk={r.risk} />
                    {r.creator_bucket && (
                      <span className={`rounded-md px-2 py-0.5 text-[11px] font-medium ${bucketChipClass(r.creator_bucket)}`}>
                        {BUCKET_LABELS[r.creator_bucket] ?? r.creator_bucket}
                      </span>
                    )}
                    <span className={`rounded-md px-2 py-0.5 text-[11px] font-medium uppercase ${labelClass(r.label)}`}>
                      {r.label}
                    </span>
                    <span className="text-[11px] text-muted">{r.decided_by}</span>
                    {r.needs_human && (
                      <span className="rounded-md bg-amber-500/15 px-1.5 py-0.5 text-[10px] text-amber-600 dark:text-amber-300">
                        review
                      </span>
                    )}
                  </div>
                  <p className="text-sm leading-relaxed text-fg">{r.text}</p>
                  {r.reasons.length > 0 && (
                    <p className="mt-2 text-[11px] leading-relaxed text-muted">{r.reasons.join(" · ")}</p>
                  )}
                  {typeof r.trace?.llm_skipped_reason === "string" && (
                    <p className="mt-1 text-[10px] text-amber-700 dark:text-amber-300">
                      LLM skipped: {r.trace.llm_skipped_reason}
                    </p>
                  )}
                  <div className="mt-3 space-y-2 border-t border-border pt-3">
                    <div className="flex flex-wrap gap-2">
                      <button
                        type="button"
                        className="btn-ghost !text-[11px]"
                        disabled={replyLoadingId === r.id || !canSuggestReply(r)}
                        onClick={() => void suggestReplyFor(r)}
                        title="Generate a style-matched draft (not posted)"
                      >
                        {replyLoadingId === r.id ? "Drafting…" : "Suggest reply"}
                      </button>
                      {replyDrafts[r.id] && (
                        <button type="button" className="btn-ghost !text-[11px]" onClick={() => copyReply(r.id)}>
                          Copy
                        </button>
                      )}
                    </div>
                    {replyDrafts[r.id] && (
                      <textarea
                        className="input-field min-h-[72px] w-full text-xs"
                        value={replyDrafts[r.id]}
                        onChange={(e) => setReplyDrafts((d) => ({ ...d, [r.id]: e.target.value }))}
                        aria-label="Suggested reply draft"
                      />
                    )}
                    <p className="text-[10px] text-muted">Copy and edit before you post — this app never posts to YouTube.</p>
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </section>
      </div>
    </div>
  );
}

function VideoPlaceholder({ message }: { message: string }) {
  return (
    <div className="panel flex aspect-video items-center justify-center border-dashed p-6 text-center text-sm text-muted">
      {message}
    </div>
  );
}

function RiskPill({ risk }: { risk: number }) {
  const tone =
    risk >= 70
      ? "bg-red-500/15 text-red-600 dark:text-red-300"
      : risk >= 40
        ? "bg-amber-500/15 text-amber-700 dark:text-amber-200"
        : "bg-emerald-500/15 text-emerald-700 dark:text-emerald-200";
  return (
    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${tone}`}>
      risk {risk}
    </span>
  );
}

function shortCommentId(id: string): string {
  if (id.length <= 12) return id;
  return `${id.slice(0, 6)}…${id.slice(-4)}`;
}

function InsightsBulletList({ title, items }: { title: string; items: string[] }) {
  return (
    <div>
      <p className="mb-1 text-[11px] font-medium text-fg">{title}</p>
      <ul className="list-disc space-y-0.5 pl-4 text-xs text-muted">
        {items.map((item, i) => (
          <li key={i}>{item}</li>
        ))}
      </ul>
    </div>
  );
}

function sentimentClass(overall: string): string {
  switch (overall) {
    case "positive":
      return "bg-emerald-500/15 text-emerald-800 dark:text-emerald-200";
    case "negative":
      return "bg-red-500/15 text-red-800 dark:text-red-200";
    case "mixed":
      return "bg-amber-500/15 text-amber-800 dark:text-amber-200";
    default:
      return "bg-zinc-500/15 text-zinc-700 dark:text-zinc-300";
  }
}

function formatCount(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function bucketChipClass(bucket: string): string {
  const map: Record<string, string> = {
    neutral_discussion: "bg-zinc-500/15 text-zinc-700 dark:text-zinc-300",
    appreciation: "bg-emerald-500/15 text-emerald-800 dark:text-emerald-200",
    question: "bg-sky-500/15 text-sky-800 dark:text-sky-200",
    critical: "bg-violet-500/15 text-violet-800 dark:text-violet-200",
    request: "bg-indigo-500/15 text-indigo-800 dark:text-indigo-200",
    discussion: "bg-zinc-500/15 text-zinc-700 dark:text-zinc-300",
    sarcastic: "bg-fuchsia-500/15 text-fuchsia-800 dark:text-fuchsia-200",
    nasty: "bg-red-500/15 text-red-800 dark:text-red-200",
    spam_scam: "bg-orange-500/15 text-orange-800 dark:text-orange-200",
    harassment: "bg-rose-500/15 text-rose-800 dark:text-rose-200",
  };
  return map[bucket] ?? "bg-zinc-500/15 text-zinc-700 dark:text-zinc-300";
}

function labelClass(label: string): string {
  if (["toxic", "threat", "identity_attack", "spam"].includes(label))
    return "bg-red-500/15 text-red-700 dark:text-red-200";
  if (label === "praise") return "bg-emerald-500/15 text-emerald-800 dark:text-emerald-200";
  if (label === "question") return "bg-sky-500/15 text-sky-800 dark:text-sky-200";
  if (label === "critique") return "bg-violet-500/15 text-violet-800 dark:text-violet-200";
  return "bg-zinc-500/15 text-zinc-700 dark:text-zinc-300";
}