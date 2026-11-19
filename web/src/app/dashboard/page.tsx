"use client";

import { useEffect, useMemo, useState } from "react";
import { loadHistory } from "@/lib/history";
import type { ModerationResult } from "@/lib/api";

type EvalScenario = {
  scenario: string;
  f1_macro_proxy?: number;
  f1_ci95_low?: number;
  f1_ci95_high?: number;
  false_positives?: number;
  benign_fpr?: number;
  transformer_pct?: number;
  transformer_ambiguous_pct?: number;
  transformer_threat_scan_pct?: number;
  latency_ms_p50?: number;
  skipped?: string;
};

type IdentityGroupRow = {
  fpr?: number | null;
  benign_rows?: number;
  false_positives?: number;
  fpr_ci95_low?: number;
  fpr_ci95_high?: number;
};

type IdentityAudit = {
  overall_fpr?: number;
  overall_fpr_ci95_low?: number;
  overall_fpr_ci95_high?: number;
  fpr_benign_identity_rows?: number;
  fpr_benign_non_identity_rows?: number;
  fpr_benign_identity_ci95_low?: number;
  fpr_benign_identity_ci95_high?: number;
  fpr_benign_non_identity_ci95_low?: number;
  fpr_benign_non_identity_ci95_high?: number;
  per_identity_group?: Record<string, IdentityGroupRow>;
};

type TaskDecomposed = {
  task_eval?: string;
  toxicity?: {
    f1?: number;
    f1_ci95_low?: number;
    f1_ci95_high?: number;
    benign_fpr?: number;
    benign_fpr_ci95_low?: number;
    benign_fpr_ci95_high?: number;
    false_positives_benign?: number;
    benign_count?: number;
  };
  threat?: { f1?: number; positives?: number; detected?: number };
};

type EvalPayload = {
  eval_generated_at?: string;
  n_test_subset?: number;
  results: EvalScenario[];
  task_metrics_decomposed?: Record<string, TaskDecomposed>;
  bucket_definitions?: Record<string, string>;
  identity_audit_jigsaw?: Record<string, IdentityAudit>;
};

type SessionPayload = {
  results: ModerationResult[];
  summary: Record<string, number | Record<string, number>>;
};

const SCENARIO_LABELS: Record<string, string> = {
  A_rules_lr: "A · Rules + LR",
  B_transformer_only: "B · Detoxify only",
  B_with_llm: "B · Detoxify + LLM",
  C_hybrid: "C · Hybrid",
  C_hybrid_with_llm: "C · Hybrid + LLM",
  D_detoxify_only: "D · Detoxify only",
};

const SCENARIO_ORDER = ["A_rules_lr", "C_hybrid", "C_hybrid_with_llm", "D_detoxify_only", "B_transformer_only", "B_with_llm"];

export default function DashboardPage() {
  const [session, setSession] = useState<SessionPayload | null>(null);
  const [evalData, setEvalData] = useState<EvalPayload | null>(null);
  const [evalError, setEvalError] = useState<string | null>(null);
  const history = useMemo(() => loadHistory(), []);

  useEffect(() => {
    try {
      const raw = localStorage.getItem("moderator:session");
      if (raw) setSession(JSON.parse(raw) as SessionPayload);
    } catch {
      /* ignore */
    }
    fetch("/eval_metrics.json")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((j) => setEvalData(j as EvalPayload))
      .catch(() => setEvalError("Model evaluation file missing. Run make eval (copies docs → web/public)."));
  }, []);

  const labels = useMemo(() => {
    if (!session) return {};
    const counts: Record<string, number> = {};
    for (const r of session.results) counts[r.label] = (counts[r.label] ?? 0) + 1;
    return counts;
  }, [session]);

  const layers = (session?.summary.layer_counts as Record<string, number>) ?? {};
  const total = session?.results.length ?? 0;
  const highRisk = session?.results.filter((r) => r.risk >= 70).length ?? 0;
  const review = session?.results.filter((r) => r.needs_human).length ?? 0;
  const decomposedRows = useMemo(() => {
    if (!evalData?.task_metrics_decomposed) return [];
    return SCENARIO_ORDER.map((id) => {
      const row = evalData.task_metrics_decomposed?.[id];
      if (!row) return null;
      return { scenario: id, ...row };
    }).filter((r): r is TaskDecomposed & { scenario: string } => Boolean(r));
  }, [evalData]);

  const hybridAudit =
    evalData?.identity_audit_jigsaw?.C_hybrid_with_llm ??
    evalData?.identity_audit_jigsaw?.C_hybrid;

  const sessionBuckets = (session?.summary.creator_bucket_counts as Record<string, number>) ?? {};

  return (
    <div className="space-y-8">
      <header>
        <h1 className="text-2xl font-bold tracking-tight text-fg">Dashboard</h1>
        <p className="mt-1 text-sm text-muted">Latest analysis session and offline model evaluation metrics.</p>
      </header>

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard label="Comments analyzed" value={total ? String(total) : "—"} hint="From last Analyze run" />
        <StatCard label="High risk (≥70)" value={total ? String(highRisk) : "—"} hint="Needs attention first" />
        <StatCard label="Human review" value={total ? String(review) : "—"} hint="Flagged by pipeline" />
        <StatCard label="History videos" value={String(history.length)} hint="Stored locally" />
      </section>

      {!session ? (
        <div className="panel p-6 text-sm text-muted">Run Analyze on the home page to populate session metrics.</div>
      ) : (
        <div className="grid gap-6 lg:grid-cols-2">
          <div className="panel p-5">
            <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-muted">Label distribution</h2>
            <ul className="space-y-2">
              {Object.entries(labels)
                .sort((a, b) => b[1] - a[1])
                .map(([k, v]) => (
                  <li key={k} className="flex items-center gap-3">
                    <span className="w-24 text-sm font-medium capitalize text-fg">{k}</span>
                    <div className="h-2 flex-1 overflow-hidden rounded-full bg-surface-2">
                      <div className="h-full rounded-full bg-accent" style={{ width: `${(v / total) * 100}%` }} />
                    </div>
                    <span className="w-10 text-right text-sm text-muted">{v}</span>
                  </li>
                ))}
            </ul>
          </div>
          <div className="panel p-5">
            <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-muted">Resolution layers</h2>
            <ul className="space-y-2 text-sm">
              {Object.entries(layers).map(([k, v]) => (
                <li key={k} className="flex justify-between border-b border-border py-2 last:border-0">
                  <span className="font-medium text-fg">{k}</span>
                  <span className="text-muted">
                    {v} ({total ? ((v / total) * 100).toFixed(1) : 0}%)
                  </span>
                </li>
              ))}
            </ul>
            {(Number(session.summary.transformer_ambiguous_pct) > 0 ||
              Number(session.summary.transformer_threat_scan_pct) > 0) && (
              <p className="mt-3 text-xs text-muted">
                % routed via full Detoxify (ambiguous LR):{" "}
                {Number(session.summary.transformer_ambiguous_pct ?? 0).toFixed(1)}% · % of comments that received a
                threat-only Detoxify scan: {Number(session.summary.transformer_threat_scan_pct ?? 0).toFixed(1)}%
              </p>
            )}
          </div>
        </div>
      )}

      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-fg">Model evaluation (offline)</h2>
        {evalError && <p className="panel p-4 text-sm text-muted">{evalError}</p>}
        {evalData && (
          <>
            <p className="text-xs text-muted">
              Civil Comments test subset · n={evalData.n_test_subset ?? "—"} · eval built{" "}
              {evalData.eval_generated_at
                ? new Date(evalData.eval_generated_at).toLocaleString()
                : "(run make eval for timestamp)"}
            </p>
            <div className="overflow-x-auto rounded-2xl border border-border">
              <table className="w-full min-w-[800px] text-left text-sm">
                <thead className="bg-surface-2 text-xs uppercase tracking-wide text-muted">
                  <tr>
                    <th className="px-4 py-3">Scenario</th>
                    <th className="px-4 py-3">Toxic F1</th>
                    <th className="px-4 py-3">95% CI</th>
                    <th className="px-4 py-3">Benign FPR</th>
                    <th className="px-4 py-3">Threat F1</th>
                    <th className="px-4 py-3">Threat det.</th>
                  </tr>
                </thead>
                <tbody>
                  {decomposedRows.map((r) => (
                    <tr key={r.scenario} className="border-t border-border">
                      <td className="px-4 py-3 font-medium text-fg">
                        {SCENARIO_LABELS[r.scenario] ?? r.scenario}
                      </td>
                      <td className="px-4 py-3">{r.toxicity?.f1?.toFixed(3) ?? "—"}</td>
                      <td className="px-4 py-3 text-muted">
                        {r.toxicity?.f1_ci95_low != null && r.toxicity?.f1_ci95_high != null
                          ? `${r.toxicity.f1_ci95_low.toFixed(3)} – ${r.toxicity.f1_ci95_high.toFixed(3)}`
                          : "—"}
                      </td>
                      <td className="px-4 py-3">{r.toxicity?.benign_fpr?.toFixed(3) ?? "—"}</td>
                      <td className="px-4 py-3">{r.threat?.f1?.toFixed(3) ?? "—"}</td>
                      <td className="px-4 py-3 text-muted">
                        {r.threat?.detected ?? "—"}/{r.threat?.positives ?? "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {(evalData.bucket_definitions || Object.keys(sessionBuckets).length > 0) && (
              <div className="grid gap-4 lg:grid-cols-2">
                {evalData.bucket_definitions && (
                  <div className="panel p-5">
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-muted">Creator buckets (definitions)</h3>
                    <ul className="mt-3 space-y-1 text-xs">
                      {Object.entries(evalData.bucket_definitions).map(([k, v]) => (
                        <li key={k}>
                          <span className="font-medium text-fg">{k}</span>
                          <span className="text-muted"> — {v}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {Object.keys(sessionBuckets).length > 0 && (
                  <div className="panel p-5">
                    <h3 className="text-xs font-semibold uppercase tracking-wider text-muted">Last session buckets</h3>
                    <ul className="mt-3 space-y-1 text-sm">
                      {Object.entries(sessionBuckets)
                        .filter(([, n]) => n > 0)
                        .sort((a, b) => b[1] - a[1])
                        .map(([k, n]) => (
                          <li key={k} className="flex justify-between">
                            <span className="text-fg">{k}</span>
                            <span className="text-muted">{n}</span>
                          </li>
                        ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
            {hybridAudit && (
              <div className="panel overflow-x-auto p-5">
                <p className="text-xs font-semibold uppercase tracking-wide text-muted">Jigsaw identity audit</p>
                <p className="mt-1 text-xs text-muted">
                  Identity FPR {(hybridAudit.fpr_benign_identity_rows ?? 0).toFixed(3)} (
                  {(hybridAudit.fpr_benign_identity_ci95_low ?? 0).toFixed(3)}–
                  {(hybridAudit.fpr_benign_identity_ci95_high ?? 0).toFixed(3)}) vs non-identity{" "}
                  {(hybridAudit.fpr_benign_non_identity_rows ?? 0).toFixed(3)}
                </p>
                {hybridAudit.per_identity_group && (
                  <table className="mt-4 w-full min-w-[640px] text-left text-xs">
                    <thead className="text-muted">
                      <tr>
                        <th className="py-2">Group</th>
                        <th className="py-2">n (benign)</th>
                        <th className="py-2">FPR</th>
                        <th className="py-2">Wilson 95% CI</th>
                        <th className="py-2">Note</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(hybridAudit.per_identity_group)
                        .filter(([, row]) => (row.benign_rows ?? 0) > 0)
                        .sort((a, b) => (b[1].fpr ?? 0) - (a[1].fpr ?? 0))
                        .map(([group, row]) => {
                          const n = row.benign_rows ?? 0;
                          const small = n < 30;
                          return (
                            <tr key={group} className="border-t border-border">
                              <td className="py-2 font-medium text-fg">{group}</td>
                              <td className="py-2">{n}</td>
                              <td className="py-2">{(row.fpr ?? 0).toFixed(3)}</td>
                              <td className="py-2 text-muted">
                                {(row.fpr_ci95_low ?? 0).toFixed(3)} – {(row.fpr_ci95_high ?? 0).toFixed(3)}
                              </td>
                              <td className="py-2">{small ? "small sample (<30)" : ""}</td>
                            </tr>
                          );
                        })}
                    </tbody>
                  </table>
                )}
              </div>
            )}
          </>
        )}
      </section>
    </div>
  );
}

function StatCard({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div className="panel p-4">
      <p className="text-xs font-medium uppercase tracking-wider text-muted">{label}</p>
      <p className="mt-2 text-3xl font-bold text-fg">{value}</p>
      <p className="mt-1 text-xs text-muted">{hint}</p>
    </div>
  );
}