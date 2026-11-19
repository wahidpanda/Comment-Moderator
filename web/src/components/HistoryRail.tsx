"use client";

import type { HistoryEntry } from "@/lib/history";

type Props = {
  items: HistoryEntry[];
  activeId: string | null;
  onSelect: (entry: HistoryEntry) => void;
  onRemove: (videoId: string) => void;
  onReanalyze: (entry: HistoryEntry) => void;
};

export function HistoryRail({ items, activeId, onSelect, onRemove, onReanalyze }: Props) {
  if (items.length === 0) return null;

  return (
    <section className="panel p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="section-kicker">Recent videos</h2>
        <span className="rounded-full bg-accent-soft px-2 py-0.5 text-[10px] font-semibold text-accent">
          {items.length}
        </span>
      </div>
      <ul className="scrollbar-thin flex gap-3 overflow-x-auto pb-1">
        {items.map((h) => {
          const active = h.videoId === activeId;
          return (
            <li
              key={h.videoId}
              className={`flex w-48 shrink-0 flex-col gap-2 rounded-2xl border p-2.5 transition ${
                active
                  ? "border-accent/50 bg-accent-soft shadow-md"
                  : "border-border bg-surface-2 hover:border-accent/30"
              }`}
            >
              <button type="button" onClick={() => onSelect(h)} className="text-left">
                <div className="relative mb-2 overflow-hidden rounded-xl">
                  <img src={h.thumbnail_url} alt="" className="aspect-video w-full object-cover" />
                  {active && (
                    <span className="absolute bottom-1.5 left-1.5 rounded-md bg-accent px-1.5 py-0.5 text-[9px] font-bold text-white">
                      Active
                    </span>
                  )}
                </div>
                <p className="line-clamp-2 text-xs font-semibold leading-snug text-fg">{h.title}</p>
                <p className="mt-1 text-[10px] text-muted">{new Date(h.lastAnalyzedAt).toLocaleString()}</p>
              </button>
              <div className="flex gap-1.5">
                <button type="button" onClick={() => onReanalyze(h)} className="btn-ghost flex-1 !py-1 !text-[10px]">
                  Refresh
                </button>
                <button
                  type="button"
                  onClick={() => onRemove(h.videoId)}
                  className="btn-ghost !px-2 !py-1 !text-[10px] text-danger"
                  aria-label="Remove"
                >
                  ×
                </button>
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}