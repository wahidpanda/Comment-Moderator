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
    <section className="panel p-3">
      <div className="mb-2 flex items-center justify-between">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-muted">Recent videos</h2>
        <span className="text-xs text-muted">{items.length}</span>
      </div>
      <ul className="flex gap-2 overflow-x-auto pb-1 scrollbar-thin">
        {items.map((h) => {
          const active = h.videoId === activeId;
          return (
            <li
              key={h.videoId}
              className={`flex w-44 shrink-0 flex-col gap-1 rounded-xl border p-2 transition ${
                active ? "border-accent bg-accent-soft" : "border-border bg-surface-2"
              }`}
            >
              <button type="button" onClick={() => onSelect(h)} className="text-left">
                <img
                  src={h.thumbnail_url}
                  alt=""
                  className="mb-1 aspect-video w-full rounded-lg object-cover"
                />
                <p className="line-clamp-2 text-[11px] font-medium leading-snug text-fg">{h.title}</p>
                <p className="text-[10px] text-muted">{new Date(h.lastAnalyzedAt).toLocaleString()}</p>
              </button>
              <div className="flex gap-1">
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