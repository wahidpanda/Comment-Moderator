# Architecture

## Layers

1. **Rules** (`engine/rules.yaml`, `engine/rules.py`) — regex spam/scam/threat, link-heavy, near-duplicate detection. Identity terms alone never flag.
2. **Logistic regression** (`engine/lr_layer.py`) — TF-IDF word 1–2 + char 3–5 grams, calibrated probabilities, `t_low`/`t_high` bands from `models/lr/v1/thresholds.json`.
3. **Transformer** (`engine/transformer_layer.py`) — Detoxify `unbiased` on CPU for ambiguous band only (optional extra: Ollama sarcasm via `OLLAMA_ENABLED`, not wired by default).

Heuristics (`engine/heuristics.py`) add `praise`, `question`, `critique` when LR says safe.

## API (`api/`)

- `POST /analyze` — batch of `{id, text}` comments
- `POST /analyze/youtube` — background job; YouTube Data API `commentThreads.list`
- `GET /jobs/{id}` — poll job status
- `GET /health` — LR model presence

SQLite (`api/db.py`) stores async YouTube jobs.

## Clients

- **Web** — YouTube URL only (API key server-side). LinkedIn not supported server-side (login wall / ToS).
- **Extension** — DOM scraping; YouTube auto-scroll with throttle; LinkedIn load-more only on explicit user click.

## Selectors

Per-platform configs: `extension/src/config/youtube.ts`, `extension/src/config/linkedin.ts` with HTML fixtures in `extension/fixtures/`.