# Output labels

## Production labels

`neutral`, `praise`, `question`, `toxic`, `threat`, `spam`, `identity_attack` — used for blocking/human review decisions.

## Experimental (UI shows “experimental”)

| Label | Status |
|-------|--------|
| **critique** | Heuristic regex only; **0%** exact label match on `creator_eval.csv`. Treat as a soft tag, not policy. |
| **sarcasm** | Not emitted by the engine today; appears only in hand-authored eval rows. Do not use for automation. |

Praise and question heuristics are retained for creator-facing tone, but harmful traffic is gated by LR + threat/spam layers.