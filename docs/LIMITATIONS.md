# Limitations

| Area | Limitation |
|------|------------|
| YouTube web | Requires Google API key; daily quota; comments disabled videos fail |
| YouTube extension | DOM selectors change; auto-scroll may not load all threads |
| LinkedIn | Extension-only; selectors marked `TODO(verify)`; no server fetch |
| Transformer | Detoxify install is heavy; CPU latency on ambiguous comments |
| Identity | Reduced FP via rules, but LR/transformer can still err — use `needs_human` |
| Actions | **No** auto-moderation on platforms |

## Non-goals

- Multilingual coverage without additional training data
- Real-time moderation at huge scale on a single laptop
- Paid LLM or cloud inference APIs