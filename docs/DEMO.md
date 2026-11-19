# Demo script (~5 min)

1. `make api` — confirm `GET /health` shows `lr_model_loaded: true` after `make train-smoke`.
2. `curl -X POST http://127.0.0.1:8000/analyze -H "Content-Type: application/json" -d "{\"comments\":[{\"id\":\"1\",\"text\":\"thank you!\"},{\"id\":\"2\",\"text\":\"I will kill you\"}]}"`
3. Web: paste a YouTube URL with comments enabled (needs `YOUTUBE_API_KEY`).
4. Extension: build, load unpacked, open a video, click **Analyze comments**.
5. LinkedIn: open a post, click **Load more comments** if needed, then **Analyze (LinkedIn)**.