# Privacy

- Comment text is sent to **your local** FastAPI instance (`127.0.0.1`) from the web app or extension.
- No telemetry or third-party inference APIs in the default configuration.
- YouTube Data API calls go to Google with your API key when using the web app URL flow.
- SQLite job results stay on disk under `data/moderator.db` unless you delete them.
- Do not commit `.env` or API keys.