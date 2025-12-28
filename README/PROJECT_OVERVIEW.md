# Project Overview — Farysene AI Telegram Bot

A modular Telegram bot that downloads and processes media from multiple platforms, offers daily music delivery, and tracks user history and stats. Designed for easy extension and safe public release.

## What it does (high level)
- Accepts links from popular platforms and returns downloadable media (audio/video/images).
- Supports multi-link messages for certain platforms (e.g., Spotify 2–3 links; Pinterest up to 5).
- Delivers curated daily songs (3 per day) to opted-in users.
- Maintains per-user download history and basic statistics.
- Provides status checks, retry/cancel controls, and simple rate management.

## Supported platforms (current)
- Spotify, Pinterest, Instagram, TikTok, YouTube, Twitter (X), Threads

## Architecture (simple)
- Framework: aiogram (v3) polling — entrypoint: `bot.py`.
- Handlers: one module per platform under `handlers/` with a shared detector for URLs (`handlers/detector.py`).
- Task orchestration: `task_manager.py` for retries and simple throttling.
- Data: SQLite by default via `db_manager.py` (Alembic present for migrations).
- Scheduling: APScheduler for daily songs (Tehran timezone by default).
- Utilities: `utils.py` (URL extraction, validation), `platform_checker.py` (status), `error_handler.py` (errors), `resource_manager.py` (auxiliary).
- Config: `.env` loaded by `config.py` for tokens and options.

## Typical flow
1) User sends a message (link or multiple links).
2) Detector identifies the platform(s) and routes to the appropriate handler.
3) Handler downloads the content (yt-dlp/APIs), applies optional post-processing (FFmpeg, metadata), and sends the result back.
4) History/stats are recorded.
5) For daily songs, APScheduler pulls a local file and broadcasts it to opted-in users with unsubscribe controls.

## Key features
- Multi-link support: Spotify (2–3), Pinterest (up to 5).
- Caching/history: avoids rework and shows recent activity to users.
- Commands: `/start`, `/help`, `/history`, `/stats`, `/status`, `/retry`, `/cancel`, `/daily_on`.
- Logging: console logging by default (file logging can be added by config).

## Setup and running
- Requirements are split:
  - `requirements.txt` — core bot and platform support.
  - `requirements-ai.txt` — optional AI features (speech-to-text, separation, recommendations). Not required for core usage.
- Environment configuration via `.env` (not tracked by Git):
  - `BOT_TOKEN` (required), optional `PROXY`, `FFMPEG_PATH`, and API keys.
- Run: `python bot.py`
- Full instructions: `README/INSTALL_GUIDE.md`.

## Security and privacy
- Tokens/credentials are read from `.env` and never committed.
- Media, caches, DBs, venvs, and IDE files are ignored by `.gitignore`.
- Respect the Terms of Service for each platform. Some features may require cookies or be subject to rate limits.

## Extending (platforms and AI)
- Add a platform: update `handlers/detector.py` with a URL pattern, then add a handler module under `handlers/` that normalizes outputs and records history.
- Add AI features: see `README/AI_UPDATE_IDEAS.md` for transcription/subtitles, translation/summaries, recommendations, thumbnail/caption generation, and moderation.

## Limitations
- Some platforms have strict ToS and anti-bot protections; features may require cookies or break over time.
- Heavy AI features (e.g., separation, Whisper) are optional and may be resource-intensive.

## License and contact
- License: MIT (see `LICENSE`).
- Contributions/issues are welcome. For direct collaboration, open an issue or leave a comment in the repository to get in touch.
