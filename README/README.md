# Farysene AI Telegram Bot

Feature-rich Telegram bot for downloading and processing media from multiple platforms with daily music delivery, history/stats, and a modular handler architecture.

This README describes the current, internal behavior of the bot and serves as user and developer documentation for the existing features. A separate document will outline AI update ideas and future filters.

## Features (current)

- Multi-platform link handling (beta):
  - Spotify, Pinterest, Instagram, TikTok, YouTube, Twitter (X), Threads
  - Link detection and routing to the correct handler
  - Multiple-links support for specific cases:
    - Spotify: 2–3 links simultaneously
    - Pinterest: up to 5 links simultaneously
- Daily songs delivery (opt‑in):
  - Sends 3 curated songs per day (morning, noon, evening)
  - Configurable via /daily_on and unsubscribe button
- History and stats:
  - /history: recent downloads per user
  - /stats: per-user download counts, platform breakdown, last 7 days overview
- Status, retry, and control commands:
  - /status: simple platform availability check
  - /retry: retry the last operation (best-effort)
  - /cancel: request to cancel the current task
- Access control:
  - ALLOWED_USERS list limits who can use the bot (development mode)
- Basic caching and persistence:
  - SQLite database (default) with migrations scaffold (Alembic present)
- Logging:
  - File log at bot.log and console logs

## Architecture

- aiogram (v3) polling-based bot (no webhook)
- Handlers per platform under handlers/ with a common detection layer (handlers/detector.py)
- Task management via task_manager for retries and simple rate control
- Database utilities in db_manager.py
- Utilities for URL extraction/validation in utils.py
- Scheduler for daily songs via APScheduler (Asia/Tehran timezone)

Entrypoint: bot.py

## Requirements

- Python 3.10+
- FFmpeg available on PATH or set via FFMPEG_PATH
- Internet connectivity; optional proxy support

Recommended: create and use a virtual environment.

## Installation (Windows PowerShell)

1) Create and activate a virtual environment

```
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
```

2) Install dependencies

```
pip install -r requirements.txt
```

3) Configure environment variables

Create a .env file at the project root:

```
BOT_TOKEN=your_telegram_bot_token
# Optional proxy (blank or remove to disable)
PROXY=
# Optional FFmpeg path if not in PATH
FFMPEG_PATH=ffmpeg
# Optional integrations
SPOTIPY_CLIENT_ID=
SPOTIPY_CLIENT_SECRET=
GENIUS_API_TOKEN=
# Optional search tuning
SPOTIFY_MARKET=US
```

4) Run the bot

```
python bot.py
```

The bot uses long polling. Make sure BOT_TOKEN is valid and the account is not used elsewhere with a different webhook.

## Commands (user)

- /start — welcome and auto-enable daily songs for allowed users
- /help — usage, limits, and contact
- /playlist — placeholder (under development)
- /history — show your last downloads
- /stats — show your statistics
- /status — platform status
- /retry — retry last process
- /cancel — try to cancel the current task
- /daily_on — enable daily songs (unsubscribe button appears on delivered items)

## Daily songs

Place files in the daily_songs/ directory using the following naming convention:

- daily_morning-<Title> - <Artist>.mp3
- daily_noon-<Title> - <Artist>.mp3
- daily_evening-<Title> - <Artist>.mp3

At the configured times (Tehran time), the bot will push the matching file to all active users.

## Data and storage

- SQLite database for caching/history (by default). Alembic present for future schema evolution.
- Media cache and downloads are kept locally (ignored by git). The repository .gitignore prevents accidental commits of large or sensitive files.

## Security and privacy

- Keep your .env file private. Never commit tokens or cookies.
- ALLOWED_USERS restricts usage during development. Adjust carefully before public deployment.
- Respect the Terms of Service of each platform. Downloading protected content may violate ToS and/or copyright law.

## Troubleshooting

- Bot does not start: verify BOT_TOKEN in .env; ensure aiogram is installed and compatible; check Python version.
- No audio/video processing: ensure FFmpeg is installed and on PATH (or set FFMPEG_PATH).
- Daily songs not delivered: verify daily_songs/ contains properly named files and that users enabled daily content.
- Rate limits/errors from platforms: try again later; consider setting a proxy via PROXY in .env.

Logs: check bot.log for detailed errors.

## Development notes

- The .gitignore excludes secrets, media, caches, venvs, and deployment artifacts to keep the repo publish‑ready.
- Webhook/Flask code has been removed; the bot uses aiogram polling only.
- For adding new platforms, extend handlers/detector.py and add a handler module under handlers/ following existing examples.
- For DB changes, add Alembic migrations and keep db_manager.py in sync.

## Roadmap (high-level)

- Consolidate media separation into a single module with stable caching
- Optional AI features: transcription/subtitles, translation/summaries, recommendations
- Per-platform rate limiting/backoff and cookie/session manager
- Lightweight admin/status page (health, queue depth, last errors)
- Tests and CI

## Legal

This software is provided for educational and personal use. You are responsible for complying with applicable laws and each platform’s Terms of Service. Avoid downloading or redistributing copyrighted content without permission.
