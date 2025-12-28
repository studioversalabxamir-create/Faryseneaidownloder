# Installation, Deployment, and Execution Guide

This guide explains how to set up, run, and (optionally) deploy the Farysene AI Telegram Bot.

## Prerequisites

- Windows 10/11, Linux, or macOS
- Python 3.10+
- FFmpeg installed and available on PATH (or set FFMPEG_PATH in .env)
- A Telegram Bot token (from @BotFather)

Optional (depending on features):
- Spotify API credentials (client id/secret) for enhanced metadata/search
- Genius API token for lyrics
- Proxy URL if your network requires it

## 1) Clone and prepare environment

```
# Clone your repository
# git clone https://github.com/<your-username>/<repo-name>.git
# cd <repo-name>

# Create a Python virtual environment
py -3 -m venv .venv

# Activate it (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Or on Linux/macOS
# source .venv/bin/activate
```

## 2) Install dependencies

Core requirements:
```
pip install -r requirements.txt
```

Optional AI features:
```
# Install only if you need AI features (speech-to-text, separation, recsys)
pip install -r requirements-ai.txt
```

Notes:
- Some AI packages are heavy (torch/demucs); consider GPU availability or CPU-only alternatives.
- For Windows ARM64 or constrained environments, skip heavy packages.

## 3) Configure environment variables

Create a .env file in the project root:
```
BOT_TOKEN=your_telegram_bot_token
# Optional proxy (leave blank to disable)
PROXY=
# Optional FFmpeg path if not in PATH
FFMPEG_PATH=ffmpeg

# Optional integrations
SPOTIPY_CLIENT_ID=
SPOTIPY_CLIENT_SECRET=
GENIUS_API_TOKEN=
OPENAI_API_KEY=

# Optional tuning
SPOTIFY_MARKET=US
```

Tip: do not commit .env. The repository .gitignore already excludes it.

## 4) Run the bot (development)

```
python bot.py
```

The bot uses long polling. Ensure the token is valid and the bot is not configured with a webhook elsewhere.

## 5) Scheduling (daily songs)

Daily songs run via APScheduler (Tehran timezone by default). Place files in daily_songs/ as:
- daily_morning-<Title> - <Artist>.mp3
- daily_noon-<Title> - <Artist>.mp3
- daily_evening-<Title> - <Artist>.mp3

Ensure the folder exists (a placeholder is included). Files are ignored by git.

## 6) Deployment options

- Simple service (Windows):
  - Use NSSM or Task Scheduler to run `python bot.py` on startup.
- Simple service (Linux):
  - Create a systemd service that runs the virtualenv python with bot.py.
- Containerization:
  - Not included by default. If you plan Docker deployment, add a Dockerfile and expose the required components.

Operational tips:
- Ensure the bot has sufficient network access (proxy if necessary).
- Monitor logs via console or add a file handler if desired.

## 7) Updating dependencies

- Pin versions in requirements.txt/requirements-ai.txt as needed.
- Test updates locally before pushing to production.

## 8) Troubleshooting

- Token issues: rotate the token and update .env.
- FFmpeg not found: install FFmpeg or set FFMPEG_PATH.
- Platform download failures: rate limits or cookies required; add proxy, rotate UA, or provide cookies via a local (ignored) file.
- Crashes/tracebacks: check console logs; add more logging where needed.

## 9) Security and ToS

- Do not commit secrets, cookies, or media.
- Respect each platform’s Terms of Service.
- Gate experimental or high-risk features behind an allowlist.

## 10) Contact and license

- License: MIT (see LICENSE in the root directory).
- To collaborate or request features, open an issue or PR. For direct contact, leave a comment in the repository and we can get in touch.
