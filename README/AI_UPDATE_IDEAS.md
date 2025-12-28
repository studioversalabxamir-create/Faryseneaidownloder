# AI Update and Platform Expansion Ideas

Goals
- Add AI-powered features that increase usefulness while staying performant.
- Expand supported platforms using modular handlers, preferring yt-dlp where possible.

## Phase 1: AI features (low–medium complexity)

- Speech-to-text + subtitles
  - Convert audio/video to text; generate subtitles (SRT/VTT) and lyrics timings (LRC).
  - Options: Faster-Whisper (local) or API.
  - UX: /transcribe (upload or link), /subtitles; inline buttons.
  - Tech: ai/transcribe.py, cache by media hash, SRT/VTT artifacts.

- Translation and TL;DR summaries
  - Translate subtitles or text to fa/en; summarize long videos or threads.
  - UX: inline buttons: “Translate to FA/EN”, “Summarize”.
  - Tech: ai/nlp.py with translate_text() and summarize_text(); API/local backend.

- Smart search (linkless)
  - “Find me song X,” “Top lofi playlist,” etc.
  - UX: /search <query>, returns 5–10 results with action buttons.
  - Tech: search/search.py providers (spotify, youtube); simple reranker.

- Audio normalization and tags
  - Normalize to -14 LUFS; embed ID3/covers; translate titles/lyrics.
  - UX: “Normalize audio” toggle.
  - Tech: audio/postprocess.py using pyloudnorm/pydub; mutagen for ID3.

## Phase 2: AI features (medium–high complexity)

- Karaoke/stems and timed lyrics
  - Extract vocals/instrumental (Demucs/Open-Unmix); align lyrics timestamps.
  - UX: “Create karaoke version (instrumental + lyrics)”.
  - Tech: audio/separation.py; content-addressed cache.

- Recommendations and “similar to”
  - Per-user embeddings and nearest-neighbor recommendations.
  - UX: “More like this” button; personalized daily suggestions.
  - Tech: ai/recsys.py; sentence-transformers + FAISS.

- Thumbnail selection and captions
  - Pick best thumbnail frame; generate captions/hashtags.
  - UX: “Best thumbnail”, “Generate caption/hashtags”.
  - Tech: video/thumbnail.py (scoring or CLIP-lite); ai/captions.py (LLM prompts).

## Phase 3: Moderation and operations

- Moderation filters
  - NSFW/unsafe classification; spam detection and blocking.
  - Tech: ai/moderation.py + rules in config.

- Predictive throttling
  - Predict rate-limit risk per platform and backoff automatically.
  - Tech: rate controller in task_manager; EWMA counters per domain.

## Platform expansion (suggested)

- Fast wins (yt-dlp): Vimeo, Dailymotion, Reddit, Twitch clips, Bilibili, Streamable, Imgur, OK.ru, VK, Rumble, Facebook (cookies).
- Music/audio: Bandcamp, Mixcloud, Audiomack (metadata-first), Deezer (metadata), Apple Music (metadata/previews).
- Text-to-PDF: X threads to PDF/Markdown; Reddit threads to PDF; Medium/Substack to PDF.

## Implementation guidelines

- Extend handlers/detector.py with new patterns.
- One handler module per platform; prefer yt-dlp, add cookies if needed.
- Normalize outputs (filenames, metadata, thumbnail); save history.
- Centralize cookies/proxies; per-domain rate limits and backoff.
- Per-job temp dirs; atomic renames; file/DB locks for cache writes.

## Config and dependencies

- requirements.txt for core; requirements-ai.txt for AI extras.
- .env flags: ENABLE_STT, ENABLE_KARAOKE, ENABLE_RECSYS, ENABLE_MODERATION; PROXY; FFMPEG_PATH.

## Acceptance criteria

- Phase 1: /transcribe + /search working; normalization option; graceful skips if tokens missing.
- Phase 2: karaoke flow stable; recommendations reasonable; thumbnail/caption working.
- Phase 3: moderation reduces unsafe content; rate-limit backoff lowers platform errors.
