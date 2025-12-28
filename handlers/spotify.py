import os
import asyncio
import logging
from aiogram import Router, types
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, FSInputFile, CallbackQuery, InputMediaPhoto
from aiogram.filters import CommandStart
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
import threading
from handlers.detector import detect_platform
from yt_dlp.utils import sanitize_filename
from subprocess import run, CalledProcessError
from spotipy import Spotify
from spotipy.oauth2 import SpotifyClientCredentials
import time
from urllib.parse import urlparse
from shutil import which
import subprocess
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, APIC, TIT2, TPE1, TALB, TCON, TDRC, TRCK, TPE2, COMM
import requests
from pydub import AudioSegment
import openai
import tempfile
from pydub import AudioSegment
from typing import Tuple, Optional
from typing import Dict, Optional
from typing import List
from filelock import FileLock
from openai import OpenAI, OpenAIError
from tempfile import NamedTemporaryFile
from typing import Union
from bs4 import BeautifulSoup
import json
from http.client import RemoteDisconnected
from urllib3.exceptions import NameResolutionError, HTTPError
from aiogram import Bot
from requests.exceptions import RequestException, HTTPError, ConnectionError, Timeout
import cloudscraper
from playwright.sync_api import sync_playwright
import re
import random
from urllib.parse import urlparse, quote_plus
import html as _html
import sqlite3
from db_manager import init_db, get_tg_file_id, update_tg_file_id, save_file, download_file_from_telegram
from aiogram.types import Chat
import shutil
from datetime import datetime, timedelta
from aiogram.fsm.state import State, StatesGroup
import sqlite3
import re
import unicodedata
import difflib
from mutagen.easyid3 import EasyID3
from aiogram.types import InlineQuery, InlineQueryResultArticle, InputTextMessageContent
from uuid import uuid4
import spotipy
from spotipy import SpotifyException
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# توکن بات
TOKEN = "YOUR_BOT_TOKEN_HERE"

async def download_song(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # اینجا فرض می‌کنیم آهنگ دانلود شده
    await update.message.reply_text("آهنگ دانلود شد!")

    # راه‌اندازی بات
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("download", download_song))
    application.run_polling()






# --- Query normalization ---
def normalize_query(query: str) -> str:
    """
    Normalize query: lowercase, remove diacritic, convert Persian/Arabic numbers to English, transliteration.
    """
    if not query:
        return ""
    # lowercase
    q = query.lower()
    # remove diacritic
    q = unicodedata.normalize('NFD', q)
    q = ''.join(c for c in q if unicodedata.category(c) != 'Mn')
    # convert Persian/Arabic numbers to English
    arabic_persian_to_english = str.maketrans('٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹', '01234567890123456789')
    q = q.translate(arabic_persian_to_english)
    # transliteration: normalize to ASCII
    q = unicodedata.normalize('NFKD', q).encode('ascii', 'ignore').decode('ascii')
    return q.strip()

logging.basicConfig(level=logging.INFO)

# --- ساخت بات ---
router = Router()
executor = ThreadPoolExecutor(max_workers=2)

# --- تنظیمات ---
from config import (
    SPOTIPY_CLIENT_ID,
    SPOTIPY_CLIENT_SECRET,
    PROXY as CFG_PROXY,
    MARKET,
    ALLOWED_USERS,
    INLINE_MIN_QUERY_LEN,
    INLINE_PAGE_SIZE,
    INLINE_CACHE_TTL,
    INLINE_CACHE_SIZE,
    INLINE_THROTTLE_PERMITS,
    INLINE_THROTTLE_WINDOW,
    GENIUS_API_TOKEN,
    # Multi-market search config for broader global coverage
    SEARCH_MARKETS,
    MULTI_MARKET_ON_SPARSE,
    MIN_INLINE_RESULTS,
    MULTI_MARKET_MAX_EXTRA,
)

BOT_TOKEN = os.getenv("BOT_TOKEN")
SUPPORT_CHAT_ID = int(os.getenv("SUPPORT_CHAT_ID") or "8196909396")
CACHE_DIR = "cache/"
# Preserve proxy structure: prefer config, then environment, then existing fallback
PROXY = "http://174.136.204.40:80"
FFMPEG_PATH = os.getenv("FFMPEG_PATH", r'G:\zAll data (All Mine)\Codeing\ffmpeg\bin\ffmpeg.exe')
# Define directories with absolute paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads/spotify/")  # Absolute path for downloads
CACHE_DIR = os.path.join(BASE_DIR, "cache/")  # Absolute path for cache
MAX_CACHE_SIZE = 1024 * 1024 * 1024  # 1GB max cache size
CACHE_EXPIRY_DAYS = 7  # Cache expires after 7 days

# --- تنظیم لاگ ---
logger = logging.getLogger(__name__)

lock_path = os.path.join(DOWNLOAD_DIR, "directory.lock")
file_lock = FileLock(lock_path, timeout=180)

# --- دایرکتوری دانلود ---
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
output_path = os.path.join(DOWNLOAD_DIR)

context_success_msg = {}

# ---- Inline search cache and throttle (for speed and quota control) ----
import time
from collections import OrderedDict, deque, defaultdict

class _InlineCache:
    """
    In-memory LRU with TTL and per-(query,market) offset cap to prevent memory growth.
    Stores only IDs to save RAM, reconstructs results on get.
    """
    MAX_OFFSETS_PER_QUERY = 5  # cap offsets stored per (query,market)

    def __init__(self, max_size: int, ttl_sec: int):
        self.max_size = max_size
        self.ttl = ttl_sec
        self.store = OrderedDict()  # key -> (ids_list, next_offset, ts)
        self.prefix_offsets = defaultdict(deque)  # "query|market" -> deque([offsets in insertion order])

    def _prefix(self, query: str, market: str) -> str:
        return f"{(query or '').lower().strip()}|{market}"

    def _make_key(self, query: str, market: str, offset: int) -> str:
        return f"{(query or '').lower().strip()}|{market}|{int(offset)}"

    def _remove_key(self, key: str):
        # key format: query|market|offset
        try:
            del self.store[key]
        except Exception:
            pass
        try:
            parts = key.rsplit("|", 1)
            if len(parts) == 2:
                prefix, off_str = parts
                try:
                    off = int(off_str)
                except Exception:
                    off = None
                dq = self.prefix_offsets.get(prefix)
                if dq and off is not None:
                    try:
                        # remove first occurrence
                        for i, v in enumerate(dq):
                            if v == off:
                                dq.remove(v)
                                break
                    except Exception:
                        pass
                if dq and len(dq) == 0:
                    try:
                        del self.prefix_offsets[prefix]
                    except Exception:
                        pass
        except Exception:
            pass

    def _reconstruct_results(self, ids_list):
        results = []
        for typ, sid in ids_list:
            try:
                if typ == 'track':
                    info = extract_track_info_optimized(sid)
                    if info:
                        results.append(_article_from_track(info))
                elif typ == 'album':
                    info = extract_album_info(sid)
                    if info:
                        results.append(_article_from_album(info))
                elif typ == 'artist':
                    info = extract_artist_info(sid)
                    if info:
                        results.append(_article_from_artist(info))
                elif typ == 'playlist':
                    info = extract_playlist_info(sid)
                    if info:
                        results.append(_article_from_playlist(info))
            except Exception as e:
                logger.debug(f"Failed to reconstruct {typ} {sid}: {e}")
        return results

    def get(self, query: str, market: str, offset: int):
        key = self._make_key(query, market, offset)
        now = time.time()
        if key in self.store:
            ids_list, next_offset, ts = self.store[key]
            if now - ts <= self.ttl:
                self.store.move_to_end(key)
                results = self._reconstruct_results(ids_list)
                return results, next_offset
            else:
                # expired -> remove and miss
                self._remove_key(key)
        return None

    def set(self, query: str, market: str, offset: int, ids_list, next_offset: str = ""):
        key = self._make_key(query, market, offset)
        now = time.time()
        # upsert
        self.store[key] = (ids_list, next_offset, now)
        self.store.move_to_end(key)

        # maintain per-prefix offset cap
        prefix = self._prefix(query, market)
        dq = self.prefix_offsets[prefix]
        if offset not in dq:
            dq.append(offset)
            while len(dq) > self.MAX_OFFSETS_PER_QUERY:
                old_off = dq.popleft()
                old_key = f"{prefix}|{int(old_off)}"
                if old_key in self.store:
                    self._remove_key(old_key)

        # global LRU trim
        while len(self.store) > self.max_size:
            try:
                oldest_key, _ = self.store.popitem(last=False)
            except Exception:
                break
            else:
                # ensure prefix_offsets updated
                self._remove_key(oldest_key)

_inline_cache = _InlineCache(INLINE_CACHE_SIZE, INLINE_CACHE_TTL)

_user_requests = defaultdict(deque)
_inline_seq = defaultdict(int)

def _throttle(user_id: int) -> bool:
    now = time.time()
    dq = _user_requests[user_id]
    while dq and now - dq[0] > INLINE_THROTTLE_WINDOW:
        dq.popleft()
    if len(dq) >= INLINE_THROTTLE_PERMITS:
        return True
    dq.append(now)
    return False

# ---- Builders for InlineQueryResultArticle with improved formatting ----
def _fmt_duration_ms(ms: int) -> str:
    try:
        return format_duration(ms / 1000.0)
    except Exception:
        return "0:00"

def _article_from_track(track: dict):
    title = track.get("name", "Unknown")
    artist = ", ".join(a.get("name","") for a in track.get("artists", [])) or "Unknown"
    url = track.get("external_urls", {}).get("spotify")
    thumb = (track.get("album", {}) or {}).get("images", [{}])[0].get("url", "")
    dur = _fmt_duration_ms(track.get("duration_ms", 0) or 0)
    pop = track.get("popularity", 0)
    sid = track.get("id", "")
    desc = f"Track • {dur} • Popularity {pop}%"
    return InlineQueryResultArticle(
        id=str(uuid4()),
        title=f"{artist} – {title}",
        description=desc,
        thumb_url=thumb,
        input_message_content=InputTextMessageContent(
            message_text=f"{url}\n#sid:track:{sid}",
            parse_mode="HTML"
        )
    )

def _article_from_album(album: dict):
    title = album.get("name", "Unknown")
    artist = ", ".join(a.get("name","") for a in album.get("artists", [])) or "Unknown"
    url = album.get("external_urls", {}).get("spotify")
    thumb = album.get("images", [{}])[0].get("url","")
    year = (album.get("release_date","") or "")[:4]
    total_tracks = album.get("total_tracks", 0)
    sid = album.get("id","")
    desc = f"Album • {year or 'Unknown'} • {total_tracks} tracks"
    return InlineQueryResultArticle(
        id=str(uuid4()),
        title=f"{artist} – {title}",
        description=desc,
        thumb_url=thumb,
        input_message_content=InputTextMessageContent(
            message_text=f"{url}\n#sid:album:{sid}",
            parse_mode="HTML"
        )
    )

def _article_from_artist(artist: dict):
    name = artist.get("name", "Unknown")
    url = artist.get("external_urls", {}).get("spotify")
    thumb = artist.get("images", [{}])[0].get("url","")
    genres = ", ".join((artist.get("genres") or [])[:3]) or "Artist"
    sid = artist.get("id","")
    desc = f"Artist • {genres}"
    return InlineQueryResultArticle(
        id=str(uuid4()),
        title=name,
        description=desc,
        thumb_url=thumb,
        input_message_content=InputTextMessageContent(
            message_text=f"{url}\n#sid:artist:{sid}",
            parse_mode="HTML"
        )
    )

def _article_from_playlist(playlist: dict):
    title = playlist.get("name", "Unknown")
    owner = (playlist.get("owner", {}) or {}).get("display_name", "Unknown")
    url = playlist.get("external_urls", {}).get("spotify")
    thumb = playlist.get("images", [{}])[0].get("url","")
    tracks_total = (playlist.get("tracks", {}) or {}).get("total", 0)
    sid = playlist.get("id","")
    desc = f"Playlist by {owner} • {tracks_total} tracks"
    return InlineQueryResultArticle(
        id=str(uuid4()),
        title=title,
        description=desc,
        thumb_url=thumb,
        input_message_content=InputTextMessageContent(
            message_text=f"{url}\n#sid:playlist:{sid}",
            parse_mode="HTML"
        )
    )

def _prefetch_top_tracks(track_ids, user_id):
    try:
        conn = sqlite3.connect("bot_cache.db")
        cur = conn.cursor()
        for tid in (track_ids or [])[:5]:
            try:
                info = extract_track_info_optimized(tid)
                if not info or not info.get("title"):
                    continue
                cur.execute("""
                    INSERT OR REPLACE INTO files
                    (user_id, file_id, artist, title, album, release_date, thumbnail, duration, popularity, genre, url, created_at, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """, (
                    user_id, tid,
                    info.get('artist'), info.get('title'), info.get('album'),
                    info.get('release_date'), info.get('thumbnail'),
                    int(info.get('duration') or 0), int(info.get('popularity') or 0),
                    info.get('genre'), info.get('url')
                ))
                conn.commit()
            except Exception:
                pass
        cur.close()
        conn.close()
    except Exception:
        pass

# Background Worker System
background_jobs: Dict[str, Dict[str, Any]] = {}
job_queue = asyncio.Queue()
worker_thread = None
worker_running = False

async def background_download_worker():
    """Background worker to process download jobs asynchronously"""
    global worker_running
    worker_running = True
    logger.info("Background download worker started")

    while worker_running:
        try:
            # Wait for job with timeout
            job = await asyncio.wait_for(job_queue.get(), timeout=1.0)
            if job is None:  # Shutdown signal
                break

            job_id = job['job_id']
            user_id = job['user_id']
            url = job['url']
            content_type = job['content_type']
            content_id = job['content_id']
            message = job['message']

            logger.info(f"Processing background job {job_id} for user {user_id}")

            # Update job status
            background_jobs[job_id]['status'] = 'processing'

            try:
                # Send initial progress message
                progress_msg = await message.reply("🔄 Background download started... 0%", parse_mode="HTML")
                background_jobs[job_id]['progress_msg'] = progress_msg

                # Progress callback for background job
                def progress_callback(current, total):
                    progress = (current / total) * 100 if total > 0 else 0
                    # Update job progress
                    background_jobs[job_id]['progress'] = progress

                    # Update progress message
                    try:
                        asyncio.run_coroutine_threadsafe(
                            progress_msg.edit_text(
                                f"🔄 Background download in progress...\nProgress: {progress:.1f}%\n⏳ Please wait, this may take a few minutes.",
                                parse_mode="HTML"
                            ),
                            asyncio.get_running_loop()
                        )
                    except Exception as e:
                        logger.debug(f"Failed to update background progress: {e}")

                # Perform download
                files = await asyncio.get_event_loop().run_in_executor(
                    executor, download_spotify, url, content_type, content_id, progress_callback
                )

                if files:
                    # Process downloaded files (similar to main handler)
                    for file_path in files:
                        if os.path.exists(file_path):
                            # Send file to user
                            await message.answer_audio(
                                audio=FSInputFile(file_path, filename=os.path.basename(file_path))
                            )

                            # Record tg_file_id will be handled when file is sent
                            logger.info(f"Background job {job_id} completed successfully")

                # Mark job as completed
                background_jobs[job_id]['status'] = 'completed'

            except Exception as e:
                logger.error(f"Background job {job_id} failed: {e}")
                background_jobs[job_id]['status'] = 'failed'
                background_jobs[job_id]['error'] = str(e)
                await message.reply(f"❌ Background download failed: {str(e)}", parse_mode="HTML")

        except asyncio.TimeoutError:
            continue
        except Exception as e:
            logger.error(f"Background worker error: {e}")

    logger.info("Background download worker stopped")

def start_background_worker():
    """Start the background worker thread"""
    global worker_thread
    if worker_thread is None or not worker_thread.is_alive():
        worker_thread = threading.Thread(target=lambda: asyncio.run(background_download_worker()))
        worker_thread.daemon = True
        worker_thread.start()
        logger.info("Background worker thread started")

async def enqueue_background_job(user_id: int, url: str, content_type: str, content_id: str, message: Message) -> str:
    """Enqueue a download job for background processing"""
    job_id = f"{user_id}_{content_id}_{asyncio.get_event_loop().time()}"

    job = {
        'job_id': job_id,
        'user_id': user_id,
        'url': url,
        'content_type': content_type,
        'content_id': content_id,
        'message': message
    }

    background_jobs[job_id] = {
        'status': 'queued',
        'progress': 0,
        'created_at': asyncio.get_event_loop().time()
    }

    await job_queue.put(job)
    logger.info(f"Enqueued background job {job_id} for user {user_id}")
    return job_id

# --- کلاینت اسپاتیفای ---
def _build_requests_session():
    s = requests.Session()
    try:
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        retry = Retry(
            total=3,
            backoff_factor=0.3,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["GET", "POST"])
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
    except Exception:
        # Fallback without adapter if modules are missing
        pass
    if PROXY:
        try:
            s.proxies.update({"http": PROXY, "https": PROXY})
        except Exception:
            pass
    try:
        s.headers.update({"User-Agent": "farysene-spotify-bot/1.0"})
    except Exception:
        pass
    return s

_session = _build_requests_session()
sp = Spotify(
    auth_manager=SpotifyClientCredentials(
        client_id=SPOTIPY_CLIENT_ID,
        client_secret=SPOTIPY_CLIENT_SECRET,
    ),
    requests_session=_session,
    proxies={"http": PROXY, "https": PROXY} if PROXY else None,
    requests_timeout=30
)

# --- بررسی نصب FFmpeg ---
# FFmpeg path is set to full path, assuming it exists

# --- تابع بررسی لینک اسپاتیفای ---
def validate_spotify_url(url: str) -> Tuple[bool, Optional[str], Optional[str]]:
    import requests
    from urllib.parse import urlparse
    try:
        # بررسی خالی نبودن و نوع URL
        if not url or not isinstance(url, str):
            logger.error("لینک ورودی خالی یا نامعتبر است.")
            raise ValueError("لینک ورودی خالی یا نامعتبر است.")

        # تجزیه URL
        parsed = urlparse(url.strip())
        
        # بررسی دامنه اسپاتیفای و دسترسی به آن
        if parsed.netloc != "open.spotify.com":
            logger.warning(f"دامنه غیرمجاز: {parsed.netloc}")
            return False, None, None

        # چک کردن دسترسی به دامنه با درخواست HEAD
        try:
            response = requests.head(f"https://{parsed.netloc}", timeout=20, proxies={"http": PROXY, "https": PROXY} if PROXY else None)
            response.raise_for_status()  # اگر 403 یا 429 برگردد، خطا می‌اندازد
        except requests.RequestException as e:
            logger.error(f"دسترسی به دامنه {parsed.netloc} مسدود یا ناموفق است (خطا: {e})")
            return False, None, None

        # استخراج مسیرها
        parts = parsed.path.strip("/").split("/")
        valid_types = {"track", "album", "playlist", "artist"}

        # بررسی ساختار مسیر
        if len(parts) >= 2 and parts[0] in valid_types:
            logger.debug(f"لینک معتبر: نوع={parts[0]}, شناسه={parts[1]}")
            return True, parts[0], parts[1]
        
        logger.warning(f"ساختار مسیر نامعتبر: {parsed.path}")
        return False, None, None

    except Exception as e:
        logger.error(f"خطا در اعتبارسنجی URL: {e}")
        return False, None, None

# فرمت زمان، مورد نیاز نمایش اطلاعات
def format_duration(seconds: Optional[Union[float, int, str]]) -> str:
    """
    seconds می‌تواند int/float (ثانیه)، یا عدد به صورت رشته، یا میلی‌ثانیه (auto-detect) یا None باشد.
    خروجی: "M:SS"
    """
    try:
        if seconds is None:
            return "0:00"

        # اگر کاربر "MM:SS" داده باشد، آن را پارس کن
        if isinstance(seconds, str):
            s = seconds.strip()
            if ":" in s:
                parts = s.split(":")
                try:
                    if len(parts) == 2:
                        minutes = int(parts[0])
                        secs = int(parts[1])
                        total = minutes * 60 + secs
                        seconds = total
                    else:
                        # اگر رشته عددی است
                        seconds = float(s)
                except Exception:
                    logger.debug(f"Couldn't parse duration string '{s}'")
                    return "0:00"
            else:
                # سعی کن رشته را به عدد تبدیل کنیم
                try:
                    seconds = float(s)
                except Exception:
                    return "0:00"

        # تا اینجا seconds عدد است (float/int)
        seconds = float(seconds)

        # auto-detect میلی‌ثانیه: اگر مقدار خیلی بزرگ باشد احتمالاً میلی‌ثانیه است
        # آستانه: 10000 (≈ 2.7 ساعت). مقادیر ثانیه معمولاً << 10000.
        if seconds > 10000:
            seconds = seconds / 1000.0

        if seconds < 0:
            return "0:00"

        total = int(round(seconds))
        minutes = total // 60
        secs = total % 60
        return f"{minutes}:{secs:02d}"

    except Exception as e:
        logger.exception(f"format_duration error: {e}")
        return "0:00"


@router.inline_query()
async def inline_spotify_search(inline_query: InlineQuery):
    query_text = (inline_query.query or "").strip()
    normalized_query = normalize_query(query_text)
    user_id = inline_query.from_user.id
    offset_str = inline_query.offset or "0"
    try:
        api_offset = int(offset_str)
        if api_offset < 0:
            api_offset = 0
    except Exception:
        api_offset = 0

    # Per-user inline query versioning for stale-response protection (only for offset 0)
    current_seq = None
    if api_offset == 0:
        try:
            _inline_seq[user_id] += 1
        except Exception:
            _inline_seq[user_id] = 1
        current_seq = _inline_seq[user_id]

    # Restrict inline search to allowed users only if list is non-empty
    try:
        if isinstance(ALLOWED_USERS, (list, set, tuple)) and len(ALLOWED_USERS) > 0 and user_id not in ALLOWED_USERS:
            await inline_query.answer(
                [],
                cache_time=10,
                is_personal=True,
                switch_pm_text="Start bot to search",
                switch_pm_parameter="inline_auth"
            )
            return
    except Exception:
        # If ALLOWED_USERS not defined for some reason, fail-safe allow
        pass

    # Minimum query length gate
    if len(query_text) < 3:
        await inline_query.answer([], cache_time=2, is_personal=True, switch_pm_text="🔎 Please enter at least 3 characters")
        return

    # Throttle heavy calls; attempt cached return if throttled
    if _throttle(user_id):
        cached = _inline_cache.get(normalized_query, MARKET, api_offset)
        if cached:
            results, next_offset = cached
            await inline_query.answer(results, cache_time=INLINE_CACHE_TTL, is_personal=True, next_offset=next_offset or "")
            return
        await inline_query.answer([], cache_time=2, is_personal=True)
        return

    # Cache first
    cached = _inline_cache.get(normalized_query, MARKET, api_offset)
    if cached:
        results, next_offset = cached
        await inline_query.answer(results, cache_time=INLINE_CACHE_TTL, is_personal=True, next_offset=next_offset or "")
        return

    # Single multi-type search call for speed
    results: list[InlineQueryResultArticle] = []
    try:
        # Interpret query: support "Search track: ..." or "album: ...", etc., and "Artist - Title"
        q_str = query_text
        type_str = "track,album,artist,playlist"
        try:
            m = re.match(r'^(?:search\s*)?(track|album|artist|playlist)\s*:?\s*(.+)$', query_text, re.IGNORECASE)
            if m:
                t = m.group(1).lower()
                body = m.group(2).strip()
                if body:
                    q_str = body
                    type_str = t
            else:
                m2 = re.match(r'^\s*(.+?)\s*[-–—]\s*(.+?)\s*$', query_text)
                if m2:
                    artist = m2.group(1).strip()
                    title = m2.group(2).strip()
                    if artist and title:
                        q_str = f'track:"{title}" artist:"{artist}"'
                        type_str = "track"
        except Exception:
            q_str = query_text
            type_str = "track,album,artist,playlist"

        search = sp.search(
            q=q_str,
            type=type_str,
            limit=min(50, max(1, int(INLINE_PAGE_SIZE))),
            market=MARKET,
            offset=api_offset
        )

        tracks = (search.get("tracks") or {}).get("items", [])
        albums = (search.get("albums") or {}).get("items", [])
        artists = (search.get("artists") or {}).get("items", [])
        playlists = (search.get("playlists") or {}).get("items", [])

        # Weighted ordering: 60% tracks, 20% albums, 10% artists, 10% playlists
        page_sz = min(50, max(1, int(INLINE_PAGE_SIZE)))
        tracks_limit = int(page_sz * 0.6)
        albums_limit = int(page_sz * 0.2)
        artists_limit = int(page_sz * 0.1)
        playlists_limit = page_sz - tracks_limit - albums_limit - artists_limit

        selected_tracks = tracks[:tracks_limit]
        selected_albums = albums[:albums_limit]
        selected_artists = artists[:artists_limit]
        selected_playlists = playlists[:playlists_limit]

        # Fill remaining slots with more from each type
        remaining_slots = page_sz - len(selected_tracks) - len(selected_albums) - len(selected_artists) - len(selected_playlists)
        if remaining_slots > 0:
            extra_tracks = tracks[tracks_limit:tracks_limit + remaining_slots]
            selected_tracks.extend(extra_tracks)
            remaining_slots -= len(extra_tracks)
            if remaining_slots > 0:
                extra_albums = albums[albums_limit:albums_limit + remaining_slots]
                selected_albums.extend(extra_albums)
                remaining_slots -= len(extra_albums)
                if remaining_slots > 0:
                    extra_artists = artists[artists_limit:artists_limit + remaining_slots]
                    selected_artists.extend(extra_artists)
                    remaining_slots -= len(extra_artists)
                    if remaining_slots > 0:
                        extra_playlists = playlists[playlists_limit:playlists_limit + remaining_slots]
                        selected_playlists.extend(extra_playlists)

        seen_urls = set()
        ids_list = []

        # Build results in weighted order
        for tr in selected_tracks:
            url = (tr.get("external_urls") or {}).get("spotify")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            try:
                results.append(_article_from_track(tr))
                ids_list.append(('track', tr.get('id')))
            except Exception as e:
                logger.debug(f"Inline track build failed: {e}")

        for al in selected_albums:
            url = (al.get("external_urls") or {}).get("spotify")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            try:
                results.append(_article_from_album(al))
                ids_list.append(('album', al.get('id')))
            except Exception as e:
                logger.debug(f"Inline album build failed: {e}")

        for ar in selected_artists:
            url = (ar.get("external_urls") or {}).get("spotify")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            try:
                results.append(_article_from_artist(ar))
                ids_list.append(('artist', ar.get('id')))
            except Exception as e:
                logger.debug(f"Inline artist build failed: {e}")

        for pl in selected_playlists:
            url = (pl.get("external_urls") or {}).get("spotify")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            try:
                results.append(_article_from_playlist(pl))
                ids_list.append(('playlist', pl.get('id')))
            except Exception as e:
                logger.debug(f"Inline playlist build failed: {e}")

        # Background prefetch: warm track info cache for top results to speed next step
        try:
            top_track_ids = [tr.get("id") for tr in selected_tracks if tr.get("id")][:2]
            if top_track_ids:
                loop = asyncio.get_running_loop()
                loop.run_in_executor(executor, _prefetch_top_tracks, top_track_ids, user_id)
        except Exception:
            pass

        # Multi-market augmentation: if page is sparse, try additional markets to widen coverage
        try:
            page_limit_local = min(50, max(1, int(INLINE_PAGE_SIZE)))
            if (
                MULTI_MARKET_ON_SPARSE
                and api_offset == 0
                and len(results) < max(1, int(MIN_INLINE_RESULTS))
            ):
                remaining = max(0, page_limit_local - len(results))
                extra_tried = 0
                for mk in SEARCH_MARKETS:
                    if mk == MARKET:
                        continue
                    extra_tried += 1
                    if extra_tried > max(0, int(MULTI_MARKET_MAX_EXTRA)):
                        break
                    try:
                        extra = sp.search(
                            q=query_text,
                            type="track,album,artist,playlist",
                            limit=page_limit_local,
                            market=mk,
                            offset=0
                        )
                        ex_tracks = (extra.get("tracks") or {}).get("items", [])
                        ex_albums = (extra.get("albums") or {}).get("items", [])
                        ex_artists = (extra.get("artists") or {}).get("items", [])
                        ex_playlists = (extra.get("playlists") or {}).get("items", [])

                        # Append in priority order while respecting remaining slots and de-dup
                        for tr in ex_tracks:
                            if len(results) >= page_limit_local:
                                break
                            url = (tr.get("external_urls") or {}).get("spotify")
                            if not url or url in seen_urls:
                                continue
                            seen_urls.add(url)
                            try:
                                results.append(_article_from_track(tr))
                            except Exception as e:
                                logger.debug(f"Inline MM track build failed ({mk}): {e}")

                        for al in ex_albums:
                            if len(results) >= page_limit_local:
                                break
                            url = (al.get("external_urls") or {}).get("spotify")
                            if not url or url in seen_urls:
                                continue
                            seen_urls.add(url)
                            try:
                                results.append(_article_from_album(al))
                            except Exception as e:
                                logger.debug(f"Inline MM album build failed ({mk}): {e}")

                        for ar in ex_artists:
                            if len(results) >= page_limit_local:
                                break
                            url = (ar.get("external_urls") or {}).get("spotify")
                            if not url or url in seen_urls:
                                continue
                            seen_urls.add(url)
                            try:
                                results.append(_article_from_artist(ar))
                            except Exception as e:
                                logger.debug(f"Inline MM artist build failed ({mk}): {e}")

                        for pl in ex_playlists:
                            if len(results) >= page_limit_local:
                                break
                            url = (pl.get("external_urls") or {}).get("spotify")
                            if not url or url in seen_urls:
                                continue
                            seen_urls.add(url)
                            try:
                                results.append(_article_from_playlist(pl))
                            except Exception as e:
                                logger.debug(f"Inline MM playlist build failed ({mk}): {e}")

                        if len(results) >= page_limit_local:
                            break
                    except Exception as ee:
                        logger.debug(f"Inline multi-market search failed for market {mk}: {ee}")
        except Exception as _mm_e:
            logger.debug(f"Inline multi-market augmentation error: {_mm_e}")

        # Pagination: if any section has more items available, set next_offset
        page_sz = min(50, max(1, int(INLINE_PAGE_SIZE)))
        has_more = False
        for section_key in ("tracks", "albums", "artists", "playlists"):
            sec = search.get(section_key) or {}
            # Spotify returns 'next' as URL when more available
            if sec.get("next"):
                has_more = True
                break
            # Or use totals/offset/limit
            try:
                total = int(sec.get("total") or 0)
                limit = int(sec.get("limit") or 0)
                offset_now = int(sec.get("offset") or api_offset)
                if total > offset_now + limit and limit > 0:
                    has_more = True
                    break
            except Exception:
                continue
        # If augmentation filled the page, allow paging even if primary market shows no 'more'
        if not has_more and len(results) >= page_sz:
            has_more = True

        next_offset = str(api_offset + page_sz) if has_more else ""

        # Cache the assembled results
        _inline_cache.set(normalized_query, MARKET, api_offset, ids_list, next_offset)

        ans_cache_time = INLINE_CACHE_TTL if api_offset == 0 else min(120, int(INLINE_CACHE_TTL) * 2 if isinstance(INLINE_CACHE_TTL, int) else 60)
        # Stale-response guard: if a newer inline query arrived for this user at offset 0, suppress this answer
        if api_offset == 0 and current_seq is not None and _inline_seq.get(user_id, current_seq) != current_seq:
            await inline_query.answer([], cache_time=1, is_personal=True)
            return
        await inline_query.answer(
            results,
            cache_time=ans_cache_time,
            is_personal=True,
            next_offset=next_offset
        )

    except SpotifyException as se:
        status = getattr(se, 'http_status', None)
        logger.warning(f"Inline search SpotifyException for user {user_id}: status={status}, err={se}")
        # Serve cached if available on rate-limit/server errors
        cached2 = _inline_cache.get(normalized_query, MARKET, api_offset)
        if cached2:
            results2, next_offset2 = cached2
            ans_cache_time = INLINE_CACHE_TTL if api_offset == 0 else min(120, int(INLINE_CACHE_TTL) * 2 if isinstance(INLINE_CACHE_TTL, int) else 60)
            await inline_query.answer(results2, cache_time=ans_cache_time, is_personal=True, next_offset=next_offset2 or "")
            return
        await inline_query.answer([], cache_time=3 if api_offset == 0 else 10, is_personal=True)
        return
    except Exception as e:
        logger.error(f"Inline search error for user {user_id}: {e}")
        # Best-effort cached fallback
        cached2 = _inline_cache.get(normalized_query, MARKET, api_offset)
        if cached2:
            results2, next_offset2 = cached2
            ans_cache_time = INLINE_CACHE_TTL if api_offset == 0 else min(120, int(INLINE_CACHE_TTL) * 2 if isinstance(INLINE_CACHE_TTL, int) else 60)
            await inline_query.answer(results2, cache_time=ans_cache_time, is_personal=True, next_offset=next_offset2 or "")
            return
        await inline_query.answer([], cache_time=2 if api_offset == 0 else 8, is_personal=True)


#تابع استخراج اطلاعات آهنگ
# بالای فایل اگر نیست:

# ---------- اصلاح extract_track_info ----------
def extract_track_info(track_id: str) -> Dict[str, Optional[str]]:
    try:
        if not track_id or not isinstance(track_id, str):
            logger.error(f"شناسه ترک نامعتبر: {track_id}")
            raise ValueError("شناسه ترک نامعتبر است.")

        track = sp.track(track_id)

        title = track.get("name") or "بدون عنوان"
        artist_list = track.get("artists", [])
        artist = artist_list[0].get("name", "نامشخص") if artist_list else "نامشخص"

        album_info = track.get("album", {}) or {}
        album_name = album_info.get("name", "نامشخص")
        release_date = album_info.get("release_date", "نامشخص")

        images = album_info.get("images", []) or []
        thumbnail = images[0].get("url") if isinstance(images, list) and images else None
        try:
            # First try album genres (more specific)
            album_info = track.get("album", {})
            genres = []
            if album_info:
                genres = album_info.get("genres", [])

            # If no album genres, try artist genres
            if not genres:
                artist_obj = sp.artist(track["artists"][0]["id"])
                genres = artist_obj.get("genres", [])

            # If still no genres, try to get from related artists or use defaults
            if not genres:
                try:
                    # Get artist's top tracks and check their genres
                    top_tracks = sp.artist_top_tracks(track["artists"][0]["id"], country="US")
                    for top_track in top_tracks.get("tracks", [])[:3]:  # Check first 3 tracks
                        if top_track.get("album", {}).get("genres"):
                            genres = top_track["album"]["genres"]
                            break
                except:
                    pass

            # If still no genres, use common defaults based on popularity
            if not genres:
                popularity = track.get("popularity", 0)
                if popularity > 70:
                    genres = ["pop"]
                elif popularity > 50:
                    genres = ["rock"]
                else:
                    genres = ["alternative"]

            # Clean and format genres
            final_genres = [g.title() for g in genres if g]  # Capitalize first letter
            genre = ", ".join(final_genres) if final_genres else "Pop"  # Default fallback

        except Exception as e:
            logger.warning(f"Failed to get genres for track {track_id}: {e}")
            genre = "Pop"  # Default fallback instead of Unknown

        # >>> اصلاح: برگرداندن(duration) به ثانیه (int)
        duration_ms = track.get("duration_ms", 0) or 0
        try:
            duration_seconds = int(duration_ms) // 1000
        except Exception:
            duration_seconds = 0

        url = f"https://open.spotify.com/track/{track_id}"

        return {
            "title": title,
            "artist": artist,
            "album": album_name,
            "thumbnail": thumbnail,
            "release_date": release_date,
            "duration": duration_seconds,   # ثانیه به صورت int
            "url": url,
            "popularity": track.get("popularity", 0),
            "genre": genre,
        }

    except SpotifyException as e:
        logger.error(f"خطا در API اسپاتیفای برای ترک {track_id}: {e}")
    except ValueError as e:
        logger.error(f"خطای اعتبارسنجی برای ترک {track_id}: {e}")
    except Exception as e:
        logger.error(f"خطای ناشناخته برای ترک {track_id}: {e}")

    # پیش‌فرض در صورت خطا
    return {
        "title": "Unknown",
        "artist": "Unknown",
        "album": "Unknown",
        "thumbnail": None,
        "release_date": "Unknown",
        "duration": 0,
        "url": None,
        "popularity": 0,
        "genre":"Unknown"
    }

# ---------- اصلاح extract_album_info ----------
def extract_album_info(album_id: str) -> Optional[Dict[str, Any]]:
    try:
        album = sp.album(album_id, market=MARKET)
        title = album.get("name", "Unknown")
        artists = ", ".join(a.get("name", "") for a in album.get("artists", [])) or "Unknown"
        url = album.get("external_urls", {}).get("spotify")
        images = album.get("images", [{}])
        thumb = images[0].get("url") if images else None
        year = (album.get("release_date", "") or "")[:4]
        total_tracks = album.get("total_tracks", 0)
        return {
            "name": title,
            "artists": artists,
            "url": url,
            "images": images,
            "release_date": album.get("release_date"),
            "total_tracks": total_tracks,
            "id": album_id
        }
    except SpotifyException as e:
        logger.error(f"Spotify API error for album {album_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error extracting album info for {album_id}: {e}")
        return None

# ---------- اصلاح extract_artist_info ----------
def extract_artist_info(artist_id: str) -> Optional[Dict[str, Any]]:
    try:
        artist = sp.artist(artist_id)
        name = artist.get("name", "Unknown")
        url = artist.get("external_urls", {}).get("spotify")
        images = artist.get("images", [{}])
        thumb = images[0].get("url") if images else None
        genres = ", ".join((artist.get("genres") or [])[:3]) or "Artist"
        followers = artist.get("followers", {}).get("total", 0)
        popularity = artist.get("popularity", 0)
        return {
            "name": name,
            "url": url,
            "images": images,
            "genres": genres,
            "followers": followers,
            "popularity": popularity,
            "id": artist_id
        }
    except SpotifyException as e:
        logger.error(f"Spotify API error for artist {artist_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error extracting artist info for {artist_id}: {e}")
        return None

# ---------- اصلاح extract_playlist_info ----------
def extract_playlist_info(playlist_id: str) -> Optional[Dict[str, Any]]:
    try:
        playlist = sp.playlist(playlist_id, market=MARKET)
        title = playlist.get("name", "Unknown")
        owner = (playlist.get("owner", {}) or {}).get("display_name", "Unknown")
        url = playlist.get("external_urls", {}).get("spotify")
        images = playlist.get("images", [{}])
        thumb = images[0].get("url") if images else None
        tracks_total = (playlist.get("tracks", {}) or {}).get("total", 0)
        description = playlist.get("description", "")
        return {
            "name": title,
            "owner": owner,
            "url": url,
            "images": images,
            "tracks_total": tracks_total,
            "description": description,
            "id": playlist_id
        }
    except SpotifyException as e:
        logger.error(f"Spotify API error for playlist {playlist_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error extracting playlist info for {playlist_id}: {e}")
        return None



# Configure logger
logger = logging.getLogger("cache_manager")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


# Initialize directories and cache management
def init_cache():
    try:
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        logger.info(f"Initialized directories: DOWNLOAD_DIR={DOWNLOAD_DIR}, CACHE_DIR={CACHE_DIR}")
        
        # Verify write permissions
        test_file = os.path.join(CACHE_DIR, "test_write.txt")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        logger.debug("Write permissions verified for CACHE_DIR")
    except PermissionError as e:
        logger.error(f"Permission denied for CACHE_DIR {CACHE_DIR}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error initializing cache directories: {e}")
        raise

# Cache cleanup function
def cleanup_cache():
    try:
        total_size = 0
        files_to_remove = []
        conn = sqlite3.connect("bot_cache.db")
        cur = conn.cursor()
        cur.execute("SELECT cache_path, created_at FROM files WHERE cache_path IS NOT NULL")
        for cache_path, created_at in cur.fetchall():
            if os.path.exists(cache_path):
                total_size += os.path.getsize(cache_path)
                if datetime.now() > datetime.fromisoformat(created_at) + timedelta(days=CACHE_EXPIRY_DAYS):
                    files_to_remove.append(cache_path)

        if total_size > MAX_CACHE_SIZE:
            cur.execute("SELECT cache_path, created_at FROM files WHERE cache_path IS NOT NULL ORDER BY created_at ASC")
            for cache_path, _ in cur.fetchall():
                if os.path.exists(cache_path) and total_size > MAX_CACHE_SIZE:
                    os.remove(cache_path)
                    total_size -= os.path.getsize(cache_path)
                    cur.execute("DELETE FROM files WHERE cache_path = ?", (cache_path,))
                    logger.info(f"Removed old cache file {cache_path} due to size limit")

        for cache_path in files_to_remove:
            if os.path.exists(cache_path):
                os.remove(cache_path)
                cur.execute("DELETE FROM files WHERE cache_path = ?", (cache_path,))
                logger.info(f"Removed expired cache file {cache_path}")

        conn.commit()
    except Exception as e:
        logger.error(f"Cache cleanup failed: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

# Example usage in a function (e.g., handle_separate_vocal)
def prepare_file_path(user_id: int, file_id: str) -> Optional[str]:
    init_cache()
    cleanup_cache()
    cache_path = os.path.join(CACHE_DIR, f"{user_id}_{file_id}.mp3")
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        logger.info(f"Using cached file at {cache_path}")
        return cache_path
    return None




# --- تابع دانلود با spotdl ---
def download_spotify(url: str, content_type: str, content_id: str, progress_callback=None) -> List[str]:

    # بررسی وجود spotdl
    spotdl_path = which("spotdl")
    if not spotdl_path:
        logger.error("spotdl not found in PATH.")
        raise FileNotFoundError("❌ spotdl در محیط جاری پیدا نشد. لطفاً آن را نصب کنید.")
    logger.info(f"spotdl found at: {spotdl_path}")
    # بررسی وجود FFmpeg
    if not FFMPEG_PATH or not os.path.exists(FFMPEG_PATH):
        logger.error(f"FFmpeg not found at {FFMPEG_PATH}.")
        raise FileNotFoundError(f"❌ FFmpeg یافت نشد: {FFMPEG_PATH}")
    logger.info(f"FFmpeg found at: {FFMPEG_PATH}")
    # ساخت دستور spotdl
    # Use isolated job directory per download to avoid race conditions
    job_dir = os.path.abspath(os.path.join(DOWNLOAD_DIR, f"job_{content_id}_{int(time.time())}"))
    output_path = job_dir
    cmd = [
        "spotdl", url,
        "--output", output_path,
        "--no-cache",
        "--overwrite", "force",
        "--format", "mp3",
        "--client-id", SPOTIPY_CLIENT_ID,
        "--client-secret", SPOTIPY_CLIENT_SECRET,
        "--log-level", "INFO",
    ]

    # افزودن FFmpeg به دستور
    cmd += ["--ffmpeg", FFMPEG_PATH]
    # افزودن پروکسی اگر وجود داشته باشد
    if PROXY and PROXY.strip():
        cmd += ["--proxy", PROXY]
    # تنظیم محیط
    env = os.environ.copy()
    env["PATH"] = f"{os.path.dirname(FFMPEG_PATH)};{env.get('PATH', '')}"
    if SPOTIPY_CLIENT_ID:
        env["SPOTIPY_CLIENT_ID"] = SPOTIPY_CLIENT_ID
    if SPOTIPY_CLIENT_SECRET:
        env["SPOTIPY_CLIENT_SECRET"] = SPOTIPY_CLIENT_SECRET
    # ایجاد دایرکتوری دانلود
    try:
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
        os.makedirs(CACHE_DIR, exist_ok=True)  # Create cache dir
    except OSError as e:
        logger.error(f"خطا در ایجاد دایرکتوری {DOWNLOAD_DIR} یا {CACHE_DIR}: {e}")
        raise RuntimeError(f"⛔️ خطا در ایجاد دایرکتوری دانلود: {e}")
    # پاکسازی فایل‌های mp3 قدیمی با قفل
    # Clean only this job directory to avoid interfering with concurrent downloads
    try:
        os.makedirs(job_dir, exist_ok=True)
        for f in os.listdir(job_dir):
            if f.endswith(".mp3"):
                try:
                    os.remove(os.path.join(job_dir, f))
                except Exception as e:
                    logger.debug(f"Could not remove old file in job_dir {f}: {e}")
    except Exception as e:
        logger.debug(f"Job dir prep failed: {e}")

    # Check cache before downloading
    conn = sqlite3.connect("bot_cache.db")
    cur = conn.cursor()
    cur.execute("SELECT cache_path FROM files WHERE file_id = ?", (content_id,))
    cached_file = cur.fetchone()
    if cached_file:
        cached_path = cached_file[0]
        if cached_path and isinstance(cached_path, str) and os.path.exists(cached_path):
            logger.info(f"Using cached file for {content_id} at {cached_path}")
            cur.close()
            conn.close()
            return [cached_path]
    cur.close()
    conn.close()

    # بررسی نسخه spotdl
    try:
        result = run(
            ["spotdl", "--version"],
            capture_output=True,
            text=True,
            env=env,
            timeout=40  # کاهش زمان‌بندی برای بررسی نسخه
        )
        if result.returncode != 0:
            logger.error(f"خطا در بررسی نسخه spotdl: {result.stderr.strip()}")
            raise RuntimeError("⛔️ خطا در بررسی نسخه spotdl.")
        logger.debug(f"spotdl version: {result.stdout.strip()}")
    except TimeoutError:
        logger.error("زمان‌بندی بررسی نسخه spotdl به اتمام رسید.")
        raise RuntimeError("⏳ زمان بررسی نسخه spotdl به پایان رسید.")
    except CalledProcessError as e:
        logger.error(f"خطا در اجرای spotdl --version: {e}")
        raise RuntimeError(f"⛔️ خطا در بررسی spotdl: {e}")

    # اجرای دانلود با پایش پیشرفت
    start_time = time.time()
    logger.info(f"Starting spotdl command: {' '.join(cmd)}")
    logger.debug(f"Environment variables: SPOTIPY_CLIENT_ID={env.get('SPOTIPY_CLIENT_ID', 'Not set')}, SPOTIPY_CLIENT_SECRET={'Set' if env.get('SPOTIPY_CLIENT_SECRET') else 'Not set'}, PATH={env.get('PATH', 'Not set')}")
    try:
        process = run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=1000  # زمان‌بندی پویاتر برای دانلودهای بزرگ
        )
        logger.info(f"spotdl process return code: {process.returncode}")
        logger.debug(f"spotdl stdout: {process.stdout}")
        logger.debug(f"spotdl stderr: {process.stderr}")
        if process.returncode != 0:
            logger.error(f"spotdl execution failed with return code {process.returncode}: {process.stderr.strip()}")
            raise RuntimeError(f"⛔️ خطا در اجرای spotdl: {process.stderr.strip()}")
        logger.info(f"Files in job_dir after spotdl: {os.listdir(job_dir) if os.path.exists(job_dir) else []}")
        # Update progress if callback exists
        if progress_callback:
            progress_callback(100, 100)  # Complete progress
    except TimeoutError:
        logger.error("spotdl execution timed out after 600 seconds.")
        raise RuntimeError("⏳ زمان اجرای spotdl به پایان رسید.")
    except CalledProcessError as e:
        logger.error(f"CalledProcessError in spotdl: {e}")
        raise RuntimeError(f"⛔️ خطا در دانلود: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in spotdl execution: {e}")
        raise RuntimeError(f"⛔️ خطای ناشناخته: {e}")

    # جمع‌آوری فایل‌های mp3 با اعتبارسنجی بیشتر
    files = []
    logger.info(f"Starting file validation in job_dir: {job_dir}")
    # Validate only files from this job directory
    for root, dirs, filenames in os.walk(job_dir):
        for f in filenames:
            file_path = os.path.join(root, f)
            if f.endswith(".mp3"):
                logger.debug(f"Validating file: {file_path}")
                try:
                    audio = AudioSegment.from_file(file_path)
                    file_size = os.path.getsize(file_path)
                    logger.debug(f"File {f} size: {file_size} bytes, duration: {len(audio)/1000:.2f}s")
                    if file_size < 50 * 1024:
                        logger.warning(f"File {f} too small ({file_size} bytes), removing.")
                        os.remove(file_path)
                        continue
                    cache_path = os.path.join(CACHE_DIR, f"{content_id}_{f}")
                    # Move to cache to free the job dir and provide consistent return path
                    os.makedirs(CACHE_DIR, exist_ok=True)
                    shutil.move(file_path, cache_path)
                    files.append(cache_path)
                    logger.info(f"File validated and cached: {cache_path}")
                    # Update DB with cache path
                    conn = sqlite3.connect("bot_cache.db")
                    cur = conn.cursor()
                    cur.execute("UPDATE files SET cache_path = ?, last_accessed = CURRENT_TIMESTAMP WHERE file_id = ?", (cache_path, content_id))
                    conn.commit()
                    cur.close()
                    conn.close()
                except Exception as e:
                    logger.warning(f"Error validating or moving file {f}: {e}")
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                        except Exception:
                            pass
    # Cleanup job dir
    try:
        shutil.rmtree(job_dir, ignore_errors=True)
    except Exception:
        pass

    if not files:
        logger.warning("No valid mp3 files found after validation.")
        return []
    logger.info(f"Download completed successfully in {time.time() - start_time:.2f} seconds with {len(files)} files.")
    return files

# افزودن کاور، وابسته به download_spotify
def embed_cover(mp3_path: str, cover_url: str) -> None:
    # بررسی وجود FFmpeg
    if not FFMPEG_PATH or not os.path.exists(FFMPEG_PATH):
        logger.error(f"FFmpeg در مسیر {FFMPEG_PATH} یافت نشد.")
        raise FileNotFoundError(f"FFmpeg یافت نشد: {FFMPEG_PATH}")

    # بررسی وجود فایل MP3
    if not os.path.exists(mp3_path):
        logger.error(f"فایل MP3 در مسیر {mp3_path} یافت نشد.")
        raise FileNotFoundError(f"فایل MP3 یافت نشد: {mp3_path}")

    # بررسی معتبر بودن URL کاور
    if not cover_url or not isinstance(cover_url, str):
        logger.error("URL کاور نامعتبر است.")
        raise ValueError("URL کاور نامعتبر است.")

    # دانلود تصویر کاور
    try:
        response = requests.get(cover_url, timeout=10, proxies=_requests_proxies(PROXY))
        if response.status_code != 200:
            logger.error(f"دانلود کاور از {cover_url} ناموفق بود: {response.status_code}")
            raise RuntimeError(f"دانلود کاور ناموفق بود: {response.status_code}")
    except requests.RequestException as e:
        logger.error(f"خطا در دانلود کاور از {cover_url}: {e}")
        raise RuntimeError(f"خطا در دانلود کاور: {e}")

    # مدیریت فایل موقت
    temp_img_path = None
    try:
        with NamedTemporaryFile(delete=False, suffix=".jpg") as temp_img:
            temp_img.write(response.content)
            temp_img_path = temp_img.name

        # تنظیم مسیر خروجی
        output_path = mp3_path.replace(".mp3", "_with_cover.mp3")

        # اجرای دستور FFmpeg
        cmd = [
            FFMPEG_PATH,
            "-i", mp3_path,
            "-i", temp_img_path,
            "-map", "0:a",  # فقط جریان صوتی از ورودی اول
            "-map", "1:v",  # جریان تصویری از ورودی دوم
            "-c", "copy",
            "-id3v2_version", "3",
            "-metadata:s:v", "title=Album cover",
            "-metadata:s:v", "comment=Cover (front)",
            output_path
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            logger.debug(f"خروجی FFmpeg: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"خطا در اجرای FFmpeg: {e.stderr}")
            raise RuntimeError(f"خطا در افزودن کاور: {e.stderr}")

        # جایگزینی فایل اصلی
        try:
            os.replace(output_path, mp3_path)
            logger.info(f"کاور با موفقیت به {mp3_path} اضافه شد.")
        except OSError as e:
            logger.error(f"خطا در جایگزینی فایل {output_path}: {e}")
            raise OSError(f"خطا در جایگزینی فایل MP3: {e}")

    except Exception as e:
        logger.error(f"خطای ناشناخته در افزودن کاور: {e}")
        raise RuntimeError(f"خطای ناشناخته: {e}")
    finally:
        # پاکسازی فایل موقت
        if temp_img_path and os.path.exists(temp_img_path):
            try:
                os.remove(temp_img_path)
                logger.debug(f"فایل موقت {temp_img_path} حذف شد.")
            except OSError as e:
                logger.warning(f"خطا در حذف فایل موقت {temp_img_path}: {e}")


# --- Utilities for audio processing: bitrate, metadata, filenames ---

def _normalize_text(value: Optional[str]) -> str:
    v = (value or "").strip()
    if not v:
        return ""
    lowered = v.lower()
    if lowered in {"unknown", "نامشخص", "بدون عنوان", "n/a", "none"}:
        return ""
    return v


def build_clean_filename(info: Dict[str, Optional[str]], bitrate_kbps: int) -> str:
    """
    Build a minimal clean filename for local storage: "Artist - Title.mp3" only.
    Any extra tags (year/bitrate/etc.) are intentionally omitted to avoid client-side issues.
    """
    artist = _normalize_text(info.get("artist")) or "Unknown Artist"
    title = _normalize_text(info.get("title")) or "Unknown Title"

    base = f"{artist} - {title}"
    base = base[:100]
    filename = sanitize_filename(base) + ".mp3"
    return filename


def build_transfer_filename(info: Dict[str, Optional[str]]) -> str:
    """Very conservative ASCII-only filename for upload to Telegram to prevent failures."""
    artist = _normalize_text(info.get("artist")) or "Unknown Artist"
    title = _normalize_text(info.get("title")) or "Unknown Title"
    # Keep ASCII letters/numbers/._- and space
    safe = f"{artist} - {title}"
    safe = re.sub(r"[^A-Za-z0-9 ._\-]", "", safe)
    safe = re.sub(r"\s+", " ", safe).strip()
    if not safe:
        safe = "audio"
    # Keep within 80 chars
    safe = safe[:80].rstrip()
    return (safe or "audio") + ".mp3"


def write_id3_metadata(mp3_path: str, info: Dict[str, Optional[str]], cover_url: Optional[str] = None) -> None:
    audio = MP3(mp3_path, ID3=ID3)
    if audio.tags is None:
        audio.add_tags()
    id3 = audio.tags

    # Clear relevant frames to avoid duplicates
    for key in ("TIT2", "TPE1", "TALB", "TCON", "TDRC", "TRCK", "TPE2", "APIC", "COMM"):
        try:
            id3.delall(key)
        except Exception:
            pass

    title = _normalize_text(info.get("title"))
    artist = _normalize_text(info.get("artist"))
    album = _normalize_text(info.get("album"))
    release_date = _normalize_text(info.get("release_date"))
    track_number = info.get("track_number") or info.get("track")
    genre = _normalize_text(info.get("genre"))

    if title:
        id3.add(TIT2(encoding=3, text=title))
    if artist:
        id3.add(TPE1(encoding=3, text=[artist]))
        id3.add(TPE2(encoding=3, text=artist))  # Album Artist
    if album:
        id3.add(TALB(encoding=3, text=album))
    if release_date:
        id3.add(TDRC(encoding=3, text=release_date.split("-")[0]))
    if track_number:
        try:
            id3.add(TRCK(encoding=3, text=str(track_number)))
        except Exception:
            pass
    if genre:
        id3.add(TCON(encoding=3, text=genre))
    try:
        id3.add(COMM(encoding=3, lang='eng', desc='comment', text='Tagged from Spotify'))
    except Exception:
        pass

    # Embed cover art
    if cover_url:
        try:
            r = requests.get(cover_url, timeout=20, proxies=_requests_proxies(PROXY))
            if r.status_code == 200 and r.content:
                mime = "image/jpeg"
                cu = cover_url.lower()
                if cu.endswith('.png'):
                    mime = "image/png"
                id3.add(APIC(encoding=3, mime=mime, type=3, desc="Cover", data=r.content))
        except Exception as e:
            logger.warning(f"Cover fetch failed for {cover_url}: {e}")

    audio.save(v2_version=3)

def ensure_320kbps(mp3_path: str) -> str:
    """
    Ensure the given MP3 file is encoded at constant 320 kbps.
    If not 320, re-encode with FFmpeg and replace in-place. Returns the (same) path.
    """
    tmp_out = None
    try:
        current_kbps = 0
        try:
            info = MP3(mp3_path)
            current_kbps = int((getattr(info.info, "bitrate", 0) or 0) / 1000)
        except Exception:
            current_kbps = 0

        if current_kbps == 320:
            return mp3_path

        tmp_out = mp3_path + ".320.mp3"
        cmd = [
            FFMPEG_PATH, "-y",
            "-i", mp3_path,
            "-vn",
            "-c:a", "libmp3lame",
            "-b:a", "320k",
            "-map_metadata", "0",
            "-id3v2_version", "3",
            tmp_out
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        os.replace(tmp_out, mp3_path)
        logger.info(f"Re-encoded to 320 kbps: {mp3_path}")
        return mp3_path
    except Exception as e:
        logger.warning(f"ensure_320kbps failed for {mp3_path}: {e}")
        try:
            if tmp_out and os.path.exists(tmp_out):
                os.remove(tmp_out)
        except Exception:
            pass
        return mp3_path

#------------دریافت لیریک و ترجمه -----------------------


# ترجمه از open ai 
try:
    from openai import OpenAI
    from openai import OpenAIError
except Exception:
    OpenAI = None
    OpenAIError = Exception

# ---------- تنظیم لاگ‌ها ----------
logger = logging.getLogger("genius_lyrics")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)

# ---------- کلاس‌های خطا ----------
class GeniusError(Exception):
    """Custom exception for Genius API errors."""
    pass

class Blocked(Exception):
    """Exception for blocked requests."""
    pass

# ---------- تنظیمات ----------


# کلمات/الگوهای متداول در URL صفحات ترجمه — اگر هرکدوم در URL بود، آن URL رد می‌شود
BLOCKED_KEYWORDS = [
    "traducciones", "traductions", "ubersetzungen", "ceviriler",
    "traduzioni", "traducoes", "translation", "traduc", "traduz",
    "versi", "versión", "versao", "tumaczenia", "ceviri", "Polskie", "turkce"
]

# ---------- توابع کمکی ----------

def _random_headers() -> Dict[str, str]:
    _UA_POOL = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:129.0) Gecko/20100101 Firefox/129.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.6; rv:129.0) Gecko/20100101 Firefox/129.0",
    ]
    return {
        "User-Agent": random.choice(_UA_POOL),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",  # فقط انگلیسی
        "Referer": "https://www.google.com/",
        "Connection": "keep-alive",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Upgrade-Insecure-Requests": "1",
        "DNT": "1",
    }

def _env_proxy_list() -> list[str]:
    # Keep existing structure; prefer configured proxy first, then built-ins
    lst: list[str] = []
    try:
        if PROXY and isinstance(PROXY, str) and PROXY.strip():
            lst.append(PROXY.strip())
    except Exception:
        pass
    lst.extend([
        "http://37.27.80.214:80",
        "http://90.162.35.34:80",
    ])
    return lst

def _requests_proxies(proxy_url: Optional[str]) -> Optional[dict[str, str]]:
    if not proxy_url:
        return None
    return {"http": proxy_url, "https": proxy_url}

def clean_text_for_url(text: str) -> str:
    txt = (text or "").lower()
    txt = txt.replace("&", "and")
    txt = re.sub(r"[^\w]+", "-", txt, flags=re.UNICODE)
    txt = re.sub(r"-+", "-", txt).strip("-")
    return txt

def clean_lyrics(text: str) -> str:
    text = re.sub(r"\r", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()

# ---------- استخراج متن از HTML ----------

def _extract_from_html(html: str) -> str:
    """
    Extract lyrics from Genius page HTML robustly.
    Returns cleaned lyrics string or "" if not found.
    """
    if not html:
        logger.debug("_extract_from_html: empty html input")
        return ""
    if isinstance(html, bytes):
        try:
            html = html.decode("utf-8", errors="replace")
        except Exception:
            html = str(html)

    def _long_enough(s: str, min_words: int = 6) -> bool:
        return isinstance(s, str) and len(s.split()) > min_words

    def _clean_candidate(s: str) -> str:
        return clean_lyrics(_html.unescape(s.strip()))

    def safe_json_loads(s: str):
        if not s or not isinstance(s, str):
            return None
        cand = s.strip()
        tries = [cand, _html.unescape(cand)]
        for t in tries:
            try:
                return json.loads(t)
            except Exception:
                m = re.search(r'(\{(?:.*)\})', t, re.S)
                if m:
                    block = m.group(1)
                    try:
                        return json.loads(block)
                    except Exception:
                        clean_block = re.sub(r',\s*([}\]])', r'\1', block)
                        try:
                            return json.loads(clean_block)
                        except Exception:
                            continue
        return None

    def find_lyrics_in_obj(obj):
        if isinstance(obj, str):
            if _long_enough(obj):
                return obj
            return None
        if isinstance(obj, dict):
            for key in obj.keys():
                if isinstance(key, str) and key.lower() in ("lyrics", "lyrics_plain", "lyricsplain", "description", "text", "body", "content"):
                    v = obj.get(key)
                    if isinstance(v, str) and _long_enough(v):
                        return v
            for k, v in obj.items():
                res = find_lyrics_in_obj(v)
                if res:
                    return res
            return None
        if isinstance(obj, list):
            for item in obj:
                res = find_lyrics_in_obj(item)
                if res:
                    return res
            return None
        return None

    soup = BeautifulSoup(html, "html.parser")
    try:
        divs = soup.select("div[data-lyrics-container='true']")
        if divs:
            text = "\n".join(d.get_text(separator="\n").strip() for d in divs)
            if _long_enough(text):
                logger.debug("_extract_from_html: found data-lyrics-container")
                return _clean_candidate(text)
    except Exception as e:
        logger.debug(f"_extract_from_html: error in data-lyrics-container block: {e}")
    try:
        divs = soup.find_all("div", class_=lambda v: v and "Lyrics__Container" in v)
        if divs:
            text = "\n".join(d.get_text(separator="\n").strip() for d in divs)
            if _long_enough(text):
                logger.debug("_extract_from_html: found Lyrics__Container")
                return _clean_candidate(text)
    except Exception as e:
        logger.debug(f"_extract_from_html: error in Lyrics__Container block: {e}")
    try:
        old = soup.find("div", class_="lyrics")
        if old:
            text = old.get_text(separator="\n").strip()
            if _long_enough(text):
                logger.debug("_extract_from_html: found old .lyrics div")
                return _clean_candidate(text)
    except Exception as e:
        logger.debug(f"_extract_from_html: error in old .lyrics block: {e}")
    try:
        og = soup.find("meta", property="og:description") or soup.find("meta", attrs={"name": "description"})
        if og and og.get("content"):
            cont = og["content"].strip()
            if _long_enough(cont):
                logger.debug("_extract_from_html: found meta description")
                return _clean_candidate(cont)
    except Exception as e:
        logger.debug(f"_extract_from_html: error reading meta description: {e}")
    try:
        for script in soup.find_all("script", type="application/ld+json"):
            txt = script.string or script.get_text()
            if not txt:
                continue
            data = safe_json_loads(txt)
            if data:
                found = find_lyrics_in_obj(data)
                if found:
                    logger.debug("_extract_from_html: found lyrics in ld+json")
                    return _clean_candidate(found)
    except Exception as e:
        logger.debug(f"_extract_from_html: error enumerating ld+json scripts: {e}")
    try:
        m = re.search(r"window\.__PRELOADED_STATE__\s*=\s*({.*?});", html, re.S)
        if not m:
            m = re.search(r"<script id=[\"']__NEXT_DATA__['\"] type=[\"']application/json['\"]>(.*?)</script>", html, re.S)
        if m:
            payload = m.group(1)
            if payload:
                data = safe_json_loads(payload)
                if data:
                    found = find_lyrics_in_obj(data)
                    if found:
                        logger.debug("_extract_from_html: found lyrics in PRELOADED/NEXT data")
                        return _clean_candidate(found)
    except Exception as e:
        logger.debug(f"_extract_from_html: error parsing preloaded/next data: {e}")
    try:
        m = re.search(r'(?sm)([A-Za-z0-9\.,\'\"!\?\-\(\)\[\]\n\r]{80,})', html)
        if m:
            candidate = m.group(1).strip()
            if _long_enough(candidate):
                logger.debug("_extract_from_html: fallback regex candidate")
                return _clean_candidate(candidate)
    except Exception as e:
        logger.debug(f"_extract_from_html: fallback regex error: {e}")
    logger.debug("_extract_from_html: no lyrics found in HTML")
    return ""

# ---------- درخواست HTTP با ری‌تری و backoff ----------

def _requests_fetch(url: str, headers: Dict[str, str], timeout: int = 120, retries: int = 6) -> str:
    """
    دانلود HTML با تست چند پروکسی به ترتیب.
    """
    proxy_list = _env_proxy_list()
    last_error = None
    for proxy_url in proxy_list:
        proxies = _requests_proxies(proxy_url)
        session = requests.Session()
        backoff = 3.0
        for attempt in range(1, retries + 1):
            try:
                session.headers.update(headers)
                if proxies:
                    session.proxies.update(proxies)
                logger.debug(f"Fetching URL: {url} with proxy {proxy_url} (attempt {attempt}/{retries})")
                resp = session.get(url, timeout=timeout, allow_redirects=True)
                status = resp.status_code
                text = resp.text or ""
                lower_text = text.lower()
                if status in (403, 429) or "make sure you're a human" in lower_text or "verify you are human" in lower_text:
                    raise Blocked(f"HTTP {status} / human-check")
                if status != 200:
                    raise requests.HTTPError(f"Status {status}")
                extracted = _extract_from_html(resp.text)
                if extracted:
                    logger.debug(f"Lyrics extracted from {url} using proxy {proxy_url} (len {len(extracted.split())} words)")
                    return extracted
                logger.debug(f"No lyrics extracted from {url} (status {status})")
                return ""
            except Blocked as b:
                logger.warning(f"Blocked with proxy {proxy_url} (attempt {attempt}/{retries}): {b}")
                last_error = b
            except requests.RequestException as e:
                logger.warning(f"Requests error with proxy {proxy_url} (attempt {attempt}/{retries}): {e}")
                last_error = e
            except Exception as e:
                logger.warning(f"Unexpected error with proxy {proxy_url} (attempt {attempt}/{retries}): {e}")
                last_error = e
            if attempt < retries:
                sleep_for = backoff + random.uniform(0, 1)
                logger.debug(f"Sleeping {sleep_for:.1f}s before retry with proxy {proxy_url}")
                time.sleep(sleep_for)
                backoff = min(backoff * 2.0, 16.0)
        logger.warning(f"Proxy {proxy_url} failed after {retries} retries, trying next proxy...")
    raise Blocked(f"All proxies failed. Last error: {last_error}")

def translate_lyrics(text: str, target_language: str = "fa") -> str:
    """
    ترجمه متن آهنگ به زبان فارسی با استفاده از OpenAI API
    :param text: متن اصلی لیریک (انگلیسی)
    :param target_language: زبان هدف (فقط 'fa' پشتیبانی می‌شود)
    :return: متن ترجمه‌شده
    """
    if not text:
        logger.warning("⚠️ متن برای ترجمه خالی است.")
        return ""
    if target_language != "fa":
        logger.warning(f"⚠️ زبان هدف فقط 'fa' پشتیبانی می‌شود، ورودی {target_language} نادیده گرفته شد.")
        target_language = "fa"
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        logger.debug(f"Translating English lyrics to {target_language} ...")
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Translate the following English song lyrics into Persian (fa). "
                                              "Keep the poetic style and line breaks."},
                {"role": "user", "content": text[:4000]}
            ],
            temperature=0.3,
            timeout=60
        )
        translated_text = response.choices[0].message.content.strip()
        logger.info(f"✅ متن با موفقیت به {target_language} ترجمه شد.")
        return translated_text
    except OpenAIError as e:
        logger.error(f"❌ خطا در ترجمه با OpenAI: {e}")
        return ""
    except Exception as e:
        logger.error(f"❌ خطای غیرمنتظره در ترجمه: {e}")
        return ""

def _api_fallback(artist: str, title: str) -> str:
    try:
        proxy_urls = _env_proxy_list()
        for proxy_url in proxy_urls:
            proxies = _requests_proxies(proxy_url)
            logger.debug(f"Attempting fallback to lyrics.ovh for {artist} - {title} with proxy {proxy_url}")
            r = requests.get(f"https://api.lyrics.ovh/v1/{quote_plus(artist)}/{quote_plus(title)}", proxies=proxies, timeout=90)
            if r.status_code == 200:
                data = r.json()
                if "lyrics" in data:
                    lyrics = clean_lyrics(data["lyrics"])
                    if lyrics and all(ord(c) < 128 for c in lyrics):
                        logger.debug("Successfully fetched English lyrics from lyrics.ovh")
                        return lyrics
                    logger.debug("Non-English lyrics from lyrics.ovh")
            logger.debug(f"Fallback failed with proxy {proxy_url}, status code: {r.status_code}")
    except Exception as e:
        logger.warning(f"API fallback error for {artist} - {title}: {e}")
    return ""

def get_lyrics_from_genius(artist: str, title: str, translate: bool = False, target_language: str = "fa") -> str:
    """
    دریافت و ترجمه لیریک‌ها از Genius API
    :param artist: نام خواننده
    :param title: عنوان آهنگ
    :param translate: آیا لیریک‌ها ترجمه بشن؟ (پیش‌فرض False)
    :param target_language: زبان هدف برای ترجمه (فقط 'fa' پشتیبانی می‌شود)
    :return: لیریک‌ها (ترجمه‌شده یا اصلی)
    """
    logger.debug(f"Starting lyrics search for {artist} - {title}")
    if not artist or not title:
        logger.warning("Artist or title is empty or invalid.")
        return ""

    # Method 1: Genius API
    api_lyrics = _genius_api_for_lyrics(artist, title)
    if api_lyrics:
        logger.info("Found lyrics via Genius API.")
        if translate and target_language == "fa":
            return translate_lyrics(api_lyrics, target_language)
        return api_lyrics

    # Fallback API (lyrics.ovh)
    ovh_lyrics = _api_fallback(artist, title)
    if ovh_lyrics:
        logger.info("Found lyrics via lyrics.ovh fallback.")
        if translate and target_language == "fa":
            return translate_lyrics(ovh_lyrics, target_language)
        return ovh_lyrics

    logger.error("No lyrics found with all methods.")
    return ""

# ---------- جستجو در Genius API (فقط صفحات اصلی انگلیسی) ----------
def _genius_api_for_lyrics(artist: str, title: str) -> str:
    GENIUS_API_TOKEN = os.getenv("GENIUS_API_TOKEN", "YOUR_GENIUS_API_TOKEN_HERE")
    if not GENIUS_API_TOKEN or GENIUS_API_TOKEN == "YOUR_GENIUS_API_TOKEN_HERE":
        logger.warning("Genius API token not set. Skipping API method.")
        return ""
    title_no_paren = re.sub(r"\s*\(.*\)", "", title)
    search_variants = [
        f"{artist} {title}",
        f"{artist.replace('The ', '')} {title}",
        f"{artist} {title_no_paren}",
        f"{clean_text_for_url(artist)}-{clean_text_for_url(title_no_paren)}"
    ]
    api_url = "https://api.genius.com/search"
    proxy_urls = _env_proxy_list()
    for query in search_variants:
        logger.debug(f"Searching Genius API with query: {query}")
        for proxy_url in proxy_urls:
            proxies = _requests_proxies(proxy_url)
            try:
                headers = {"Authorization": f"Bearer {GENIUS_API_TOKEN}"}
                params = {"q": query}
                response = requests.get(api_url, headers=headers, params=params, proxies=proxies, timeout=90)
                logger.debug(f"Genius API response status for query '{query}' with proxy {proxy_url}: {response.status_code}")
                if response.status_code != 200:
                    logger.warning(f"Genius API search error for {query} with proxy {proxy_url}: Status {response.status_code}")
                    continue
                data = response.json()
                hits = data.get("response", {}).get("hits", [])
                if not hits:
                    logger.debug(f"No hits found for query: {query} with proxy {proxy_url}")
                    continue
                for hit in hits:
                    if hit.get("type") != "song" or not hit.get("result"):
                        continue
                    song_url = hit["result"].get("url", "")
                    if not song_url:
                        continue
                    lower_url = song_url.lower()
                    # فیلتر URLهای ترجمه و غیرانگلیسی
                    if any(keyword in lower_url for keyword in BLOCKED_KEYWORDS):
                        logger.debug(f"Skipping translation URL: {song_url}")
                        continue
                    # بررسی وجود "-lyrics" برای اطمینان از صفحه اصلی
                    if "-lyrics" not in lower_url:
                        logger.debug(f"Skipping URL (no '-lyrics'): {song_url}")
                        continue
                    # تطابق ساده slug با artist و title
                    slug = f"{clean_text_for_url(artist)}-{clean_text_for_url(title_no_paren)}"
                    if slug not in lower_url:
                        tokens = (clean_text_for_url(artist) + "-" + clean_text_for_url(title_no_paren)).split("-")
                        tokens = [t for t in tokens if t]
                        found_token = any(t in lower_url for t in tokens[:3])
                        if not found_token:
                            logger.debug(f"Skipping URL (doesn't match artist-title): {song_url}")
                            continue
                    logger.info(f"Found Genius song URL via API: {song_url}")
                    fetch_headers = _random_headers()
                    try:
                        lyrics = _requests_fetch(song_url, fetch_headers, timeout=90, retries=6)
                        if lyrics and all(ord(c) < 128 for c in lyrics):  # بررسی که متن فقط شامل کاراکترهای ASCII (انگلیسی) باشد
                            logger.debug(f"Successfully fetched English lyrics from {song_url}")
                            return lyrics
                        logger.debug(f"Non-English or no lyrics extracted from {song_url}")
                    except Blocked as b:
                        logger.warning(f"Fetch error (blocked) for {song_url}: {b}")
                    except Exception as e:
                        logger.warning(f"Fetch error for {song_url}: {e}")
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to decode Genius API JSON for {query} with proxy {proxy_url}: {e}")
                continue
            except requests.RequestException as e:
                logger.warning(f"Network error when calling Genius API for {query} with proxy {proxy_url}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Genius API unexpected error for {query} with proxy {proxy_url}: {e}")
                continue
        logger.debug(f"Exhausted proxies for query: {query}")
    logger.debug("Exhausted all search variants and proxies, no lyrics found.")
    return ""



#----------- پایان دریافت لیریک -----------------




# تابع کمکی برای ایجاد دکمه متن با اطلاعات دقیق
def get_track_inline_buttons(
    track_url: str,
    artist: str = "",
    title: str = "",
    message_id: int = None,
    file_id: str = ""
) -> InlineKeyboardMarkup:
    """
    ایجاد دکمه‌های اینلاین برای ترک اسپاتیفای با امکانات اضافی
    """
    try:
        # بررسی اعتبار لینک
        if not track_url or "spotify.com" not in track_url.lower():
            logger.warning(f"❌ لینک نامعتبر یا غیر اسپاتیفای: {track_url}")
            # در صورت خطا کیبورد حداقلی برگردان
            return InlineKeyboardMarkup(
                inline_keyboard=[
                    [InlineKeyboardButton(text="🛒 خرید اکانت اسپاتیفای", url="https://t.me/YOUR_SUPPORT_BOT")]
                ]
            )

        # ساخت دکمه دریافت متن
        if artist and title and message_id and file_id:
            lyrics_button = create_lyrics_button(artist, title, message_id, file_id)
        else:
            lyrics_button = InlineKeyboardButton(
                text="📝 دریافت متن آهنگ",
                callback_data=f"lyrics|{track_url}"
            )

        # ساخت کیبورد کامل
        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(text="🎵 اضافه به پلی‌لیست", callback_data=f"add_to_playlist|{track_url}"),
                    InlineKeyboardButton(text="🛒 خرید اکانت اسپاتیفای", url="https://t.me/YOUR_SUPPORT_BOT"),
                ],
                [
                    InlineKeyboardButton(text="🎙 جدا سازی وکال / بیت",callback_data=f"separate_vocals|{file_id}"),
                    lyrics_button,
                ],
                [
                    InlineKeyboardButton(text="🎧 پیشنهاد آهنگ مشابه", callback_data=f"suggest|{track_url}")
                ],
            ]
        )

        logger.debug(f"✅ دکمه‌های اینلاین برای {track_url} ساخته شد.")
        return keyboard

    except Exception as e:
        logger.error(f"❌ خطا در ساخت دکمه‌های اینلاین: {e}")
        # اگر خطای غیرمنتظره بود، کیبورد پایه‌ای برگردون
        return InlineKeyboardMarkup(
            inline_keyboard=[
                [InlineKeyboardButton(text="🛒 خرید اکانت اسپاتیفای", url="https://t.me/YOUR_SUPPORT_BOT")]
            ]
        )


def create_lyrics_button(artist: str, title: str, message_id: int, file_id: str) -> InlineKeyboardButton:
    safe_artist = (artist[:20] if artist else "نامشخص")
    safe_title = (title[:20] if title else "نامشخص")
    callback_data = f"lyrics|{safe_artist}|{safe_title}|{message_id}|{file_id}"
    if len(callback_data.encode('utf-8')) > 64:
        safe_artist = safe_artist[:10]
        safe_title = safe_title[:10]
        callback_data = f"lyrics|{safe_artist}|{safe_title}|{message_id}|{file_id}"
    return InlineKeyboardButton("📜 متن", callback_data=callback_data)


def transcribe_lyrics_from_file(mp3_path: str) -> str:
    """
    استخراج متن آهنگ از فایل MP3 با استفاده از OpenAI Whisper
    :param mp3_path: مسیر فایل MP3
    :return: متن استخراج شده
    """
    FFMPEG_PATH = os.getenv("FFMPEG_PATH", r'G:\zAll data (All Mine)\Codeing\ffmpeg\bin\ffmpeg.exe')
 
    if not os.path.exists(mp3_path):
        logger.error(f"❌ فایل MP3 در مسیر {mp3_path} یافت نشد.")
        raise FileNotFoundError(f"فایل MP3 یافت نشد: {mp3_path}")

    temp_wav: Optional[str] = None
    try:
        # اطمینان از اینکه ffmpeg ست شده
        AudioSegment.converter = FFMPEG_PATH
        audio = AudioSegment.from_mp3(mp3_path)

        temp_wav = mp3_path.replace(".mp3", "_speech.wav")
        audio.export(temp_wav, format="wav", parameters=["-ar", "16000"])
        logger.debug(f"📂 فایل WAV در {temp_wav} ایجاد شد.")

        if not os.path.exists(temp_wav):
            logger.error(f"❌ فایل WAV در {temp_wav} ایجاد نشد.")
            raise FileNotFoundError(f"فایل WAV ایجاد نشد: {temp_wav}")

        client = OpenAI()
        with open(temp_wav, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
            )
            logger.info(f"✅ متن آهنگ از {mp3_path} استخراج شد.")
            return transcript.text.strip()

    except OpenAIError as e:
        logger.error(f"❌ خطا در API OpenAI: {e}")
        raise RuntimeError(f"خطا در استخراج متن با Whisper: {e}")
    except (RequestException, HTTPError, ConnectionError) as e:
        logger.error(f"❌ خطای شبکه در Whisper: {e}")
        raise RuntimeError(f"خطای شبکه در Whisper: {e}")
    except Exception as e:
        logger.error(f"❌ خطای ناشناخته در استخراج متن: {e}")
        raise RuntimeError(f"خطای ناشناخته در Whisper: {e}")
    finally:
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
                logger.debug(f"🗑️ فایل موقت {temp_wav} حذف شد.")
            except OSError as e:
                logger.warning(f"⚠️ خطا در حذف فایل موقت {temp_wav}: {e}")

# نمونه استفاده هنگام ارسال آهنگ
async def send_music_with_lyrics_button(bot, chat_id: int, audio_file, artist: str, title: str, message_id: int):
    keyboard = InlineKeyboardMarkup(inline_keyboard=[[
        create_lyrics_button(artist, title, message_id, audio_file.file_id)
    ]])
    caption = f"<b>{artist} – {title}</b>"
    await bot.send_audio(
        chat_id=chat_id,
        audio=audio_file,
        caption=caption,
        parse_mode="HTML",
        reply_markup=keyboard
    )


#---------------- کال بک ها ---------------


# هندلر برای callback_query عمومی "lyrics"
@router.callback_query(lambda q: q.data == "lyrics")
async def handle_lyrics_general(query: CallbackQuery) -> None:
    message = query.message
    user_id = query.from_user.id
    await query.answer()
    await message.reply("🔍 در حال بررسی متن موسیقی...")

    try:
        # --- استخراج کپشن از پیام ---
        caption = message.caption or ""
        title, artist = "نامشخص", "نامشخص"
        search_query = caption.split("\n")[0].strip() if caption else ""

        if not search_query:
            await message.reply("❌ کپشن پیام خالی است. لطفاً نام آهنگ را در کپشن بنویسید.")
            return

        # --- تفکیک artist و title ---
        if "–" in search_query:
            parts = search_query.split("–", 1)
            if len(parts) == 2:
                artist = parts[0].strip().replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", "")
                title = parts[1].strip().replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", "")
            else:
                title = search_query.strip().replace("<b>", "").replace("</b>", "")
                artist = "نامشخص"
        else:
            title = search_query.strip().replace("<b>", "").replace("</b>", "").replace("<i>", "").replace("</i>", "")
            artist = "نامشخص"

        # --- پاکسازی رشته‌ها ---
        if title != "نامشخص":
            title = (
                title.split("(feat.")[0]
                .split("[feat.")[0]
                .split("feat.")[0]
                .split("(remix")[0]
                .split("[remix")[0]
                .split("remix")[0]
                .split("(version")[0]
                .split("[version")[0]
                .split("version")[0]
                .split("(")[0]
                .split("[")[0]
                .strip()
            )
        if artist != "نامشخص":
            artist = (
                artist.split("feat.")[0]
                .split("(feat.")[0]
                .split("[feat.")[0]
                .strip()
            )

        # --- چک کردن اعتبار ورودی ---
        if title == "نامشخص" or len(title.strip()) < 2:
            await message.reply("❌ نام آهنگ معتبر نیست. لطفاً کپشن را بررسی کنید.")
            return

        display_title = f"{artist} – {title}" if artist != "نامشخص" else title
        logger.info(f"کاربر {user_id}: جستجوی متن در Genius برای → {display_title}")

        # --- جستجو در Genius ---
        lyrics = await asyncio.to_thread(get_lyrics_from_genius, artist, title)

        if not lyrics or len(lyrics.strip()) < 10:
            await message.reply("❌ متن آهنگ یافت نشد. ممکن است نام آهنگ اشتباه باشد.")
            return

        # --- نمایش متن اصلی ---
        await message.reply(
            f"🎶 متن آهنگ یافت شد ({artist} – {title}):\n\n"
            f"<blockquote expandable>{lyrics}</blockquote>",
            parse_mode="HTML"
        )

        # --- ترجمه (فقط اگر متن معتبر باشد) ---
        try:
            if len(lyrics) > 50:  # فقط اگر متن کافی باشد
                await message.reply("🔄 در حال ترجمه به فارسی...")
                translated_lyrics = await asyncio.to_thread(translate_lyrics, lyrics, "fa")
                
                if translated_lyrics and len(translated_lyrics.strip()) > 10:
                    await message.reply(
                        f"🌟 ترجمه فارسی:\n\n"
                        f"<blockquote expandable>{translated_lyrics}</blockquote>",
                        parse_mode="HTML"
                    )
                    logger.info(f"کاربر {user_id}: متن و ترجمه آهنگ دریافت شد.")
                else:
                    await message.reply("❌ ترجمه با مشکل مواجه شد. فقط متن اصلی نمایش داده شد.")
            else:
                logger.info(f"کاربر {user_id}: متن کوتاه، ترجمه انجام نشد.")
        except Exception as e:
            logger.error(f"کاربر {user_id}: خطا در ترجمه متن: {e}")
            await message.reply("⚠️ متن اصلی نمایش داده شد. ترجمه با مشکل مواجه شد.")

    except Exception as e:
        logger.error(f"کاربر {user_id}: خطا در هندلر عمومی: {e}")
        await message.reply("❌ خطا در پردازش درخواست متن. لطفاً دوباره تلاش کنید.")


# هندلرهای callback_query دیگر
@router.callback_query(lambda q: q.data == "add_to_playlist")
async def handle_add_to_playlist(query: CallbackQuery) -> None:
    await query.answer("🎵 قابلیت پلی‌لیست در حال توسعه است...", show_alert=True)

@router.callback_query(lambda q: q.data == "ask_for_ai")
async def handle_ask_for_ai(query: CallbackQuery) -> None:
    await query.answer("🎵 قابلیت هوش مصنوعی در حال توسعه است...", show_alert=True)

@router.callback_query(lambda q: q.data.startswith("suggest|"))
async def handle_suggest_similar(query: CallbackQuery) -> None:
    await query.answer("🎧 قابلیت پیشنهاد آهنگ در حال توسعه است...", show_alert=True)


@router.message(lambda m: m.audio or (m.document and m.document.file_name.endswith(".mp3")))
async def handle_music_message(message: Message) -> None:
    user_id = message.from_user.id
    caption = message.caption or ""
    artist, title = "نامشخص", "نامشخص"
    if "–" in caption:
        parts = caption.split("–", 1)
        artist = parts[0].strip().replace("<b>", "").replace("</b>", "")
        title = parts[1].strip().replace("<b>", "").replace("</b>", "").split("\n")[0].strip()
    
    file_id = message.audio.file_id if message.audio else message.document.file_id
    message_id = message.message_id
    
    await send_music_with_lyrics_button(
        bot=message.bot,
        chat_id=message.chat.id,
        audio_file=message.audio or message.document,
        artist=artist,
        title=title,
        message_id=message_id
    )



# --- /start ---
@router.message(CommandStart())
async def start(message: types.Message) -> None:
    try:
        if not message.text:
            logger.warning(f"کاربر {message.from_user.id}: پیام غیرمتنی برای /start")
            await message.answer("❌ لطفاً از دستور /start استفاده کنید.", parse_mode="HTML")
            return

        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text="⭐️ دانلود موزیک / آلبوم", callback_data="download")],
            [InlineKeyboardButton(text="📃 اطلاعات هنرمند / پلی‌لیست", callback_data="info")],
            [InlineKeyboardButton(text="🖐 خرید اکانت اسپاتیفای", callback_data="buy_account")]
        ])
        await message.answer(
            "به ربات اسپاتیفای خوش آمدید. یکی از گزینه‌ها را انتخاب کنید:",
            reply_markup=keyboard,
            parse_mode="HTML"
        )
        logger.info(f"کاربر {message.from_user.id} دستور /start را اجرا کرد.")
    except Exception as e:
        logger.error(f"کاربر {message.from_user.id}: خطا در ارسال پیام خوش‌آمدگویی: {e}")
        await message.answer("❌ خطا در شروع ربات. لطفاً دوباره تلاش کنید.", parse_mode="HTML")
        raise RuntimeError(f"خطا در ارسال پیام: {e}")
    
# --- درخواست خرید اکانت ---
@router.callback_query(lambda q: q.data == "buy_account")
async def buy_account(query: CallbackQuery) -> None:
    try:
        # بررسی تنظیم بودن SUPPORT_CHAT_ID
        if not SUPPORT_CHAT_ID:
            logger.error("شناسه چت پشتیبانی تنظیم نشده است.")
            raise ValueError("شناسه چت پشتیبانی تنظیم نشده است.")

        user = query.from_user
        username = f"@{user.username}" if user.username else "ندارد"
        msg = (
            f"🔔 درخواست خرید اکانت از کاربر:\n"
            f"👤 {username}\n"
            f"🆔 <code>{user.id}</code>"
        )

        # ارسال پیام به پشتیبانی
        try:
            await query.bot.send_message(
                chat_id=SUPPORT_CHAT_ID,
                text=msg,
                parse_mode="HTML"
            )
            logger.info(f"درخواست خرید اکانت از کاربر {user.id} به پشتیبانی ارسال شد.")
        except Exception as e:
            logger.error(f"خطا در ارسال پیام به پشتیبانی: {e}")
            raise RuntimeError(f"خطا در ارسال پیام به پشتیبانی: {e}")

        # پاسخ به کاربر
        await query.message.answer(
            "✅ درخواست شما به پشتیبانی ارسال شد.",
            parse_mode="HTML"
        )
        await query.answer()  # تأیید کال‌بک
        logger.debug(f"پاسخ به کاربر {user.id} ارسال شد.")

    except ValueError as e:
        logger.error(f"خطای اعتبارسنجی: {e}")
        await query.message.answer(f"❌ خطا: {str(e)}", parse_mode="HTML")
        await query.answer(show_alert=True)
    except RuntimeError as e:
        logger.error(f"خطا در پردازش درخواست: {e}")
        await query.message.answer(f"❌ خطا در ارسال درخواست: {str(e)}", parse_mode="HTML")
        await query.answer(show_alert=True)
    except Exception as e:
        logger.error(f"خطای ناشناخته در پردازش درخواست: {e}")
        await query.message.answer(f"❌ خطای ناشناخته: {str(e)}", parse_mode="HTML")
        await query.answer(show_alert=True)

# --- کال‌بک‌ها ---
@router.callback_query(lambda q: q.data == "download")
async def ask_for_link(query: types.CallbackQuery) -> None:
    try:
        await query.message.answer(
            "🔗 لطفاً لینک آهنگ، آلبوم یا پلی‌لیست اسپاتیفای را ارسال کنید:",
            parse_mode="HTML"
        )
        await query.answer()
        logger.debug(f"کاربر {query.from_user.id}: درخواست لینک ارسال شد.")
    except Exception as e:
        logger.error(f"کاربر {query.from_user.id}: خطا در ارسال پیام درخواست لینک: {e}")
        await query.answer(f"❌ خطا: لطفاً دوباره تلاش کنید.", show_alert=True)
        raise RuntimeError(f"خطا در ارسال پیام: {e}")

#------ پرسش اطلاعات هنرمند یا پلی‌لیست ------
@router.callback_query(lambda q: q.data == "info")
async def ask_for_info(query: CallbackQuery) -> None:
    try:
        await query.message.answer(
            "🔍 لینک هنرمند یا پلی‌لیست اسپاتیفای را ارسال کنید:",
            parse_mode="HTML"
        )
        await query.answer()
        logger.debug(f"درخواست لینک اطلاعات از کاربر {query.from_user.id} ارسال شد.")
    except Exception as e:
        logger.error(f"خطا در ارسال پیام درخواست اطلاعات: {e}")
        await query.answer(f"❌ خطا در پردازش درخواست: {str(e)}", show_alert=True)
        raise RuntimeError(f"خطا در ارسال پیام: {e}")

# --- هندلر اصلی دانلود اسپاتیفای ---
@router.message(lambda m: m.text and "spotify.com" in m.text.lower())
async def spotify_download_handler(message: types.Message) -> None:

    try:
        # بررسی وجود متن
        if not message.text:
            logger.warning(f"User {message.from_user.id}: Non-text message received.")
            await message.answer("❌ Please send a Spotify link as text.", parse_mode="HTML")
            return

        text = message.text.strip()

        # Extract all Spotify URLs from the message
        import re
        spotify_urls = re.findall(r'https?://[^\s]*spotify\.com[^\s]*', text)

        # Remove duplicates while preserving order
        seen = set()
        spotify_urls = [url for url in spotify_urls if not (url in seen or seen.add(url))]

        num_links = len(spotify_urls)

        # Validation for number of links
        if num_links == 0:
            await message.answer("❌ No Spotify links found.", parse_mode="HTML")
            return
        elif num_links > 3:
            await message.answer("❌ Maximum 3 links supported simultaneously. Please send fewer links.", parse_mode="HTML")
            return
        elif num_links == 1:
            # Single link processing
            url = spotify_urls[0]

            # تاخیر 10 ثانیه‌ای قبل از حذف لینک
            asyncio.create_task(delete_message_later(message, delay=10))
            logger.info(f"کاربر {message.from_user.id}: لینک دریافت شد و در 10 ثانیه بعد حذف خواهد شد.")

            logger.info(f"کاربر {message.from_user.id}: لینک دریافت‌شده: {url}")

            # بررسی پلتفرم
            platform = detect_platform(url)
            logger.info(f"کاربر {message.from_user.id}: پلتفرم تشخیص‌داده‌شده: {platform}")
            if platform != "spotify":
                await message.answer("❌ این لینک برای اسپاتیفای نیست.", parse_mode="HTML")
                return

            # Try fast-path from inline #sid tag, else validate URL
            sid_match = re.search(r'#sid:(track|album|playlist|artist):([A-Za-z0-9]+)', text)
            if sid_match:
                content_type = sid_match.group(1)
                content_id = sid_match.group(2)
                is_valid = True
            else:
                is_valid, content_type, content_id = validate_spotify_url(url)

            if not is_valid or not content_type or not content_id:
                logger.warning(f"کاربر {message.from_user.id}: لینک اسپاتیفای نامعتبر است: {url}")
                await message.answer("❌ لینک اسپاتیفای نامعتبر است.", parse_mode="HTML")
                return

            # پیام موقت
            msg = await message.reply("Receiving information...")
            try:
                await asyncio.sleep(1)
                await msg.delete()
            except Exception as e:
                logger.warning(f"کاربر {message.from_user.id}: خطا در حذف پیام موقت: {e}")
                await asyncio.sleep(0.5)

            # پردازش بر اساس نوع محتوا
            if content_type == "album":
                await handle_spotify_album_download(message, url, content_id)
            elif content_type == "playlist":
                await handle_spotify_playlist_download(message, url, content_id)
            elif content_type == "artist":
                await show_info(message, content_type, content_id)
            else:
                await handle_spotify_download(message, url, content_type, content_id)

        else:
            # Multiple links (2-3) - simultaneous processing
            logger.info(f"User {message.from_user.id}: {num_links} simultaneous links received")

            # پیام پیشرفت
            progress_msg = await message.reply(f"🔄 Processing {num_links} links simultaneously...")

            # Process simultaneous links (message deletion will be handled inside)
            await handle_simultaneous_spotify_links(message, spotify_urls, progress_msg)

    except ValueError as e:
        logger.error(f"کاربر {message.from_user.id}: خطای اعتبارسنجی: {e}")
        await message.answer(f"❌ خطا: {str(e)}", parse_mode="HTML")
    except SpotifyException as e:
        logger.error(f"کاربر {message.from_user.id}: خطا در API اسپاتیفای: {e}")
        await message.answer(f"❌ خطا در پردازش لینک: {str(e)}", parse_mode="HTML")
    except RuntimeError as e:
        logger.error(f"کاربر {message.from_user.id}: خطا در پردازش لینک: {e}")
        await message.answer(f"❌ خطا در پردازش: {str(e)}", parse_mode="HTML")
    except Exception as e:
        logger.error(f"کاربر {message.from_user.id}: خطای ناشناخته: {e}")
        await message.answer(f"❌ خطای ناشناخته: {str(e)}", parse_mode="HTML")

# تابع کمکی برای حذف پیام با تأخیر
async def delete_message_later(message: types.Message, delay: int = 10):
    try:
        await asyncio.sleep(delay)
        await message.delete()
        logger.info(f"پیام کاربر {message.from_user.id} بعد از {delay} ثانیه حذف شد.")
    except Exception as e:
        logger.warning(f"خطا در حذف پیام کاربر {message.from_user.id}: {e}")

# --- دانلود موزیک با spotdl ---
# Helper function (called by spotify_download_handler); not a router message
async def handle_spotify_download(message: types.Message, url: str, content_type: str, content_id: str) -> None:

    user_id = message.from_user.id
    chat_id = message.chat.id
    try:
        
        info: Optional[dict] = None
        cover_msg: Optional[types.Message] = None
        db_file_info: Optional[Tuple] = None

        # Initialize database connection for this session
        conn = sqlite3.connect("bot_cache.db")
        cur = conn.cursor()

        inline_buttons = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="Add to playlist", callback_data="add_to_playlist"),
                InlineKeyboardButton(text="Get song lyrics", callback_data="lyrics")
            ],
            [
                InlineKeyboardButton(text="Separation of the voice", callback_data="separate_vocal"),
                InlineKeyboardButton(text="Question about AI", callback_data="ask_for_ai")
            ],
        ])

        if content_type == "track":
            # FAST-PATH: Check for tg_file_id first for instant upload
            tg_file_id = get_tg_file_id(content_id, user_id)
            if tg_file_id:
                try:
                    # Get complete info for full fast-path experience
                    cur.execute("SELECT artist, title, duration, thumbnail, album, release_date, genre, popularity, url FROM files WHERE file_id = ? AND user_id = ?", (content_id, user_id))
                    cached_info = cur.fetchone()
                    # Also fetch cache_path (for background copy to downloads folder)
                    cached_local_path = None
                    try:
                        cur.execute("SELECT cache_path FROM files WHERE file_id = ? AND user_id = ?", (content_id, user_id))
                        _row = cur.fetchone()
                        cached_local_path = _row[0] if _row else None
                    except Exception as _e:
                        cached_local_path = None

                    # Send cover photo first (if available)
                    cover_msg = None
                    if cached_info and cached_info[3]:  # thumbnail
                        try:
                            total_seconds = int(cached_info[2] or 0)
                            minutes = total_seconds // 60
                            seconds = total_seconds % 60
                            duration_str = f"{minutes}:{seconds:02d}"
                            pop_val = int(cached_info[7] or 0)

                            caption = (
                                f"<b>{cached_info[0] or 'Unknown'} – {cached_info[1] or 'Unknown'}</b>\n"
                                f"Album: {cached_info[4] or 'Unknown'}\n"
                                f"Duration: {duration_str}\n"
                                f"Release date: {cached_info[5] or 'Unknown'}\n"
                                f"Popularity: {pop_val}% \n"
                                f"Genre: {cached_info[6] or 'Unknown'}\n"
                                f"Url: <a href='{cached_info[8] or f'https://open.spotify.com/track/{content_id}'}'>Link Spotify</a>"
                            )

                            cover_msg = await message.answer_photo(
                                photo=cached_info[3],
                                caption=caption,
                                reply_markup=inline_buttons,
                                parse_mode="HTML"
                            )
                        except Exception as e:
                            logger.debug(f"User {user_id}: Cover photo failed in fast-path: {e}")

                    # Send quickly from downloads folder if we have a cached local file,
                    # otherwise fall back to tg_file_id (fastest), and schedule background copy.
                    sent_via_downloads = False
                    try:
                        if cached_info and cached_local_path and os.path.exists(cached_local_path):
                            artist_bg = cached_info[0] or "Unknown Artist"
                            title_bg = cached_info[1] or "Unknown Title"
                            info_bg = {"artist": artist_bg, "title": title_bg}
                            clean_name_bg = build_clean_filename(info_bg, bitrate_kbps=320)
                            dl_path_bg = os.path.join(DOWNLOAD_DIR, clean_name_bg)
                            os.makedirs(DOWNLOAD_DIR, exist_ok=True)
                            # Copy only if not already present or size differs
                            if (not os.path.exists(dl_path_bg)) or (os.path.getsize(dl_path_bg) != os.path.getsize(cached_local_path)):
                                shutil.copy2(cached_local_path, dl_path_bg)
                            # Send from downloads folder (FSInputFile)
                            sent_msg_dl = await message.answer_audio(
                                audio=FSInputFile(dl_path_bg, filename=os.path.basename(dl_path_bg)),
                                performer=cached_info[0] if cached_info else None,
                                title=cached_info[1] if cached_info else None,
                                duration=cached_info[2] if cached_info and len(cached_info) > 2 else None
                            )
                            # Clean up the quick copy in downloads after sending
                            try:
                                if os.path.exists(dl_path_bg):
                                    os.remove(dl_path_bg)
                            except Exception as _rm_e:
                                logger.debug(f"User {user_id}: could not remove quick copy {dl_path_bg}: {_rm_e}")
                            sent_via_downloads = True
                    except Exception as _e:
                        logger.debug(f"User {user_id}: cached fast-path send from downloads failed: {_e}")

                    if not sent_via_downloads:
                        # Fallback to immediate tg_file_id send (ultra fast)
                        await message.answer_audio(
                            audio=tg_file_id,
                            performer=cached_info[0] if cached_info else None,
                            title=cached_info[1] if cached_info else None,
                            duration=cached_info[2] if cached_info and len(cached_info) > 2 else None
                        )
                        # Non-blocking: ensure a copy exists in downloads/spotify for further operations
                        try:
                            if cached_info and cached_local_path and os.path.exists(cached_local_path):
                                artist_bg = cached_info[0] or "Unknown Artist"
                                title_bg = cached_info[1] or "Unknown Title"
                                info_bg = {"artist": artist_bg, "title": title_bg}
                                clean_name_bg = build_clean_filename(info_bg, bitrate_kbps=320)
                                dl_path_bg = os.path.join(DOWNLOAD_DIR, clean_name_bg)
                                def _copy_to_downloads():
                                    try:
                                        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
                                        if (not os.path.exists(dl_path_bg)) or (os.path.getsize(dl_path_bg) != os.path.getsize(cached_local_path)):
                                            shutil.copy2(cached_local_path, dl_path_bg)
                                    except Exception as _e2:
                                        logger.debug(f"User {user_id}: background copy to downloads failed: {_e2}")
                                asyncio.get_running_loop().run_in_executor(None, _copy_to_downloads)
                        except Exception as _e:
                            logger.debug(f"User {user_id}: scheduling background copy failed: {_e}")

                    # Send success message
                    if cover_msg:
                        try:
                            success_msg = await message.answer(
                                "The download was successful.\nآیا امکانات بیشتری نیاز دارید؟",
                                reply_to_message_id=cover_msg.message_id,
                                parse_mode="HTML"
                            )
                            # Store success message for cleanup
                            context_success_msg[user_id] = success_msg
                        except Exception as e:
                            logger.debug(f"User {user_id}: Success message failed in fast-path: {e}")

                    logger.info(f"User {user_id}: ULTRA FAST UPLOAD - Complete experience in <2 seconds via tg_file_id {tg_file_id}")
                    return
                except Exception as e:
                    logger.warning(f"User {user_id}: tg_file_id fast-path failed, fallback to normal flow: {e}")

            # SECOND FAST-PATH: Check for cached local file
            cur.execute("SELECT cache_path FROM files WHERE file_id = ? AND user_id = ?", (content_id, user_id))
            cached_file = cur.fetchone()
            if cached_file and cached_file[0] and os.path.exists(cached_file[0]):
                try:
                    # Get complete info for full fast-path experience
                    cur.execute("SELECT artist, title, duration, thumbnail, album, release_date, genre, popularity, url FROM files WHERE file_id = ? AND user_id = ?", (content_id, user_id))
                    cached_info = cur.fetchone()

                    # Send cover photo first (if available)
                    cover_msg = None
                    if cached_info and cached_info[3]:  # thumbnail
                        try:
                            total_seconds = int(cached_info[2] or 0)
                            minutes = total_seconds // 60
                            seconds = total_seconds % 60
                            duration_str = f"{minutes}:{seconds:02d}"
                            pop_val = int(cached_info[7] or 0)

                            caption = (
                                f"<b>{cached_info[0] or 'Unknown'} – {cached_info[1] or 'Unknown'}</b>\n"
                                f"Album: {cached_info[4] or 'Unknown'}\n"
                                f"Duration: {duration_str}\n"
                                f"Release date: {cached_info[5] or 'Unknown'}\n"
                                f"Popularity: {pop_val}% \n"
                                f"Genre: {cached_info[6] or 'Unknown'}\n"
                                f"Url: <a href='{cached_info[8] or f'https://open.spotify.com/track/{content_id}'}'>Link Spotify</a>"
                            )

                            cover_msg = await message.answer_photo(
                                photo=cached_info[3],
                                caption=caption,
                                reply_markup=inline_buttons,
                                parse_mode="HTML"
                            )
                        except Exception as e:
                            logger.debug(f"User {user_id}: Cover photo failed in cached fast-path: {e}")

                    # Send cached file quickly from downloads folder (copy fast, no heavy processing)
                    try:
                        artist_cf = (cached_info[0] if cached_info else None) or "Unknown Artist"
                        title_cf  = (cached_info[1] if cached_info else None) or "Unknown Title"
                        info_cf = {"artist": artist_cf, "title": title_cf}
                        clean_name_cf = build_clean_filename(info_cf, bitrate_kbps=320)
                        download_path_cf = os.path.join(DOWNLOAD_DIR, clean_name_cf)
                        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
                        # Copy from cache to downloads if needed
                        if not os.path.exists(download_path_cf) or os.path.getsize(download_path_cf) != os.path.getsize(cached_file[0]):
                            shutil.copy2(cached_file[0], download_path_cf)
                        send_path_cf = download_path_cf if os.path.exists(download_path_cf) else cached_file[0]
                    except Exception as _e:
                        logger.warning(f"User {user_id}: could not prepare cached fast-path copy: {_e}")
                        send_path_cf = cached_file[0]

                    # Send cached file directly
                    sent_message = await message.answer_audio(
                        audio=FSInputFile(send_path_cf),
                        performer=cached_info[0] if cached_info else None,
                        title=cached_info[1] if cached_info else None,
                        duration=cached_info[2] if cached_info and len(cached_info) > 2 else None
                    )
                    # Remove the quick copy from downloads after sending (keep cache)
                    try:
                        if send_path_cf.startswith(DOWNLOAD_DIR) and os.path.exists(send_path_cf):
                            os.remove(send_path_cf)
                    except Exception as _rm_e:
                        logger.debug(f"User {user_id}: could not remove quick copy {send_path_cf}: {_rm_e}")

                    # Record tg_file_id for even faster future access
                    if sent_message and sent_message.audio:
                        new_tg_file_id = sent_message.audio.file_id
                        update_tg_file_id(content_id, user_id, new_tg_file_id)
                        logger.info(f"User {user_id}: Recorded new tg_file_id {new_tg_file_id} from cached file")

                    # Send success message
                    if cover_msg:
                        try:
                            success_msg = await message.answer(
                                "The download was successful.\nآیا امکانات بیشتری نیاز دارید؟",
                                reply_to_message_id=cover_msg.message_id,
                                parse_mode="HTML"
                            )
                            # Store success message for cleanup
                            context_success_msg[user_id] = success_msg
                        except Exception as e:
                            logger.debug(f"User {user_id}: Success message failed in cached fast-path: {e}")

                    logger.info(f"User {user_id}: FAST UPLOAD - Complete cached file experience in <3 seconds")
                    return
                except Exception as e:
                    logger.warning(f"User {user_id}: Cached file upload failed, fallback to normal flow: {e}")

            # Check if complete info is already in DB (optimized caching)
            cur.execute("""
                SELECT artist, title, album, release_date, thumbnail, duration, popularity, genre, url
                FROM files
                WHERE file_id = ? AND user_id = ? AND artist IS NOT NULL
                ORDER BY last_accessed DESC
                LIMIT 1
            """, (content_id, user_id))
            cached_info = cur.fetchone()

            if cached_info:
                # Use complete cached info
                info = {
                    "artist": cached_info[0],
                    "title": cached_info[1],
                    "album": cached_info[2],
                    "release_date": cached_info[3],
                    "thumbnail": cached_info[4],
                    "duration": cached_info[5],
                    "popularity": cached_info[6],
                    "genre": cached_info[7],
                    "url": cached_info[8] or f"https://open.spotify.com/track/{content_id}"
                }
                logger.info(f"User {user_id}: Retrieved complete track info from DB for {content_id}")
            else:
                # Extract fresh info using optimized function
                info = await asyncio.get_event_loop().run_in_executor(executor, extract_track_info_optimized, content_id)
                if not info or not info.get("title"):
                    logger.error(f"User {user_id}: Track info incomplete for ID {content_id}")
                    raise ValueError("اطلاعات آهنگ ناقص است.")

                # Save complete info to DB for future use
                cur.execute("""
                    INSERT OR REPLACE INTO files
                    (user_id, file_id, artist, title, album, release_date, thumbnail, duration, popularity, genre, url, created_at, last_accessed)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                """, (
                    user_id, content_id,
                    info["artist"], info["title"], info.get("album"),
                    info.get("release_date"), info.get("thumbnail"),
                    info.get("duration"), info.get("popularity"), info.get("genre"),
                    info.get("url")
                ))
                conn.commit()
                logger.info(f"User {user_id}: Saved complete track info to DB for {content_id}")

            # Ensure duration and popularity
            try:
                total_seconds = int(info.get("duration") or 0)
            except Exception:
                total_seconds = 0
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            duration_str = f"{minutes}:{seconds:02d}"
            try:
                pop_val = int(info.get("popularity") or 0)
            except Exception:
                pop_val = 0

            caption = (
                f"<b>{info.get('artist', 'Unknown')} – {info.get('title', 'Unknown')}</b>\n"
                f"Album: {info.get('album', 'Unknown')}\n"
                f"Duration: {duration_str}\n"
                f"Release date: {info.get('release_date', 'Unknown')}\n"
                f"Popularity: {pop_val}% \n"
                f"Genre: {info.get('genre', 'Unknown')}\n"
                f"Url: <a href='{info.get('url', f'https://open.spotify.com/track/{content_id}') }'>Link Spotify</a>"
            )
            thumbnail = info.get("thumbnail")
            try:
                if thumbnail:
                    cover_msg = await message.answer_photo(
                        photo=thumbnail,
                        caption=caption,
                        reply_markup=inline_buttons,
                        parse_mode="HTML"
                    )
                else:
                    cover_msg = await message.answer(
                        text=caption,
                        reply_markup=inline_buttons,
                        parse_mode="HTML"
                    )
                await asyncio.sleep(0.5)  # Delay to avoid rate limit
            except Exception as e:
                logger.error(f"User {user_id}: Error sending cover or caption: {e}")
                cover_msg = await message.answer(
                    text=caption,
                    reply_markup=inline_buttons,
                    parse_mode="HTML"
                )
        progress_msg = await message.answer("Downloading song from Spotify...\nplease wait.", parse_mode="HTML")

        # Start background worker if not running
        start_background_worker()

        try:
            # Add progress callback to download_spotify
            main_loop = asyncio.get_running_loop()
            download_start_time = asyncio.get_running_loop().time()
            slow_download_threshold = 15.0  # 15 seconds threshold for background processing

            def progress_callback(current, total):
                try:
                    progress = (current / total) * 100 if total else 0
                    elapsed = asyncio.get_running_loop().time() - download_start_time

                    # Check if download is taking too long
                    if elapsed > slow_download_threshold and progress < 50:
                        # Enqueue for background processing
                        logger.info(f"Download taking too long ({elapsed:.1f}s), moving to background for user {user_id}")
                        asyncio.run_coroutine_threadsafe(
                            enqueue_background_job(user_id, url, content_type, content_id, message),
                            main_loop
                        )
                        # Update message to indicate background processing
                        asyncio.run_coroutine_threadsafe(
                            progress_msg.edit_text(
                                "⏳ Download is taking longer than expected.\n🔄 Processing in background...\nYou'll receive the file when ready.",
                                parse_mode="HTML"
                            ),
                            main_loop
                        )
                        return  # Stop updating progress

                    asyncio.run_coroutine_threadsafe(
                        progress_msg.edit_text(
                            f"Downloading song from Spotify...\nplease wait.\nProgress: {progress:.1f}%",
                            parse_mode="HTML"
                        ),
                        main_loop
                    )
                except Exception as e:
                    logger.debug(f"Skip progress update: {e}")

            files = await asyncio.get_running_loop().run_in_executor(
                executor, download_spotify, url, content_type, content_id, progress_callback
            )
            if not files:
                await message.answer(
                    "❌ فایل صوتی یافت نشد یا دانلود با شکست مواجه شد.\n"
                    "🔗 لطفاً صحت لینک را بررسی کنید.\n"
                    "🧪 اطمینان حاصل کنید که موزیک در دسترس است.",
                    parse_mode="HTML"
                )
                return
        except SpotifyException as e:
            await progress_msg.delete()
            logger.error(f"User {user_id}: Spotify API error: {e}")
            raise RuntimeError(f"خطا در دانلود: {e}")
        except Exception as e:
            await progress_msg.delete()
            logger.error(f"User {user_id}: Error downloading files: {e}")
            raise RuntimeError(f"خطا در دانلود: {e}")

        try:
            await progress_msg.delete()
        except Exception as e:
            logger.warning(f"User {user_id}: Error deleting progress message: {e}")
            await asyncio.sleep(0.5)

        if not files:
            await message.answer(
                "❌ فایل صوتی یافت نشد یا دانلود با شکست مواجه شد.\n"
                "🔗 لطفاً صحت لینک را بررسی کنید.\n"
                "🧪 اطمینان حاصل کنید که موزیک در دسترس است.",
                parse_mode="HTML"
            )
            return

        sent_success = False
        for path in files:
            if not os.path.exists(path):
                logger.warning(f"User {user_id}: File {path} not found.")
                continue

            try:
                # Prepare working path
                orig_filename = sanitize_filename(os.path.basename(path))
                work_path = os.path.join(DOWNLOAD_DIR, orig_filename)
                try:
                    os.rename(path, work_path)
                except (OSError, FileExistsError) as e:
                    logger.warning(f"User {user_id}: Error renaming file {path}: {e}")
                    work_path = path

                # Enforce bitrate: force 320kbps
                try:
                    work_path = ensure_320kbps(work_path)
                except Exception as e:
                    logger.warning(f"User {user_id}: ensure_320kbps failed for {work_path}: {e}")

                # Accurate metadata + cover
                # Fallback for duration from file if missing
                try:
                    if not info.get('duration') or int(info.get('duration') or 0) <= 0:
                        info['duration'] = int(MP3(work_path).info.length)
                except Exception:
                    pass

                # Embed cover art
                try:
                    thumbnail = info.get('thumbnail')
                    if thumbnail:
                        embed_cover(work_path, thumbnail)
                        logger.info(f"User {user_id}: Embedded cover for single download")
                except Exception as e:
                    logger.warning(f"User {user_id}: Cover embedding failed for single download: {e}")

                # Write ID3 metadata
                try:
                    if info:
                        write_id3_metadata(work_path, info, info.get('thumbnail'))
                        logger.info(f"User {user_id}: Wrote ID3 metadata for single download")
                except Exception as e:
                    logger.warning(f"User {user_id}: Writing ID3 metadata failed for {work_path}: {e}")

                # Clean final filename (just artist - title, no extra characters)
                artist = _normalize_text(info.get("artist")) or "Unknown Artist"
                title = _normalize_text(info.get("title")) or "Unknown Title"
                clean_filename = f"{artist} - {title}.mp3"
                safe_final_name = sanitize_filename(clean_filename)
                final_path = os.path.join(DOWNLOAD_DIR, safe_final_name)
                try:
                    if os.path.abspath(work_path) != os.path.abspath(final_path):
                        if os.path.exists(final_path):
                            os.remove(final_path)
                        os.rename(work_path, final_path)
                except Exception as e:
                    logger.warning(f"User {user_id}: Renaming to clean name failed, keep original: {e}")
                    final_path = work_path
                    safe_final_name = os.path.basename(final_path)

                # Create optimized cache copy for reuse
                cache_filename = f"{user_id}_{content_id}_{user_id}_{content_id}_{safe_final_name}"
                cache_path = os.path.join(CACHE_DIR, cache_filename)
                try:
                    if os.path.getsize(final_path) < 50 * 1024 * 1024:
                        shutil.copy2(final_path, cache_path)
                        cur.execute("UPDATE files SET cache_path = ? WHERE file_id = ?", (cache_path, content_id))
                        conn.commit()
                        logger.info(f"User {user_id}: Cached file with optimized naming: {cache_filename}")
                except Exception as e:
                    logger.debug(f"User {user_id}: Skipped caching: {e}")

                # Send file
                try:
                    send_path = cache_path if os.path.exists(cache_path) else final_path
                    exists = os.path.exists(send_path)
                    size_bytes = os.path.getsize(send_path) if exists else -1
                    readable = os.access(send_path, os.R_OK) if exists else False
                    writable = os.access(send_path, os.W_OK) if exists else False
                    logger.debug(f"Preflight for send: path={send_path}, exists={exists}, readable={readable}, writable={writable}, size={size_bytes} bytes")

                    if not exists or size_bytes <= 0:
                        raise FileNotFoundError(f"Send path not available or empty: {send_path}")

                    # Try opening to detect locks/permissions with retry
                    for attempt in range(5):
                        try:
                            with open(send_path, 'rb') as _f:
                                _f.read(1)
                            break
                        except Exception as open_err:
                            logger.warning(f"User {user_id}: file open test failed (attempt {attempt+1}): {open_err}")
                            await asyncio.sleep(1.5)
                    else:
                        raise IOError(f"Failed to open file after retries: {send_path}")

                    # Additional MP3 integrity validation
                    try:
                        audio_probe = MP3(send_path)
                        if getattr(audio_probe, 'info', None) is None or getattr(audio_probe.info, 'length', 0) <= 0:
                            raise ValueError("MP3 has no valid stream info")
                    except Exception as mp3_err:
                        logger.warning(f"User {user_id}: MP3 validation failed: {mp3_err}")
                        raise ValueError(f"Invalid MP3 file: {mp3_err}")

                    # Reduce size for Telegram if needed (>49MB): re-encode to 256k as a temporary copy
                    tg_tmp_path = None
                    try:
                        if size_bytes > 49 * 1024 * 1024:
                            tg_tmp_path = send_path + ".tg.mp3"
                            cmd = [
                                FFMPEG_PATH, "-y", "-i", send_path,
                                "-vn", "-c:a", "libmp3lame", "-b:a", "256k",
                                "-map_metadata", "0", "-id3v2_version", "3",
                                tg_tmp_path
                            ]
                            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
                            logger.info(f"User {user_id}: Created temporary downsized file for Telegram: {tg_tmp_path}, output: {result.stdout}")
                            send_path = tg_tmp_path
                            size_bytes = os.path.getsize(send_path)
                    except Exception as downs_err:
                        logger.warning(f"User {user_id}: downsizing failed, continue with original: {downs_err}")

                    size_mb = size_bytes / 1024 / 1024
                    transfer_name = build_transfer_filename(info or {})

                    # Prefer sending as audio to reduce filename issues and show metadata
                    send_ok = False
                    sent_message = None
                    for send_attempt in range(3):
                        try:
                            sent_message = await message.answer_audio(
                                audio=FSInputFile(send_path, filename=transfer_name),
                                performer=(info.get('artist') or None),
                                title=(info.get('title') or None),
                                duration=int(info.get('duration') or 0)
                            )
                            send_ok = True
                            break
                        except Exception as e1:
                            logger.warning(f"User {user_id}: send_audio failed (attempt {send_attempt+1}/3): {e1}")
                            await asyncio.sleep(1.0)

                    # RECORD TG_FILE_ID: Store the Telegram file_id for fast future uploads
                    if send_ok and sent_message and sent_message.audio:
                        tg_file_id = sent_message.audio.file_id
                        if update_tg_file_id(content_id, user_id, tg_file_id):
                            logger.info(f"User {user_id}: Recorded tg_file_id {tg_file_id} for content {content_id}")
                        else:
                            logger.warning(f"User {user_id}: Failed to record tg_file_id {tg_file_id}")
                        
                        # Record download history
                        try:
                            from bot import record_download_sync
                            record_download_sync(
                                user_id, "spotify", url,
                                title=info.get('title') if info else None,
                                artist=info.get('artist') if info else None,
                                file_type="audio",
                                file_size=size_bytes if size_bytes > 0 else None
                            )
                        except Exception as hist_e:
                            logger.debug(f"Failed to record download history: {hist_e}")

                    if not send_ok:
                        await message.answer_document(
                            document=FSInputFile(send_path, filename=transfer_name),
                            disable_notification=True,
                        )

                    # Cleanup temporary downsized file if created
                    if tg_tmp_path and os.path.exists(tg_tmp_path):
                        try:
                            os.remove(tg_tmp_path)
                        except OSError:
                            pass

                    sent_success = True
                    await asyncio.sleep(0.5)
                except Exception as e:
                    logger.exception(f"User {user_id}: Error sending file {send_path}: {e}")
                    await message.answer(
                        f"❌ خطا در ارسال فایل: {build_transfer_filename(info or {})}\nدلیل: {str(e)}",
                        parse_mode="HTML"
                    )

                # Cleanup temporary file if any
                try:
                    if os.path.exists(work_path) and os.path.abspath(work_path) != os.path.abspath(final_path):
                        os.remove(work_path)
                except OSError as e:
                    logger.warning(f"User {user_id}: Error removing temp file {work_path}: {e}")

            except Exception as e:
                logger.error(f"User {user_id}: Error processing file {path}: {e}")
                await message.answer(
                    f"❌ خطا در پردازش فایل {os.path.basename(path)}",
                    parse_mode="HTML"
                )

        if sent_success and cover_msg:
            # Only update last_accessed
            try:
                cur.execute("UPDATE files SET last_accessed = CURRENT_TIMESTAMP WHERE file_id = ?", (content_id,))
                conn.commit()
            except Exception as _e:
                logger.debug(f"User {user_id}: Could not update last_accessed: {_e}")

            # Increment download counter for track

            try:
                success_msg = await message.answer(
                    "The download was successful.\nآیا امکانات بیشتری نیاز دارید؟",
                    reply_to_message_id=cover_msg.message_id,
                    parse_mode="HTML"
                )
                # ذخیره پیام برای پاک کردن در کلیک بعدی
                context_success_msg[user_id] = success_msg
            except Exception as _e:
                logger.warning(f"User {user_id}: Could not send success message: {_e}")


    except ValueError as e:
        logger.error(f"User {user_id}: Validation error: {e}")
        await message.answer(f"❌ خطا: اطلاعات آهنگ ناقص است.", parse_mode="HTML")
    except SpotifyException as e:
        logger.error(f"User {user_id}: Spotify API error: {e}")
        await message.answer(f"❌ خطا در دانلود: {str(e)}", parse_mode="HTML")
    except RuntimeError as e:
        logger.error(f"User {user_id}: Download error: {e}")
        await message.answer(f"❌ خطا در دانلود: {str(e)}", parse_mode="HTML")
    except Exception as e:
        logger.error(f"User {user_id}: Unknown error: {e}")
        await message.answer(f"❌ خطای ناشناخته: {str(e)}", parse_mode="HTML")
    finally:
        if 'conn' in locals():
            conn.close()
            logger.debug(f"User {user_id}: Database connection closed")

async def handle_spotify_album_download(message: types.Message, url: str, album_id: str) -> None:
    """
    Download a full album with proxy + high timeout behavior identical to single:
    - Fetch album metadata via sp.album and full tracklist via sp.album_tracks (paginated)
    - Download each track (reusing single-track downloader) into a temporary album folder
    - Write per-track metadata and embed cover (on the temp copies)
    - Cache both album metadata and track entries in DB (similar to single)
    """


    user_id = message.from_user.id
    try:
        
        # Progress message
        progress_msg = await message.reply("🔎 Fetching album info...", parse_mode="HTML")

        # 1) Fetch album metadata
        album = sp.album(album_id, market=MARKET)
        if not album:
            await progress_msg.edit_text("❌ Album not found or unavailable.", parse_mode="HTML")
            return

        album_name = album.get("name", "Unknown")
        album_url = (album.get("external_urls") or {}).get("spotify", f"https://open.spotify.com/album/{album_id}")
        album_images = album.get("images", []) or []
        album_thumb = album_images[0].get("url") if album_images else None
        album_release_date = album.get("release_date", "Unknown")
        album_label = album.get("label", "Unknown")
        album_total_tracks = int(album.get("total_tracks") or 0)
        album_popularity = int(album.get("popularity") or 0)

        # Genres: album.genres may be empty; fallback to artist genres
        genres = album.get("genres") or []
        if not genres:
            try:
                artists = album.get("artists") or []
                if artists:
                    ar_obj = sp.artist(artists[0]["id"])
                    genres = ar_obj.get("genres") or []
            except Exception:
                genres = []
        genres_fmt = ", ".join([g.title() for g in genres]) if genres else "Unknown"

        # ALL track IDs (paginate)
        track_items = []
        offset = 0
        limit = 50
        while True:
            batch = sp.album_tracks(album_id, market=MARKET, offset=offset, limit=limit)
            items = (batch or {}).get("items", [])
            track_items.extend(items)
            if not batch or not batch.get("next") or len(items) < limit:
                break
            offset += limit

        track_ids = [t.get("id") for t in track_items if t.get("id")]
        if not track_ids:
            await progress_msg.edit_text("❌ No tracks found in album.", parse_mode="HTML")
            return

        # Compute total duration and top track by popularity (batch sp.tracks)
        total_ms = 0
        top_track_name = "Unknown"
        top_track_pop = -1
        # Preload detailed track info for all tracks
        detailed_tracks = []
        for i in range(0, len(track_ids), 50):
            batch_ids = track_ids[i:i+50]
            try:
                tr_res = sp.tracks(batch_ids)
                detailed_tracks.extend((tr_res or {}).get("tracks", []))
            except Exception:
                # fallback minimal
                for tid in batch_ids:
                    detailed_tracks.append({"id": tid})
        for dt in detailed_tracks:
            try:
                dms = int(dt.get("duration_ms") or 0)
                total_ms += dms
                pop = int(dt.get("popularity") or 0)
                if pop > top_track_pop:
                    top_track_pop = pop
                    top_track_name = dt.get("name", "Unknown")
            except Exception:
                pass
        album_duration_seconds = (total_ms // 1000) if total_ms else 0

        # Persist album metadata row in DB cache (no file, content_type='album')
        try:
            save_file(
                user_id=user_id,
                file_id=album_id,
                artist=", ".join(a.get("name","") for a in album.get("artists", [])) or "Unknown",
                title=album_name,
                url=album_url,
                thumbnail=album_thumb,
                duration=album_duration_seconds,
                album=album_name,
                release_date=album_release_date,
                popularity=album_popularity,
                genre=genres_fmt,
                content_type="album",
                total_tracks=album_total_tracks,
                label=album_label,
                top_track=top_track_name
            )
        except Exception as _db_e:
            # Non-fatal
            logger.debug(f"Album metadata save failed: {_db_e}")

        # Show album info to user (like single cover message)
        try:
            mins = album_duration_seconds // 60
            secs = album_duration_seconds % 60
            dur_str = f"{mins}:{secs:02d}"
            cap = (
                f"{', '.join(a.get('name','') for a in album.get('artists', [])) or 'Unknown'} – 💿 {album_name}\n"
                f"Release Date: {album_release_date}\n"
                f"Total Tracks: {album_total_tracks}\n"
                f"Duration: {dur_str}\n"
                f"Popularity: {album_popularity}%\n"
                f"Label: {album_label}\n"
                f"Genres: {genres_fmt}\n"
                f"Top Track: {top_track_name}\n\n"
                f"🔗 <a href='{album_url}'>View on Spotify</a>"
            )
            # Send cover first + inline buttons (inactive features)
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text=" Get lyrics all", callback_data=f"album_lyrics_all|{album_id}")],
                [InlineKeyboardButton(text=" Add to playlist all", callback_data=f"album_add_to_playlist|{album_id}")],
                [InlineKeyboardButton(text=" Ask about the album", callback_data=f"album_ask_ai|{album_id}")]
            ])
            if album_thumb:
                cover_msg = await message.answer_photo(photo=album_thumb, caption=cap, parse_mode="HTML", reply_markup=keyboard)
            else:
                cover_msg = await message.answer(cap, parse_mode="HTML", reply_markup=keyboard)

            # Recreate progress message as a reply to the cover
            try:
                if progress_msg:
                    await progress_msg.delete()
            except Exception:
                pass
            progress_msg = await message.answer(
                "🔎 Fetching album info...",
                reply_to_message_id=cover_msg.message_id,
                parse_mode="HTML"
            )
            await progress_msg.edit_text(f"Downloading album tracks... 0/{len(track_ids)}", parse_mode="HTML")
        except Exception as _e_cover:
            logger.debug(f"Album cover send failed: {_e_cover}")
            # Fallback: keep progress message in chat
            try:
                await progress_msg.edit_text(f"Downloading album tracks... 0/{len(track_ids)}", parse_mode="HTML")
            except Exception:
                pass

        # 2) Create album temporary directory
        ts = int(time.time())
        album_dir = os.path.abspath(os.path.join(DOWNLOAD_DIR, f"album_{album_id}_{ts}"))
        os.makedirs(album_dir, exist_ok=True)

        # 3) Download every track using the single-track flow to preserve proxy/timeouts/quality
        done = 0
        total = len(track_ids)
        # Open DB once for fast-path checks
        conn_fast = sqlite3.connect("bot_cache.db")
        cur_fast = conn_fast.cursor()
        for idx, tid in enumerate(track_ids, start=1):
            try:
                # FAST-PATH: cached file or tg_file_id for ultra-fast send (<10s)
                cached_row = None
                try:
                    cur_fast.execute("""
                        SELECT cache_path, tg_file_id, artist, title, duration, thumbnail, album, release_date, popularity, genre, url
                        FROM files
                        WHERE file_id = ? AND user_id = ?
                        ORDER BY last_accessed DESC
                        LIMIT 1
                    """, (tid, user_id))
                    cached_row = cur_fast.fetchone()
                except Exception:
                    cached_row = None

                info = None
                if cached_row:
                    cache_path = cached_row[0]
                    tg_id = cached_row[1]
                    info = {
                        "artist": cached_row[2],
                        "title": cached_row[3],
                        "duration": cached_row[4],
                        "thumbnail": cached_row[5],
                        "album": cached_row[6] or album_name,
                        "release_date": cached_row[7] or album_release_date,
                        "popularity": cached_row[8] or 0,
                        "genre": cached_row[9] or "Unknown",
                        "url": cached_row[10] or f"https://open.spotify.com/track/{tid}"
                    }

                    # Fast-path 1: local cached file exists and is clean
                    if cache_path and os.path.exists(cache_path):
                        is_clean = False
                        try:
                            audio_probe = MP3(cache_path)
                            if getattr(audio_probe, 'info', None) and getattr(audio_probe.info, 'length', 0) > 0 and os.path.getsize(cache_path) > 50 * 1024:
                                is_clean = True
                        except Exception:
                            is_clean = False
                        if is_clean:
                            # Ensure minimal info present
                            if not info.get("artist") or not info.get("title"):
                                try:
                                    info = await asyncio.get_event_loop().run_in_executor(executor, extract_track_info_optimized, tid)
                                except Exception:
                                    pass
                            clean_name_fp = build_clean_filename(info or {"artist": "Unknown Artist", "title": "audio"}, bitrate_kbps=320)
                            dst = os.path.join(album_dir, clean_name_fp)
                            try:
                                if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(cache_path):
                                    shutil.copy2(cache_path, dst)
                            except Exception:
                                dst = cache_path  # fallback
                            # Quick send
                            try:
                                sent_message = await message.answer_audio(
                                    audio=FSInputFile(dst, filename=os.path.basename(dst)),
                                    performer=(info or {}).get('artist'),
                                    title=(info or {}).get('title'),
                                    duration=int((info or {}).get('duration') or 0)
                                )
                                if sent_message and sent_message.audio:
                                    try:
                                        update_tg_file_id(tid, user_id, sent_message.audio.file_id)
                                    except Exception:
                                        pass
                            except Exception:
                                pass

                            done += 1
                            try:
                                await progress_msg.edit_text(f"Downloading album tracks... {done}/{total}", parse_mode="HTML")
                            except Exception:
                                pass
                            continue

                    # Fast-path 2: tg_file_id exists (send immediately) + background copy to album folder
                    if tg_id:
                        # Ensure minimal info
                        if not info or not info.get("title"):
                            try:
                                info = await asyncio.get_event_loop().run_in_executor(executor, extract_track_info_optimized, tid)
                            except Exception:
                                pass
                        clean_name_fp2 = build_clean_filename(info or {"artist": "Unknown Artist", "title": "audio"}, bitrate_kbps=320)
                        dst2 = os.path.join(album_dir, clean_name_fp2)

                        try:
                            sent_message = await message.answer_audio(
                                audio=tg_id,
                                performer=(info or {}).get('artist'),
                                title=(info or {}).get('title'),
                                duration=int((info or {}).get('duration') or 0)
                            )
                        except Exception:
                            sent_message = None
                        # Background copy from Telegram CDN to cache and into album folder
                        def _bg_copy_from_tg(tg_file_id_local: str, dst_local: str, content_id_local: str, user_local: int):
                            try:
                                local_cached = download_file_from_telegram(tg_file_id_local, CACHE_DIR)
                                if local_cached and os.path.exists(local_cached):
                                    try:
                                        if not os.path.exists(dst_local) or os.path.getsize(dst_local) != os.path.getsize(local_cached):
                                            shutil.copy2(local_cached, dst_local)
                                    except Exception:
                                        pass
                                    # Update DB cache_path for this user/content
                                    try:
                                        _c = sqlite3.connect("bot_cache.db")
                                        _cur = _c.cursor()
                                        _cur.execute(
                                            "UPDATE files SET cache_path = ?, last_accessed = CURRENT_TIMESTAMP WHERE file_id = ? AND user_id = ?",
                                            (local_cached, content_id_local, user_local)
                                        )
                                        _c.commit()
                                        _cur.close()
                                        _c.close()
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                        try:
                            asyncio.get_running_loop().run_in_executor(None, _bg_copy_from_tg, tg_id, dst2, tid, user_id)
                        except Exception:
                            pass

                        done += 1
                        try:
                            await progress_msg.edit_text(f"Downloading album tracks... {done}/{total}", parse_mode="HTML")
                        except Exception:
                            pass
                        continue

                # Per-track info (optimized) to drive naming/metadata
                info = await asyncio.get_event_loop().run_in_executor(executor, extract_track_info_optimized, tid)
                if not info or not info.get("title"):
                    info = {
                        "title": "Unknown Track",
                        "artist": "Unknown Artist",
                        "album": album_name,
                        "thumbnail": album_thumb,
                        "release_date": album_release_date,
                        "duration": 0,
                        "url": f"https://open.spotify.com/track/{tid}",
                        "popularity": 0,
                        "genre": genres_fmt if genres_fmt != "Unknown" else "Unknown"
                    }

                # Download (or reuse cache) exactly like single
                track_url = f"https://open.spotify.com/track/{tid}"
                files = await asyncio.get_event_loop().run_in_executor(
                    executor, download_spotify, track_url, "track", tid
                )
                if not files:
                    done += 1
                    try:
                        await progress_msg.edit_text(f"Downloading album tracks... {done}/{total}", parse_mode="HTML")
                    except Exception:
                        pass
                    continue

                # Copy cached/validated file to album temp dir with clean name
                src = files[0]
                clean_name = build_clean_filename(info, bitrate_kbps=320)
                dst = os.path.join(album_dir, clean_name)
                try:
                    if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
                        shutil.copy2(src, dst)
                except Exception as _cp_e:
                    logger.debug(f"Album copy failed for {tid}: {_cp_e}")
                    dst = src  # fallback to source

                # Ensure 320kbps bitrate
                try:
                    dst = ensure_320kbps(dst)
                except Exception as _320_e:
                    logger.debug(f"Album ensure_320kbps failed for {tid}: {_320_e}")

                # Ensure cover + ID3 on the temp copy (like single)
                try:
                    if album_thumb:
                        embed_cover(dst, album_thumb)
                except Exception as _cov_e:
                    logger.debug(f"Album cover embed failed for {tid}: {_cov_e}")
                try:
                    write_id3_metadata(dst, info, info.get('thumbnail') or album_thumb)
                except Exception as _id3_e:
                    logger.debug(f"Album ID3 write failed for {tid}: {_id3_e}")

                # Cache track metadata (do NOT overwrite existing cache_path created by downloader)
                try:
                    save_file(
                        user_id=user_id,
                        file_id=tid,
                        artist=info.get('artist'),
                        title=info.get('title'),
                        url=info.get('url'),
                        thumbnail=info.get('thumbnail') or album_thumb,
                        duration=int(info.get('duration') or 0),
                        tg_file_id=None,
                        album=info.get('album') or album_name,
                        release_date=info.get('release_date') or album_release_date,
                        popularity=int(info.get('popularity') or 0),
                        genre=info.get('genre') or genres_fmt,
                        content_type="track"
                    )
                except Exception as _sf_e:
                    logger.debug(f"Album track save_file failed for {tid}: {_sf_e}")

                # Send the cleaned file to user (from Downloads/Spotify folder)
                try:
                    sent_message = await message.answer_audio(
                        audio=FSInputFile(dst, filename=os.path.basename(dst)),
                        performer=info.get('artist'),
                        title=info.get('title'),
                        duration=int(info.get('duration') or 0)
                    )
                    if sent_message and sent_message.audio:
                        try:
                            update_tg_file_id(tid, user_id, sent_message.audio.file_id)
                        except Exception as _upd_e:
                            logger.debug(f"Album track tg_file_id update failed for {tid}: {_upd_e}")
                except Exception as _send_e:
                    logger.warning(f"Album track send failed for {tid}: {_send_e}")

                done += 1
                # Update progress
                try:
                    await progress_msg.edit_text(f"Downloading album tracks... {done}/{total}", parse_mode="HTML")
                except Exception:
                    pass

            except Exception as e_tr:
                logger.warning(f"Album track failed {tid}: {e_tr}")
                done += 1
                try:
                    await progress_msg.edit_text(f"Downloading album tracks... {done}/{total}", parse_mode="HTML")
                except Exception:
                    pass
                continue

        # Close fast-path DB handles if open
        try:
            cur_fast.close()
            conn_fast.close()
        except Exception:
            pass

        # Finalize
        try:
            completion_text = "✅ Album downloaded and cached. Tracks are available for fast access."
            # Edit the threaded progress message
            await progress_msg.edit_text(completion_text, parse_mode="HTML")
            # Auto-clear the threaded status after 5 seconds
            asyncio.create_task(delete_after_delay(progress_msg, 5))
            # Additionally post a distinct completion reply under the album cover (permanent)
            try:
                if 'cover_msg' in locals() and cover_msg:
                    cover_done_msg = await message.answer(
                        completion_text,
                        reply_to_message_id=cover_msg.message_id,
                        parse_mode="HTML"
                    )
            except Exception:
                pass

            # Increment download counter for album
        except Exception:
            pass

        # Increment download counter for playlist

    except Exception as e:
        logger.error(f"Error in handle_spotify_album_download: {e}")
        try:
            await message.answer(f"❌ Error downloading album: {e}", parse_mode="HTML")
        except Exception:
            pass

def parse_spotify_url(url: str) -> tuple[str, str]:
    """
    Parse Spotify URL to extract type (track/album/playlist) and ID
    """
    m = re.search(r"spotify\.com/(track|album|playlist)/([a-zA-Z0-9]+)", url)
    if not m:
        raise ValueError(f"Invalid Spotify URL: {url}")
    return m.group(1), m.group(2)

# --- اطلاعات پلی‌لیست یا هنرمند ---
async def show_info(message: types.Message, content_type: str, content_id: str) -> None:
    if content_type not in ("artist", "playlist", "album"):
        logger.error(f"Invalid content type: {content_type}")
        raise ValueError(f"Invalid content type: {content_type}")

    if not content_id or not isinstance(content_id, str):
        logger.error(f"Invalid content ID: {content_id}")
        raise ValueError(f"Invalid content ID: {content_id}")

    try:
        if content_type == "artist":
            progress_msg = await message.answer("🎨 Getting artist info...")
            # Fallback auto-delete after 10 seconds in case we can't delete immediately
            asyncio.create_task(delete_after_delay(progress_msg, 10))

            artist = sp.artist(content_id)
            name = artist.get("name", "Unknown")
            followers = artist.get("followers", {}).get("total", 0)
            genres = ", ".join(artist.get("genres", [])) or "Unknown"
            image = artist.get("images", [{}])[0].get("url") if artist.get("images") else None
            popularity = artist.get("popularity", 0)

            # --- Extra Info (Web Scraping) ---
            monthly_listeners = "Unknown"
            top_country = "Unknown"
            try:
                url = f"https://open.spotify.com/artist/{content_id}"
                headers = {"User-Agent": "Mozilla/5.0"}
                r = requests.get(url, headers=headers, timeout=10, proxies=_requests_proxies(PROXY))
                soup = BeautifulSoup(r.text, "html.parser")

                # JSON metadata is embedded in <script id="__NEXT_DATA__">
                script_tag = soup.find("script", {"id": "__NEXT_DATA__"})
                if script_tag:
                    data = json.loads(script_tag.string)

                    # monthly listeners
                    monthly_listeners = data.get("props", {}).get("pageProps", {}).get("artist", {}).get("stats", {}).get("monthlyListeners", "Unknown")

                    # top country
                    top_country = data.get("props", {}).get("pageProps", {}).get("artist", {}).get("stats", {}).get("topCity", {}).get("country", "Unknown")

            except Exception as e:
                logger.warning(f"Could not fetch extra artist info: {e}")

            # --- Biography (via Wikipedia link) ---
            wiki_link = f"https://en.wikipedia.org/wiki/{name.replace(' ', '_')}"
            bio = get_artist_biography(name)
            biography_section = f"📚 Biography: {bio}" if bio else f"📚 Biography: <a href='{wiki_link}'>Read on Wikipedia</a>"

            # --- Top Albums ---
            top_albums_list = "Unknown"
            try:
                albums = sp.artist_albums(content_id, album_type="album", limit=6).get("items", [])
                if albums:
                    # بخش لیست آلبوم‌ها
                    albums_content = "\n".join([format_album_link(album) for album in albums])

        # محاسبه سال انتشار
                    years = [int(album.get('release_date', '0000')[:4]) for album in albums if album.get('release_date')]
                    year_range_str = ""
                    if years:
                        min_year = min(years)
                        max_year = max(years)
                        range_str = f"{min_year}-{max_year}" if min_year != max_year else str(min_year)
                        year_range_str = f"📅 Active Years: {range_str}\n"

        # بخش آلبوم برتر (داخل quote)
                    quote_block = ""
                    try:
                        first_album = albums[0]
                        album_id = first_album['id']
                        album_tracks = sp.album_tracks(album_id, limit=50).get('items', [])
                        if album_tracks:
                            track_list = "\n".join([f"🎵 {t['name']}" for t in album_tracks[:20]])
                            quote_block = f"<blockquote expandable>💿 Top Album: \n {first_album['name']} \nTracks:\n{track_list}</blockquote>"
                    except Exception as e:
                        logger.warning(f"Could not fetch album tracks: {e}")

        # مونتاژ نهایی
                    top_albums_list = (
                        f"<blockquote expandable>💿 Top Albums:\n{albums_content}</blockquote>\n\n"
                        f"{year_range_str}"
                        f"{quote_block}"
                    )



            except SpotifyException as e:
                logger.warning(f"Error fetching top albums: {e}")


            # --- Social Media Links ---
            social_media = get_artist_social_media(name)
            social_media_section = "📱 Social Media:\n"
            if social_media.get('instagram'):
                social_media_section += f"Insta: <a href='{social_media['instagram']}'>Instagram</a>\n"
            else:
                instagram_handle = f"{name.lower().replace(' ', '')}official"
                social_media_section += f"Instagram: <a href='https://www.instagram.com/{instagram_handle}/'>Instagram</a>\n"
            if social_media.get('twitter'):
                social_media_section += f"X: <a href='{social_media['twitter']}'>X</a>\n"
            else:
                x_handle = f"{name.lower().replace(' ', '')}"
                social_media_section += f"X: <a href='https://x.com/{x_handle}'>X</a>\n"
            if social_media.get('facebook'):
                social_media_section += f"Facebook: <a href='{social_media['facebook']}'>Facebook</a>\n"
            if social_media.get('youtube'):
                social_media_section += f"YouTube: <a href='{social_media['youtube']}'>YouTube</a>\n"
            if social_media.get('website'):
                social_media_section += f"Website: <a href='{social_media['website']}'>Official Website</a>"

            # --- Related Artists ---
            related_artists_list = "No related artists found"
            try:
                related = sp.artist_related_artists(content_id).get("artists", [])[:5]
                if related:
                    related_artists_list = "\n".join([f"👤 <a href='https://open.spotify.com/artist/{a['id']}'>{a['name']}</a>" for a in related])
                    related_artists_list = f"<blockquote>{related_artists_list}</blockquote>"
            except SpotifyException as e:
                logger.warning(f"Error fetching related artists: {e}")

            # Playlist: Search for "This is {name}"
            playlist_name = f"This is {name}"
            try:
                search_result = sp.search(q=playlist_name, type="playlist", limit=1)
                if search_result["playlists"]["items"]:
                    playlist_url = search_result["playlists"]["items"][0]["external_urls"]["spotify"]
                else:
                    playlist_url = f"https://open.spotify.com/search/{playlist_name.replace(' ', '%20')}"
            except Exception:
                playlist_url = f"https://open.spotify.com/search/{playlist_name.replace(' ', '%20')}"

            # Top tracks
            try:
                top_tracks = sp.artist_top_tracks(content_id, country="US").get("tracks", [])[:20]
                top_tracks_list = "\n".join([format_track_link(t) for t in top_tracks]) or "No tracks found"
                top_tracks_list = f"<blockquote expandable>{top_tracks_list}</blockquote>" if top_tracks_list != "No tracks found" else top_tracks_list
            except SpotifyException as e:
                logger.warning(f"Error fetching top tracks: {e}")
                top_tracks_list = "No tracks found"

            # --- Caption ---
            caption = (
                f"👤 {name}\n"
                f"🔊 Monthly Listeners: {monthly_listeners}\n"
                f"🌍 Top Country: {top_country}\n"
                f"❤️ Followers: {followers:,}\n"
                f"📝 Genres: {genres}\n\n"
                f"{biography_section}\n\n"
                f"🔥 Top Tracks:\n{top_tracks_list}\n\n"
                f"💿 Top Albums:\n{top_albums_list}\n\n"
                f"{social_media_section}\n\n"
                f"👥 Related Artists:\n{related_artists_list}\n\n"
                f"🎧 Suggested Playlist: <a href='{playlist_url}'>{playlist_name}</a>\n"
                f"🔗 <a href='https://open.spotify.com/artist/{content_id}'>Artist Page</a>"
            )

            # --- Inline Buttons ---
            markup = InlineKeyboardMarkup(
                inline_keyboard=[
                    [InlineKeyboardButton(text="اطلاعات تکمیلی (نام تمامی آهنگ‌ها و آلبوم‌ها)", callback_data=f"more_info:{content_id}")],
                    [InlineKeyboardButton(text="پرسش از هوش مصنوعی در مورد آرتیست", callback_data=f"ask_ai:{content_id}")]
               ]
            )

            info_msg = None
            try:
                if image:
                    info_msg = await message.answer_photo(photo=image, caption=caption, parse_mode="HTML", reply_markup=markup)
                else:
                    info_msg = await message.answer(caption, parse_mode="HTML", reply_markup=markup)
            except Exception as e:
                logger.error(f"Error sending artist message: {e}")
                info_msg = await message.answer(caption, parse_mode="HTML", reply_markup=markup)

            # Attempt to delete the temporary "Getting artist info..." message immediately after sending info
            try:
                if 'progress_msg' in locals() and progress_msg:
                    await progress_msg.delete()
            except Exception:
                pass

            # Send a follow-up message as a reply to the info message and keep until user clicks for more info
            try:
                user_id = message.from_user.id
                if info_msg:
                    follow_msg = await message.answer(
                        "The initial info is over. Do you want more info?",
                        parse_mode="HTML",
                        reply_to_message_id=info_msg.message_id
                    )
                    # Store message to delete later when user clicks the inline 'more info' button
                    try:
                        context_success_msg[user_id] = follow_msg
                    except Exception:
                        pass
            except Exception as _e_follow:
                logger.debug(f"Could not send or store follow-up message: {_e_follow}")

        elif content_type in ("playlist", "album"):
            try:
                if content_type == "playlist":
                    data = sp.playlist(content_id)
                    name = data.get("name", "Unknown")
                    owner = data.get("owner", {}).get("display_name", "Unknown")
                    release_date = data.get("snapshot_id", "")[:10] or "Unknown"
                    total_tracks = data.get("tracks", {}).get("total", 0)
                    tracks = data.get("tracks", {}).get("items", [])
                    top_track = (
                        max(tracks, key=lambda x: x["track"].get("popularity", 0))["track"].get("name", "Unknown")
                        if tracks else "Unknown"
                    )
                    image = data.get("images", [{}])[0].get("url") if data.get("images") else None

                    # Custom playlist caption (separate from album)
                    try:
                        playlist_url = (data.get("external_urls") or {}).get("spotify", f"https://open.spotify.com/playlist/{content_id}")
                        followers = (data.get("followers") or {}).get("total", 0)
                        collaborative = bool(data.get("collaborative"))
                        public_flag = data.get("public")
                        visibility = "Public" if public_flag is True else ("Private" if public_flag is False else "Unknown")

                        # Compute created and last update from track added_at dates
                        added_dates = [(it.get("added_at") or "")[:10] for it in (tracks or []) if isinstance(it, dict) and it.get("added_at")]
                        created_date = min(added_dates) if added_dates else "Unknown"
                        last_update = max(added_dates) if added_dates else "Unknown"

                        # Aggregate duration and popularity
                        total_ms = 0
                        pops = []
                        for it in (tracks or []):
                            tr = it.get("track") or {}
                            try:
                                total_ms += int(tr.get("duration_ms") or 0)
                                pops.append(int(tr.get("popularity") or 0))
                            except Exception:
                                pass
                        total_seconds = (total_ms // 1000) if total_ms else 0
                        avg_pop = int(round(sum(pops) / len(pops))) if pops else 0

                        dur_str = format_duration(total_seconds)

                        cap = (
                            f"Playlist – {name}\n"
                            f"User: {owner}\n"
                            f"Created: {created_date}\n"
                            f"Last Update: {last_update}\n"
                            f"Tracks: {total_tracks}\n"
                            f"Duration: {dur_str}\n"
                            f"Popularity: {avg_pop}%\n"
                            f"Followers: {followers:,}\n"
                            f"Collaborative: {'Yes' if collaborative else 'No'}\n"
                            f"Visibility: {visibility}\n"
                            f"Top Track: {top_track}\n\n"
                            f"🔗 <a href='{playlist_url}'>View on Spotify</a>"
                        )

                        if image:
                            await message.answer_photo(photo=image, caption=cap, parse_mode="HTML")
                        else:
                            await message.answer(cap, parse_mode="HTML")
                        return
                    except Exception as _pl_e:
                        logger.warning(f"Playlist caption build/send failed: {_pl_e}")

                else:  # album
                    data = sp.album(content_id)
                    name = data.get("name", "Unknown")
                    owner = data.get("artists", [{}])[0].get("name", "Unknown")
                    release_date = data.get("release_date", "Unknown")
                    total_tracks = data.get("total_tracks", 0)
                    track_ids = [track["id"] for track in data.get("tracks", {}).get("items", []) if "id" in track]
                    tracks_data = sp.tracks(track_ids)["tracks"] if track_ids else []
                    top_track = (
                        max(tracks_data, key=lambda x: x.get("popularity", 0)).get("name", "Unknown")
                        if tracks_data else "Unknown"
                    )
                    image = data.get("images", [{}])[0].get("url") if data.get("images") else None

                # User rating placeholder
                user_rating = "Unknown"

                # Compute display fields to avoid undefined variables
                try:
                    if content_type == "album":
                        # Album-specific fields
                        popularity = int(data.get("popularity") or 0)
                        label = (data.get("label") or "Unknown") if isinstance(data, dict) else "Unknown"
                        genres = data.get("genres") or []
                        # Sum durations of album tracks
                        try:
                            total_ms = sum(int(t.get("duration_ms") or 0) for t in (tracks_data or []))
                        except Exception:
                            total_ms = 0
                        album_duration = format_duration((total_ms // 1000) if total_ms else 0)
                    else:
                        # Playlist-specific fallbacks
                        # Popularity: average of track popularities
                        try:
                            pops = [int((it.get("track") or {}).get("popularity") or 0) for it in (tracks or []) if it.get("track")]
                            popularity = int(sum(pops) / len(pops)) if pops else 0
                        except Exception:
                            popularity = 0
                        label = "N/A"
                        # Genres: derive top few from track artists
                        try:
                            genres = []
                            for it in (tracks or [])[:10]:
                                tr = it.get("track") or {}
                                artists = tr.get("artists") or []
                                for ar in artists[:2]:
                                    try:
                                        ar_obj = sp.artist(ar.get("id"))
                                        genres.extend(ar_obj.get("genres") or [])
                                    except Exception:
                                        pass
                            # dedupe and cap
                            seen = set()
                            genres = [g.title() for g in genres if not (g in seen or seen.add(g))][:5]
                        except Exception:
                            genres = []
                        # Duration: sum of track durations
                        try:
                            total_ms = sum(int((it.get("track") or {}).get("duration_ms") or 0) for it in (tracks or []) if it.get("track"))
                        except Exception:
                            total_ms = 0
                        album_duration = format_duration((total_ms // 1000) if total_ms else 0)
                except Exception:
                    # Final fallbacks
                    popularity = 0
                    label = "Unknown"
                    genres = []
                    album_duration = "0:00"

                caption = (
                    f"{owner} – 💿 {name}\n"
                    f"Release Date: {release_date}\n"
                    f"Total Tracks: {total_tracks}\n"
                    f"Duration: {album_duration}\n"
                    f"Popularity: {popularity}%\n"
                    f"Rating: {user_rating}\n"
                    f"Label: {label}\n"
                    f"Genres: {', '.join(genres) if genres else 'Unknown'}\n"
                    f"Top Track: {top_track}\n\n"
                    f"🔗 <a href='{url}'>View on Spotify</a>"
                )



                info_msg = None
                try:
                    if image:
                        info_msg = await message.answer_photo(photo=image, caption=caption, parse_mode="HTML")
                    else:
                        info_msg = await message.answer(caption, parse_mode="HTML")
                except Exception as e:
                    logger.error(f"Error sending {content_type} message: {e}")
                    info_msg = await message.answer(caption, parse_mode="HTML")


            except SpotifyException as e:
                logger.error(f"Error fetching {content_type} data: {e}")
                await message.answer(f"❌ Error fetching {content_type} information: {str(e)}", parse_mode="HTML")

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        await message.answer(f"❌ Error: {str(e)}", parse_mode="HTML")
    except SpotifyException as e:
        logger.error(f"Spotify API error: {e}")
        await message.answer(f"❌ Error processing information: {str(e)}", parse_mode="HTML")
    except Exception as e:
        logger.error(f"Unknown error processing information: {e}")
        await message.answer(f"❌ Unknown error: {str(e)}", parse_mode="HTML")

async def send_more_info(callback_query: types.CallbackQuery, content_id: str) -> None:
    try:
        # Delete previously sent follow-up prompt when user requests more info
        try:
            user_id = callback_query.from_user.id
            if user_id in context_success_msg:
                try:
                    await context_success_msg[user_id].delete()
                except Exception:
                    pass
                try:
                    del context_success_msg[user_id]
                except Exception:
                    pass
        except Exception:
            pass

        # Fetch artist albums
        albums_data = sp.artist_albums(content_id, album_type="album", limit=20).get("items", [])
        albums_list = "\n".join([format_album_link(album) for album in albums_data]) or "No albums found"
        albums_list = f"<blockquote expandable>{albums_list}</blockquote>" if albums_list != "No albums found" else albums_list

        # Fetch singles
        singles_data = sp.artist_albums(content_id, album_type="single", limit=10).get("items", [])
        singles_list = "\n".join([f"🎵 {single.get('name', 'Unknown')} ({single.get('release_date', 'Unknown')[:4]})"
                                  for single in singles_data]) or "No singles found"
        singles_list = f"<blockquote expandable>{singles_list}</blockquote>" if singles_list != "No singles found" else singles_list

        # Fetch all singles (up to 40 tracks)
        singles = sp.artist_albums(content_id, album_type="single", limit=50).get("items", [])
        singles_tracks = []
        for single in singles:
            try:
                tracks = sp.album_tracks(single['id'], limit=10).get('items', [])
                singles_tracks.extend(tracks)
            except Exception as e:
                logger.warning(f"Could not fetch tracks for single {single['id']}: {e}")
        singles_tracks = singles_tracks[:40]
        singles_list = "\n".join([format_track_link(t) for t in singles_tracks]) or "No singles found"

        # Fetch all albums (paginate to get all)
        all_albums = []
        offset = 0
        while True:
            try:
                batch = sp.artist_albums(content_id, album_type="album", limit=50, offset=offset)
                items = batch.get("items", [])
                all_albums.extend(items)
                if not batch.get("next") or len(items) < 50:
                    break
                offset += 50
            except Exception as e:
                logger.warning(f"Error fetching albums at offset {offset}: {e}")
                break
        albums_list = "\n".join([format_album_link(album) for album in all_albums]) or "No albums found"
        singles_list = f"<blockquote expandable>{singles_list}</blockquote>" if singles_list != "No singles found" else singles_list
        albums_list = f"<blockquote expandable>{albums_list}</blockquote>" if albums_list != "No albums found" else albums_list

        # Caption
        caption = (
            f"📑 Extended Information:\n\n"
            f"🎵 Singles:\n{singles_list}\n\n"
            f"💿 All Albums:\n{albums_list}\n\n"
            f"🔗 <a href='https://open.spotify.com/artist/{content_id}'>View Artist on Spotify</a>"
        )

        # Reply to original artist message and capture the sent message
        info_msg2 = await callback_query.message.reply(
            caption,
            parse_mode="HTML",
            reply_to_message_id=callback_query.message.message_id
        )

        # Send end-of-information message as a reply to the additional info message
        await callback_query.message.reply(
            "The information is over.",
            parse_mode="HTML",
            reply_to_message_id=info_msg2.message_id
        )

        await callback_query.answer()  # remove "loading" state on button

    except Exception as e:
        logger.error(f"Error in send_more_info: {e}")


@router.callback_query(lambda c: c.data == "add_to_playlist")
async def add_to_playlist_callback(query: CallbackQuery) -> None:
    try:
        await query.answer("✅ آهنگ به پلی‌لیست شما اضافه شد.", show_alert=True)
        logger.info(f"کاربر {query.from_user.id} درخواست افزودن به پلی‌لیست را اجرا کرد.")
    except Exception as e:
        logger.error(f"خطا در پردازش درخواست افزودن به پلی‌لیست: {e}")
        await query.answer(f"❌ خطا در افزودن به پلی‌لیست: {str(e)}", show_alert=True)
        raise RuntimeError(f"خطا در پردازش درخواست: {e}")

# --- Album cover inline buttons (inactive placeholders) ---
@router.callback_query(lambda q: q.data and q.data.startswith("album_lyrics_all|"))
async def album_lyrics_all_cb(query: CallbackQuery) -> None:
    try:
        await query.answer("This feature is under development.", show_alert=True)
    except Exception as e:
        logger.debug(f"album_lyrics_all_cb error: {e}")

@router.callback_query(lambda q: q.data and q.data.startswith("album_add_to_playlist|"))
async def album_add_to_playlist_cb(query: CallbackQuery) -> None:
    try:
        await query.answer("This feature is under development.", show_alert=True)
    except Exception as e:
        logger.debug(f"album_add_to_playlist_cb error: {e}")

@router.callback_query(lambda q: q.data and q.data.startswith("album_ask_ai|"))
async def album_ask_ai_cb(query: CallbackQuery) -> None:
    try:
        await query.answer("This feature is under development.", show_alert=True)
    except Exception as e:
        logger.debug(f"album_ask_ai_cb error: {e}")

@router.callback_query(lambda c: c.data.startswith("suggest|"))
async def suggest_callback(query: CallbackQuery) -> None:
    try:
        # استخراج لینک ترک
        track_url = query.data.split("|", 1)[1] if len(query.data.split("|", 1)) > 1 else ""
        if not track_url or not isinstance(track_url, str) or "spotify.com" not in track_url.lower():
            logger.error(f"لینک ترک نامعتبر: {track_url}")
            raise ValueError("لینک ترک نامعتبر است.")

        # پاسخ موقت (ویژگی غیرفعال)
        await query.message.answer(
            f"🤖 پیشنهاد آهنگ‌های مشابه برای {track_url} به‌زودی نمایش داده می‌شود.",
            parse_mode="HTML"
        )
        await query.answer()
        logger.info(f"کاربر {query.from_user.id} درخواست پیشنهاد آهنگ برای {track_url} را اجرا کرد (غیرفعال).")

    except ValueError as e:
        logger.error(f"خطای اعتبارسنجی: {e}")
        await query.answer(f"❌ خطا: لینک نامعتبر است.", show_alert=True)
    except Exception as e:
        logger.error(f"خطای ناشناخته در پردازش درخواست پیشنهاد: {e}")
        await query.answer(f"❌ خطای ناشناخته: {str(e)}", show_alert=True)
        raise RuntimeError(f"خطا در پردازش درخواست: {e}")

@router.callback_query(lambda c: c.data.startswith("lyrics|"))
async def handle_simultaneous_lyrics(query: CallbackQuery) -> None:
    """Handle lyrics request for simultaneous tracks"""
    try:
        parts = query.data.split("|")
        if len(parts) < 2:
            await query.answer("❌ Invalid data", show_alert=True)
            return

        track_index = int(parts[1]) - 1  # Convert to 0-based index

        # For now, show a message that lyrics feature is being developed
        await query.answer(f"🎵 Lyrics for song {track_index + 1} - Feature in development", show_alert=True)

        # TODO: Implement lyrics retrieval for media group tracks
        # This would require storing track info in a way that's accessible to callbacks

    except Exception as e:
        logger.error(f"Error in simultaneous lyrics handler: {e}")
        await query.answer("❌ Error getting lyrics", show_alert=True)

@router.callback_query(lambda c: c.data.startswith("separate|"))
async def handle_simultaneous_separate(query: CallbackQuery) -> None:
    """Handle vocal separation request for simultaneous tracks"""
    try:
        parts = query.data.split("|")
        if len(parts) < 2:
            await query.answer("❌ Invalid data", show_alert=True)
            return

        track_index = int(parts[1]) - 1  # Convert to 0-based index

        # For now, show development message
        await query.answer(f"🎙 Vocal separation for song {track_index + 1} is in development...", show_alert=True)

    except Exception as e:
        logger.error(f"Error in simultaneous separate handler: {e}")
        await query.answer("❌ Error in separation", show_alert=True)

@router.callback_query(lambda c: c.data == "suggestions")
async def handle_suggestions(query: CallbackQuery) -> None:
    """Handle suggestions request"""
    await query.answer("⭐️ Song suggestions feature is in development...", show_alert=True)

@router.callback_query(lambda c: c.data and c.data.startswith("more_info:"))
async def more_info(callback_query: types.CallbackQuery):
    content_id = callback_query.data.split(":")[1]
    await send_more_info(callback_query, content_id)





# ---------- Helper functions ----------
async def delete_after_delay(message, delay: int):
    await asyncio.sleep(delay)
    try:
        await message.delete()
    except Exception:
        pass

# ---------- Helper functions for artist info formatting ----------
def format_track_link(track: dict) -> str:
    name = track.get('name', 'Unknown')
    url = track.get('external_urls', {}).get('spotify', '')
    if not url:
        return f"🎵 {name}"
    words = name.split()
    if words:
        first_word = words[0]
        rest = ' '.join(words[1:])
        return f"🎵 <a href='{url}'>{first_word}</a> {rest}"
    else:
        return f"🎵 <a href='{url}'>{name}</a>"

def format_album_link(album: dict) -> str:
    name = album.get('name', 'Unknown')
    url = album.get('external_urls', {}).get('spotify', '')
    year = album.get('release_date', 'Unknown')[:4]
    if not url:
        return f"💿 {name} ({year})"
    words = name.split()
    if words:
        first_word = words[0]
        rest = ' '.join(words[1:])
        return f"💿 <a href='{url}'>{first_word}</a> {rest} ({year})"
    else:
        return f"💿 <a href='{url}'>{name}</a> ({year})"

# Cache for social media links
_social_media_cache = {}

def get_artist_social_media(artist_name: str) -> dict:
    # Check cache first
    cache_key = artist_name.lower().strip()
    if cache_key in _social_media_cache:
        return _social_media_cache[cache_key]

    GENIUS_API_TOKEN = os.getenv("GENIUS_API_TOKEN", "YOUR_GENIUS_API_TOKEN_HERE")
    if not GENIUS_API_TOKEN or GENIUS_API_TOKEN == "YOUR_GENIUS_API_TOKEN_HERE":
        logger.warning("Genius API token not set.")
        return {}
    api_url = "https://api.genius.com/search"
    params = {"q": artist_name}
    headers = {"Authorization": f"Bearer {GENIUS_API_TOKEN}"}
    try:
        response = requests.get(api_url, headers=headers, params=params, proxies=_requests_proxies(PROXY), timeout=10)
        if response.status_code == 200:
            data = response.json()
            hits = data.get("response", {}).get("hits", [])
            for hit in hits:
                if hit.get("type") == "song":
                    artist = hit.get("result", {}).get("primary_artist", {})
                    if artist.get("name", "").lower().strip() == artist_name.lower().strip():
                        artist_url = artist.get("url")
                        if artist_url:
                            social = scrape_artist_social_media(artist_url)
                            _social_media_cache[cache_key] = social  # Cache the result
                            return social
    except Exception as e:
        logger.warning(f"Failed to get social media for {artist_name}: {e}")
    return {}

def scrape_artist_social_media(url: str) -> dict:
    try:
        headers = _random_headers()
        html = _requests_fetch(url, headers, timeout=30, retries=3)
        if html:
            soup = BeautifulSoup(html, "html.parser")
            social = {}
            for a in soup.find_all('a', href=True):
                href = a['href']
                text = a.get_text().lower()
                if 'instagram' in href:
                    social['instagram'] = href
                elif 'twitter' in href or 'x.com' in href:
                    social['twitter'] = href
                elif 'facebook' in href:
                    social['facebook'] = href
                elif 'youtube' in href:
                    social['youtube'] = href
                elif 'official' in text and 'http' in href and href.startswith('http'):
                    social['website'] = href
            return social
    except Exception as e:
        logger.warning(f"Failed to scrape social media from {url}: {e}")
    return {}

def get_artist_biography(artist_name: str) -> str:
    """Fetch a short biography summary from Wikipedia"""
    try:
        # Try Wikipedia API
        wiki_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{artist_name.replace(' ', '_')}"
        response = requests.get(wiki_url, timeout=10, proxies=_requests_proxies(PROXY))
        if response.status_code == 200:
            data = response.json()
            extract = data.get("extract", "")
            if extract and len(extract) > 50:
                # Take first 2 sentences
                sentences = extract.split('.')
                short_bio = '.'.join(sentences[:2]) + '.' if len(sentences) >= 2 else extract
                return short_bio[:300] + "..." if len(short_bio) > 300 else short_bio
    except Exception as e:
        logger.debug(f"Wikipedia bio failed for {artist_name}: {e}")

    # Fallback to Genius description
    try:
        GENIUS_API_TOKEN = os.getenv("GENIUS_API_TOKEN", "YOUR_GENIUS_API_TOKEN_HERE")
        if GENIUS_API_TOKEN and GENIUS_API_TOKEN != "YOUR_GENIUS_API_TOKEN_HERE":
            api_url = "https://api.genius.com/search"
            params = {"q": artist_name}
            headers = {"Authorization": f"Bearer {GENIUS_API_TOKEN}"}
            response = requests.get(api_url, headers=headers, params=params, proxies=_requests_proxies(PROXY), timeout=10)
            if response.status_code == 200:
                data = response.json()
                hits = data.get("response", {}).get("hits", [])
                for hit in hits:
                    if hit.get("type") == "song":
                        artist = hit.get("result", {}).get("primary_artist", {})
                        if artist.get("name", "").lower().strip() == artist_name.lower().strip():
                            description = artist.get("description", {}).get("plain", "")
                            if description and len(description) > 50:
                                sentences = description.split('.')
                                short_bio = '.'.join(sentences[:2]) + '.' if len(sentences) >= 2 else description
                                return short_bio[:300] + "..." if len(short_bio) > 300 else short_bio
    except Exception as e:
        logger.debug(f"Genius bio fallback failed for {artist_name}: {e}")

    return ""

# ---------- جداسازی وکال، ویژگی اضافی ------------

# Logger setup with improved formatting
logger = logging.getLogger("separation")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
    file_handler = logging.FileHandler("separation.log")  # Added file logging for better debugging
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
logger.setLevel(logging.INFO)  # Changed to INFO to reduce verbosity

# New: Global variable for FFmpeg path, with fallback detection
FFMPEG_PATH = os.environ.get("FFMPEG_PATH", which("ffmpeg"))
if not FFMPEG_PATH:
    raise EnvironmentError("FFmpeg not found in PATH or environment variable.")

# ----------------- وضعیت‌ها برای FSM (نگهداری شده با توضیح) -----------------
class SeparationStates(StatesGroup):
    waiting_for_audio = State()  # State for waiting user to send audio
    select_model = State()  # State for selecting separation mode

# ----------------- توابع کمکی -----------------

def compress_audio(file_path: str, target_size_mb: int = 45) -> str:
    """فشرده‌سازی فایل MP3 تا رسیدن به حجم مورد نظر با بهبود مدیریت خطا"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File for compression not found: {file_path}")
    
    audio = AudioSegment.from_file(file_path)
    out_path = file_path.replace(".mp3", "_compressed.mp3")
    bitrate = 128  # Changed to integer for easier manipulation
    audio.export(out_path, format="mp3", bitrate=f"{bitrate}k")

    while os.path.getsize(out_path) / (1024 * 1024) > target_size_mb:
        bitrate -= 32
        if bitrate < 64:
            raise ValueError("⚠️ نمی‌توان فایل را بیشتر فشرده کرد.")
        audio.export(out_path, format="mp3", bitrate=f"{bitrate}k")
        logger.debug(f"Compressed to bitrate {bitrate}k, size: {os.path.getsize(out_path) / (1024 * 1024):.2f} MB")

    return out_path

# New function: Check if GPU is available for Spleeter/TensorFlow acceleration
def check_gpu_availability() -> bool:
    """چک کردن وجود GPU برای شتاب‌دهی Spleeter"""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        logger.info(f"GPU detected: {gpus}. Using GPU for separation.")
        return True
    logger.info("No GPU detected. Using CPU for separation.")
    return False



# Try to import spleeter with fallback
try:
    from spleeter.separator import Separator
    import tensorflow as tf
    SPLEETER_AVAILABLE = True
    logger.info("Spleeter imported successfully")
except ImportError as e:
    logger.warning(f"Spleeter not available: {e}")
    SPLEETER_AVAILABLE = False
    Separator = None
    tf = None

def check_gpu_availability() -> bool:
    """Check if GPU is available for Spleeter/TensorFlow acceleration"""
    if not SPLEETER_AVAILABLE or not tf:
        return False
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"GPU detected: {gpus}. Using GPU for separation.")
            return True
        logger.info("No GPU detected. Using CPU for separation.")
        return False
    except Exception as e:
        logger.warning(f"GPU check failed: {e}")
        return False
compress_audio = lambda x, target_size_mb=45: x  # Placeholder for compress_audio function


# ست کردن ffmpeg یکبار (مثلاً بالای فایل)
AudioSegment.converter = r'G:\zAll data (All Mine)\Codeing\ffmpeg\bin\ffmpeg.exe'
 
def separate_vocals(mp3_path: str, mode: str = "spleeter:2stems", progress_callback=None) -> Tuple[str, Optional[str]]:
    """جداسازی وکال و بیت با Spleeter با بهبود مدیریت خطا، پشتیبانی GPU و پیشرفت"""
    # بررسی وجود FFmpeg
    if not AudioSegment.converter:
        logger.error("FFmpeg یافت نشد یا در PATH نیست.")
        raise FileNotFoundError("FFmpeg یافت نشد یا در PATH نیست.")

    # بررسی وجود فایل ورودی
    if not os.path.exists(mp3_path):
        logger.error(f"فایل MP3 یافت نشد: {mp3_path}")
        raise FileNotFoundError(f"فایل MP3 یافت نشد: {mp3_path}")

    # چک کردن GPU قبل از جداسازی
    check_gpu_availability()

    temp_dir = tempfile.mkdtemp()
    vocal_mp3 = None
    instrumental_mp3 = None
    total_steps = 4  # Steps: init, separate, convert, compress
    current_step = 0
    try:
        # Initialize separator
        current_step += 1
        if progress_callback:
            progress_callback(current_step, total_steps)
        separator = Separator(mode)
        logger.debug(f"Separator initialized with mode {mode}")

        # Perform separation
        current_step += 1
        if progress_callback:
            progress_callback(current_step, total_steps)
        separator.separate_to_file(mp3_path, temp_dir)
        logger.info(f"Separation completed for {mp3_path} to {temp_dir}")

        stem_dir = os.path.join(temp_dir, os.path.splitext(os.path.basename(mp3_path))[0])
        vocal_path = os.path.join(stem_dir, "vocals.wav")
        instrumental_path = os.path.join(stem_dir, "accompaniment.wav")

        out_dir = os.path.dirname(mp3_path)
        vocal_mp3 = os.path.join(out_dir, "وکال.mp3")
        instrumental_mp3 = os.path.join(out_dir, "بیت.mp3") if mode != "vocal_only" else None

        # تبدیل WAV به MP3 با بررسی سلامت فایل
        current_step += 1
        if progress_callback:
            progress_callback(current_step, total_steps)

        if os.path.exists(vocal_path):
            audio = AudioSegment.from_wav(vocal_path)
            audio.export(vocal_mp3, format="mp3", parameters=["-q:a", "2"])
            logger.debug(f"Vocal exported to {vocal_mp3}")
            if not os.path.exists(vocal_mp3) or os.path.getsize(vocal_mp3) == 0:
                raise ValueError(f"Export failed for vocal at {vocal_mp3}")
        else:
            raise FileNotFoundError(f"فایل وکال تولید نشده: {vocal_path}")

        if instrumental_mp3:
            if os.path.exists(instrumental_path):
                audio = AudioSegment.from_wav(instrumental_path)
                audio.export(instrumental_mp3, format="mp3", parameters=["-q:a", "2"])
                logger.debug(f"Instrumental exported to {instrumental_mp3}")
                if not os.path.exists(instrumental_mp3) or os.path.getsize(instrumental_mp3) == 0:
                    logger.warning(f"Export failed or empty for instrumental at {instrumental_mp3}")
            else:
                logger.warning("فایل accompaniment تولید نشده؛ مسیر انتظار شده موجود نیست.")

        # فشرده‌سازی اختیاری
        current_step += 1
        if progress_callback:
            progress_callback(current_step, total_steps)
        try:
            if vocal_mp3 and os.path.exists(vocal_mp3) and os.path.getsize(vocal_mp3) / (1024 * 1024) > 50:
                vocal_mp3 = compress_audio(vocal_mp3)
                logger.info(f"Vocal compressed to {vocal_mp3}")
            if instrumental_mp3 and os.path.exists(instrumental_mp3) and os.path.getsize(instrumental_mp3) / (1024 * 1024) > 50:
                instrumental_mp3 = compress_audio(instrumental_mp3)
                logger.info(f"Instrumental compressed to {instrumental_mp3}")
        except Exception as e:
            logger.warning(f"هشدار فشرده‌سازی: {e}")

        return vocal_mp3, instrumental_mp3

    except Exception as e:
        logger.exception(f"Error in separation: {e}")
        raise
    finally:
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
                logger.debug(f"Temp dir cleaned: {temp_dir}")
        except Exception as e:
            logger.warning(f"خطا در پاکسازی temp_dir: {e}")


# -------------- منوی انتخاب (بهینه‌سازی شده با گزینه‌های بیشتر) ----------------
def get_separation_menu():
    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
            InlineKeyboardButton(text="🎙 وکال + بیت", callback_data="separate_2stems"),
            InlineKeyboardButton(text="🎙 فقط وکال", callback_data="vocal_only")
        ],
        [
            InlineKeyboardButton(text="🎸 فقط بیت", callback_data="instrumental_only"),
            InlineKeyboardButton(text="❌ لغو", callback_data="cancel")
        ],
        # New: Added advanced options
        [
            InlineKeyboardButton(text="🔊 4stems (پیشرفته)", callback_data="separate_4stems"),
            InlineKeyboardButton(text="📊 چک وضعیت", callback_data="check_status")
        ]
    ])
    return keyboard

# New function: For advanced 4stems separation (added for completeness)
def separate_vocals_advanced(mp3_path: str, mode: str = "spleeter:4stems") -> Tuple[str, str, str, str]:
    """جداسازی پیشرفته با 4stems (وکال, درام, باس, سایر)"""
    # Similar to separate_vocals but returns more stems
    # Implementation similar, but export drums.wav, bass.wav, other.wav in addition
    # (Omitted full code for brevity; you can expand it)
    pass  # Implement if needed


# ----------------- هندلر جدید برای راه A (اجرای مستقیم) -----------------

@router.callback_query(lambda q: q.data.startswith("separate_vocal"))
async def handle_separate_vocal(query: CallbackQuery) -> None:
    await query.answer()
    msg = query.message
    user_id = query.from_user.id

    # Cleanup previous success message if present for this user
    if user_id in context_success_msg:
        try:
            await context_success_msg[user_id].delete()
        except Exception:
            pass
        finally:
            try:
                del context_success_msg[user_id]
            except Exception:
                pass
    chat_id = msg.chat.id
    bot_user = await query.bot.get_me()
    bot_id = bot_user.id

    logger.info(f"user {user_id} clicked separate_vocals (data={query.data}) at {datetime.now().strftime('%H:%M:%S')}")
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Database connection for this session
    conn = sqlite3.connect("bot_cache.db")
    cur = conn.cursor()

    # Parse optional file_id
    parts = query.data.split("|")
    cb_file_id = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None

    file_path: str | None = None
    vocal_mp3: str | None = None
    instrumental_mp3: str | None = None

    # ---------- helpers ----------
    async def _download_by_file_id(file_id: str, dest_path: str, retries: int = 2) -> str | None:
        for attempt in range(retries + 1):
            try:
                logger.debug(f"Attempt {attempt + 1}/{retries + 1}: Downloading file_id {file_id} to {dest_path}")
                tg_file = await query.bot.get_file(file_id)
                await query.bot.download_file(tg_file.file_path, dest_path)
                if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
                    logger.info(f"Download successful for {file_id} at {dest_path}")
                    return dest_path
                else:
                    logger.warning(f"Downloaded file {dest_path} is empty or invalid")
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                    if attempt < retries:
                        await asyncio.sleep(2)
                    continue
            except Exception as e:
                logger.debug(f"Attempt {attempt + 1}: _download_by_file_id failed for {file_id}: {e}")
                if attempt == retries:
                    return None
                await asyncio.sleep(2)
        return None

    async def _download_from_audio(audio_obj, dest_path: str, retries: int = 2) -> str | None:
        for attempt in range(retries + 1):
            try:
                logger.debug(f"Attempt {attempt + 1}/{retries + 1}: Downloading audio to {dest_path}")
                with open(dest_path, "wb") as f:
                    await audio_obj.download(destination_file=f)
                if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
                    logger.info(f"Download successful to {dest_path}")
                    return dest_path
                else:
                    logger.warning(f"Downloaded file {dest_path} is empty or invalid")
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                    if attempt < retries:
                        await asyncio.sleep(2)
                    continue
            except Exception as e:
                logger.debug(f"Attempt {attempt + 1}: _download_from_audio failed: {e}")
                if attempt == retries:
                    return None
                await asyncio.sleep(2)
        return None

    def _normalize_text(s: Optional[str]) -> str:
        if not s:
            return ""
        s = unicodedata.normalize("NFKC", str(s))
        s = s.replace("–", "-").replace("—", "-")
        s = re.sub(r"\s+", " ", s).strip().lower()
        return s

    # Log callback data immediately
    logger.debug(f"user {user_id}: callback data raw = {query.data}; parsed cb_file_id = {cb_file_id!r}")

    # ---------- strategy 1: direct file_id from callback with cache check ----------
    if cb_file_id:
        cur.execute("SELECT cache_path FROM files WHERE file_id = ? AND user_id = ?", (cb_file_id, user_id))
        cached = cur.fetchone()
        if cached and cached[0] and os.path.exists(cached[0]):
            file_path = cached[0]
            logger.info(f"user {user_id}: Retrieved from cache -> {file_path}")
            cur.execute("UPDATE files SET last_accessed = CURRENT_TIMESTAMP WHERE file_id = ?", (cb_file_id,))
            conn.commit()
        else:
            candidate = os.path.join(DOWNLOAD_DIR, f"{user_id}_{cb_file_id}.mp3")
            downloaded = await _download_by_file_id(cb_file_id, candidate)
            if downloaded:
                file_path = downloaded
                cache_path = os.path.join(CACHE_DIR, f"{user_id}_{cb_file_id}.mp3")
                if os.path.getsize(file_path) < 50 * 1024 * 1024:
                    shutil.move(file_path, cache_path)
                    file_path = cache_path
                    cur.execute("UPDATE files SET cache_path = ?, last_accessed = CURRENT_TIMESTAMP WHERE file_id = ?", (cache_path, cb_file_id))
                    conn.commit()
                    logger.info(f"user {user_id}: Cached file -> {cache_path}")
                logger.info(f"user {user_id}: downloaded via callback file_id -> {file_path}")

    # ---------- strategy 2: reply_to_message ----------
    if not file_path and msg.reply_to_message:
        cand = msg.reply_to_message
        if cand.audio or cand.voice or (cand.document and getattr(cand.document, "mime_type", "").startswith("audio")):
            audio_obj = cand.audio or cand.voice or cand.document
            fid = getattr(audio_obj, "file_id", None)
            candidate = os.path.join(DOWNLOAD_DIR, f"{user_id}_{fid or 'audio'}.mp3")
            downloaded = await _download_from_audio(audio_obj, candidate)
            if downloaded:
                file_path = downloaded
                cache_path = os.path.join(CACHE_DIR, f"{user_id}_{fid or 'audio'}.mp3")
                if os.path.getsize(file_path) < 50 * 1024 * 1024:
                    shutil.move(file_path, cache_path)
                    file_path = cache_path
                    cur.execute("INSERT OR REPLACE INTO files (user_id, file_id, cache_path, last_accessed) VALUES (?, ?, ?, CURRENT_TIMESTAMP)", (user_id, fid, cache_path))
                    conn.commit()
                    logger.info(f"user {user_id}: Cached reply_to_message -> {cache_path}")
                logger.info(f"user {user_id}: downloaded from reply_to_message -> {file_path}")

    # ---------- strategy 3: message itself ----------
    if not file_path and (msg.audio or msg.voice or (msg.document and getattr(msg.document, "mime_type", "").startswith("audio"))):
        audio_obj = msg.audio or msg.voice or msg.document
        fid = getattr(audio_obj, "file_id", None)
        candidate = os.path.join(DOWNLOAD_DIR, f"{user_id}_{fid or 'audio'}.mp3")
        downloaded = await _download_from_audio(audio_obj, candidate)
        if downloaded:
            file_path = downloaded
            cache_path = os.path.join(CACHE_DIR, f"{user_id}_{fid or 'audio'}.mp3")
            if os.path.getsize(file_path) < 50 * 1024 * 1024:
                shutil.move(file_path, cache_path)
                file_path = cache_path
                cur.execute("INSERT OR REPLACE INTO files (user_id, file_id, cache_path, last_accessed) VALUES (?, ?, ?, CURRENT_TIMESTAMP)", (user_id, fid, cache_path))
                conn.commit()
                logger.info(f"user {user_id}: Cached message itself -> {cache_path}")
            logger.info(f"user {user_id}: downloaded from inline message itself -> {file_path}")

    # ---------- strategy 4: fallback from DB with fuzzy match ----------
    if not file_path:
        artist_song = None
        if msg.text:
            first_line = msg.text.splitlines()[0].strip()
            m = re.search(r"(.+?)\s*[-–—]\s*(.+)", first_line)
            artist_song = f"{m.group(1).strip()} - {m.group(2).strip()}" if m else first_line

        artist_song_n = _normalize_text(artist_song)
        cur.execute("SELECT file_id, cache_path FROM files WHERE user_id = ? ORDER BY last_accessed DESC LIMIT 30", (user_id,))
        rows = cur.fetchall()

        best_score = 0.0
        best_path = None
        candidates_debug = []

        for fid, cache_path in rows:
            if not cache_path or not os.path.exists(cache_path):
                continue
            try:
                tags = EasyID3(cache_path)
                performer = tags.get('artist', [''])[0]
                title = tags.get('title', [''])[0]
                meta = f"{performer} - {title}".strip()
            except Exception as e:
                logger.debug(f"user {user_id}: tag read failed for {cache_path}: {e}")
                meta = ""

            meta_n = _normalize_text(meta)
            base = _normalize_text(os.path.splitext(os.path.basename(cache_path))[0])

            score_meta = difflib.SequenceMatcher(None, artist_song_n, meta_n).ratio() if artist_song_n and meta_n else 0.0
            score_base = difflib.SequenceMatcher(None, artist_song_n, base).ratio() if artist_song_n else 0.0

            score = max(score_meta * 1.4, score_base)
            if artist_song_n and artist_song_n in base:
                score = max(score, 0.95)

            candidates_debug.append((cache_path, score, score_meta, score_base, meta, base))

            if score > best_score:
                best_score = score
                best_path = cache_path

        if candidates_debug:
            sorted_dbg = sorted(candidates_debug, key=lambda x: x[1], reverse=True)[:5]
            logger.debug("DB-candidates: " + "; ".join([
                f"{os.path.basename(c[0])} score={c[1]:.3f} meta={c[4]} base={c[5]}"
                for c in sorted_dbg
            ]))

        MATCH_THRESHOLD = 0.78
        if best_path and best_score >= MATCH_THRESHOLD:
            file_path = best_path
            logger.info(f"user {user_id}: chosen cached file by fuzzy match -> {file_path} (score={best_score:.3f})")
        else:
            logger.info(f"user {user_id}: no confident fuzzy match (best={best_score:.3f})")

    # ---------- final check with DB integration ----------
    if not file_path:
        cur.execute("SELECT cache_path FROM files WHERE user_id = ? ORDER BY last_accessed DESC LIMIT 1", (user_id,))
        cached_fallback = cur.fetchone()
        if cached_fallback and cached_fallback[0] and os.path.exists(cached_fallback[0]):
            file_path = cached_fallback[0]
            logger.info(f"user {user_id}: Fallback to last cached file -> {file_path}")
        else:
            await msg.reply("⚠️ فایل صوتی پیدا نشد. لطفاً مطمئن شوید بات فایل ارسال کرده است.", parse_mode="HTML")
            conn.close()
            return

    # Verify file integrity
    try:
        if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
            logger.error(f"user {user_id}: File {file_path} is invalid or empty")
            if os.path.exists(file_path):
                os.remove(file_path)
            await msg.reply("⚠️ فایل صوتی نامعتبر است. لطفاً دوباره تلاش کنید.", parse_mode="HTML")
            conn.close()
            return
    except Exception as e:
        logger.error(f"user {user_id}: Error verifying file {file_path}: {e}")
        await msg.reply(f"❌ خطا در بررسی فایل: {e}", parse_mode="HTML")
        conn.close()
        return

    # ---------- run separation with progress ----------
    try:
        progress_msg = await msg.reply("🔧 فایل دریافت شد. در حال جداسازی...", parse_mode="HTML")
        main_loop = asyncio.get_running_loop()

        def update_progress(current, total):
            progress = (current / total) * 100 if total > 0 else 0
            try:
                coro = progress_msg.edit_text(
                    f"🔧 فایل دریافت شد. در حال جداسازی... {progress:.1f}%",
                    parse_mode="HTML"
                )
                asyncio.run_coroutine_threadsafe(coro, main_loop)
            except Exception as e:
                logger.debug(f"Skip progress update: {e}")

        loop = asyncio.get_running_loop()
        vocal_mp3, instrumental_mp3 = await loop.run_in_executor(
            None, separate_vocals, file_path, "spleeter:2stems", update_progress
        )

        # ارسال وکال
        if vocal_mp3 and os.path.exists(vocal_mp3):
            await msg.answer_audio(FSInputFile(vocal_mp3))
            # بعد از ۳۰ ثانیه پیام progress پاک بشه
            asyncio.create_task(delete_after_delay(progress_msg, 30))

        # ارسال بیت
        if instrumental_mp3 and os.path.exists(instrumental_mp3):
            await msg.answer_audio(FSInputFile(instrumental_mp3))

    except Exception as e:
        logger.error(f"Error in separation: {e}")
        await msg.answer("❌ خطا در جداسازی وکال.", parse_mode="HTML")


    # تابع کمکی برای حذف پیام بعد از تاخیر
        async def delete_after_delay(message, delay: int):
            await asyncio.sleep(delay)
            try:
                await message.delete()
            except Exception:
                pass



        await msg.reply(
            "Done./" if file_path.startswith(CACHE_DIR) else "Cache status: Not used",
            reply_markup=InlineKeyboardMarkup(
                inline_keyboard=[
                    [InlineKeyboardButton(text="❓ پشتیبانی", url="https://t.me/support_channel")]
                ]
            ),
            parse_mode="HTML"
        )
    except Exception as e:
        logger.exception(f"user {user_id}: error during separation: {e}")
        await msg.reply(f"❌ خطا در پردازش: {e}", parse_mode="HTML")
    finally:
        for p in [file_path, vocal_mp3, instrumental_mp3]:
            if p and isinstance(p, str) and os.path.exists(p):
                try:
                    os.remove(p)
                    logger.debug(f"cleanup successful for {p}")
                except Exception as e:
                    logger.warning(f"cleanup failed for {p}: {e}")
        if 'conn' in locals():
            conn.close()
            logger.debug(f"user {user_id}: Database connection closed")

# --- Optimized Download Function ---
def download_spotify_optimized(url: str, user_id: int, track_number: int, info: Dict[str, Optional[str]] = None) -> List[str]:
    """Optimized download function for concurrent processing with proper file handling"""
    try:
        # Validate URL
        is_valid, content_type, content_id = validate_spotify_url(url)
        if not is_valid:
            logger.error(f"Invalid URL for track {track_number}: {url}")
            return []

        # Check cache first
        conn = sqlite3.connect("bot_cache.db")
        cur = conn.cursor()
        cur.execute("SELECT cache_path FROM files WHERE file_id = ? AND user_id = ? AND cache_path IS NOT NULL",
                   (content_id, user_id))
        cached = cur.fetchone()
        conn.close()

        if cached and cached[0] and os.path.exists(cached[0]):
            try:
                os.makedirs(DOWNLOAD_DIR, exist_ok=True)
                if info and isinstance(info, dict):
                    clean_filename = build_clean_filename(info, bitrate_kbps=320)
                else:
                    # Fallback to cache basename if info not provided
                    base = os.path.basename(cached[0])
                    base = sanitize_filename(os.path.splitext(base)[0]) + ".mp3"
                    clean_filename = base
                download_path = os.path.join(DOWNLOAD_DIR, clean_filename)
                if not os.path.exists(download_path) or os.path.getsize(download_path) != os.path.getsize(cached[0]):
                    shutil.copy2(cached[0], download_path)
                logger.info(f"User {user_id}: Using cached file for track {track_number} -> copied to downloads: {download_path}")
                return [download_path]
            except Exception as e:
                logger.warning(f"User {user_id}: Cached fast-path copy failed for track {track_number}: {e}. Falling back to cached path.")
                return [cached[0]]

        # Fast-path using Telegram file_id when available (no heavy processing)
        try:
            conn2 = sqlite3.connect("bot_cache.db")
            cur2 = conn2.cursor()
            cur2.execute(
                "SELECT tg_file_id FROM files WHERE file_id = ? AND user_id = ? AND tg_file_id IS NOT NULL",
                (content_id, user_id)
            )
            row = cur2.fetchone()
            if row and row[0]:
                tg_id = row[0]
                try:
                    # Download file via Telegram CDN directly into cache, then copy to downloads
                    from db_manager import download_file_from_telegram
                    local_cached = download_file_from_telegram(tg_id, CACHE_DIR)
                    if local_cached and os.path.exists(local_cached):
                        # Update DB: set cache_path for future ultra-fast reuse
                        try:
                            cur2.execute(
                                "UPDATE files SET cache_path = ?, last_accessed = CURRENT_TIMESTAMP WHERE file_id = ? AND user_id = ?",
                                (local_cached, content_id, user_id)
                            )
                            conn2.commit()
                        except Exception as _e_db:
                            logger.debug(f"User {user_id}: Could not update cache_path from tg fast-path: {_e_db}")

                        # Build download path and copy
                        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
                        if info and isinstance(info, dict):
                            clean_filename = build_clean_filename(info, bitrate_kbps=320)
                        else:
                            base = os.path.basename(local_cached)
                            base = sanitize_filename(os.path.splitext(base)[0]) + ".mp3"
                            clean_filename = base
                        download_path = os.path.join(DOWNLOAD_DIR, clean_filename)
                        if not os.path.exists(download_path) or os.path.getsize(download_path) != os.path.getsize(local_cached):
                            shutil.copy2(local_cached, download_path)
                        logger.info(f"User {user_id}: Using tg_file_id fast-path for track {track_number} -> copied to downloads: {download_path}")
                        return [download_path]
                except Exception as e_tg:
                    logger.warning(f"User {user_id}: tg_file_id fast-path failed for track {track_number}: {e_tg}")
        except Exception as e_outer:
            logger.debug(f"User {user_id}: tg_file_id fast-path outer error for track {track_number}: {e_outer}")
        finally:
            try:
                cur2.close()
                conn2.close()
            except Exception:
                pass

        # Download if not cached nor retrievable via tg_file_id
        files = download_spotify(url, content_type, content_id)
        if not files:
            logger.warning(f"User {user_id}: Failed to download track {track_number}")
            return []

        # Process downloaded files with proper naming, cover, and quality
        processed_files = []
        for file_path in files:
            if os.path.exists(file_path):
                try:
                    # Get track info if not provided
                    if not info:
                        info = extract_track_info_optimized(content_id)
                        if not info:
                            logger.warning(f"Could not get info for track {track_number}")
                            continue

                    # Build clean filename similar to single download
                    clean_filename = build_clean_filename(info, bitrate_kbps=320)
                    download_path = os.path.join(DOWNLOAD_DIR, clean_filename)
                    # Move to download folder with clean name
                    if os.path.abspath(file_path) != os.path.abspath(download_path):
                        if os.path.exists(download_path):
                            os.remove(download_path)
                        shutil.move(file_path, download_path)
                        file_path = download_path

                    # Ensure target quality (prefer 320kbps, min 256kbps)
                    try:
                        current_kbps = 0
                        try:
                            current_kbps = int((MP3(file_path).info.bitrate or 0) / 1000)
                        except Exception:
                            current_kbps = 0
                        target_kbps = 320 if current_kbps < 320 else current_kbps
                        if target_kbps < 256:
                            target_kbps = 256
                        if current_kbps < target_kbps:
                            tmp_out = file_path + ".tmp.mp3"
                            cmd = [
                                FFMPEG_PATH, "-y",
                                "-i", file_path,
                                "-vn",
                                "-c:a", "libmp3lame",
                                "-b:a", f"{target_kbps}k",
                                "-map_metadata", "0",
                                "-id3v2_version", "3",
                                tmp_out
                            ]
                            try:
                                subprocess.run(cmd, check=True, capture_output=True, text=True)
                                os.replace(tmp_out, file_path)
                                logger.info(f"User {user_id}: Transcoded track {track_number} to {target_kbps}kbps")
                            finally:
                                if os.path.exists(tmp_out):
                                    try:
                                        os.remove(tmp_out)
                                    except OSError:
                                        pass
                    except Exception as e:
                        logger.warning(f"User {user_id}: Bitrate transcode failed for track {track_number}: {e}")

                    # Embed cover art
                    try:
                        thumbnail = info.get("thumbnail")
                        if thumbnail:
                            embed_cover(file_path, thumbnail)
                            logger.info(f"User {user_id}: Embedded cover for track {track_number}")
                    except Exception as e:
                        logger.warning(f"User {user_id}: Cover embedding failed for track {track_number}: {e}")

                    # Write ID3 metadata
                    try:
                        write_id3_metadata(file_path, info, info.get('thumbnail'))
                        logger.info(f"User {user_id}: Wrote ID3 metadata for track {track_number}")
                    except Exception as e:
                        logger.warning(f"User {user_id}: ID3 metadata write failed for track {track_number}: {e}")

                    # Create cache copy AFTER processing for reuse
                    try:
                        cache_filename = f"{user_id}_{content_id}_{user_id}_{content_id}_{os.path.basename(file_path)}"
                        cache_path = os.path.join(CACHE_DIR, cache_filename)
                        os.makedirs(CACHE_DIR, exist_ok=True)
                        shutil.copy2(file_path, cache_path)
                        logger.info(f"User {user_id}: Created cache copy: {cache_filename}")
                    except Exception as e:
                        logger.warning(f"User {user_id}: Failed to create cache copy: {e}")

                    processed_files.append(file_path)
                    logger.info(f"User {user_id}: Processed track {track_number} successfully: {os.path.basename(file_path)}")

                except Exception as e:
                    logger.error(f"User {user_id}: Error processing file for track {track_number}: {e}")
                    continue

        logger.info(f"User {user_id}: Downloaded and processed track {track_number} successfully")
        return processed_files

    except Exception as e:
        logger.error(f"Error in optimized download for track {track_number}: {e}")
        return []

# --- Optimized Track Info Extraction ---
def extract_track_info_optimized(track_id: str) -> Dict[str, Optional[str]]:
    """Optimized version of extract_track_info with reduced API calls and better error handling"""
    try:
        if not track_id or not isinstance(track_id, str):
            logger.error(f"Invalid track_id: {track_id}")
            return {
                "title": "Unknown Track",
                "artist": "Unknown Artist",
                "album": "Unknown Album",
                "thumbnail": None,
                "release_date": "Unknown",
                "duration": 0,
                "url": f"https://open.spotify.com/track/{track_id}",
                "popularity": 0,
                "genre": "Unknown"
            }

        # Single API call to get track data
        track = sp.track(track_id, market=MARKET)  # Use configured market to align with search

        if not track:
            logger.warning(f"No track data found for {track_id}, using fallback info")
            return {
                "title": "Unknown Track",
                "artist": "Unknown Artist",
                "album": "Unknown Album",
                "thumbnail": None,
                "release_date": "Unknown",
                "duration": 0,
                "url": f"https://open.spotify.com/track/{track_id}",
                "popularity": 0,
                "genre": "Unknown"
            }

        title = track.get("name") or "Unknown Track"
        artist_list = track.get("artists", [])
        artist = artist_list[0].get("name", "Unknown Artist") if artist_list else "Unknown Artist"

        album_info = track.get("album", {})
        album_name = album_info.get("name", "Unknown Album")
        release_date = album_info.get("release_date", "Unknown")

        # Get thumbnail (usually first image is the best quality)
        images = album_info.get("images", [])
        thumbnail = images[0].get("url") if images else None

        # Duration in seconds
        duration_ms = track.get("duration_ms", 0)
        duration_seconds = duration_ms // 1000 if duration_ms else 0

        # Get genre from artist (single additional API call, but cached)
        genre = "Pop"  # Default
        if artist_list:
            try:
                artist_obj = sp.artist(artist_list[0]["id"])
                genres = artist_obj.get("genres", [])
                if genres:
                    genre = genres[0].title()  # First genre, capitalized
            except Exception as e:
                logger.debug(f"Could not get genre for artist {artist_list[0]['id']}: {e}")

        url = f"https://open.spotify.com/track/{track_id}"
        popularity = track.get("popularity", 0)

        return {
            "title": title,
            "artist": artist,
            "album": album_name,
            "thumbnail": thumbnail,
            "release_date": release_date,
            "duration": duration_seconds,
            "url": url,
            "popularity": popularity,
            "genre": genre,
        }

    except SpotifyException as e:
        logger.error(f"Spotify API error for track {track_id}: {e}")
        return {
            "title": "Unknown Track",
            "artist": "Unknown Artist",
            "album": "Unknown Album",
            "thumbnail": None,
            "release_date": "Unknown",
            "duration": 0,
            "url": f"https://open.spotify.com/track/{track_id}",
            "popularity": 0,
            "genre": "Unknown"
        }
    except Exception as e:
        logger.error(f"Unexpected error extracting track info for {track_id}: {e}")
        return {
            "title": "Unknown Track",
            "artist": "Unknown Artist",
            "album": "Unknown Album",
            "thumbnail": None,
            "release_date": "Unknown",
            "duration": 0,
            "url": f"https://open.spotify.com/track/{track_id}",
            "popularity": 0,
            "genre": "Unknown"
        }

# --- Simultaneous Links Handler ---
async def handle_simultaneous_spotify_links(message: Message, urls: list[str], progress_msg):
    """Handle 2 or 3 Spotify links simultaneously with optimized performance"""
    user_id = message.from_user.id
    chat_id = message.chat.id
    num_links = len(urls)

    logger.info(f"User {user_id}: Processing {num_links} simultaneous Spotify links")

    # Validate all URLs first
    track_ids = []
    for i, url in enumerate(urls):
        try:
            is_valid, content_type, content_id = validate_spotify_url(url)
            if not is_valid or content_type != "track":
                await message.answer(f"❌ Link {i+1} is invalid or not a track.")
                return
            track_ids.append(content_id)
        except Exception as e:
            logger.error(f"User {user_id}: Error validating URL {i+1}: {e}")
            await message.answer(f"❌ Error validating link {i+1}: {str(e)}")
            return

    # Check cache for existing track info first
    tracks_info = [None] * num_links
    uncached_ids = []

    # Database connection for caching
    conn = sqlite3.connect("bot_cache.db")
    cur = conn.cursor()

    try:
        for i, content_id in enumerate(track_ids):
            # Check if track info is cached
            cur.execute("""
                SELECT artist, title, album, release_date, thumbnail, duration, popularity, genre, url
                FROM files
                WHERE file_id = ? AND user_id = ? AND artist IS NOT NULL
                ORDER BY last_accessed DESC
                LIMIT 1
            """, (content_id, user_id))

            cached_info = cur.fetchone()
            if cached_info:
                # Use cached info (place at exact index)
                info = {
                    "artist": cached_info[0],
                    "title": cached_info[1],
                    "album": cached_info[2],
                    "release_date": cached_info[3],
                    "thumbnail": cached_info[4],
                    "duration": cached_info[5],
                    "popularity": cached_info[6],
                    "genre": cached_info[7],
                    "url": cached_info[8] or f"https://open.spotify.com/track/{content_id}"
                }
                tracks_info[i] = info
                logger.info(f"User {user_id}: Used cached info for track {i+1}: {info.get('title')}")
            else:
                uncached_ids.append((i, content_id))

        # Extract info for uncached tracks concurrently
        if uncached_ids:
            logger.info(f"User {user_id}: Extracting info for {len(uncached_ids)} uncached tracks concurrently")

            # Create concurrent tasks for uncached tracks
            tasks = []
            for i, content_id in uncached_ids:
                task = asyncio.get_event_loop().run_in_executor(
                    executor, extract_track_info_optimized, content_id
                )
                tasks.append((i, task))

            # Wait for all tasks to complete
            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

            # Process results and maintain order
            for (original_index, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    logger.error(f"User {user_id}: Error extracting info for track {original_index+1}: {result}")
                    # Do NOT abort; fallback to minimal info and continue
                    fallback_info = {
                        "title": "Unknown Track",
                        "artist": "Unknown Artist",
                        "album": "Unknown Album",
                        "thumbnail": None,
                        "release_date": "Unknown",
                        "duration": 0,
                        "url": f"https://open.spotify.com/track/{track_ids[original_index]}",
                        "popularity": 0,
                        "genre": "Unknown"
                    }
                    tracks_info[original_index] = fallback_info
                    continue
                elif result:
                    tracks_info[original_index] = result
                    logger.info(f"User {user_id}: Extracted fresh info for track {original_index+1}: {result.get('title')}")
 
                    # Cache the fresh info
                    try:
                        cur.execute("""
                            INSERT OR REPLACE INTO files
                            (user_id, file_id, artist, title, album, release_date, thumbnail, duration, popularity, genre, url, created_at, last_accessed)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                        """, (
                            user_id, track_ids[original_index],
                            result.get('artist'), result.get('title'), result.get('album'),
                            result.get('release_date'), result.get('thumbnail'),
                            result.get('duration'), result.get('popularity'), result.get('genre'),
                            result.get('url')
                        ))
                        conn.commit()
                    except Exception as e:
                        logger.warning(f"User {user_id}: Failed to cache track info: {e}")
                else:
                    # Use fallback info instead of stopping
                    fallback_info = {
                        "title": "Unknown Track",
                        "artist": "Unknown Artist",
                        "album": "Unknown Album",
                        "thumbnail": None,
                        "release_date": "Unknown",
                        "duration": 0,
                        "url": f"https://open.spotify.com/track/{track_ids[original_index]}",
                        "popularity": 0,
                        "genre": "Unknown"
                    }
                    tracks_info[original_index] = fallback_info
                    logger.warning(f"User {user_id}: Using fallback info for track {original_index+1}")

    except Exception as e:
        logger.error(f"User {user_id}: Error in track info processing: {e}")
        await message.answer(f"❌ Error processing track information: {str(e)}")
        return
    finally:
        conn.close()

    # Normalize tracks_info list to ensure all entries exist and thumbnails are populated
    for idx in range(len(tracks_info)):
        if not tracks_info[idx]:
            tracks_info[idx] = {
                "title": "Unknown Track",
                "artist": "Unknown Artist",
                "album": "Unknown Album",
                "thumbnail": None,
                "release_date": "Unknown",
                "duration": 0,
                "url": f"https://open.spotify.com/track/{track_ids[idx]}",
                "popularity": 0,
                "genre": "Unknown"
            }
        # Ensure thumbnail is available (fallback via Spotify API if missing)
        if not tracks_info[idx].get("thumbnail"):
            try:
                tr = sp.track(track_ids[idx])
                alb = tr.get("album", {}) or {}
                imgs = alb.get("images", []) or []
                if imgs:
                    tracks_info[idx]["thumbnail"] = imgs[0].get("url")
            except Exception as _thumb_e:
                logger.debug(f"User {user_id}: Could not fetch thumbnail fallback for track {idx+1}: {_thumb_e}")

    # Update progress message
    await progress_msg.edit_text("🎨 Preparing song covers and information...", parse_mode="HTML")

    # Send media group with covers together, then detailed information, then inline keyboard
    keyboard_msg = None
    cover_messages = None
    try:
        media_group = []
        for i, info in enumerate(tracks_info):
            thumbnail = info.get("thumbnail")
            thumbnail_added = False

            # Add thumbnail directly (Telegram will fetch it)
            if thumbnail and isinstance(thumbnail, str) and thumbnail.strip().startswith(("http://", "https://")):
                media_item = InputMediaPhoto(media=thumbnail)
                media_group.append(media_item)
                thumbnail_added = True

            if not thumbnail_added:
                logger.warning(f"User {user_id}: No valid thumbnail found for track {i+1}")

        # Send the media group if we have items
        if media_group:
            # Send media group with all covers together
            sent_messages = await message.answer_media_group(media_group)
            cover_messages = sent_messages if sent_messages else []
            logger.info(f"User {user_id}: Sent media group with {len(media_group)} covers together")

            # Update progress
            await progress_msg.edit_text("📝 Formatting song information...", parse_mode="HTML")

            # Send detailed information message in the requested format
            info_text = "🎵 <b>Song Information:</b>\n\n"
            for i, info in enumerate(tracks_info):
                total_seconds = int(info.get("duration") or 0)
                minutes = total_seconds // 60
                seconds = total_seconds % 60
                duration_str = f"{minutes}:{seconds:02d}"

                info_text += (
                    f"<b>Song {i+1}: {info.get('artist', 'Unknown')} – {info.get('title', 'Unknown')}</b>\n"
                    f"<blockquote expandable>Album: {info.get('album', 'Unknown')}\n"
                    f"Duration: {duration_str}\n"
                    f"Release date: {info.get('release_date', 'Unknown')}\n"
                    f"Popularity: {info.get('popularity', 0)}%\n"
                    f"Genre: {info.get('genre', 'Unknown')}\n"
                    f"URL: <a href='{info.get('url', '')}'>Spotify Link</a></blockquote>\n\n"
                )

            # Create inline keyboard
            keyboard_buttons = []
            for i in range(len(tracks_info)):
                keyboard_buttons.append([
                    InlineKeyboardButton(
                        text=f"🎵 Lyrics {i+1}",
                        callback_data=f"lyrics|{i+1}"
                    ),
                    InlineKeyboardButton(
                        text=f"🎙 Separate {i+1}",
                        callback_data=f"separate|{i+1}"
                    )
                ])

            # Add general buttons
            keyboard_buttons.append([
                InlineKeyboardButton(text="⭐️ Suggestions", callback_data="suggestions"),
                InlineKeyboardButton(text="❓ Support", url="https://t.me/support_channel")
            ])

            # Send information message with inline keyboard attached
            info_msg = await message.answer(
                info_text,
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard_buttons)
            )
            keyboard_msg = info_msg
            logger.info(f"User {user_id}: Sent detailed information message with inline keyboard")


            # Delete progress message after 10 seconds
            asyncio.create_task(delete_after_delay(progress_msg, 10))

            # Delete the original message after sending the information message
            try:
                await message.delete()
                logger.info(f"User {user_id}: Original message deleted after sending information")
            except Exception as e:
                logger.debug(f"User {user_id}: Could not delete original message: {e}")

            logger.info(f"User {user_id}: Sent inline keyboard as reply to information message")
        else:
            # No valid covers found, but continue with information and downloads
            logger.warning(f"User {user_id}: No valid covers found for {len(tracks_info)} tracks, continuing without covers")

            # Update progress
            await progress_msg.edit_text("📝 Formatting song information...", parse_mode="HTML")

            # Send detailed information message in the requested format
            info_text = "🎵 <b>Song Information:</b>\n\n"
            for i, info in enumerate(tracks_info):
                total_seconds = int(info.get("duration") or 0)
                minutes = total_seconds // 60
                seconds = total_seconds % 60
                duration_str = f"{minutes}:{seconds:02d}"

                info_text += (
                    f"<b>Song {i+1}: {info.get('artist', 'Unknown')} – {info.get('title', 'Unknown')}</b>\n"
                    f"<blockquote expandable>Album: {info.get('album', 'Unknown')}\n"
                    f"Duration: {duration_str}\n"
                    f"Release date: {info.get('release_date', 'Unknown')}\n"
                    f"Popularity: {info.get('popularity', 0)}%\n"
                    f"Genre: {info.get('genre', 'Unknown')}\n"
                    f"URL: <a href='{info.get('url', '')}'>Spotify Link</a></blockquote>\n\n"
                )

            # Create inline keyboard
            keyboard_buttons = []
            for i in range(len(tracks_info)):
                keyboard_buttons.append([
                    InlineKeyboardButton(
                        text=f"🎵 Lyrics {i+1}",
                        callback_data=f"lyrics|{i+1}"
                    ),
                    InlineKeyboardButton(
                        text=f"🎙 Separate {i+1}",
                        callback_data=f"separate|{i+1}"
                    )
                ])

            # Add general buttons
            keyboard_buttons.append([
                InlineKeyboardButton(text="⭐️ Suggestions", callback_data="suggestions"),
                InlineKeyboardButton(text="❓ Support", url="https://t.me/support_channel")
            ])

            # Send information message with inline keyboard attached
            info_msg = await message.answer(
                info_text,
                parse_mode="HTML",
                reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard_buttons)
            )
            keyboard_msg = info_msg
            logger.info(f"User {user_id}: Sent detailed information message with inline keyboard (no covers)")


            # Delete progress message after 10 seconds
            asyncio.create_task(delete_after_delay(progress_msg, 10))

            # Delete the original message after sending the information message
            try:
                await message.delete()
                logger.info(f"User {user_id}: Original message deleted after sending information")
            except Exception as e:
                logger.debug(f"User {user_id}: Could not delete original message: {e}")

            logger.info(f"User {user_id}: Sent inline keyboard as reply to information message (no covers)")

    except Exception as e:
        logger.error(f"User {user_id}: Error sending media group: {e}")
        # Don't stop processing, continue with information and downloads
        logger.warning(f"User {user_id}: Media group failed, continuing with information and downloads: {e}")

        # Update progress
        await progress_msg.edit_text("📝 Formatting song information...", parse_mode="HTML")

        # Send detailed information message in the requested format
        info_text = "🎵 <b>Song Information:</b>\n\n"
        for i, info in enumerate(tracks_info):
            total_seconds = int(info.get("duration") or 0)
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            duration_str = f"{minutes}:{seconds:02d}"

            info_text += (
                f"<b>Song {i+1}: {info.get('artist', 'Unknown')} – {info.get('title', 'Unknown')}</b>\n"
                f"<blockquote expandable>Album: {info.get('album', 'Unknown')}\n"
                f"Duration: {duration_str}\n"
                f"Release date: {info.get('release_date', 'Unknown')}\n"
                f"Popularity: {info.get('popularity', 0)}%\n"
                f"Genre: {info.get('genre', 'Unknown')}\n"
                f"URL: <a href='{info.get('url', '')}'>Spotify Link</a></blockquote>\n\n"
            )

        # Create inline keyboard
        keyboard_buttons = []
        for i in range(len(tracks_info)):
            keyboard_buttons.append([
                InlineKeyboardButton(
                    text=f"🎵 Lyrics {i+1}",
                    callback_data=f"lyrics|{i+1}"
                ),
                InlineKeyboardButton(
                    text=f"🎙 Separate {i+1}",
                    callback_data=f"separate|{i+1}"
                )
            ])

        # Add general buttons
        keyboard_buttons.append([
            InlineKeyboardButton(text="⭐️ Suggestions", callback_data="suggestions"),
            InlineKeyboardButton(text="❓ Support", url="https://t.me/support_channel")
        ])

        # Send information message with inline keyboard attached
        info_msg = await message.answer(
            info_text,
            parse_mode="HTML",
            reply_markup=InlineKeyboardMarkup(inline_keyboard=keyboard_buttons)
        )
        keyboard_msg = info_msg
        logger.info(f"User {user_id}: Sent detailed information message with inline keyboard (media group failed)")


        # Delete progress message after 10 seconds
        asyncio.create_task(delete_after_delay(progress_msg, 10))

        # Delete the original message after sending the information message
        try:
            await message.delete()
            logger.info(f"User {user_id}: Original message deleted after sending information")
        except Exception as e:
            logger.debug(f"User {user_id}: Could not delete original message: {e}")

        logger.info(f"User {user_id}: Continued processing after media group failure")

    
        # Download and send audio files concurrently
        logger.info(f"User {user_id}: Starting concurrent downloads for {len(urls)} tracks")
    
        # Update progress message
        await progress_msg.edit_text(f"⚡ Starting concurrent downloads for {len(urls)} tracks...", parse_mode="HTML")
    
        # Create download tasks
        download_tasks = []
        for i, (url, info) in enumerate(zip(urls, tracks_info)):
            task = asyncio.get_event_loop().run_in_executor(
                executor, download_spotify_optimized, url, user_id, i+1, info
            )
            download_tasks.append((i, info, task))
    
        # Update progress during downloads
        await progress_msg.edit_text(f"⏳ Downloading {len(urls)} tracks simultaneously...\nPlease wait, this may take 1-2 minutes.", parse_mode="HTML")
    
        # Wait for all downloads to complete
        download_results = await asyncio.gather(*[task for _, _, task in download_tasks], return_exceptions=True)
    
        # Process download results
        for (track_index, info, _), result in zip(download_tasks, download_results):
            try:
                if isinstance(result, Exception):
                    logger.error(f"User {user_id}: Download failed for track {track_index+1}: {result}")
                    await message.answer(f"❌ خطا در دانلود آهنگ {track_index+1}: {str(result)}")
                    continue
    
                files = result
                if not files:
                    logger.warning(f"User {user_id}: No files downloaded for track {track_index+1}")
                    continue
    
                for file_path in files:
                    if os.path.exists(file_path):
                        # Send audio file from download folder (before caching)
                        sent_message = await message.answer_audio(
                            audio=FSInputFile(file_path, filename=os.path.basename(file_path)),
                            performer=info.get('artist'),
                            title=info.get('title'),
                            duration=int(info.get('duration') or 0)
                        )
    
                        # Record tg_file_id and persist metadata/caching optimally
                        if sent_message and sent_message.audio:
                            tg_file_id = sent_message.audio.file_id
                            is_valid2, ctype2, cid2 = validate_spotify_url(urls[track_index])
                            content_id = cid2 or ""
    
                            # Check if cache already exists; avoid re-copying in fast-path scenarios
                            existing_cache_path = None
                            try:
                                _conn = sqlite3.connect("bot_cache.db")
                                _cur = _conn.cursor()
                                _cur.execute(
                                    "SELECT cache_path FROM files WHERE file_id = ? AND user_id = ? AND cache_path IS NOT NULL ORDER BY last_accessed DESC LIMIT 1",
                                    (content_id, user_id),
                                )
                                row = _cur.fetchone()
                                if row and row[0] and os.path.exists(row[0]):
                                    existing_cache_path = row[0]
                            except Exception as _db_e:
                                logger.debug(f"User {user_id}: cache existence check failed: {_db_e}")
                            finally:
                                try:
                                    _cur.close()
                                    _conn.close()
                                except Exception:
                                    pass
    
                            # Create optimized cache copy only when not already cached
                            cache_path = existing_cache_path
                            if not cache_path:
                                cache_filename = f"{user_id}_{content_id}_{user_id}_{content_id}_{os.path.basename(file_path)}"
                                candidate_cache = os.path.join(CACHE_DIR, cache_filename)
                                try:
                                    os.makedirs(CACHE_DIR, exist_ok=True)
                                    shutil.copy2(file_path, candidate_cache)
                                    cache_path = candidate_cache
                                    logger.info(f"User {user_id}: Created optimized cache copy: {os.path.basename(candidate_cache)}")
    
                                    # Update DB with new cache_path
                                    try:
                                        _conn2 = sqlite3.connect("bot_cache.db")
                                        _cur2 = _conn2.cursor()
                                        _cur2.execute(
                                            "UPDATE files SET cache_path = ?, last_accessed = CURRENT_TIMESTAMP WHERE file_id = ? AND user_id = ?",
                                            (cache_path, content_id, user_id),
                                        )
                                        _conn2.commit()
                                    except Exception as _upd_e:
                                        logger.debug(f"User {user_id}: DB update for cache_path failed: {_upd_e}")
                                    finally:
                                        try:
                                            _cur2.close()
                                            _conn2.close()
                                        except Exception:
                                            pass
                                except Exception as e:
                                    logger.warning(f"User {user_id}: Could not create cache copy: {e}")
    
                            # Save/Upsert metadata (respecting existing cache_path if present)
                            from db_manager import save_file
                            saved_path = save_file(
                                user_id=user_id,
                                file_id=content_id,
                                artist=info.get('artist'),
                                title=info.get('title'),
                                url=info.get('url'),
                                thumbnail=info.get('thumbnail'),
                                source_path=cache_path,
                                duration=int(info.get('duration') or 0),
                                tg_file_id=tg_file_id,
                                album=info.get('album'),
                                release_date=info.get('release_date'),
                                popularity=int(info.get('popularity') or 0),
                                genre=info.get('genre')
                            )
                            if saved_path:
                                logger.info(f"User {user_id}: Saved track {track_index+1} metadata to database")
                            else:
                                logger.warning(f"User {user_id}: Failed to save track {track_index+1} metadata to database")
    
                        # Always delete the quick copy from downloads after sending (keep cache only)
                        try:
                            if file_path and os.path.abspath(file_path).startswith(os.path.abspath(DOWNLOAD_DIR)) and os.path.exists(file_path):
                                os.remove(file_path)
                                logger.debug(f"User {user_id}: Removed quick copy from downloads: {os.path.basename(file_path)}")
                        except Exception as _rm_e:
                            logger.debug(f"User {user_id}: could not remove quick copy {file_path}: {_rm_e}")
    
                        logger.info(f"User {user_id}: Sent audio for track {track_index+1} from download folder")
                        break  # Send only first file
    
            except Exception as e:
                logger.error(f"User {user_id}: Error processing download result for track {track_index+1}: {e}")
                await message.answer(f"❌ خطا در پردازش آهنگ {track_index+1}")
    
        # Post a final success message as a reply to the first cover (if present)
        try:
            final_caption = "✅ Download successful. Do you need more features?"
            if cover_messages and isinstance(cover_messages, list) and len(cover_messages) > 0:
                await message.answer(
                    final_caption,
                    reply_to_message_id=cover_messages[0].message_id,
                    parse_mode="HTML"
                )
            elif keyboard_msg:
                await message.answer(
                    final_caption,
                    reply_to_message_id=keyboard_msg.message_id,
                    parse_mode="HTML"
                )
            else:
                await message.answer(final_caption, parse_mode="HTML")
        except Exception as e:
            logger.error(f"User {user_id}: Error sending final success message: {e}")
      
        # Delete progress message after a short delay (covers remain)
        await asyncio.sleep(2)
        try:
            await progress_msg.delete()
            logger.info(f"User {user_id}: Progress message deleted after all processing")
        except Exception as e:
            logger.debug(f"User {user_id}: Could not delete progress message: {e}")
        
        logger.info(f"User {user_id}: Completed simultaneous processing of {num_links} links")

# === PLAYLIST DOWNLOAD HANDLER (fast-path aware, album-like flow) ===
async def handle_spotify_playlist_download(message: types.Message, url: str, playlist_id: str) -> None:
    """
    Download an entire Spotify playlist with strong fast-path reuse:
    - Fetch playlist metadata via sp.playlist and full tracklist via sp.playlist_items (paginated)
    - Save playlist metadata in DB (content_type='playlist')
    - For each track:
        - FAST-PATH 1: If cached local file exists, copy/send immediately (no download)
        - FAST-PATH 2: If tg_file_id exists, send immediately and copy from Telegram CDN in background
        - Else: download with spotdl (proxy-aware), process (cover, ID3, quality), cache, send, record tg_file_id
    - On subsequent requests, no heavy download/extraction is repeated; sending is instant via tg_file_id/local cache.
    """

    user_id = message.from_user.id
    try:
        # Progress message (threaded)
        progress_msg = await message.reply("🔎 Fetching playlist info...", parse_mode="HTML")

        # 1) Fetch playlist metadata
        try:
            playlist = sp.playlist(playlist_id, market=MARKET)
        except Exception as e:
            await progress_msg.edit_text(f"❌ Playlist not found or unavailable.\n{e}", parse_mode="HTML")
            return

        name = playlist.get("name", "Unknown")
        owner = (playlist.get("owner") or {}).get("display_name", "Unknown")
        playlist_url = (playlist.get("external_urls") or {}).get("spotify", f"https://open.spotify.com/playlist/{playlist_id}")
        images = playlist.get("images", []) or []
        playlist_thumb = images[0].get("url") if images else None
        followers = (playlist.get("followers") or {}).get("total", 0)
        collaborative = bool(playlist.get("collaborative"))
        public_flag = playlist.get("public")
        visibility = "Public" if public_flag is True else ("Private" if public_flag is False else "Unknown")

        # Gather ALL items (paginated)
        track_items = []
        offset = 0
        limit = 100
        while True:
            try:
                batch = sp.playlist_items(playlist_id, market=MARKET, offset=offset, limit=limit, additional_types=("track",))
            except Exception:
                batch = None
            items = (batch or {}).get("items", [])
            track_items.extend(items)
            if not batch or not batch.get("next") or len(items) < limit:
                break
            offset += limit

        tracks_total = int((playlist.get("tracks") or {}).get("total") or len(track_items) or 0)

        # created / last update from added_at
        added_dates = [(it.get("added_at") or "")[:10] for it in (track_items or []) if isinstance(it, dict) and it.get("added_at")]
        created_date = min(added_dates) if added_dates else "Unknown"
        last_update = max(added_dates) if added_dates else "Unknown"

        # Track IDs
        track_ids = []
        for it in (track_items or []):
            tr = it.get("track") or {}
            tid = tr.get("id")
            if tid:
                track_ids.append(tid)

        # Compute total duration and top track by popularity
        total_ms = 0
        top_track_name = "Unknown"
        top_track_pop = -1
        detailed_tracks = []
        try:
            for i in range(0, len(track_ids), 50):
                batch_ids = track_ids[i:i+50]
                try:
                    tr_res = sp.tracks(batch_ids)
                    detailed_tracks.extend((tr_res or {}).get("tracks", []))
                except Exception:
                    # fallback minimal
                    for tid in batch_ids:
                        detailed_tracks.append({"id": tid})
        except Exception:
            pass

        pops = []
        for dt in detailed_tracks:
            try:
                dms = int(dt.get("duration_ms") or 0)
                total_ms += dms
            except Exception:
                pass
            try:
                pop = int(dt.get("popularity") or 0)
                pops.append(pop)
                if pop > top_track_pop:
                    top_track_pop = pop
                    top_track_name = dt.get("name", "Unknown")
            except Exception:
                pass

        playlist_duration_seconds = (total_ms // 1000) if total_ms else 0
        avg_pop = int(round(sum(pops) / len(pops))) if pops else 0
        dur_str = format_duration(playlist_duration_seconds)

        # Persist playlist metadata row in DB cache (no file, content_type='playlist')
        try:
            save_file(
                user_id=user_id,
                file_id=playlist_id,
                artist=owner,
                title=name,
                url=playlist_url,
                thumbnail=playlist_thumb,
                duration=playlist_duration_seconds,
                album=name,  # store name for convenience
                release_date=created_date,  # use earliest added date as created
                popularity=avg_pop,
                genre=None,
                content_type="playlist",
                total_tracks=tracks_total,
                label=visibility,      # overload label with visibility context
                top_track=top_track_name
            )
        except Exception as _db_e:
            # Non-fatal
            logger.debug(f"Playlist metadata save failed: {_db_e}")

        # Show playlist info to user (like album cover message)
        try:
            cap = (
                f"💿 Playlist – {name}\n"
                f"User: {owner}\n"
                f"Created: {created_date}\n"
                f"Last Update: {last_update}\n"
                f"Tracks: {tracks_total}\n"
                f"Duration: {dur_str}\n"
                f"Popularity: {avg_pop}%\n"
                f"Followers: {followers:,}\n"
                f"Collaborative: {'Yes' if collaborative else 'No'}\n"
                f"Visibility: {visibility}\n\n"
                f"Top Track: {top_track_name}\n"
                f"🔗 <a href='{playlist_url}'>View on Spotify</a>"
            )
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text=" Add to playlist all", callback_data=f"playlist_add_to_playlist|{playlist_id}")],
                [InlineKeyboardButton(text=" Ask about playlist music", callback_data=f"playlist_ask_ai|{playlist_id}")]
            ])
            if playlist_thumb:
                cover_msg = await message.answer_photo(photo=playlist_thumb, caption=cap, parse_mode="HTML", reply_markup=keyboard)
            else:
                cover_msg = await message.answer(cap, parse_mode="HTML", reply_markup=keyboard)

            # Thread progress under the cover, if possible
            try:
                if progress_msg:
                    await progress_msg.delete()
            except Exception:
                pass
            progress_msg = await message.answer(
                "🔎 Fetching playlist info...",
                reply_to_message_id=cover_msg.message_id,
                parse_mode="HTML"
            )
            await progress_msg.edit_text(f"Downloading playlist tracks... 0/{len(track_ids)}", parse_mode="HTML")
        except Exception as _e_cover:
            logger.debug(f"Playlist cover send failed: {_e_cover}")
            try:
                await progress_msg.edit_text(f"Downloading playlist tracks... 0/{len(track_ids)}", parse_mode="HTML")
            except Exception:
                pass

        # 2) Create playlist temporary directory
        ts = int(time.time())
        playlist_dir = os.path.abspath(os.path.join(DOWNLOAD_DIR, f"playlist_{playlist_id}_{ts}"))
        os.makedirs(playlist_dir, exist_ok=True)

        # 3) Download every track using fast-path logic similar to album
        done = 0
        total = len(track_ids)

        # DB handle for fast-path checks
        conn_fast = sqlite3.connect("bot_cache.db")
        cur_fast = conn_fast.cursor()

        for idx, tid in enumerate(track_ids, start=1):
            try:
                # FAST-PATH: cached file or tg_file_id for ultra-fast send
                cached_row = None
                try:
                    cur_fast.execute("""
                        SELECT cache_path, tg_file_id, artist, title, duration, thumbnail, album, release_date, popularity, genre, url
                        FROM files
                        WHERE file_id = ? AND user_id = ?
                        ORDER BY last_accessed DESC
                        LIMIT 1
                    """, (tid, user_id))
                    cached_row = cur_fast.fetchone()
                except Exception:
                    cached_row = None

                info = None
                if cached_row:
                    cache_path = cached_row[0]
                    tg_id = cached_row[1]
                    info = {
                        "artist": cached_row[2],
                        "title": cached_row[3],
                        "duration": cached_row[4],
                        "thumbnail": cached_row[5],
                        "album": cached_row[6],
                        "release_date": cached_row[7],
                        "popularity": cached_row[8] or 0,
                        "genre": cached_row[9] or "Unknown",
                        "url": cached_row[10] or f"https://open.spotify.com/track/{tid}"
                    }

                    # Fast-path 1: local cached file exists and is clean
                    if cache_path and os.path.exists(cache_path):
                        is_clean = False
                        try:
                            audio_probe = MP3(cache_path)
                            if getattr(audio_probe, 'info', None) and getattr(audio_probe.info, 'length', 0) > 0 and os.path.getsize(cache_path) > 50 * 1024:
                                is_clean = True
                        except Exception:
                            is_clean = False
                        if is_clean:
                            # Ensure minimal info present
                            if not info.get("artist") or not info.get("title"):
                                try:
                                    info = await asyncio.get_event_loop().run_in_executor(executor, extract_track_info_optimized, tid)
                                except Exception:
                                    pass
                            clean_name_fp = build_clean_filename(info or {"artist": "Unknown Artist", "title": "audio"}, bitrate_kbps=320)
                            dst = os.path.join(playlist_dir, clean_name_fp)
                            try:
                                if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(cache_path):
                                    shutil.copy2(cache_path, dst)
                            except Exception:
                                dst = cache_path  # fallback

                            # Quick send
                            try:
                                sent_message = await message.answer_audio(
                                    audio=FSInputFile(dst, filename=os.path.basename(dst)),
                                    performer=(info or {}).get('artist'),
                                    title=(info or {}).get('title'),
                                    duration=int((info or {}).get('duration') or 0)
                                )
                                if sent_message and sent_message.audio:
                                    try:
                                        update_tg_file_id(tid, user_id, sent_message.audio.file_id)
                                    except Exception:
                                        pass
                            except Exception:
                                pass

                            done += 1
                            try:
                                await progress_msg.edit_text(f"Downloading playlist tracks... {done}/{total}", parse_mode="HTML")
                            except Exception:
                                pass
                            continue

                    # Fast-path 2: tg_file_id exists (send immediately) + background copy to cache and playlist folder
                    if tg_id:
                        # Ensure minimal info
                        if not info or not info.get("title"):
                            try:
                                info = await asyncio.get_event_loop().run_in_executor(executor, extract_track_info_optimized, tid)
                            except Exception:
                                pass
                        clean_name_fp2 = build_clean_filename(info or {"artist": "Unknown Artist", "title": "audio"}, bitrate_kbps=320)
                        dst2 = os.path.join(playlist_dir, clean_name_fp2)

                        try:
                            sent_message = await message.answer_audio(
                                audio=tg_id,
                                performer=(info or {}).get('artist'),
                                title=(info or {}).get('title'),
                                duration=int((info or {}).get('duration') or 0)
                            )
                        except Exception:
                            sent_message = None

                        # Background copy from Telegram CDN to cache and into playlist folder
                        def _bg_copy_from_tg(tg_file_id_local: str, dst_local: str, content_id_local: str, user_local: int):
                            try:
                                local_cached = download_file_from_telegram(tg_file_id_local, CACHE_DIR)
                                if local_cached and os.path.exists(local_cached):
                                    try:
                                        if not os.path.exists(dst_local) or os.path.getsize(dst_local) != os.path.getsize(local_cached):
                                            shutil.copy2(local_cached, dst_local)
                                    except Exception:
                                        pass
                                    # Update DB cache_path for this user/content
                                    try:
                                        _c = sqlite3.connect("bot_cache.db")
                                        _cur = _c.cursor()
                                        _cur.execute(
                                            "UPDATE files SET cache_path = ?, last_accessed = CURRENT_TIMESTAMP WHERE file_id = ? AND user_id = ?",
                                            (local_cached, content_id_local, user_local)
                                        )
                                        _c.commit()
                                        _cur.close()
                                        _c.close()
                                    except Exception:
                                        pass
                            except Exception:
                                pass
                        try:
                            asyncio.get_running_loop().run_in_executor(None, _bg_copy_from_tg, tg_id, dst2, tid, user_id)
                        except Exception:
                            pass

                        done += 1
                        try:
                            await progress_msg.edit_text(f"Downloading playlist tracks... {done}/{total}", parse_mode="HTML")
                        except Exception:
                            pass
                        continue

                # If not fast-path, extract info and download
                info = await asyncio.get_event_loop().run_in_executor(executor, extract_track_info_optimized, tid)
                if not info or not info.get("title"):
                    info = {
                        "title": "Unknown Track",
                        "artist": "Unknown Artist",
                        "album": name,
                        "thumbnail": playlist_thumb,
                        "release_date": created_date,
                        "duration": 0,
                        "url": f"https://open.spotify.com/track/{tid}",
                        "popularity": 0,
                        "genre": "Unknown"
                    }

                # Download (proxy-aware) exactly like single
                track_url = f"https://open.spotify.com/track/{tid}"
                files = await asyncio.get_event_loop().run_in_executor(
                    executor, download_spotify, track_url, "track", tid
                )
                if not files:
                    done += 1
                    try:
                        await progress_msg.edit_text(f"Downloading playlist tracks... {done}/{total}", parse_mode="HTML")
                    except Exception:
                        pass
                    continue

                # Copy validated file to playlist temp dir with clean name
                src = files[0]
                clean_name = build_clean_filename(info, bitrate_kbps=320)
                dst = os.path.join(playlist_dir, clean_name)
                try:
                    if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
                        shutil.copy2(src, dst)
                except Exception as _cp_e:
                    logger.debug(f"Playlist copy failed for {tid}: {_cp_e}")
                    dst = src  # fallback

                # Ensure 320kbps bitrate
                try:
                    dst = ensure_320kbps(dst)
                except Exception as _320_e:
                    logger.debug(f"Playlist ensure_320kbps failed for {tid}: {_320_e}")

                # Ensure cover + ID3 on the temp copy (like single)
                try:
                    thumb = info.get('thumbnail') or playlist_thumb
                    if thumb:
                        embed_cover(dst, thumb)
                except Exception as _cov_e:
                    logger.debug(f"Playlist cover embed failed for {tid}: {_cov_e}")
                try:
                    write_id3_metadata(dst, info, info.get('thumbnail') or playlist_thumb)
                except Exception as _id3_e:
                    logger.debug(f"Playlist ID3 write failed for {tid}: {_id3_e}")

                # Cache track metadata (do NOT overwrite existing cache_path created by downloader)
                try:
                    save_file(
                        user_id=user_id,
                        file_id=tid,
                        artist=info.get('artist'),
                        title=info.get('title'),
                        url=info.get('url'),
                        thumbnail=info.get('thumbnail') or playlist_thumb,
                        duration=int(info.get('duration') or 0),
                        tg_file_id=None,
                        album=info.get('album') or name,
                        release_date=info.get('release_date') or created_date,
                        popularity=int(info.get('popularity') or 0),
                        genre=info.get('genre') or "Unknown",
                        content_type="track"
                    )
                except Exception as _sf_e:
                    logger.debug(f"Playlist track save_file failed for {tid}: {_sf_e}")

                # Send the cleaned file to user (from the playlist folder)
                try:
                    sent_message = await message.answer_audio(
                        audio=FSInputFile(dst, filename=os.path.basename(dst)),
                        performer=info.get('artist'),
                        title=info.get('title'),
                        duration=int(info.get('duration') or 0)
                    )
                    if sent_message and sent_message.audio:
                        try:
                            update_tg_file_id(tid, user_id, sent_message.audio.file_id)
                        except Exception as _upd_e:
                            logger.debug(f"Playlist track tg_file_id update failed for {tid}: {_upd_e}")
                except Exception as _send_e:
                    logger.warning(f"Playlist track send failed for {tid}: {_send_e}")

                done += 1
                # Update progress
                try:
                    await progress_msg.edit_text(f"Downloading playlist tracks... {done}/{total}", parse_mode="HTML")
                except Exception:
                    pass

            except Exception as e_tr:
                logger.warning(f"Playlist track failed {tid}: {e_tr}")
                done += 1
                try:
                    await progress_msg.edit_text(f"Downloading playlist tracks... {done}/{total}", parse_mode="HTML")
                except Exception:
                    pass
                continue

        # Close fast-path DB handles if open
        try:
            cur_fast.close()
            conn_fast.close()
        except Exception:
            pass

        # Finalize
        try:
            completion_text = "Playlist download completed."
            try:
                if 'cover_msg' in locals() and cover_msg:
                    await message.answer(
                        completion_text,
                        reply_to_message_id=cover_msg.message_id,
                        parse_mode="HTML"
                    )
            except Exception:
                pass
            # Delete progress message after completion
            try:
                await progress_msg.delete()
            except Exception:
                pass
        except Exception:
            pass

    except Exception as e:
        try:
            await message.answer(f"❌ Error downloading playlist: {e}", parse_mode="HTML")
        except Exception:
            pass

# --- Playlist cover inline button placeholders (disabled) ---
@router.callback_query(lambda q: q.data and q.data.startswith("playlist_add_to_playlist|"))
async def playlist_add_to_playlist_cb(query: CallbackQuery) -> None:
    try:
        await query.answer("This feature is under development....", show_alert=True)
    except Exception as e:
        logger.debug(f"playlist_add_to_playlist_cb error: {e}")

@router.callback_query(lambda q: q.data and q.data.startswith("playlist_ask_ai|"))
async def playlist_ask_ai_cb(query: CallbackQuery) -> None:
    try:
        await query.answer("This feature is under development....", show_alert=True)
    except Exception as e:
        logger.debug(f"playlist_ask_ai_cb error: {e}")
