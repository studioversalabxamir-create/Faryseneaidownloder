import os
import io
import sys
import time
import json
import shutil
import logging
import tempfile
import asyncio
import pathlib
import random
from aiogram import Dispatcher
from typing import List, Optional, Tuple, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
from aiogram import Router, types
from aiogram.types import FSInputFile, InputMediaPhoto, InputMediaVideo, InputMediaDocument
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import inspect
from aiogram.filters import Command
import asyncio
import inspect
import html
from aiogram.enums import ParseMode
from aiogram.types import Message
from pathlib import Path
import subprocess, tempfile, shutil, json, sys
from typing import Optional
import re
from aiogram.types import BufferedInputFile
import pathlib
import mimetypes
from PIL import Image
from io import BytesIO

import instaloader
from instaloader import Profile, Post, Instaloader, Hashtag, StoryItem, Highlight

from fake_useragent import UserAgent


# Optional translator
try:
    from googletrans import Translator  # type: ignore
    _HAS_GOOGLETRANS = True
except Exception:
    _HAS_GOOGLETRANS = False

# Optional ffmpeg wrapper (used only if present)
FFMPEG_PATH = os.getenv("FFMPEG_PATH")  # if None, compression/skipping will be disabled

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Router + executor for blocking operations
router = Router()
_executor = ThreadPoolExecutor(max_workers=3)

# Storage folders
BASE_DOWNLOAD_DIR = pathlib.Path("downloads/instagram")
BASE_DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Instaloader session file location
SESSION_DIR = pathlib.Path("instaloader_sessions")
SESSION_DIR.mkdir(parents=True, exist_ok=True)

FFMPEG_BIN = os.getenv("FFMPEG_PATH", "ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_PATH", "ffprobe")

# UserAgent for web fallbacks
ua = UserAgent()

# Proxy configuration - import from centralized config
try:
    from config import PROXY
    proxies = PROXY
except ImportError:
    proxies = None


# Rate-limiting / queue basic implementation
_download_lock = asyncio.Lock()

MAX_PHOTO_BYTES = 9_500_000  # ~9.5MB: اگر بزرگ‌تر بود به‌جای sendPhoto با document بفرست
MAX_CAPTION_LENGTH = 1024  # Telegram caption limit


# ---------- Helpers / Utils ----------

# تعریف تابع نمونه
SHORTCODE_RX = re.compile(
    r"(?:instagram\.com/(?:p|reel|tv)/)(?P<code>[A-Za-z0-9_-]{5,})", re.IGNORECASE
)

# Assuming url is passed as a parameter or defined earlier


def extract_shortcode(url: str) -> Optional[str]:
    """
    سعی می‌کند شورت‌کد را از انواع لینک‌های اینستا (p/reel/tv) درآورد.
    """
    if not url:
        return None
    url = url.strip()
    m = SHORTCODE_RX.search(url)
    if m:
        return m.group("code")
    # fallback خیلی ساده برای زمانی که کاربر فقط کد را بدهد
    tail = url.rstrip("/").split("/")[-1].split("?")[0]
    if re.fullmatch(r"[A-Za-z0-9_-]{5,}", tail):
        return tail
    return None

def classify_instagram_input(text: str) -> Tuple[str, str]:
    """
    ورودی را دسته‌بندی می‌کند:
    returns (kind, value)
    kind ∈ {"post","reel","profile","story","highlight","unknown"}
    - برای پست/ریل: value = url
    - برای پروفایل/استوری/هایلایت: value = username (بدون @ و بدون آدرس)
    """
    t = (text or "").strip()
    if not t:
        return "unknown", ""
    lower = t.lower()

    # لینک پست/ریل/tv؟
    if "instagram.com/p/" in lower or "instagram.com/reel/" in lower or "instagram.com/tv/" in lower:
        return "post", t

    # پروفایل/استوری/هایلایت بر اساس الگوهای معمول
    # @username یا لینک پروفایل
    if lower.startswith("@") or "instagram.com/" in lower:
        u = lower.replace("https://www.instagram.com/", "").replace("http://www.instagram.com/", "")
        u = u.strip("/@ ").split("?")[0]
        if not u:
            return "unknown", ""
        # اگر کاربر خودش گفت استوری/هایلایت، می‌توانی از فرمان‌ها استفاده کنی؛
        # در حالت auto، ما استوری عمومی را دانلود می‌کنیم وگرنه پروفایل.
        if "/stories/" in lower:
            return "story", u
        return "profile", u

    # اگر متن ساده و شبیه یوزرنیم است
    if re.fullmatch(r"[a-z0-9_.]+", t):
        return "profile", t

    return "unknown", t

def _ffprobe_duration(path: str) -> Optional[float]:
    try:
        cmd = [
            FFPROBE_BIN, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "format=duration", "-of", "json", path
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
        data = json.loads(out.decode("utf-8", "ignore"))
        dur = data.get("format", {}).get("duration")
        return float(dur) if dur is not None else None
    except Exception:
        return None

def _safe_ext(path: str) -> str:
    return os.path.splitext(path)[1].lower()

def ensure_telegram_compatible_video(src_path: str) -> str:
    """
    اگر duration=0 بود یا فرمت مناسب نبود، با ffmpeg ریموکس/ترنسکُد می‌کنه.
    خروجی یک mp4 سازگار با تلگرام هست.
    """
    if not os.path.exists(src_path):
        raise FileNotFoundError(src_path)

    ext = _safe_ext(src_path)
    dur = _ffprobe_duration(src_path)  # ممکنه None یا 0.0 برگرده

    # مسیر خروجی موقت
    out_dir = tempfile.mkdtemp(prefix="tgfix_")
    fixed_mp4 = os.path.join(out_dir, "fixed.mp4")

    def _try_copy_remux():
        # سریع‌ترین حالت: فقط ریموکس و faststart
        cmd = [
            FFMPEG_BIN, "-y", "-i", src_path,
            "-c", "copy", "-movflags", "+faststart", fixed_mp4
        ]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _transcode_full():
        # اگر ریموکس کافی نبود یا duration صفر بود، ترنسکُد کامل
        cmd = [
            FFMPEG_BIN, "-y", "-i", src_path,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart",
            fixed_mp4
        ]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        # اگر WebM/3gp/mov … یا duration مشکوک: اول ریموکس، بعد در صورت نیاز ترنسکُد
        needs_transcode = (ext not in (".mp4", ".m4v")) or (dur is None or dur <= 0.01)

        if not needs_transcode:
            # تلاش برای ریموکس
            _try_copy_remux()
            new_dur = _ffprobe_duration(fixed_mp4)
            if new_dur is None or new_dur <= 0.01:
                # اگر هنوز صفره، ترنسکُد کامل
                _transcode_full()
        else:
            # مستقیم ترنسکُد
            _transcode_full()

        # در نهایت اگر موفق بود، مسیر fixed_mp4 را برگردون
        return fixed_mp4

    except Exception:
        # آخرین تلاش: ترنسکُد کامل
        try:
            _transcode_full()
            return fixed_mp4
        except Exception as e:
            # شکست خورد → همون فایل اصلی رو برگردونیم تا هندلر تصمیم بگیره
            try:
                shutil.rmtree(out_dir, ignore_errors=True)
            except Exception:
                pass
            raise e
      
def make_requests_session(timeout: int = 60) -> requests.Session:
    """
    Create a requests.Session with retries, standard headers and random proxy from list.
    Return the session object to be used by Instaloader and internal requests.
    """
    session = requests.Session()
    # استاندارد هدرها (تیپ مرورگر واقعی)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.instagram.com/",
        "Accept": "*/*"
    })

    # Retry strategy
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["HEAD", "GET", "OPTIONS", "POST"]),
        raise_on_status=False,
        respect_retry_after_header=True
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    # Select random proxy from list
    if proxies:
        selected_proxy = random.choice(proxies)
        session.proxies.update({"http": selected_proxy, "https": selected_proxy})

    # small convenience attribute
    session.request_timeout = timeout
    return session

def _get_instaloader_instance(session_username: Optional[str] = None,
                              download_videos: bool = True,
                              save_metadata: bool = False) -> Instaloader:
    """
    Return configured Instaloader instance with random proxy.
    """
    L = instaloader.Instaloader(
        dirname_pattern=str(BASE_DOWNLOAD_DIR),
        download_videos=download_videos,
        download_video_thumbnails=False,
        download_geotags=False,
        save_metadata=save_metadata,
        compress_json=False,
        post_metadata_txt_pattern="",
    )

    # create and attach a requests session with retries + random proxy
    try:
        session = make_requests_session(timeout=60)
        # attach session to instaloader context so all internal HTTP uses it
        # instaloader uses L.context._session internally (requests.Session)
        L.context._session = session
        logger.debug("Instaloader session attached with random proxy")
    except Exception as e:
        logger.warning(f"Failed to attach custom session to Instaloader: {e}")

    # try to load stored session (login), if requested
    if session_username:
        session_file = SESSION_DIR / f"{session_username}.session"
        if session_file.exists():
            try:
                L.load_session_from_file(session_username, filename=str(session_file))
                logger.debug(f"Loaded session for {session_username}")
            except Exception as e:
                logger.warning(f"Failed to load session file for {session_username}: {e}")
    return L

def instaloader_login(username: str, password: str) -> Tuple[bool, str]:
    """
    Perform a login and store session file. Returns (success, message).
    """
    try:
        L = _get_instaloader_instance(download_videos=True)
        L.context.log("Logging in...")
        L.login(username, password)
        session_file = SESSION_DIR / f"{username}.session"
        L.save_session_to_file(filename=str(session_file))
        return True, f"Logged in and session saved to {session_file}"
    except Exception as e:
        logger.exception("Login failed")
        return False, str(e)

def _make_pdf_from_text(text: str, out_path: pathlib.Path, title: Optional[str] = None) -> str:
    """
    Create a simple PDF with the provided text using reportlab.
    Returns path as string.
    """
    text = text or ""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out_path), pagesize=letter)
    width, height = letter
    if title:
        c.setFont("Helvetica-Bold", 14)
        c.drawString(40, height - 50, title)
        y = height - 80
    else:
        y = height - 50
    c.setFont("Helvetica", 11)
    lines = []
    for paragraph in text.splitlines():
        # simple wrapping
        while paragraph:
            # estimate char count by approximate width
            max_chars = 95
            lines.append(paragraph[:max_chars])
            paragraph = paragraph[max_chars:]
    for line in lines:
        if y < 60:
            c.showPage()
            y = height - 40
            c.setFont("Helvetica", 11)
        c.drawString(40, y, line)
        y -= 14
    c.save()
    return str(out_path)

def _translate_text(text: str, dest: str = "en") -> Tuple[str, bool]:
    if not text:
        return "", False
    if not _HAS_GOOGLETRANS:
        return text, False
    try:
        translator = Translator()
        res = translator.translate(text, dest=dest)
        return res.text, True
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return text, False

def _compress_video_with_ffmpeg(input_path: str, output_path: str, crf: int = 28) -> Optional[str]:
    """
    Compress video using ffmpeg if FFMPEG_PATH is set. Returns output_path if success else None.
    """
    if not FFMPEG_PATH:
        return None
    try:
        cmd = [
            FFMPEG_PATH,
            "-y",
            "-i", input_path,
            "-vcodec", "libx264",
            "-crf", str(crf),
            "-preset", "medium",
            "-acodec", "aac",
            "-movflags", "+faststart",
            output_path
        ]
        import subprocess
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return output_path
    except Exception as e:
        logger.warning(f"ffmpeg compression failed: {e}")
        return None


# ---------- Core download functions (blocking) ----------
# These run inside ThreadPoolExecutor to avoid blocking the event loop.

def _download_url_to_file(url: str, dest_path: pathlib.Path, timeout: int = 60) -> bool:
    try:
        session = make_requests_session(timeout=timeout)
        with session.get(url, stream=True) as r:
            r.raise_for_status()
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_path, "wb") as fh:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        fh.write(chunk)
        # verify file size > 0
        return dest_path.exists() and dest_path.stat().st_size > 0
    except Exception as e:
        logger.exception(f"_download_url_to_file failed for {url}: {e}")
        try:
            if dest_path.exists():
                dest_path.unlink()
        except Exception:
            pass
        return False

def download_post(url: str, session_username: Optional[str] = None, keep_files: bool = False) -> dict:
    """
    Returns a dict with keys:
      - files: list of downloaded file paths (strings) in order
      - caption: caption string to attach to media
      - shortcode, metadata
      - progress_text: suggested "loading" message
      - final_text: suggested "done" message
      - cleanup_paths: list of files to remove after sending (empty if keep_files True)
      - error: if any error occurred
    """
    out = {"files": [], "caption": None, "shortcode": None, "metadata": {},
           "progress_text": "We are downloading your content...", "final_text": "✅ Download completed.",
           "cleanup_paths": []}
    try:
        L = _get_instaloader_instance(session_username=session_username, download_videos=True, save_metadata=True)

        shortcode = extract_shortcode(url)
        if not shortcode:
            raise ValueError("Could not determine post shortcode from URL.")

        out["shortcode"] = shortcode
        post = Post.from_shortcode(L.context, shortcode)
        username = getattr(post, "owner_username", "unknown")
        out["caption"] = (post.caption or "").strip()
        out["metadata"] = {
            "owner_username": username,
            "owner_id": getattr(post, "owner_id", None),
            "date_utc": getattr(post, "date_utc", None).isoformat() if getattr(post, "date_utc", None) else None,
            "is_video": getattr(post, "is_video", False),
            "typename": getattr(post, "typename", None),
            "likes": getattr(post, "likes", None),
            "comments": getattr(post, "comments", None),
        }

        target_dir = pathlib.Path(BASE_DOWNLOAD_DIR)
        target_dir.mkdir(parents=True, exist_ok=True)

        def _file_path(name: str) -> pathlib.Path:
            return target_dir / name

        # download logic (photo/video/sidecar)
        if getattr(post, "typename", "") == "GraphSidecar":
            nodes = list(post.get_sidecar_nodes())
            for idx, node in enumerate(nodes):
                if getattr(node, "is_video", False):
                    url_src = getattr(node, "video_url", None)
                    if not url_src:
                        continue
                    fname = f"{username}_{shortcode}_{idx}.mp4"
                    fpath = _file_path(fname)
                    ok = _download_url_to_file(url_src, fpath)
                    if ok:
                        out["files"].append(str(fpath))
                else:
                    url_src = getattr(node, "display_url", None)
                    if not url_src:
                        continue
                    fname = f"{username}_{shortcode}_{idx}.jpg"
                    fpath = _file_path(fname)
                    ok = _download_url_to_file(url_src, fpath)
                    if ok:
                        out["files"].append(str(fpath))
        else:
            if getattr(post, "is_video", False):
                url_src = getattr(post, "video_url", None)
                if url_src:
                    fname = f"{username}_{shortcode}.mp4"
                    fpath = _file_path(fname)
                    ok = _download_url_to_file(url_src, fpath)
                    if ok:
                        out["files"].append(str(fpath))
                else:
                    try:
                        L.download_post(post, target=str(target_dir))
                        for p in target_dir.glob(f"{username}_{shortcode}*.mp4"):
                            if p.stat().st_size > 0:
                                out["files"].append(str(p))
                    except Exception:
                        pass
            else:
                url_src = getattr(post, "url", None)
                if url_src:
                    fname = f"{username}_{shortcode}.jpg"
                    fpath = _file_path(fname)
                    ok = _download_url_to_file(url_src, fpath)
                    if ok:
                        out["files"].append(str(fpath))
                else:
                    try:
                        L.download_post(post, target=str(target_dir))
                        for p in target_dir.glob(f"{username}_{shortcode}*.jpg"):
                            if p.stat().st_size > 0:
                                out["files"].append(str(p))
                    except Exception:
                        pass

        # اطمینان از سازگاری ویدیوها با تلگرام
        compressed_files = []
        for p in list(out["files"]):
            if p.lower().endswith(".mp4"):
                try:
                    comp_path = ensure_telegram_compatible_video(p)
                    if comp_path and os.path.exists(comp_path) and os.path.getsize(comp_path) > 0:
                        compressed_files.append(comp_path)
                        out["files"].remove(p)
                        out["files"].append(comp_path)
                    else:
                        logger.warning(f"Video compatibility fix failed for {p}, keeping original")
                        compressed_files.append(p)
                except Exception as e:
                    logger.warning(f"ensure_telegram_compatible_video failed for {p}: {e}")
                    compressed_files.append(p)

        if not out["files"]:
            out["error"] = "No valid media files were downloaded."
        # prepare cleanup list
        if not keep_files:
            out["cleanup_paths"] = list(out["files"])
        else:
            out["cleanup_paths"] = []

        return out
    except Exception as e:
        logger.exception("download_post failed")
        return {"error": str(e)}

def download_reel(url: str, session_username: Optional[str] = None, keep_files: bool = False) -> dict:
    """
    Download a Reel given its URL. Uses instaloader or yt-dlp fallback.
    """
    try:
        # Reels share same shortcode format as posts
        return download_post(url, session_username=session_username, keep_files=keep_files)
    except Exception as e:
        logger.exception("download_reel failed")
        return {"error": str(e)}

def download_profile_stories(username: str, session_username: Optional[str] = None, keep_files: bool = False) -> dict:
    """
    Download public stories for a profile. Returns dict with file list.
    """
    out = {"files": [], "username": username}
    try:
        L = _get_instaloader_instance(session_username=session_username, download_videos=True, save_metadata=True)
        profile = Profile.from_username(L.context, username)

        target_dir = BASE_DOWNLOAD_DIR

        stories = L.get_stories(userids=[profile.userid])
        count = 0
        for story in stories:
            for item in story.get_items():
                date = item.date_local
                if item.is_video:
                    fname = f"{username}_story_{count}.mp4"
                    path = target_dir / fname
                    L.download_storyitem(item, str(path))
                    out["files"].append(str(path))
                else:
                    fname = f"{username}_story_{count}.jpg"
                    path = target_dir / fname
                    L.download_storyitem(item, str(path))
                    out["files"].append(str(path))
                count += 1
        return out
    except Exception as e:
        logger.exception("download_profile_stories failed")
        return {"error": str(e)}

def download_highlights(username: str, highlight_id: Optional[str] = None, session_username: Optional[str] = None,
                        keep_files: bool = False) -> dict:
    """
    Download highlights. If highlight_id is None, download all highlights for user.
    """
    out = {"files": [], "username": username}
    try:
        L = _get_instaloader_instance(session_username=session_username, download_videos=True, save_metadata=True)
        profile = Profile.from_username(L.context, username)
        # iterate highlights
        target_dir = BASE_DOWNLOAD_DIR / f"highlights_{username}_{int(time.time())}"
        target_dir.mkdir(parents=True, exist_ok=True)
        highlights = L.get_highlights(profile.userid)
        for highlight in highlights:
            # highlight is Highlight object
            if highlight is None:
                continue
            if highlight_id and str(highlight.pk) != str(highlight_id):
                continue
            for item in highlight.get_items():
                if item.is_video:
                    fname = f"{username}_highlight_{highlight.title}_{item.media_id}.mp4"
                    path = target_dir / fname
                    L.download_storyitem(item, str(path))
                    out["files"].append(str(path))
                else:
                    fname = f"{username}_highlight_{highlight.title}_{item.media_id}.jpg"
                    path = target_dir / fname
                    L.download_storyitem(item, str(path))
                    out["files"].append(str(path))
        return out
    except Exception as e:
        logger.exception("download_highlights failed")
        return {"error": str(e)}

def export_all_posts(username: str, session_username: Optional[str] = None, limit: Optional[int] = None,
                     keep_files: bool = False) -> dict:
    """
    Export all posts of a user. WARNING: Potentially large.
    Use confirmation on caller side before invoking.
    """
    out = {"files": [], "username": username, "count": 0}
    try:
        L = _get_instaloader_instance(session_username=session_username, download_videos=True, save_metadata=True)
        profile = Profile.from_username(L.context, username)
        posts = profile.get_posts()
        target_dir = BASE_DOWNLOAD_DIR / f"archive_{username}_{int(time.time())}"
        target_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        for post in posts:
            if limit and count >= limit:
                break
            shortcode = post.shortcode
            res = download_post(f"https://www.instagram.com/p/{shortcode}/", session_username=session_username, keep_files=keep_files)
            if res.get("files"):
                out["files"].extend(res["files"])
            count += 1
        out["count"] = count
        return out
    except Exception as e:
        logger.exception("export_all_posts failed")
        return {"error": str(e)}

def _fetch_profile_info(username: str) -> dict:
    try:
        L = _get_instaloader_instance()
        profile = Profile.from_username(L.context, username)
        return {
            "username": profile.username,
            "biography": profile.biography,
            "followers": profile.followers,
            "profile_pic_url": profile.profile_pic_url
        }
    except Exception as e:
        return {"error": str(e)}

# ---------- Async handlers (aiogram) ----------
# Each handler offloads the heavy work to ThreadPoolExecutor via run_in_executor.

async def instagram_download_handler(message: types.Message):
    url = (message.text or "").strip()
    if not url or "instagram.com" not in url:
        await message.answer("❌ Please provide a valid Instagram link.")
        return
    # ارسال پیام لودینگ
    loading_msg = await message.reply("We are downloading your content...")
    try:
        # تشخیص نوع ورودی
        media_type, identifier = classify_instagram_input(url)
        result = None
        profile_info = None
        pdf_path = None
        caption = None
        sent_message = None
        sent_files = []
        
        # دانلود بر اساس نوع
        if media_type == "post":
            result = download_post(url)
        elif media_type == "reel":
            result = download_reel(url)
        elif media_type == "story":
            result = download_profile_stories(identifier)
        elif media_type == "highlights":
            result = download_highlights(identifier)
        elif media_type == "profile":
            profile_info = _fetch_profile_info(identifier)
            caption = f"👤 {profile_info.get('username')}\n📌 {profile_info.get('bio', '')}\n👥 {profile_info.get('followers')} followers"
        elif media_type == "export":
            result = export_all_posts(identifier)
        elif media_type == "caption_pdf":
            # ساخت PDF کپشن
            caption_text = download_post(url).get("caption", "")
            if caption_text:
                pdf_path = _make_pdf_from_text(caption_text, "caption.pdf", title="Instagram Caption")
            else:
                raise ValueError("⚠️ No caption was found.")
        else:
            raise ValueError("⚠️ The link or input type is not supported.")
        
        # حذف پیام لودینگ پس از 3 ثانیه
        await asyncio.sleep(3)
        try:
            await loading_msg.delete()
        except Exception:
            pass
        
        # هندل ارسال بر اساس نوع
        if media_type == "profile":
            if profile_info.get("profile_pic"):
                sent_message = await message.answer_photo(profile_info["profile_pic"], caption=caption)
                sent_files.append(profile_info["profile_pic"])
            else:
                await message.answer(caption)
        elif media_type == "caption_pdf":
            sent_message = await message.answer_document(types.FSInputFile(pdf_path))
            sent_files.append(pdf_path)
        elif result and isinstance(result.get("files"), list):
            files = result["files"]
            original_caption = result.get("caption", "")
            descriptions_text = original_caption
            caption = f"Descriptions:\n{descriptions_text}\n\nOriginal link: <a href='{url}'>your link</a>\n\nDownload by <a href='https://t.me/Faryseneaidownloder_bot'>Faryseneaidownloderbot</a>"
            use_quote = len(caption) > MAX_CAPTION_LENGTH
            if not files:
                raise ValueError("⚠️ No files were found to send.")

            # helper to wrap path -> FSInputFile
            def _fs(fpath: str):
                return types.FSInputFile(fpath)

            if len(files) > 1:  # Multiple files (media group)
                media = []
                for i, file_path in enumerate(files):
                    lower = file_path.lower()
                    if lower.endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
                        media.append(
                            InputMediaPhoto(
                                media=_fs(file_path),
                                caption=(caption if i == 0 and not use_quote else ""),
                                parse_mode=ParseMode.HTML
                            )
                        )
                    elif lower.endswith((".mp4", ".mov", ".webm")):
                        media.append(
                            InputMediaVideo(
                                media=_fs(file_path),
                                caption=(caption if i == 0 and not use_quote else ""),
                                parse_mode=ParseMode.HTML
                            )
                        )
                    else:
                        media.append(
                            InputMediaDocument(
                                media=_fs(file_path),
                                caption=(caption if i == 0 and not use_quote else ""),
                                parse_mode=ParseMode.HTML
                            )
                        )
                sent_msgs = await message.answer_media_group(media=media)
                if sent_msgs:
                    sent_message = sent_msgs[0]
                    sent_files.extend(files)
                    if use_quote:
                        await message.answer(caption, parse_mode=ParseMode.HTML)
            else:  # Single file
                file_path = files[0]
                lower = file_path.lower()
                if lower.endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
                    sent_message = await message.answer_photo(
                        _fs(file_path),
                        caption=(caption if not use_quote else ""),
                        parse_mode=ParseMode.HTML
                    )
                elif lower.endswith((".mp4", ".mov", ".webm")):
                    sent_message = await message.answer_video(
                        _fs(file_path),
                        caption=(caption if not use_quote else ""),
                        parse_mode=ParseMode.HTML
                    )
                else:
                    sent_message = await message.answer_document(
                        _fs(file_path),
                        caption=(caption if not use_quote else ""),
                        parse_mode=ParseMode.HTML
                    )
                sent_files.append(file_path)
                if use_quote:
                    await message.reply(caption, reply_to_message_id=sent_message.message_id, parse_mode=ParseMode.HTML)
        
        # حذف فایل‌های ارسال شده
        for fpath in sent_files:
            try:
                if os.path.exists(fpath):
                    os.remove(fpath)
            except Exception:
                pass

        # Delete the user's message after 20 seconds
        await asyncio.sleep(20)
        try:
            await message.delete()
        except Exception:
            pass
    
    except Exception as e:
        try:
            await loading_msg.edit_text(f"❌ Error during processing: {e}")
        except Exception:
            await message.answer(f"❌ Error during processing: {e}")

@router.message(Command("instagram"))
async def handle_instagram(message: types.Message):
    await instagram_download_handler(message)

@router.message(Command("insta_login"))
async def insta_login_handler(message: types.Message):
    """
    Usage: /insta_login <username> <password>
    (password ممکنه شامل فاصله باشه — از split(maxsplit=2) استفاده شده)
    """
    text = message.text or ""
    args = text.split(maxsplit=2)
    if len(args) < 3:
        await message.reply("Usage: /insta_login <username> <password>")
        return

    username = args[1].strip()
    password = args[2].strip()

    waiting_msg = await message.reply("⏳ Logging in and saving session...")

    try:
        if inspect.iscoroutinefunction(instaloader_login):
            ok, msg = await instaloader_login(username, password)
        elif hasattr(asyncio, "to_thread"):
            ok, msg = await asyncio.to_thread(instaloader_login, username, password)
        else:
            loop = asyncio.get_running_loop()
            ok, msg = await loop.run_in_executor(None, instaloader_login, username, password)

        if ok:
            await waiting_msg.edit_text(f"✅ {msg}")
        else:
            await waiting_msg.edit_text(f"❌ Login failed: {msg}")
    except Exception as e:
        await waiting_msg.edit_text(f"❌ Error during login: {e}")

# --- helpers (drop these near the top of your instagram.py, above handlers) ---


async def _send_media(
    message: types.Message,
    fpath: Optional[str] = None,
    photo_group: Optional[List[InputMediaPhoto]] = None,
    caption: str = ""
) -> Union[types.Message, List[types.Message], None]:
    """
    Send media (photo, video, or photo group) with unified caption behavior.
    Automatically wraps long captions inside <blockquote expandable>.
    Sends media as standalone messages (not replies).
    """
    if not fpath and not photo_group:
        logger.error("No file path or photo group provided")
        return None

    # ✅ کپشن طولانی تبدیل به quote expandable
    if caption and len(caption) > 1024:
        caption = f"<blockquote expandable>\n{caption.strip()}\n</blockquote>"

    # ✅ گروه عکس‌ها
    if photo_group:
        if len(photo_group) == 1:
            # فقط یک عکس → ارسال مستقیم
            media = photo_group[0].media
            caption = photo_group[0].caption or caption
            if isinstance(media, FSInputFile):
                fpath = media.path
            else:
                return await message.bot.send_photo(
                    chat_id=message.chat.id,
                    photo=media,
                    caption=caption or None,
                    parse_mode=ParseMode.HTML
                )
        else:
            try:
                # چند عکس → گروهی ارسال کن
                return await message.bot.send_media_group(
                    chat_id=message.chat.id,
                    media=photo_group
                )
            except Exception as e:
                logger.warning(f"send_media_group failed ({e}); falling back to single sends")
                sent_messages = []
                for item in photo_group:
                    media = item.media
                    item_caption = item.caption or caption
                    try:
                        msg = await message.bot.send_photo(
                            chat_id=message.chat.id,
                            photo=media,
                            caption=item_caption or None,
                            parse_mode=ParseMode.HTML
                        )
                        sent_messages.append(msg)
                    except Exception as e2:
                        logger.error(f"single photo send failed: {e2}")
                        continue
                return sent_messages if sent_messages else None

    # ✅ تک فایل
    if not fpath or not os.path.exists(fpath):
        logger.error(f"File not found: {fpath}")
        return None

    mime_type, _ = mimetypes.guess_type(fpath)
    is_photo = mime_type and mime_type.startswith("image/")
    is_video = mime_type and mime_type.startswith("video/")
    size = os.path.getsize(fpath)

    logger.debug(f"Sending file: {fpath}, size={size}, MIME={mime_type}")

    # اگر عکس بزرگ بود → document
    if is_photo and size > MAX_PHOTO_BYTES:
        logger.warning(f"Photo too large ({fpath}), sending as document")
        return await message.bot.send_document(
            chat_id=message.chat.id,
            document=FSInputFile(fpath),
            caption=caption or None,
            parse_mode=ParseMode.HTML
        )

    try:
        result = None
        if is_photo:
            result = await message.bot.send_photo(
                chat_id=message.chat.id,
                photo=FSInputFile(fpath),
                caption=caption or None,
                parse_mode=ParseMode.HTML
            )
        elif is_video:
            result = await message.bot.send_video(
                chat_id=message.chat.id,
                video=FSInputFile(fpath),
                caption=caption or None,
                parse_mode=ParseMode.HTML
            )
        else:
            result = await message.bot.send_document(
                chat_id=message.chat.id,
                document=FSInputFile(fpath),
                caption=caption or None,
                parse_mode=ParseMode.HTML
            )
        
        # Record download history after successful send
        if result:
            try:
                from bot import record_download
                file_size = os.path.getsize(fpath) if os.path.exists(fpath) else None
                # Extract URL from message if available
                url = getattr(message, 'text', '') or ''
                if not url or 'instagram.com' not in url:
                    url = fpath  # Fallback to file path
                await record_download(
                    message.from_user.id, "instagram", url,
                    file_type="photo" if is_photo else "video" if is_video else "document",
                    file_size=file_size
                )
            except Exception as hist_e:
                logger.debug(f"Failed to record download history: {hist_e}")
        
        return result

    except Exception as e:
        logger.warning(f"FSInputFile send failed ({e}), fallback to BytesIO")

        try:
            with open(fpath, "rb") as f:
                bio = BytesIO(f.read())
                bio.name = os.path.basename(fpath)
            result = None
            if is_photo:
                result = await message.bot.send_photo(chat_id=message.chat.id, photo=bio, caption=caption or None, parse_mode=ParseMode.HTML)
            elif is_video:
                result = await message.bot.send_video(chat_id=message.chat.id, video=bio, caption=caption or None, parse_mode=ParseMode.HTML)
            else:
                result = await message.bot.send_document(chat_id=message.chat.id, document=bio, caption=caption or None, parse_mode=ParseMode.HTML)
            
            # Record download history after successful send
            if result:
                try:
                    from bot import record_download
                    file_size = os.path.getsize(fpath) if os.path.exists(fpath) else None
                    # Extract URL from message if available
                    url = getattr(message, 'text', '') or ''
                    if not url:
                        url = fpath  # Fallback to file path
                    await record_download(
                        message.from_user.id, "instagram", url,
                        file_type=file_type or ("photo" if is_photo else "video" if is_video else "document"),
                        file_size=file_size
                    )
                except Exception as hist_e:
                    logger.debug(f"Failed to record download history: {hist_e}")
            
            return result
        except Exception as e2:
            logger.error(f"BytesIO send failed: {e2}")
            try:
                return await message.bot.send_document(
                    chat_id=message.chat.id,
                    document=FSInputFile(fpath),
                    caption=caption or None,
                    parse_mode=ParseMode.HTML
                )
            except Exception as e3:
                logger.error(f"Final send attempt failed: {e3}")
                return None



# -------------------- UPDATED HANDLER --------------------

# تابع برای مدیریت مسیرها در PyInstaller
def resource_path(relative_path: str) -> str:
    """تبدیل مسیر نسبی به مسیر مطلق در محیط PyInstaller"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

@router.message(Command("insta_post"))
async def insta_post_handler(message: types.Message):
    """
    Usage: /insta_post <post_url>
    Downloads and sends Instagram post media (photos/videos) safely.
    Behavior:
      - Sends a temporary "downloading..." message and deletes it after 3 seconds.
      - Sends media as a single message (media group for photos with caption on first item,
        or single video/photo with caption attached).
      - Sends a caption replying to the sent media if caption length exceeds MAX_CAPTION_LENGTH.
      - Cleans up local files that were successfully sent (unless download_post returned keep_files).
    """
    text = (message.text or "").strip()
    parts = text.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await message.reply("Usage: /insta_post <post_url>")
        return
    url = parts[1].strip()
    loading_msg = await message.reply("We are downloading your content...")
    try:
        # Run sync download_post in background thread
        async with _download_lock:
            if hasattr(asyncio, "to_thread"):
                res: Dict = await asyncio.to_thread(download_post, url)
            else:
                loop = asyncio.get_running_loop()
                res = await loop.run_in_executor(_executor, download_post, url)
        
        # Delete loading message after 3 seconds
        await asyncio.sleep(3)
        try:
            await loading_msg.delete()
        except Exception:
            pass

        if not isinstance(res, dict):
            await message.reply("❌ Error: invalid response from download_post")
            return
        if res.get("error"):
            await message.reply(f"❌ Error: {res['error']}")
            return
        files: List[str] = res.get("files", []) or []
        original_caption = res.get("caption", "")
        descriptions_text = original_caption
        caption = f"Descriptions:\n{descriptions_text}\n\nOriginal link: <a href='{url}'>your link</a>\n\nDownload by <a href='https://t.me/Faryseneaidownloder_bot'>Faryseneaidownloderbot</a>"
        use_quote = len(caption) > MAX_CAPTION_LENGTH
        if not files:
            await message.reply("❌ No files were returned for this post.")
            return

        sent_message = None
        sent_files = []

        # Helper to wrap path -> FSInputFile safely
        def _fs(fpath: str):
            return types.FSInputFile(resource_path(fpath))

        # Handle media sending
        if len(files) > 1:  # Multiple files (use media group)
            media = []
            for i, f in enumerate(files):
                lower = f.lower()
                if lower.endswith((".jpg", ".jpeg", ".png", ".webp")):
                    media.append(
                        InputMediaPhoto(
                            media=_fs(f),
                            caption=(caption if i == 0 and not use_quote else ""),
                            parse_mode=ParseMode.HTML
                        )
                    )
                elif lower.endswith((".mp4", ".mov", ".webm", ".mkv", ".avi")):
                    media.append(
                        InputMediaVideo(
                            media=_fs(f),
                            caption=(caption if i == 0 and not use_quote else ""),
                            parse_mode=ParseMode.HTML
                        )
                    )
                else:
                    media.append(
                        InputMediaDocument(
                            media=_fs(f),
                            caption=(caption if i == 0 and not use_quote else ""),
                            parse_mode=ParseMode.HTML
                        )
                    )
            try:
                sent_msgs = await message.answer_media_group(media=media)
                if sent_msgs:
                    sent_message = sent_msgs[0]  # Use first message for reply
                    sent_files.extend(files)
                    if use_quote:
                        await message.answer(caption, parse_mode=ParseMode.HTML)
            except Exception as e:
                await message.reply(f"❌ Error sending media group: {e}")
                return
        else:  # Single file
            f = files[0]
            lower = f.lower()
            try:
                if lower.endswith((".jpg", ".jpeg", ".png", ".webp")):
                    sent_message = await message.reply_photo(
                        _fs(f),
                        caption=(caption if not use_quote else ""),
                        parse_mode=ParseMode.HTML
                    )
                elif lower.endswith((".mp4", ".mov", ".webm", ".mkv", ".avi")):
                    sent_message = await message.reply_video(
                        _fs(f),
                        caption=(caption if not use_quote else ""),
                        parse_mode=ParseMode.HTML
                    )
                else:
                    sent_message = await message.reply_document(
                        _fs(f),
                        caption=(caption if not use_quote else ""),
                        parse_mode=ParseMode.HTML
                    )
                sent_files.append(f)
                if use_quote:
                    await message.reply(caption, reply_to_message_id=sent_message.message_id, parse_mode=ParseMode.HTML)
            except Exception as e:
                await message.reply(f"❌ Error sending file: {e}")
                return

        # Cleanup: remove only successfully sent files
        if not res.get("keep_files", False):
            for fpath in sent_files:
                try:
                    rp = resource_path(fpath)
                    if rp and os.path.exists(rp):
                        os.remove(rp)
                        logger.debug(f"Deleted file: {rp}")
                except Exception:
                    logger.exception(f"Failed to delete file: {fpath}")

        # Delete the user's message after 20 seconds
        await asyncio.sleep(20)
        try:
            await message.delete()
        except Exception:
            pass

    except Exception as exc:
        logger.exception(f"Error in insta_post_handler: {exc}")
        try:
            await loading_msg.edit_text(f"❌ Error during processing: {exc}")
        except Exception:
            await message.reply(f"❌ Error during processing: {exc}")

@router.message(Command("insta_reel"))
async def insta_reel_handler(message: types.Message):
    """
    Usage: /insta_reel <reel_url>
    Downloads and sends Instagram reel media (photos/videos) with description.
    """
    text = (message.text or "").strip()
    parts = text.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await message.answer("Usage: /insta_reel <reel_url>")
        return

    url = parts[1].strip()
    status_msg = await message.answer("⏳ Downloading reel...")

    try:
        async with _download_lock:
            if hasattr(asyncio, "to_thread"):
                res: Dict = await asyncio.to_thread(download_reel, url)
            else:
                loop = asyncio.get_running_loop()
                res = await loop.run_in_executor(_executor, download_reel, url)

        if not isinstance(res, dict):
            await status_msg.edit_text("❌ Error: invalid response from download_reel")
            return

        if res.get("error"):
            await status_msg.edit_text(f"❌ Error: {res['error']}")
            return

        files: List[str] = res.get("files", []) or []
        original_caption = (res.get("caption", "") or "").strip()

        # Build caption
        if original_caption:
            caption = (
                f"{original_caption}\n\n"
                f"🔗 Download by <a href='https://t.me/Faryseneaidownloder_bot'>Faryseneaidownloderbot</a>"
            )
        else:
            caption = f"🔗 Download by <a href='https://t.me/Faryseneaidownloder_bot'>Faryseneaidownloderbot</a>"

        use_quote = len(caption) > MAX_CAPTION_LENGTH

        if not files:
            await status_msg.edit_text("❌ No files were returned for this reel.")
            return
        
        sent_files = []
        media_group = []

        try:
            # Delete status message before sending media
            await asyncio.sleep(3)
            try:
                await status_msg.delete()
            except Exception:
                pass

            # Collect all media items with caption on first item only
            for idx, fpath in enumerate(files):
                fpath = resource_path(fpath)
                if not fpath or not os.path.exists(fpath):
                    logger.error(f"File not found: {fpath}")
                    continue

                lower = fpath.lower()
                is_video = lower.endswith((".mp4", ".mov", ".webm", ".mkv", ".avi"))
                is_photo = lower.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp"))

                # Ensure video compatibility
                if is_video:
                    try:
                        compatible_path = ensure_telegram_compatible_video(fpath)
                        if compatible_path and os.path.exists(compatible_path):
                            fpath = compatible_path
                        sent_files.append(fpath)
                    except Exception as e:
                        logger.warning(f"ensure_telegram_compatible_video failed for {fpath}: {e}")
                        sent_files.append(fpath)

                # Add to media group with caption only on first item if not quoting
                caption_for_item = (caption if idx == 0 and not use_quote else None)
                parse_mode_for_item = (ParseMode.HTML if caption_for_item else None)
                if is_photo:
                    media_group.append(InputMediaPhoto(
                        media=FSInputFile(fpath),
                        caption=caption_for_item,
                        parse_mode=parse_mode_for_item
                    ))
                    sent_files.append(fpath)
                elif is_video:
                    media_group.append(InputMediaVideo(
                        media=FSInputFile(fpath),
                        caption=caption_for_item,
                        parse_mode=parse_mode_for_item
                    ))
                else:
                    media_group.append(InputMediaDocument(
                        media=FSInputFile(fpath),
                        caption=caption_for_item,
                        parse_mode=parse_mode_for_item
                    ))
                    sent_files.append(fpath)

            # Send media
            if not media_group:
                await message.answer("❌ No valid files to send.")
                return

            sent_message = None
            if len(media_group) == 1:
                item = media_group[0]
                if isinstance(item, InputMediaVideo):
                    sent_message = await message.answer_video(item.media, caption=item.caption, parse_mode=item.parse_mode)
                elif isinstance(item, InputMediaPhoto):
                    sent_message = await message.answer_photo(item.media, caption=item.caption, parse_mode=item.parse_mode)
                else:
                    sent_message = await message.answer_document(item.media, caption=item.caption, parse_mode=item.parse_mode)
            else:
                sent_messages = await message.bot.send_media_group(chat_id=message.chat.id, media=media_group)
                if sent_messages:
                    sent_message = sent_messages[0]

            # Send caption separately if too long
            if use_quote and sent_message:
                await message.reply(caption, reply_to_message_id=sent_message.message_id, parse_mode=ParseMode.HTML)

        except Exception as send_exc:
            logger.exception(f"Error sending files: {send_exc}")
            await message.answer(f"❌ Error sending files: {send_exc}")
            raise

        finally:
            # Clean up temporary files
            for fpath in sent_files:
                try:
                    if fpath and os.path.exists(fpath):
                        os.remove(fpath)
                except Exception:
                    logger.warning(f"Failed to delete file: {fpath}")

            # Delete user's message after 20 seconds
            await asyncio.sleep(20)
            try:
                await message.delete()
            except Exception:
                pass

    except Exception as exc:
        logger.exception(f"Error in insta_reel_handler: {exc}")
        try:
            await message.answer(f"❌ Error during processing: {exc}")
        except Exception:
            pass



@router.message(Command("insta_story"))
async def insta_story_handler(message: types.Message):
    """
    Usage: /insta_story <@username or profile url>
    Downloads and sends Instagram stories for the given username.
    """
    text = (message.text or "").strip()
    parts = text.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await message.reply("Usage: /insta_story <username or profile url>")
        return

    raw = parts[1].strip()
    # استخراج username از ورودی‌هایی مثل @user یا https://www.instagram.com/user/
    username = raw.replace("https://www.instagram.com/", "").replace("http://www.instagram.com/", "")
    username = username.strip("/@ ").split("?")[0]  # حذف پارامترهای احتمالی

    status_msg = await message.reply(f"⏳ Downloading public stories for {username} ...")

    try:
        # دانلود استوری‌ها در thread
        async with _download_lock:
            if hasattr(asyncio, "to_thread"):
                res: Dict = await asyncio.to_thread(download_profile_stories, username)
            else:
                loop = asyncio.get_running_loop()
                res = await loop.run_in_executor(_executor, download_profile_stories, username)

        if not isinstance(res, dict):
            await status_msg.edit_text("❌ Error: invalid response from download_profile_stories")
            return

        if res.get("error"):
            await status_msg.edit_text(f"❌ Error: {res['error']}")
            return

        files: List[str] = res.get("files", []) or []
        original_caption = res.get("caption", "")
        descriptions_text = original_caption
        links_line = f"Original link: <a href='https://www.instagram.com/{username}/'>your link</a>"
        caption = f"Descriptions:\n{descriptions_text}\n\n{links_line}\n\nDownload by <a href='https://t.me/Faryseneaidownloder_bot'>Faryseneaidownloderbot</a>"
        use_quote = len(caption) > MAX_CAPTION_LENGTH
        if not files:
            await status_msg.edit_text("❌ No stories were found or downloaded.")
            return

        sent_files = []
        photo_group = []
        sent_any = False
        last_sent_message = None

        try:
            for idx, fpath in enumerate(files):
                # بررسی وجود فایل
                fpath = resource_path(fpath)
                if not fpath or not os.path.exists(fpath):
                    logger.error(f"File not found: {fpath}")
                    continue

                logger.debug(f"Attempting to send file: {fpath}, exists: {os.path.exists(fpath)}")
                lower = fpath.lower()
                is_video = lower.endswith((".mp4", ".mov", ".webm", ".mkv", ".avi"))
                is_photo = lower.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp"))

                if is_photo:
                    # جمع‌آوری عکس‌ها برای ارسال گروهی
                    photo_group.append(InputMediaPhoto(
                        media=FSInputFile(fpath),
                        caption=(caption if not sent_any and len(photo_group) == 0 and not use_quote else ""),
                        parse_mode=ParseMode.HTML
                    ))
                    sent_files.append(fpath)
                    need_flush = (len(photo_group) == 10) or (idx == len(files) - 1)
                    if need_flush:
                        if len(photo_group) == 1:
                            last_sent_message = await message.reply_photo(
                                photo=photo_group[0].media,
                                caption=photo_group[0].caption,
                                parse_mode=ParseMode.HTML
                            )
                        else:
                            sent_msgs = await message.bot.send_media_group(
                                chat_id=message.chat.id,
                                media=photo_group
                            )
                            if sent_msgs:
                                last_sent_message = sent_msgs[0]
                        sent_any = True
                        photo_group = []

                elif is_video:
                    # اطمینان از سازگاری ویدیو با تلگرام
                    try:
                        compatible_path = ensure_telegram_compatible_video(fpath)
                        if compatible_path and os.path.exists(compatible_path):
                            fpath = compatible_path
                            sent_files.append(fpath)
                        else:
                            logger.warning(f"Video compatibility fix failed for {fpath}, using original")
                            sent_files.append(fpath)
                    except Exception as e:
                        logger.warning(f"ensure_telegram_compatible_video failed for {fpath}: {e}")
                        sent_files.append(fpath)

                    # flush عکس‌های جمع‌شده
                    if photo_group:
                        if len(photo_group) == 1:
                            last_sent_message = await message.reply_photo(
                                photo=photo_group[0].media,
                                caption=photo_group[0].caption,
                                parse_mode=ParseMode.HTML
                            )
                        else:
                            sent_msgs = await message.bot.send_media_group(
                                chat_id=message.chat.id,
                                media=photo_group
                            )
                            if sent_msgs:
                                last_sent_message = sent_msgs[0]
                        sent_any = True
                        photo_group = []

                    cap = caption if not sent_any and not use_quote else ""
                    last_sent_message = await message.reply_video(
                        video=FSInputFile(fpath),
                        caption=cap,
                        parse_mode=ParseMode.HTML
                    )
                    sent_any = True

                else:
                    # flush عکس‌های جمع‌شده
                    if photo_group:
                        if len(photo_group) == 1:
                            last_sent_message = await message.reply_photo(
                                photo=photo_group[0].media,
                                caption=photo_group[0].caption,
                                parse_mode=ParseMode.HTML
                            )
                        else:
                            sent_msgs = await message.bot.send_media_group(
                                chat_id=message.chat.id,
                                media=photo_group
                            )
                            if sent_msgs:
                                last_sent_message = sent_msgs[0]
                        sent_any = True
                        photo_group = []

                    cap = caption if not sent_any and not use_quote else ""
                    last_sent_message = await message.reply_document(
                        document=FSInputFile(fpath),
                        caption=cap,
                        parse_mode=ParseMode.HTML
                    )
                    sent_files.append(fpath)
                    sent_any = True

            # ارسال کپشن جداگانه در صورت طولانی بودن
            if use_quote and sent_any and last_sent_message:
                await message.answer(caption, parse_mode=ParseMode.HTML)

        except Exception as send_exc:
            logger.exception(f"Error sending files: {send_exc}")
            await status_msg.edit_text(f"❌ Error sending files: {send_exc}")
            raise

        finally:
            # حذف فقط فایل‌هایی که با موفقیت ارسال شده‌اند
            for fpath in sent_files:
                try:
                    if fpath and os.path.exists(fpath):
                        os.remove(fpath)
                        logger.debug(f"Deleted file: {fpath}")
                except Exception:
                    logger.warning(f"Failed to delete file: {fpath}")

            # Delete the user's message after 20 seconds
            await asyncio.sleep(20)
            try:
                await message.delete()
            except Exception:
                pass

    except Exception as exc:
        logger.exception(f"Error in insta_story_handler: {exc}")
        await status_msg.edit_text(f"❌ Error during processing: {exc}")

@router.message(Command("insta_highlights"))
async def insta_highlights_handler(message: types.Message):
    """
    Usage: /insta_highlights <username> [highlight_id]
    Downloads and sends Instagram highlights for the given username.
    """
    text = (message.text or "").strip()
    parts = text.split(maxsplit=2)  # maxsplit=2 تا highlight_id هم با فاصله مدیریت بشه
    if len(parts) < 2 or not parts[1].strip():
        await message.reply("Usage: /insta_highlights <username> [highlight_id]")
        return

    raw_username = parts[1].strip()
    highlight_id = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None

    # پاک‌سازی username از URL یا علامت @
    username = raw_username.replace("https://www.instagram.com/", "").replace("http://www.instagram.com/", "")
    username = username.strip("/@ ").split("?")[0]

    status_msg = await message.reply(f"⏳ Downloading highlights for {username} ...")

    try:
        async with _download_lock:
            # اجرای blocking function در thread
            if getattr(asyncio, "to_thread", None):
                res: Dict = await asyncio.to_thread(download_highlights, username, highlight_id)
            else:
                loop = asyncio.get_running_loop()
                res = await loop.run_in_executor(_executor, download_highlights, username, highlight_id)

        if not isinstance(res, dict):
            await status_msg.edit_text("❌ Error: invalid response from download_highlights")
            return

        if res.get("error"):
            await status_msg.edit_text(f"❌ Error: {res['error']}")
            return

        files: List[str] = res.get("files", []) or []
        original_caption = res.get("caption", "")
        descriptions_text = original_caption
        links_line = f"Original link: <a href='https://www.instagram.com/{username}/'>your link</a>"
        caption = f"Descriptions:\n{descriptions_text}\n\n{links_line}\n\nDownload by <a href='https://t.me/Faryseneaidownloder_bot'>Faryseneaidownloderbot</a>"
        use_quote = len(caption) > MAX_CAPTION_LENGTH
        if not files:
            await status_msg.edit_text("❌ No highlight files were found.")
            return

        # ارسال فایل‌ها — استفاده از media_group برای عکس‌ها
        photo_group: List[types.InputMedia] = []
        sent_files = []
        sent_any = False
        last_sent_message = None

        try:
            for idx, fpath in enumerate(files):
                fpath = resource_path(fpath)
                if not fpath or not os.path.exists(fpath):
                    logger.error(f"File not found: {fpath}")
                    continue

                logger.debug(f"Attempting to send file: {fpath}, exists: {os.path.exists(fpath)}")
                lower = fpath.lower()
                is_video = lower.endswith((".mp4", ".mov", ".webm", ".mkv", ".avi"))
                is_photo = lower.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp"))

                if is_photo:
                    caption_for_item = caption if not sent_any and len(photo_group) == 0 and not use_quote else ""
                    photo_group.append(InputMediaPhoto(
                        media=FSInputFile(fpath),
                        caption=caption_for_item,
                        parse_mode=ParseMode.HTML
                    ))
                    sent_files.append(fpath)
                    need_flush = (len(photo_group) == 10) or (idx == len(files) - 1)
                    if need_flush:
                        if len(photo_group) == 1:
                            last_sent_message = await message.reply_photo(
                                photo=photo_group[0].media,
                                caption=photo_group[0].caption,
                                parse_mode=ParseMode.HTML
                            )
                        else:
                            sent_msgs = await message.bot.send_media_group(
                                chat_id=message.chat.id,
                                media=photo_group
                            )
                            if sent_msgs:
                                last_sent_message = sent_msgs[0]
                        sent_any = True
                        photo_group = []

                elif is_video:
                    # اطمینان از سازگاری ویدیو با تلگرام
                    try:
                        compatible_path = ensure_telegram_compatible_video(fpath)
                        if compatible_path and os.path.exists(compatible_path):
                            fpath = compatible_path
                            sent_files.append(fpath)
                        else:
                            logger.warning(f"Video compatibility fix failed for {fpath}, using original")
                            sent_files.append(fpath)
                    except Exception as e:
                        logger.warning(f"ensure_telegram_compatible_video failed for {fpath}: {e}")
                        sent_files.append(fpath)

                    # flush عکس‌های جمع‌شده اگر وجود دارند
                    if photo_group:
                        if len(photo_group) == 1:
                            last_sent_message = await message.reply_photo(
                                photo=photo_group[0].media,
                                caption=photo_group[0].caption,
                                parse_mode=ParseMode.HTML
                            )
                        else:
                            sent_msgs = await message.bot.send_media_group(
                                chat_id=message.chat.id,
                                media=photo_group
                            )
                            if sent_msgs:
                                last_sent_message = sent_msgs[0]
                        sent_any = True
                        photo_group = []

                    cap = caption if not sent_any and not use_quote else ""
                    last_sent_message = await message.reply_video(
                        FSInputFile(fpath),
                        caption=cap,
                        parse_mode=ParseMode.HTML
                    )
                    sent_any = True

                else:
                    # فرمت ناشناخته -> flush عکس‌ها و سپس ارسال به عنوان document
                    if photo_group:
                        if len(photo_group) == 1:
                            last_sent_message = await message.reply_photo(
                                photo=photo_group[0].media,
                                caption=photo_group[0].caption,
                                parse_mode=ParseMode.HTML
                            )
                        else:
                            sent_msgs = await message.bot.send_media_group(
                                chat_id=message.chat.id,
                                media=photo_group
                            )
                            if sent_msgs:
                                last_sent_message = sent_msgs[0]
                        sent_any = True
                        photo_group = []

                    cap = caption if not sent_any and not use_quote else ""
                    last_sent_message = await message.reply_document(
                        FSInputFile(fpath),
                        caption=cap,
                        parse_mode=ParseMode.HTML
                    )
                    sent_files.append(fpath)
                    sent_any = True

            # ارسال کپشن جداگانه در صورت طولانی بودن
            if use_quote and sent_any and last_sent_message:
                await message.answer(caption, parse_mode=ParseMode.HTML)

        except Exception as send_exc:
            logger.exception(f"Error sending files: {send_exc}")
            await status_msg.edit_text(f"❌ Error sending files: {send_exc}")
            raise

        finally:
            # پاک‌سازی فایل‌ها حتی در صورت خطا
            for fpath in sent_files:
                try:
                    if fpath and os.path.exists(fpath):
                        os.remove(fpath)
                        logger.debug(f"Deleted file: {fpath}")
                except Exception:
                    logger.warning(f"Failed to delete file: {fpath}")

            # Delete the user's message after 20 seconds
            await asyncio.sleep(20)
            try:
                await message.delete()
            except Exception:
                pass

    except Exception as exc:
        logger.exception(f"Error in insta_highlights_handler: {exc}")
        await status_msg.edit_text(f"❌ Error during processing: {exc}")

@router.message(Command("insta_profile"))
async def insta_profile_handler(message: types.Message):
    """
    Usage: /insta_profile <@username or profile url>
    """
    text = (message.text or "").strip()
    parts = text.split(maxsplit=1)
    if len(parts) < 2 or not parts[1].strip():
        await message.reply("Usage: /insta_profile <username or profile url>")
        return

    raw = parts[1].strip()
    # پاک‌سازی ورودی از URL یا @
    username = raw.replace("https://www.instagram.com/", "").replace("http://www.instagram.com/", "")
    username = username.strip("/@ ").split("?")[0]

    status_msg = await message.reply(f"⏳ Fetching profile {username} ...")

    try:
        # اجرا در thread تا event loop بلاک نشه
        if getattr(asyncio, "to_thread", None):
            info = await asyncio.to_thread(_fetch_profile_info, username)
        else:
            loop = asyncio.get_running_loop()
            info = await loop.run_in_executor(None, _fetch_profile_info, username)

        # اعتبارسنجی پاسخ
        if not isinstance(info, dict) or not info.get("username"):
            await status_msg.edit_text("❌ Error: failed to fetch profile info.")
            return

        # امن‌سازی متن برای استفاده در parse_mode="HTML"
        esc_username = html.escape(info.get("username", ""))
        esc_bio = html.escape(info.get("biography", "") or "No biography")
        followers = info.get("followers")
        followers_text = str(followers) if followers is not None else "N/A"
        profile_pic_url = info.get("profile_pic_url")

        caption = (
            f"<b>{esc_username}</b>\n"
            f"📝 {esc_bio}\n"
            f"👥 Followers: {followers_text}\n"
            f"🔗 https://www.instagram.com/{esc_username}/"
        )

        # اگر URL عکس پروفایل موجود است، آن را به عنوان photo ارسال کن (تلگرام از URL پشتیبانی می‌کند)
        if profile_pic_url:
            await status_msg.delete()  # پیام وضعیت را حذف می‌کنیم تا اسپم نشود
            await message.reply_photo(profile_pic_url, caption=caption, parse_mode=ParseMode.HTML)
        else:
            await status_msg.edit_text(caption, parse_mode=ParseMode.HTML)

    except Exception as e:
        # لاگ خطا برای دیباگ
        try:
            logger.exception("insta_profile failed")
        except Exception:
            pass
        # پیام قابل فهم برای کاربر
        await status_msg.edit_text(f"❌ Error fetching profile: {e}")

@router.message(Command("insta_caption_pdf"))
async def insta_caption_pdf_handler(message: types.Message):
    """
    /insta_caption_pdf <post_url> [--translate=<lang>]
    """
    text = (message.text or "").strip()
    parts = text.split()
    if len(parts) < 2:
        await message.reply("Usage: /insta_caption_pdf <post_url> [--translate=<lang>]")
        return

    url = parts[1].strip()
    translate_to = None
    for p in parts[2:]:
        if p.startswith("--translate="):
            translate_to = p.split("=", 1)[1].strip() or None

    status_msg = await message.reply("⏳ Extracting caption and creating PDF...")

    try:
        # دانلود پست در thread
        if getattr(asyncio, "to_thread", None):
            res = await asyncio.to_thread(download_post, url)
        else:
            loop = asyncio.get_running_loop()
            res = await loop.run_in_executor(_executor, download_post, url)

        if not isinstance(res, dict) or res.get("error"):
            await status_msg.edit_text(f"❌ Error extracting post: {res.get('error', 'unknown error')}")
            return

        caption = res.get("caption") or ""

        # ترجمه در thread
        if translate_to:
            translated, ok = await asyncio.to_thread(_translate_text, caption, translate_to)
            if ok:
                caption = translated

        # ایجاد PDF در thread
        pdf_path = BASE_DOWNLOAD_DIR / f"caption_pdf_{int(time.time())}.pdf"
        await asyncio.to_thread(_make_pdf_from_text, caption, pdf_path, title="Instagram Caption")

        # ارسال PDF
        await message.reply_document(FSInputFile(str(pdf_path)))
        await status_msg.edit_text("✅ PDF created successfully.")

    except Exception as e:
        try:
            logger.exception("PDF creation failed")
        except Exception:
            pass
        await status_msg.edit_text(f"❌ PDF creation failed: {e}")

    finally:
        # پاک‌سازی PDF
        try:
            if pdf_path.exists():
                pdf_path.unlink()
        except Exception:
            pass

@router.message()
async def insta_confirm_export_handler(message: Message):
    text = (message.text or "").strip()
    if not text.startswith("YES_EXPORT ") or len(text.split(maxsplit=1)) < 2 or not text.split(maxsplit=1)[1].strip():
        await message.reply("⚠️ Invalid confirmation format. Use: YES_EXPORT <username>")
        return

    username = text.split(maxsplit=1)[1].strip()
    status_msg = await message.reply(f"⏳ Exporting posts for {username} (this may take long)...", parse_mode=ParseMode.HTML)

    try:
        async with _download_lock:
            if getattr(asyncio, "to_thread", None):
                res = await asyncio.to_thread(export_all_posts, username)
            else:
                loop = asyncio.get_running_loop()
                res = await loop.run_in_executor(_executor, export_all_posts, username)

        if not isinstance(res, dict):
            await status_msg.edit_text("❌ Error: invalid response from export_all_posts")
            return

        if res.get("error"):
            await status_msg.edit_text(f"❌ Error exporting: {res['error']}")
            return

        count = res.get("count", 0)
        await status_msg.edit_text(
            f"✅ Exported {count} posts. Files stored temporarily on server.\n"
            f"Note: Files are not auto-sent to avoid flooding; user can request specific files or a zip.",
            parse_mode=ParseMode.HTML
        )

    except Exception as exc:
        await status_msg.edit_text(f"❌ Error during export: {exc}")

