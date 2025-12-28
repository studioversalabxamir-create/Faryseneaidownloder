import os
import sys
import json
import re
import pathlib
import tempfile
import logging
import asyncio
import requests
from datetime import datetime
from typing import Tuple, Dict, List, Optional
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
from tiktok_downloader import snaptik
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from http.client import RemoteDisconnected

from aiogram import Router, types
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

# ---------- Logging ----------
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------- Router & Executor ----------
router = Router()
executor = ThreadPoolExecutor(max_workers=3)

# ---------- Proxies ----------
# Import from centralized config
try:
    from config import PROXY
except ImportError:
    PROXY = "http://174.136.204.40:80"

# ---------- Temp & Downloads ----------
tempfile.tempdir = tempfile.gettempdir()  # درصورت نیاز می‌توانی "C:\\Temp" بگذاری
DOWNLOAD_DIR = pathlib.Path("downloads/tiktok")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ---------- UA ----------
ua = UserAgent()

# ---------- Try import TikTok downloader libs (two variants) ----------
ttd = None
try:
    import tiktok_downloader as _ttd
    ttd = _ttd
except Exception:
    try:
        import tiktok_downloader as _ttd
        ttd = _ttd
    except Exception:
        ttd = None

# ---------- Helpers ----------

def _common_headers() -> Dict[str, str]:
    return {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Referer": "https://www.tiktok.com/",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "*/*",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
    }

def _is_direct_video(url: str) -> bool:
    u = url.lower()
    return u.endswith((".mp4", ".mov", ".webm", ".m4v")) or "mime_type=video" in u

def _safe_filename(s: str, maxlen: int = 80) -> str:
    s = re.sub(r"[\\/:*?\"<>|]+", "_", (s or "").strip())
    return (s[:maxlen] or "tiktok_video").strip("_")


# ====== Session factory with retries ======
def create_session(timeout: int = 30) -> requests.Session:
    """
    Create a requests.Session with Retry (exponential backoff) and proper headers.
    Use this session for HEAD/GET requests and for downloading stream.
    """
    session = requests.Session()

    # Common headers (server-friendly)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        "Referer": "https://www.tiktok.com/",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "*/*",
    })

    # Retry strategy
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,               # 1s, 2s, 4s, 8s...
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["HEAD", "GET", "OPTIONS"]),
        raise_on_status=False,
        respect_retry_after_header=True
    )
    adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    # Set reasonable timeout default via attribute for our wrappers (we still pass timeout explicitly)
    session.request_timeout = timeout
    return session

# ====== Resolve final URL (safe) ======
def _resolve_final_url(url: str) -> str:
    """
    Resolve short redirects. Use session HEAD first, fallback to GET if HEAD fails.
    Uses create_session() with retries.
    """
    session = create_session(timeout=20)
    try:
        # prefer HEAD (lightweight), but some servers drop HEAD -> fallback to GET
        r = session.head(url, allow_redirects=True, proxies=PROXY, timeout=20)
        if r.ok and r.url:
            return r.url
    except Exception as e_head:
        logger.debug(f"HEAD failed resolving URL ({url}): {e_head}")

    # fallback to GET (safer)
    try:
        r = session.get(url, allow_redirects=True, proxies=PROXY, timeout=25)
        if r.ok and r.url:
            return r.url
    except Exception as e_get:
        logger.debug(f"GET fallback failed resolving URL ({url}): {e_get}")

    # if both failed, return original
    return url

# ====== Robust download with retries & handling RemoteDisconnected ======
def _download_via_requests(file_url: str, save_path: pathlib.Path, headers: Optional[Dict[str, str]] = None) -> str:
    """
    Download file_url to save_path using a session with retries and streaming.
    Handles RemoteDisconnected by retrying via session adapter.
    """
    session = create_session(timeout=120)
    # merge headers (session headers have defaults)
    req_headers = session.headers.copy()
    if headers:
        req_headers.update(headers)

    tmp_path = str(save_path) + ".part"
    try:
        with session.get(file_url, headers=req_headers, proxies=PROXY, stream=True, timeout=120) as r:
            r.raise_for_status()
            # write streaming
            with open(tmp_path, "wb") as fd:
                for chunk in r.iter_content(chunk_size=1024 * 256):  # 256KB chunks
                    if chunk:
                        fd.write(chunk)
        # atomic rename
        os.replace(tmp_path, str(save_path))
        return str(save_path)
    except RemoteDisconnected as rd:
        logger.warning(f"RemoteDisconnected when downloading {file_url}: {rd}. Retrying once with fresh session...")
        # one more attempt with a fresh session (sometimes helps)
        try:
            session2 = create_session(timeout=120)
            with session2.get(file_url, headers=req_headers, proxies=PROXY, stream=True, timeout=120) as r2:
                r2.raise_for_status()
                with open(tmp_path, "wb") as fd:
                    for chunk in r2.iter_content(chunk_size=1024 * 256):
                        if chunk:
                            fd.write(chunk)
            os.replace(tmp_path, str(save_path))
            return str(save_path)
        except Exception as e2:
            logger.error(f"Retry after RemoteDisconnected failed for {file_url}: {e2}")
            # cleanup partial
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            raise
    except Exception as e:
        logger.error(f"Download failed for {file_url}: {e}")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        raise

def _extract_metadata_html(url: str) -> Dict:
    """
    متادیتا را از صفحه HTML (OG و JSON-LD) استخراج می‌کند.
    """
    info = {
        "text": "",
        "uploader": "",
        "uploader_id": "",
        "uploader_image": None,
        "upload_date": None,
    }
    try:
        r = requests.get(url, headers=_common_headers(), proxies=PROXY, timeout=45)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")

        # Extract uploader_id from URL
        parsed = urlparse(url)
        path_parts = parsed.path.strip('/').split('/')
        if path_parts and path_parts[0].startswith('@'):
            info["uploader_id"] = path_parts[0][1:]  # remove @

        # caption/description
        meta_desc = soup.find("meta", {"property": "og:description"}) or soup.find("meta", {"name": "description"})
        if meta_desc and meta_desc.get("content"):
            info["text"] = meta_desc.get("content")[:1000]

        # uploader from JSON-LD or og:title
        ld_tags = soup.find_all("script", {"type": "application/ld+json"})
        for tag in ld_tags:
            try:
                data = json.loads(tag.string or "")
                if isinstance(data, dict) and data.get("@type") == "VideoObject":
                    if data.get("description"):
                        info["text"] = data["description"][:1000]
                    if data.get("uploadDate"):
                        try:
                            dt = datetime.fromisoformat(data["uploadDate"].replace("Z", "+00:00"))
                            info["upload_date"] = dt.strftime("%Y-%m-%d")
                        except Exception:
                            info["upload_date"] = data.get("uploadDate")
                    auth = data.get("author")
                    if isinstance(auth, dict):
                        info["uploader"] = auth.get("name") or info["uploader"]
                    elif isinstance(auth, str):
                        info["uploader"] = auth or info["uploader"]
                    break
            except Exception:
                continue

        if not info["uploader"]:
            meta_title = soup.find("meta", {"property": "og:title"}) or soup.find("meta", {"name": "title"})
            if meta_title and meta_title.get("content"):
                info["uploader"] = meta_title.get("content")

        if not info["uploader_image"]:
            meta_img = soup.find("meta", {"property": "og:image"})
            if meta_img and meta_img.get("content"):
                info["uploader_image"] = meta_img.get("content")

    except Exception as e:
        logger.debug(f"Failed to extract HTML metadata: {e}")

    return info

def _download_with_ttd(url: str, save_dir: pathlib.Path) -> Optional[str]:
    """
    تلاش برای دانلود با کتابخانه tiktok_downloader / tik_tok_downloader (اگر نصب باشد).
    سعی می‌کنیم APIهای متداول را به صورت داک-تایپ صدا بزنیم.
    """
    if not ttd:
        return None
    try:
        # حالت 1: کلاس TikTokDownloader(save_path=...).download(url) → path
        if hasattr(ttd, "TikTokDownloader"):
            try:
                dl = ttd.TikTokDownloader(save_path=str(save_dir))
            except TypeError:
                dl = ttd.TikTokDownloader()
            # متدهای محتمل
            cand_methods = ["download", "download_video", "get_video"]
            for m in cand_methods:
                if hasattr(dl, m):
                    out = getattr(dl, m)(url)
                    # اگر خود فایل را دانلود کرده باشد و مسیر بدهد
                    if isinstance(out, str) and os.path.exists(out):
                        return out
                    # اگر URL ویدیو بدون واترمارک دهد
                    if isinstance(out, dict):
                        vid_u = (
                            out.get("video_no_watermark")
                            or out.get("video_url_no_watermark")
                            or out.get("video_url")
                            or out.get("play")
                        )
                        if vid_u:
                            fname = "tiktok_video.mp4"
                            save_path = save_dir / fname
                            return _download_via_requests(vid_u, save_path)
                    # اگر لیست مسیر/آبجکت برگرداند
                    if isinstance(out, list) and out:
                        # اولویت: فایل موجود
                        for item in out:
                            if isinstance(item, str) and os.path.exists(item):
                                return item
                        # یا URL
                        for item in out:
                            if isinstance(item, str) and item.startswith("http"):
                                fname = "tiktok_video.mp4"
                                save_path = save_dir / fname
                                return _download_via_requests(item, save_path)
        # حالت 2: ماژول فانکشنال (دانلود مستقیم URL)
        for name in ["download", "download_video", "save"]:
            if hasattr(ttd, name):
                out = getattr(ttd, name)(url, str(save_dir))
                if isinstance(out, str) and os.path.exists(out):
                    return out
    except Exception as e:
        logger.debug(f"TTD download failed: {e}")
    return None

def _download_via_tikwm(url: str, save_dir: pathlib.Path) -> Optional[str]:
    """
    فال‌بک به API tikwm.com برای لینک بدون واترمارک و دانلود با requests
    """
    try:
        api_url = "https://www.tikwm.com/api/"
        resp = requests.post(api_url, data={"url": url}, headers={"User-Agent": _common_headers()["User-Agent"]}, proxies=PROXY, timeout=30)
        data = resp.json()
        if data.get("data"):
            video_url = data["data"].get("play") or data["data"].get("wmplay")
            if video_url:
                fname = "tiktok_video.mp4"
                save_path = save_dir / fname
                return _download_via_requests(video_url, save_path)
    except Exception as e:
        logger.debug(f"TikWM fallback failed: {e}")
    return None

def _download_via_page_sniff(url: str, save_dir: pathlib.Path) -> Optional[str]:
    """
    آخرین راه: از HTML هر mp4 معتبری را پیدا کن و دانلود کن.
    """
    try:
        r = requests.get(url, headers=_common_headers(), proxies=None, timeout=45)
        r.raise_for_status()
        mp4s = re.findall(r'https?://[^\s"\']+\.mp4[^\s"\']*', r.text)
        for m in mp4s:
            if any(h in m for h in ("tiktokcdn", "v16.tiktokcdn.com", "tikcdn", "bytecdn")) or _is_direct_video(m):
                save_path = save_dir / ( _safe_filename("tiktok_video") + ".mp4" )
                return _download_via_requests(m, save_path)
    except Exception as e:
        logger.debug(f"Page sniff fallback failed: {e}")
    return None

def fetch_tiktok_media_and_info(tiktok_url: str) -> Tuple[str, Dict]:
    """
    تابع همگام برای فچ متادیتا و لینک/فایل دانلود.
    """
    text = ""
    uploader = ""
    uploader_id = ""
    uploader_image = None
    upload_date = None
    media_urls: List[str] = []
    raw_info = {}
    content_type = "text-only"

    final_url = _resolve_final_url(tiktok_url)

    # Extract metadata (HTML)
    meta = _extract_metadata_html(final_url)
    text = meta.get("text") or ""
    uploader = meta.get("uploader") or ""
    uploader_image = meta.get("uploader_image")
    upload_date = meta.get("upload_date")

    # Try TikTok downloader library first (preferred)
    file_path = _download_with_ttd(final_url, DOWNLOAD_DIR)

    # Fallback 1: tikwm API
    if not file_path:
        file_path = _download_via_tikwm(final_url, DOWNLOAD_DIR)

    # Fallback 2: sniff .mp4 in page
    if not file_path:
        file_path = _download_via_page_sniff(final_url, DOWNLOAD_DIR)

    # If we ended up with a local file, set media_urls to that local file path marker
    # Handler below will read the file directly (not by URL)
    if file_path and os.path.exists(file_path):
        media_urls = [f"file://{file_path}"]
        content_type = "video"
    else:
        # اگر هیچ فایلی نبود، حداقل لینک صفحه را برگردانیم (برای لاگ/تشخیص)
        media_urls = [final_url] if final_url else []
        content_type = "text-only" if not media_urls else "video"  # سعی می‌کنیم ویدیو باشد اگر لینک داریم

    result_info = {
        "text": (text or "")[:1000],
        "uploader": uploader,
        "uploader_id": uploader_id,
        "uploader_image": uploader_image,
        "upload_date": upload_date,
        "media_urls": media_urls,
        "raw_info": raw_info,
    }
    return content_type, result_info

def _sendable_caption(info: Dict) -> str:
    parts = []
    if info.get("text"):
        parts.append(f"Description: {info['text']}")
    if info.get("uploader"):
        parts.append(f"👤 Uploader: {info['uploader']}")
    if info.get("upload_date"):
        parts.append(f"📅 Date: {info['upload_date']}")
    parts.append("\n\nDownload by [Faryseneaidownloderbot](https://t.me/Faryseneaidownloderbot)")
    return "\n".join(parts)

def _materialize_media_to_path(media_url: str) -> str:
    """
    اگر media_url با file:// شروع شود یعنی از قبل دانلود شده و همان مسیر را برمی‌گردانیم،
    در غیر این صورت سعی می‌کنیم آن را دانلود کنیم (برای عکس/گیف احتمالی).
    """
    if media_url.startswith("file://"):
        return media_url.replace("file://", "", 1)

    fname = _safe_filename(os.path.basename(urlparse(media_url).path)) or "media"
    if not os.path.splitext(fname)[1]:
        # حدس پسوند
        if _is_direct_video(media_url):
            fname += ".mp4"
        else:
            fname += ".jpg"
    save_path = DOWNLOAD_DIR / fname
    return _download_via_requests(media_url, save_path, headers=_common_headers())

# ---------- Handlers ----------

async def handle_multiple_tiktok_links(message: types.Message, urls: list[str]):
    loading_message = await message.answer(f"We are processing your {len(urls)} TikTok links...")

    try:
        loop = asyncio.get_running_loop()

        # Fetch all concurrently
        tasks = [loop.run_in_executor(executor, fetch_tiktok_media_and_info, url) for url in urls]
        results = await asyncio.gather(*tasks)

        # Build media group
        media_group = []
        descriptions = []
        links_line = " ".join([f"<a href='{url}'>video {i+1}</a>" for i, url in enumerate(urls)])

        for i, (content_type, info) in enumerate(results):
            media_urls = info.get("media_urls", [])
            if media_urls and content_type == "video":
                file_path = await loop.run_in_executor(executor, _materialize_media_to_path, media_urls[0])
                try:
                    media_group.append(types.InputMediaVideo(media=types.FSInputFile(file_path), caption="" if i > 0 else None))
                finally:
                    try:
                        if file_path.startswith(str(DOWNLOAD_DIR)) and os.path.exists(file_path):
                            os.remove(file_path)
                    except Exception:
                        pass

            # Collect descriptions
            desc = f"Description: {info.get('text', '')}" if info.get("text") else ""
            descriptions.append(desc)

        # Set caption on first media
        if media_group:
            descriptions_text = "\n".join([f"<blockquote>{desc}</blockquote>" for desc in descriptions])
            media_group[0].caption = f"{links_line}\n\nDescriptions:\n{descriptions_text}\n\nDownload by [Faryseneaidownloderbot](https://t.me/Faryseneaidownloderbot)"
            media_group[0].parse_mode = "HTML"

        # Send media group
        if media_group:
            await message.bot.send_media_group(chat_id=message.chat.id, media=media_group)
        await message.bot.delete_message(chat_id=message.chat.id, message_id=loading_message.message_id)

        # Delete user message after 10 seconds
        await asyncio.sleep(10)
        try:
            await message.delete()
        except Exception:
            pass

    except Exception as e:
        logger.exception(f"Multiple TikTok Error: {e}")
        await message.bot.edit_message_text(
            chat_id=message.chat.id,
            message_id=loading_message.message_id,
            text="Sorry, an error occurred while processing your TikTok links. Please try again later."
        )

@router.message()
async def tiktok_download_handler(message: types.Message):
    text = message.text.strip()
    urls = re.findall(r'https?://[^\s]+', text)
    tiktok_urls = [u for u in urls if 'tiktok.com' in u or 'vm.tiktok.com' in u]

    # Handle multiple links (2-5)
    if 2 <= len(tiktok_urls) <= 5:
        await handle_multiple_tiktok_links(message, tiktok_urls)
        return
    elif len(tiktok_urls) != 1:
        await message.answer("⚠️ Please send exactly 1 or 2–5 TikTok links.")
        return

    url = tiktok_urls[0]
    loading = await message.answer("⏳ Downloading from TikTok...", parse_mode=None)
    try:
        loop = asyncio.get_running_loop()
        content_type, info = await loop.run_in_executor(executor, fetch_tiktok_media_and_info, url)
        media_urls: List[str] = info.get("media_urls", [])
        caption_text = f"TikTok: [link]({url})\n" + _sendable_caption(info)

        sent = None
        if media_urls:
            # تمرکز روی ویدیو؛ ابتدا فایل‌های ویدیویی
            video_like = [m for m in media_urls if _is_direct_video(m) or m.startswith("file://")]
            others = [m for m in media_urls if m not in video_like]

            # ارسال ویدیوها
            for idx, media_url in enumerate(video_like):
                file_path = await loop.run_in_executor(executor, _materialize_media_to_path, media_url)
                try:
                    sent = await message.answer_video(
                        types.FSInputFile(file_path),
                        caption=caption_text if idx == 0 else "",
                        parse_mode="Markdown"
                    )
                    # Record download history
                    if sent:
                        try:
                            from bot import record_download
                            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else None
                            url = getattr(message, 'text', '') or media_url
                            await record_download(
                                message.from_user.id, "tiktok", url,
                                file_type="video",
                                file_size=file_size
                            )
                        except Exception as hist_e:
                            logger.debug(f"Failed to record download history: {hist_e}")
                finally:
                    try:
                        if file_path.startswith(str(DOWNLOAD_DIR)) and os.path.exists(file_path):
                            os.remove(file_path)
                    except Exception:
                        pass

            # بقیه مدیا (عکس/گیف) در صورت وجود
            for media_url in others:
                file_path = await loop.run_in_executor(executor, _materialize_media_to_path, media_url)
                ext = os.path.splitext(file_path)[1].lower()
                try:
                    if ext == ".gif":
                        sent = await message.answer_animation(types.FSInputFile(file_path), parse_mode="Markdown")
                    else:
                        sent = await message.answer_photo(types.FSInputFile(file_path), parse_mode="Markdown")
                    # Record download history
                    if sent:
                        try:
                            from bot import record_download
                            file_size = os.path.getsize(file_path) if os.path.exists(file_path) else None
                            url = getattr(message, 'text', '') or media_url
                            await record_download(
                                message.from_user.id, "tiktok", url,
                                file_type="gif" if ext == ".gif" else "photo",
                                file_size=file_size
                            )
                        except Exception as hist_e:
                            logger.debug(f"Failed to record download history: {hist_e}")
                finally:
                    try:
                        if file_path.startswith(str(DOWNLOAD_DIR)) and os.path.exists(file_path):
                            os.remove(file_path)
                    except Exception:
                        pass
        else:
            # بدون مدیا، متن را بفرست
            if info.get("text"):
                await message.answer(info["text"], parse_mode="Markdown")
            else:
                await message.answer("⚠️ No content found.", parse_mode="Markdown")

        # Download completed, no additional message

    except Exception as e:
        logger.exception(f"TikTok download handler error: {e}")
        await message.answer(f"❌ Error downloading TikTok: {e}", parse_mode=None)
    finally:
        try:
            await loading.delete()
        except Exception:
            pass

    # Delete user message after 10 seconds
    await asyncio.sleep(10)
    try:
        await message.delete()
    except Exception:
        pass

# ---------- Optional: TikTok profile (basic HTML scrape) ----------

def fetch_tiktok_profile(url: str) -> Dict:
    headers = _common_headers()
    profile = {"username": None, "displayname": None, "avatar": None, "followers": None, "likes": None, "bio": None, "link": url}
    try:
        r = requests.get(url, headers=headers, proxies=None, timeout=40)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        meta_title = soup.find("meta", {"property": "og:title"}) or soup.find("meta", {"name": "title"})
        if meta_title and meta_title.get("content"):
            profile["displayname"] = meta_title.get("content")
        meta_desc = soup.find("meta", {"property": "og:description"}) or soup.find("meta", {"name": "description"})
        if meta_desc and meta_desc.get("content"):
            profile["bio"] = meta_desc.get("content")
        meta_img = soup.find("meta", {"property": "og:image"})
        if meta_img and meta_img.get("content"):
            profile["avatar"] = meta_img.get("content")
    except Exception as e:
        logger.error(f"Failed to fetch profile for {url}: {e}")
    return profile

@router.message()
async def tiktok_profile_handler(message: types.Message):
    raw = (message.text or "").strip()
    parts = raw.split()
    url = parts[1].strip() if parts and parts[0].startswith("/tiktok_profile") and len(parts) > 1 else raw
    if not url or "tiktok.com" not in url:
        await message.answer("❌ Please send a valid TikTok profile link.", parse_mode=None)
        return

    loading = await message.answer("⏳ Fetching TikTok profile info...", parse_mode=None)
    try:
        loop = asyncio.get_running_loop()
        profile_info = await loop.run_in_executor(executor, fetch_tiktok_profile, url)
        caption_lines = [
            f"<b>{profile_info.get('displayname') or 'TikTok User'}</b>",
            f"🔗 {profile_info.get('link')}"
        ]
        if profile_info.get("bio"):
            caption_lines.append(f"📝 {profile_info.get('bio')}")
        if profile_info.get("followers") is not None:
            caption_lines.append(f"👥 Followers: {profile_info.get('followers')}")
        if profile_info.get("likes") is not None:
            caption_lines.append(f"❤️ Likes: {profile_info.get('likes')}")
        caption = "\n".join(caption_lines)

        if profile_info.get("avatar"):
            await message.answer_photo(profile_info.get("avatar"), caption=caption, parse_mode="HTML")
        else:
            await message.answer(caption, parse_mode="HTML")
    except Exception as e:
        logger.exception(f"Error fetching TikTok profile: {e}")
        await message.answer(f"❌ Error fetching TikTok profile: {e}", parse_mode=None)
    finally:
        try:
            await loading.delete()
        except Exception:
            pass
