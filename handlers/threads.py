import os
import requests
from bs4 import BeautifulSoup
import json
import re
from typing import Tuple, Dict, List, Optional
from fake_useragent import UserAgent
import yt_dlp
import asyncio
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor
from aiogram import Router, types
from urllib.parse import urlparse
try:
    from config import PROXY
except ImportError:
    PROXY = "http://174.136.204.40:80"

# ---------- Logging ----------
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------- Router & Executor ----------
router = Router()
executor = ThreadPoolExecutor(max_workers=3)

# ---------- Proxies / UA ----------
PROXIES = {
    "http": "http://5.161.133.32:80	",
    "https": "http://5.161.133.32:80"
}
ua = UserAgent()


# ---------- Helpers ----------
def _is_direct_media_url(url: str) -> bool:
    path = urlparse(url).path or ""
    ext = os.path.splitext(path)[1].lower()
    return ext in (".jpg", ".jpeg", ".png", ".gif", ".webp", ".mp4", ".mov", ".webm", ".m4v", ".m4a")

def download_media_to_temp(media_url: str) -> str:
    headers = {"User-Agent": ua.random}
    try:
        if _is_direct_media_url(media_url):
            r = requests.get(media_url, headers=headers, proxies=PROXIES, stream=True, timeout=30)
            r.raise_for_status()
            path_ext = os.path.splitext(urlparse(media_url).path)[1] or ""
            if not path_ext:
                # try content-type
                ctype = r.headers.get("content-type", "")
                if "jpeg" in ctype:
                    path_ext = ".jpg"
                elif "png" in ctype:
                    path_ext = ".png"
                elif "gif" in ctype:
                    path_ext = ".gif"
                elif "mp4" in ctype:
                    path_ext = ".mp4"
                else:
                    path_ext = ".bin"
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=path_ext)
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    tf.write(chunk)
            tf.flush()
            tf.close()
            return tf.name
        else:
            # use yt-dlp to download (handles pages / streaming urls)
            ydl_opts = {
                "outtmpl": os.path.join(tempfile.gettempdir(), "%(id)s.%(ext)s"),
                "quiet": True,
                "no_warnings": True,
                "proxy": PROXY
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(media_url, download=True)
                file_path = ydl.prepare_filename(info)
            return file_path
    except Exception as e:
        logger.exception(f"download_media_to_temp error for {media_url}: {e}")
        raise

def fetch_threads_media_and_info(thread_url: str) -> Tuple[str, Dict]:
    text = ""
    uploader = ""
    uploader_id = ""
    uploader_image = None
    upload_date = None
    media_urls: List[str] = []
    content_type = "text-only"
    raw_info = {}

    # 1) تلاش با yt-dlp (ممکن است اطلاعات مدیا مستقیم فراهم کند)
    try:
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "proxy": PROXY
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(thread_url, download=False) or {}
            raw_info = info

        # متادیتا
        text = (info.get("title") or info.get("description") or "") or ""
        uploader = info.get("uploader") or ""
        uploader_id = info.get("uploader_id") or ""
        uploader_image = info.get("thumbnail") or None
        upload_date = info.get("upload_date") or None

        # ویدیوها (فرمت‌هایی که مربوط به ویدیو هستند)
        formats = info.get("formats") or []
        if isinstance(formats, list):
            video_urls = []
            for f in formats:
                url = f.get("url")
                if not url:
                    continue
                # قبول هر فرمت که شامل ویدیو باشد (vcodec != none) یا ext mp4
                if (f.get("vcodec") and f.get("vcodec") != "none") or (str(f.get("ext", "")).lower() == "mp4"):
                    video_urls.append(url)
            if video_urls:
                # dedupe preserve order
                seen = set()
                for v in video_urls:
                    if v and v not in seen:
                        seen.add(v)
                        media_urls.append(v)
                content_type = "video"

        # تصاویر از thumbnails
        thumbs = info.get("thumbnails") or []
        if isinstance(thumbs, list):
            for t in thumbs:
                url = t.get("url") if isinstance(t, dict) else None
                if url and url not in media_urls:
                    media_urls.append(url)
            if media_urls and content_type != "video":
                content_type = "images"

    except Exception:
        # ignore yt-dlp extraction errors, ادامه با HTML fallback
        pass

    # 2) HTML fallback — همیشه چک کنیم (برخلاف گذشته که فقط وقتی ویدیو نبود انجام می‌شد)
    try:
        headers = {"User-Agent": ua.random}
        r = requests.get(thread_url, headers=headers, proxies=PROXIES, timeout=20)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")

        # متن (اگر yt-dlp متن نیاورده)
        if not text:
            meta_desc = soup.find("meta", {"property": "og:description"}) or soup.find("meta", {"name": "description"})
            if meta_desc and meta_desc.get("content"):
                text = meta_desc.get("content")

        # uploader
        meta_title = soup.find("meta", {"property": "og:title"}) or soup.find("meta", {"name": "title"})
        if meta_title and meta_title.get("content"):
            uploader = uploader or meta_title.get("content")

        # uploader image
        meta_image = soup.find("meta", {"property": "og:image"})
        if meta_image and meta_image.get("content"):
            uploader_image = uploader_image or meta_image.get("content")

        # collect og:image (may be many)
        img_tags = soup.find_all("meta", property="og:image")
        for tag in img_tags:
            val = tag.get("content")
            if val and val not in media_urls:
                media_urls.append(val)

        # collect og:video (may be present)
        og_video_props = ["og:video", "og:video:url", "og:video:secure_url", "twitter:player:stream"]
        for prop in og_video_props:
            tag = soup.find("meta", {"property": prop}) or soup.find("meta", {"name": prop})
            if tag and tag.get("content"):
                v = tag.get("content")
                if v and v not in media_urls:
                    media_urls.append(v)

        # sniff mp4 urls inside page JS (useful for GIFs as mp4)
        mp4_matches = re.findall(r'https?://[^\s"\']+\.mp4[^\s"\']*', r.text)
        for m in mp4_matches:
            if ("threads" in m or "cdn" in m or "twimg" in m or "video" in m) and m not in media_urls:
                media_urls.append(m)

        # determine content_type after HTML fallback
        if any(u.lower().endswith((".mp4", ".mov", ".webm")) for u in media_urls):
            content_type = "video"
        elif any(u.lower().endswith((".jpg", ".jpeg", ".png", ".gif", ".webp")) for u in media_urls):
            if content_type != "video":
                content_type = "images"

    except Exception:
        pass

    # Deduplicate & normalize
    final_media = []
    seen = set()
    for u in media_urls:
        if not u:
            continue
        cu = u.strip()
        if cu.startswith("//"):
            cu = "https:" + cu
        if cu not in seen:
            seen.add(cu)
            final_media.append(cu)

    # Final text length limit
    text = (text or "")[:1000]

    result_info = {
        "text": text,
        "uploader": uploader,
        "uploader_id": uploader_id,
        "uploader_image": uploader_image,
        "upload_date": upload_date,
        "media_urls": final_media,
        "raw_info": raw_info
    }

    if not final_media and text:
        content_type = "text-only"

    return content_type, result_info

# ---------- Aiogram handler ----------
@router.message()
async def threads_download_handler(message: types.Message):
    if not message.text:
        return

    parts = message.text.split()
    if not parts:
        return

    # Accept both command forms: "/threads_download <url>" or direct url message
    if parts[0].startswith("/threads_download"):
        if len(parts) < 2:
            await message.reply("Usage: /threads_download <url>")
            return
        thread_url = parts[1].strip()
    else:
        thread_url = message.text.strip()

    if "threads.com" not in thread_url:
        await message.reply("❌ Please send a valid Threads link.")
        return

    loading = await message.answer("⏳ Downloading content from Threads…")
    try:
        loop = asyncio.get_running_loop()
        content_type, info = await loop.run_in_executor(executor, fetch_threads_media_and_info, thread_url)
        media_urls: List[str] = info.get("media_urls", [])
        caption_text = info.get("text", "")
        # send media files
        if media_urls:
            for media_url in media_urls:
                try:
                    file_path = await loop.run_in_executor(executor, download_media_to_temp, media_url)
                except Exception as e:
                    logger.warning(f"Failed to download media {media_url}: {e}")
                    continue

                try:
                    ext = os.path.splitext(file_path)[1].lower()
                    if ext in (".mp4", ".mov", ".webm", ".m4v"):
                        await message.answer_video(types.FSInputFile(file_path), caption=caption_text)
                    elif ext in (".gif",):
                        await message.answer_animation(types.FSInputFile(file_path), caption=caption_text)
                    else:
                        await message.answer_photo(types.FSInputFile(file_path), caption=caption_text)
                    # After first media, clear caption to avoid repeating long captions on each file
                    caption_text = ""
                finally:
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass
        else:
            # no media — send text only
            await message.reply(info.get("text", "No content found."))

        # optionally send uploader info
        uploader = info.get("uploader")
        if uploader:
            await message.reply(f"👤 {uploader}\n🔗 {thread_url}")

    except Exception as e:
        logger.exception(f"Error downloading Threads: {e}")
        await message.reply(f"❌ Error downloading: {e}")
    finally:
        try:
            await loading.delete()
        except Exception:
            pass
