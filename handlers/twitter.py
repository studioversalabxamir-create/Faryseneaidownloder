import os
import json
import asyncio
import logging
import tempfile
import requests
import time
from typing import List, Dict, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor
from aiogram import Router, types
from fake_useragent import UserAgent
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
try:
    from config import PROXY
except ImportError:
    PROXY = "http://174.136.204.40:80"
import re
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from datetime import datetime
import http.cookies

# ---------- Logging ----------
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------- Router & Executor ----------
router = Router()
executor = ThreadPoolExecutor(max_workers=3)

# ---------- Proxies / UA ----------
PROXIES = {"http": PROXY, "https": PROXY} if PROXY else None
ua = UserAgent()

# ---------- Helpers ----------

def _download_to_tempfile(media_url: str, preferred_ext: Optional[str] = None) -> str:
    """
    Download media_url via requests (using PROXIES) to a temporary file and return path.
    """
    headers = {"User-Agent": ua.random}
    r = requests.get(media_url, headers=headers, proxies=PROXIES, stream=True, timeout=60)
    r.raise_for_status()
    # determine extension
    if preferred_ext:
        ext = preferred_ext
    else:
        # try from content-type
        ctype = r.headers.get("content-type", "")
        if "jpeg" in ctype:
            ext = ".jpg"
        elif "png" in ctype:
            ext = ".png"
        elif "gif" in ctype:
            ext = ".gif"
        elif "mpeg" in ctype or "mp4" in ctype:
            ext = ".mp4"
        else:
            # fallback to .bin
            ext = os.path.splitext(media_url.split("?")[0])[1] or ".bin"

    tf = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    try:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                tf.write(chunk)
        tf.flush()
    finally:
        tf.close()
    return tf.name

def _create_pdf_from_text(text: str) -> str:
    """
    Create a simple PDF from given text using reportlab. Return temp pdf path.
    """
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tf.close()
    c = canvas.Canvas(tf.name, pagesize=letter)
    width, height = letter
    margin = 40
    y = height - margin
    line_height = 12
    # simple wrapping
    for paragraph in text.split("\n"):
        words = paragraph.split(" ")
        line = ""
        for w in words:
            test_line = (line + " " + w).strip()
            if c.stringWidth(test_line, "Helvetica", 10) > (width - 2*margin):
                c.setFont("Helvetica", 10)
                c.drawString(margin, y, line)
                y -= line_height
                line = w
                if y < margin:
                    c.showPage()
                    y = height - margin
            else:
                line = test_line
        if line:
            c.setFont("Helvetica", 10)
            c.drawString(margin, y, line)
            y -= line_height
            if y < margin:
                c.showPage()
                y = height - margin
    c.save()
    return tf.name


def _load_cookies_from_file(cookie_file: str) -> str:
    """Load cookies from Netscape format file and convert to requests format (cookie string)."""
    cookies = {}
    if not os.path.exists(cookie_file):
        return ""

    try:
        with open(cookie_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) >= 7:
                    try:
                        name = parts[5]
                        value = parts[6]
                        cookies[name] = value
                    except (ValueError, IndexError):
                        continue
    except Exception as e:
        logger.warning(f"Failed to load cookies from {cookie_file}: {e}")
    cookie_str = "; ".join([f"{k}={v}" for k, v in cookies.items()])
    return cookie_str

def _fetch_tweet_data(tweet_url: str) -> Tuple[str, Dict]:
    """
    Fetch tweet data using Playwright for JavaScript-rendered content under 15 seconds.
    """
    info = {
        "text": "",
        "uploader": "",
        "uploader_id": "",
        "uploader_image": "",
        "upload_date": "",
        "media_urls": [],
        "stats": {"likes": 0, "retweets": 0, "replies": 0, "views": 0, "bookmarks": 0},
        "profile": {"bio": ""},
        "thread_tweets": [],
        "best_reply": {"text": "", "user": ""},
        "poll": {"options": []},
        "raw_info": {}
    }
    content_type = "text-only"

    # مسیر اصلاح‌شده به فایل کوکی در ریشه پروژه
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cookie_file = os.path.join(BASE_DIR, 'xcookies.txt')
    logger.debug(f"Checking cookie file at: {cookie_file}")
    if not os.path.exists(cookie_file):
        logger.error(f"Cookie file not found at: {cookie_file}")
        return content_type, info

    cookie_str = _load_cookies_from_file(cookie_file)

    try:
        start_time = time.time()
        with sync_playwright() as p:
            # Configure browser with proxy and cookies
            browser = p.chromium.launch(proxy={"server": PROXIES["https"]}, headless=True)
            context = browser.new_context(
                user_agent=ua.random,
                extra_http_headers={"Accept-Language": "en-US,en;q=0.9", "Referer": "https://x.com/"},
                viewport={"width": 1280, "height": 720},
                ignore_https_errors=True
            )
            # Set cookies - Fixed conversion from string to list of dicts
            playwright_cookies = []
            if cookie_str:
                cookie_pairs = re.split(r'\s*;\s*', cookie_str.strip())
                for pair in cookie_pairs:
                    if '=' in pair:
                        name, value = pair.split('=', 1)
                        name = name.strip()
                        value = value.strip()
                        if name and value:
                            playwright_cookies.append({
                                "name": name,
                                "value": value,
                                "domain": ".x.com",
                                "path": "/"
                            })
            logger.debug(f"Loaded {len(playwright_cookies)} cookies for Playwright")
            context.add_cookies(playwright_cookies)

            page = context.new_page()
            page.goto(tweet_url, wait_until="networkidle", timeout=15000)  # 15-second timeout

            # Wait for content to load with modern selectors
            page.wait_for_selector('[data-testid="tweet"]', state="visible", timeout=10000)

            # Scroll to load more content
            page.evaluate('window.scrollTo(0, document.body.scrollHeight)')
            page.wait_for_timeout(2000)

            # Parse page HTML with BeautifulSoup
            html = page.content()
            soup = BeautifulSoup(html, 'html.parser')

            # Log page info for debugging
            page_title = page.title()
            page_url = page.url
            logger.debug(f"Page loaded: title='{page_title}', url='{page_url}'")
            if page_url != tweet_url:
                logger.warning(f"URL mismatch: expected '{tweet_url}', got '{page_url}' - possible redirect")

            # Check if logged in
            login_check = page.evaluate('''() => {
                const loginElements = document.querySelectorAll('a[href*="login"], button[data-testid*="login"], [data-testid*="signin"]');
                return loginElements.length > 0;
            }''')
            logger.debug(f"Login elements detected: {login_check}")
            if login_check:
                logger.warning("Login required - cookies may be invalid or expired")

            # Extract text using JavaScript with modern selectors
            text_result = page.evaluate('''() => {
                const selectors = [
                    '[data-testid="tweetText"]',
                    '[data-testid="cellInnerDiv"] div[dir="ltr"]',
                    'article div[aria-label*="Tweet"]',
                    'div[data-testid="Tweet-User-Text"]',
                    '[role="article"] div[dir="auto"]'
                ];
                for (const selector of selectors) {
                    const el = document.querySelector(selector);
                    if (el && el.textContent.trim()) {
                        return { value: el.textContent.trim(), selector: selector, found: true };
                    }
                }
                return { value: '', selector: '', found: false };
            }''')
            info["text"] = text_result['value']
            logger.debug(f"Text extraction: found={text_result['found']}, selector='{text_result['selector']}', value='{info['text'][:100]}...'")

            # Extract uploader using JavaScript with modern selectors
            uploader_data = page.evaluate('''() => {
                const selectors = [
                    '[data-testid="User-Name"]',
                    '[data-testid="Tweet-User-Name"]',
                    'article a[href*="/"] span[dir="ltr"]',
                    '[role="link"][href*="/"] span'
                ];
                for (const selector of selectors) {
                    const el = document.querySelector(selector);
                    if (el && el.textContent.trim()) {
                        const link = el.closest('a');
                        const href = link ? link.href : '';
                        const id = href ? href.split('/').pop() : '';
                        return { name: el.textContent.trim(), id: id, selector: selector, found: true };
                    }
                }
                return { name: '', id: '', selector: '', found: false };
            }''')
            info["uploader"] = uploader_data['name']
            info["uploader_id"] = uploader_data['id']
            logger.debug(f"Uploader extraction: found={uploader_data['found']}, selector='{uploader_data['selector']}', name='{info['uploader']}', id='{info['uploader_id']}'")

            # Extract avatar using JavaScript with modern selectors
            info["uploader_image"] = page.evaluate('''() => {
                const selectors = [
                    'img[src*="profile_images"]',
                    '[data-testid="UserAvatar-Container"] img',
                    'article img[alt*="profile"]'
                ];
                for (const selector of selectors) {
                    const el = document.querySelector(selector);
                    if (el && el.src) {
                        return el.src;
                    }
                }
                return '';
            }''')
            logger.debug(f"Extracted avatar: '{info['uploader_image']}'")

            # Extract bio using JavaScript with modern selectors
            info["profile"]["bio"] = page.evaluate('''() => {
                const selectors = [
                    '[data-testid="UserDescription"]',
                    '[data-testid="UserProfileBio"]',
                    'div[aria-label="Bio"]'
                ];
                for (const selector of selectors) {
                    const el = document.querySelector(selector);
                    if (el && el.textContent.trim()) {
                        return el.textContent.trim();
                    }
                }
                return '';
            }''')
            logger.debug(f"Extracted bio: '{info['profile']['bio']}'")

            # Extract date using JavaScript with modern selectors
            info["upload_date"] = page.evaluate('''() => {
                const selectors = [
                    'time[data-testid="tweetTimestamp"]',
                    '[data-testid="Tweet-User-Timestamp"]',
                    'article time[datetime]'
                ];
                for (const selector of selectors) {
                    const el = document.querySelector(selector);
                    if (el && el.getAttribute('datetime')) {
                        return el.getAttribute('datetime');
                    }
                }
                return '';
            }''')
            logger.debug(f"Extracted date: '{info['upload_date']}'")

            # Extract media URLs using JavaScript with modern selectors
            media_urls = page.evaluate('''() => {
                const urls = [];
                // Images
                const imgs = document.querySelectorAll('img[src*="pbs.twimg.com/media/"], img[data-src*="pbs.twimg.com/media/"]');
                imgs.forEach(img => {
                    const src = img.src || img.getAttribute('data-src') || '';
                    if (src && src.includes('pbs.twimg.com/media/')) {
                        const highQuality = src.replace(/name=\w+/, 'name=orig');
                        urls.push(highQuality);
                    }
                });
                // Videos
                const videos = document.querySelectorAll('video[src], [data-testid="videoComponent"] video');
                videos.forEach(video => {
                    if (video.src) {
                        let src = video.src.replace('amplify_video_thumb', 'amplify_video');
                        if (!src.includes('&tag=14')) {
                            src += '&tag=14';
                        }
                        urls.push(src);
                    }
                    const sources = video.querySelectorAll('source[src]');
                    sources.forEach(source => {
                        if (source.src) {
                            let src = source.src.replace('amplify_video_thumb', 'amplify_video');
                            if (!src.includes('&tag=14')) {
                                src += '&tag=14';
                            }
                            urls.push(src);
                        }
                    });
                });
                return [...new Set(urls)];
            }''')
            info["media_urls"] = media_urls
            logger.debug(f"Extracted media URLs: {info['media_urls']}")

            # Fallback: Extract from JSON in scripts if fields are empty
            if not info["text"] or not info["uploader"]:
                json_data = page.evaluate('''() => {
                    const scripts = document.querySelectorAll('script[type="application/json"], script:not([src])');
                    for (const script of scripts) {
                        if (script.textContent && (script.textContent.includes('__INITIAL_STATE__') || script.textContent.includes('data-items') || script.textContent.includes('window.__INITIAL_DATA__'))) {
                            try {
                                const jsonStr = script.textContent.split('=')[1]?.trim().replace(/;$/, '') || script.textContent.trim();
                                return JSON.parse(jsonStr);
                            } catch (e) {
                                continue;
                            }
                        }
                    }
                    return null;
                }''')
                if json_data:
                    # Attempt to parse JSON data
                    if 'entry' in json_data and 'content' in json_data['entry']:
                        info["text"] = json_data['entry']['content']['itemContent']['item']['content'] or ''
                    if 'user' in json_data and 'legacy' in json_data['user']:
                        info["uploader"] = json_data['user']['legacy']['name'] or ''
                        info["uploader_id"] = json_data['user']['screen_name'] or ''
                    logger.debug(f"Parsed JSON data: text='{info['text'][:200]}...', uploader='{info['uploader']}'")

            # Extract stats with modern selectors
            stats = page.evaluate('''() => {
                const stats = {};
                const statElements = document.querySelectorAll('[data-testid*="metric"]');
                statElements.forEach(el => {
                    const testId = el.getAttribute('data-testid');
                    const text = el.textContent.trim().replace(',', '');
                    if (testId.includes('like')) stats.likes = parseInt(text) || 0;
                    if (testId.includes('retweet')) stats.retweets = parseInt(text) || 0;
                    if (testId.includes('reply')) stats.replies = parseInt(text) || 0;
                    if (testId.includes('view')) stats.views = parseInt(text) || 0;
                    if (testId.includes('bookmark')) stats.bookmarks = parseInt(text) || 0;
                });
                return stats;
            }''')
            info["stats"].update(stats or {})

            # Extract poll with modern selectors
            poll_data = page.evaluate('''() => {
                const poll = document.querySelector('[data-testid="cellInnerDiv"] [data-testid="card.poll"]');
                if (!poll) return null;
                const options = [];
                poll.querySelectorAll('[role="progressbar"]').forEach(opt => {
                    const label = opt.textContent.trim().split('%')[0].trim();
                    const votesMatch = opt.textContent.match(/(\d+)%/);
                    const votes = votesMatch ? parseInt(votesMatch[1]) : 0;
                    options.push({ label, votes });
                });
                return { options };
            }''')
            if poll_data and poll_data.get('options'):
                info["poll"] = poll_data

            # Extract thread tweets with modern selectors
            thread_tweets = page.evaluate('''() => {
                const tweets = [];
                const articles = document.querySelectorAll('[data-testid="tweet"]');
                articles.forEach((article, i) => {
                    const text = Array.from(article.querySelectorAll('[data-testid="tweetText"], [dir="ltr"]'))
                        .find(el => el.textContent.trim()).textContent.trim();
                    const date = article.querySelector('time[datetime]')?.getAttribute('datetime') || '';
                    const user = Array.from(article.querySelectorAll('[data-testid="User-Name"], [role="link"] span[dir="ltr"]'))
                        .find(el => el.textContent.trim()).textContent.trim();
                    const media = Array.from(article.querySelectorAll('img[src*="pbs.twimg.com/media/"]'))
                        .map(img => img.src.replace(/name=\w+/, 'name=orig'));
                    if (text) tweets.push({ id: `thread_${i}`, text, date, user, media_urls: media });
                });
                return tweets;
            }''')
            info["thread_tweets"] = thread_tweets or []

            # Extract best reply with modern selectors
            best_reply = page.evaluate('''() => {
                const replies = document.querySelectorAll('[data-testid="cellInnerDiv"] [data-testid="tweet"]');
                let best = { text: "", user: "", likes: 0 };
                replies.forEach(reply => {
                    const text = reply.querySelector('[data-testid="tweetText"]')?.textContent.trim() || '';
                    const user = reply.querySelector('[data-testid="User-Name"]')?.textContent.trim() || '';
                    const likes = parseInt(reply.querySelector('[data-testid="like"]')?.textContent.replace(',', '') || '0');
                    if (likes > best.likes) best = { text, user, likes };
                });
                return best.text ? best : null;
            }''')
            if best_reply and best_reply["text"]:
                info["best_reply"] = {"text": best_reply["text"], "user": best_reply["user"]}

            # Determine content type
            content_type = "text-only"
            if info["media_urls"]:
                if any('.mp4' in url for url in info["media_urls"]):
                    content_type = "video"
                else:
                    content_type = "images"

            logger.debug(f"Extracted tweet data for {tweet_url} in {time.time() - start_time:.2f} seconds")
            logger.debug(f"Final extracted info: text='{info['text'][:200]}...', uploader='{info['uploader']}', media_count={len(info['media_urls'])}")

            # Fallback for text if empty
            if not info["text"]:
                fallback_text = page.evaluate('''() => {
                    const metaDesc = document.querySelector('meta[name="description"]');
                    if (metaDesc && metaDesc.content) {
                        return metaDesc.content;
                    }
                    return document.title;
                }''')
                if fallback_text and len(fallback_text) > 10:
                    info["text"] = fallback_text
                    logger.debug(f"Used fallback text: '{info['text'][:200]}...'")

            # Clean up
            context.close()
            browser.close()

    except Exception as e:
        logger.error(f"Failed to fetch tweet data for {tweet_url}: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")

    return content_type, info


# ---------- Core fetch function ----------
def fetch_tweet_media_and_info(tweet_url: str) -> Tuple[str, Dict]:
    return _fetch_tweet_data(tweet_url)

# ---------- Handlers ----------

@router.message()
async def twitter_download_handler(message: types.Message):
    """
    Expect message.text to contain a Twitter/X tweet URL.
    Downloads and sends all content (text, media, thread, poll, best reply) with a formatted caption.
    """

    url = (message.text or "").strip()
    if not url or ("twitter.com" not in url and "x.com" not in url):
        await message.answer("❌ Please send a valid X (Twitter) link.", parse_mode="HTML")
        return

    loading = await message.answer("Loading and retrieving content...")

    def _format_caption(info: Dict, url: str) -> str:
        parts = []
        uploader = info.get("uploader", "Unknown")
        uploader_id = info.get("uploader_id", uploader)
        parts.append(f"👤 <a href='https://x.com/{uploader_id}'>@{uploader}</a>")
        if info.get("upload_date"):
            parts.append(f"📅 {info['upload_date']}")
        stats = info.get("stats", {})
        parts.append(
            f"❤️ {stats.get('likes', 0)} | 🔄 {stats.get('retweets', 0)} | "
            f"💬 {stats.get('replies', 0)} | 👁️ {stats.get('views', 0)} | 🔖 {stats.get('bookmarks', 0)}"
        )
        profile = info.get("profile", {})
        if profile.get("bio"):
            parts.append(f"📚 Bio: {textwrap_short(profile['bio'], 100)}")
        if info.get("text"):
            parts.append(f"\n{textwrap_short(info['text'], 400)}")
        if info.get("thread_tweets"):
            parts.append(f"🧵 Thread: {len(info['thread_tweets'])} tweets")
        if info.get("best_reply"):
            parts.append(f"💡 Top reply: {textwrap_short(info['best_reply']['text'], 100)} (@{info['best_reply']['user']})")
        if info.get("poll"):
            parts.append("📊 Poll:")
            for opt in info["poll"].get("options", []):
                parts.append(f"  - {opt['label']}: {opt.get('votes', 0)} votes")
        parts.append(f"<a href='{url}'>X(Twitter)</a>")
        parts.append(f"\n\nDownload by <a href='https://t.me/Faryseneaidownloderbot'>Faryseneaidownloderbot</a>")
        return "\n".join(parts)

    try:
        start_time = time.time()
        loop = asyncio.get_running_loop()
        # Retry logic for rate limits
        max_retries = 3
        for attempt in range(max_retries):
            try:
                content_type, info = await loop.run_in_executor(executor, fetch_tweet_media_and_info, url)
                if time.time() - start_time > 15:
                    raise TimeoutError("Extraction took longer than 15 seconds")
                break
            except Exception as e:
                error_str = str(e).lower()
                if ("rate limit" in error_str or "429" in error_str or "too many requests" in error_str) and attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # exponential backoff
                    logger.warning(f"Rate limit or login error detected, retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise e
        caption = _format_caption(info, url)

        sent_messages = []
        if info.get("media_urls"):
            media_group = []
            for i, media_url in enumerate(info["media_urls"]):
                file_path = await loop.run_in_executor(executor, _download_to_tempfile, media_url, ".mp4" if ".mp4" in media_url.lower() else None)
                try:
                    ext = os.path.splitext(file_path)[1].lower()
                    media_caption = caption if i == 0 else ""
                    if content_type == "video" or ext == ".mp4":
                        media = types.InputMediaVideo(media=types.FSInputFile(file_path), caption=media_caption, parse_mode="HTML")
                    elif ext == ".gif":
                        media = types.InputMediaAnimation(media=types.FSInputFile(file_path), caption=media_caption, parse_mode="HTML")
                    else:
                        media = types.InputMediaPhoto(media=types.FSInputFile(file_path), caption=media_caption, parse_mode="HTML")
                    media_group.append(media)
                finally:
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass
            if media_group:
                sent_messages = await message.answer_media_group(media_group)
        else:
            # فقط متن
            sent_messages = [await message.answer(caption, parse_mode="HTML")]

        # ارسال thread به صورت PDF اگر بیش از یک توییت باشد
        if info.get("thread_tweets") and len(info["thread_tweets"]) > 1:
            thread_text = "\n\n".join(
                f"Tweet {i+1} (@{t['user']}, {t['date']}):\n{t['text']}\n"
                f"Media: {', '.join(t['media_urls']) if t['media_urls'] else 'None'}" for i, t in enumerate(info["thread_tweets"])
            )
            full_text = f"Main tweet:\n{info['text']}\n\nThread:\n{thread_text}"
            pdf_path = await loop.run_in_executor(executor, _create_pdf_from_text, full_text)
            try:
                sent_messages.append(
                    await message.answer_document(
                        types.FSInputFile(pdf_path), caption="📜 Tweet thread PDF", parse_mode="HTML"
                    )
                )
            finally:
                os.remove(pdf_path)


        # Schedule user message deletion after 15 seconds
        async def delete_user_message():
            await asyncio.sleep(15)
            try:
                await message.delete()
            except Exception:
                pass
        asyncio.create_task(delete_user_message())

    except Exception as e:
        logger.exception(f"Twitter download error: {e}")
        await message.answer("❌ Error: Unable to load tweet content.", parse_mode="HTML")
    finally:
        try:
            await loading.delete()
        except Exception:
            pass

def textwrap_short(text: str, limit: int = 900) -> str:
    """Shorten text for caption safely."""
    if not text:
        return ""
    t = text.strip()
    if len(t) > limit:
        return t[:limit-3] + "..."
    return t

@router.message()
async def twitter_tweet_to_pdf_handler(message: types.Message):
    """
    Turn tweet text into PDF and send.
    Expect message.text contains tweet URL.
    """
    url = (message.text or "").strip()
    if not url or ("twitter.com" not in url and "x.com" not in url):
        await message.answer("❌ Please send a valid X (Twitter) link.")
        return

    loading = await message.answer("⏳ Generating PDF from tweet text...")
    try:
        loop = asyncio.get_running_loop()
        _, info = await loop.run_in_executor(executor, fetch_tweet_media_and_info, url)
        txt = info.get("text") or "(No tweet text available)"
        # create PDF
        pdf_path = await loop.run_in_executor(executor, _create_pdf_from_text, txt)
        try:
            await message.answer_document(types.FSInputFile(pdf_path), caption="📄 Tweet text PDF")
        finally:
            try:
                os.remove(pdf_path)
            except Exception:
                pass
        await message.answer("✅ PDF created and sent.")
    except Exception as e:
        logger.exception(f"Error creating PDF: {e}")
        await message.answer(f"❌ Error creating PDF: {e}")
    finally:
        try:
            await loading.delete()
        except Exception:
            pass

@router.message()
async def twitter_profile_handler(message: types.Message):
    """
    Send profile info: avatar, bio, follower count, and a few latest tweets.
    Input: message.text should be profile link or @username.
    """
    raw = (message.text or "").strip()
    if not raw:
        await message.answer("❌ Please send a username or profile link.")
        return

    # normalize username
    if raw.startswith("http"):
        parsed = raw.rstrip("/").split("/")
        username = parsed[-1] if parsed[-1] else parsed[-2]
    else:
        username = raw.lstrip("@")

    profile_url = f"https://x.com/{username}"

    loading = await message.answer("⏳ Fetching profile info...")
    try:
        # Use the same fetching method for profile
        _, info = await asyncio.get_running_loop().run_in_executor(executor, _fetch_tweet_data, profile_url)
        caption = (
            f"<b>{info['uploader']} (@{info['uploader_id']})</b>\n"
            f"📚 Bio: {info['profile'].get('bio', '—')}\n"
            f"👥 Followers: {info['stats'].get('followers', 0)}\n"
            f"🔗 <a href='{profile_url}'>{profile_url}</a>\n\n"
            f"<b>Latest tweets:</b>\n" + ("\n".join([t['text'][:200] for t in info['thread_tweets'][:5]]) if info['thread_tweets'] else "None found.")
        )

        if info.get("uploader_image"):
            await message.answer_photo(info["uploader_image"], caption=caption, parse_mode="HTML")
        else:
            await message.answer(caption, parse_mode="HTML")

    except Exception as e:
        logger.exception(f"Error fetching profile: {e}")
        await message.answer(f"❌ Error fetching profile: {e}")
    finally:
        try:
            await loading.delete()
        except Exception:
            pass
