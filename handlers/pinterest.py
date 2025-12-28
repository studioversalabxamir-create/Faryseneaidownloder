import os
import requests
import asyncio
import aiohttp
# from config import PROXY  # No longer needed - using rotating proxies
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from aiogram import Router, types
from fake_useragent import UserAgent
import logging
from urllib.parse import urlparse
import json
import re
import time  # برای زمان‌سنجی

# تنظیم لاگینگ
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# روتر و تردپول
router = Router()
executor = ThreadPoolExecutor(max_workers=5)  # افزایش به 5 برای موازی‌سازی بهتر

# Proxy configuration - import from centralized config
try:
    from config import PROXY
    WORKING_PROXIES = [PROXY] if PROXY else []
except ImportError:
    WORKING_PROXIES = []

# شاخص پروکسی فعلی برای چرخش
proxy_index = 0

def get_next_proxy():
    """دریافت پروکسی بعدی برای چرخش"""
    global proxy_index
    if not WORKING_PROXIES:
        return None
    proxy = WORKING_PROXIES[proxy_index]
    proxy_index = (proxy_index + 1) % len(WORKING_PROXIES)
    return proxy

ua = UserAgent()

def normalize_pin_url(url):
    """Normalize Pinterest CDN URLs to get original quality images"""
    return url.replace("/236x/", "/originals/") \
              .replace("/474x/", "/originals/") \
              .replace("/736x/", "/originals/")

async def pinterest_download(url, ext):
    """
    Pinterest-safe download function with fallback support
    Returns local file path
    """
    # Ensure temp directory exists
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Generate unique filename
    filename = f"pin_{int(time.time())}{ext}"
    local_path = os.path.join(temp_dir, filename)
    
    # Normalize URL for better quality
    url = normalize_pin_url(url)
    
    # Real browser headers
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": "https://www.pinterest.com/",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive"
    }
    
    # First attempt: no cookie
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if response.status == 200:
                with open(local_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        f.write(chunk)
                return local_path
            
            # Fallback: retry with cookies if 403
            if response.status == 403:
                headers_with_cookie = headers.copy()
                headers_with_cookie["Cookie"] = "session=valid; csrftoken=valid"
                
                async with session.get(url, headers=headers_with_cookie) as retry_response:
                    if retry_response.status == 200:
                        with open(local_path, 'wb') as f:
                            async for chunk in retry_response.content.iter_chunked(8192):
                                f.write(chunk)
                        return local_path
    
    raise Exception("Pinterest download failed")

# تابع استخراج توضیحات
def extract_description(html: str):
    if 'og:description" content="' in html:
        start = html.find('og:description" content="') + len('og:description" content="')
        end = html.find('"', start)
        return html[start:end]
    return None

from task_manager import task_manager

import asyncio
import re

# مطمئن شو اینها رو از ماژول‌های مربوطه وارد کردی:
# from utils import fetch_pinterest_content, fetch_pinterest_profile, handle_multiple_pinterest_links
# from config import executor, errors_total, logger

@router.message(lambda m: m.text and ("pinterest.com" in m.text.lower() or "pin.it" in m.text.lower()))
async def pinterest_download_handler(message: types.Message):
    """
    پردازش لینک‌های Pinterest با پشتیبانی از cancel (فلگ در task_manager)
    """

    async def process_pinterest_download():
        user_id = message.from_user.id

        text = message.text.strip()
        urls = re.findall(r'https?://[^\s]+', text)
        pinterest_urls = [u for u in urls if 'pinterest.com' in u or 'pin.it' in u]

        # بررسی سریع فلگ لغو قبل از شروع
        if getattr(task_manager, "cancel_flags", {}).get(user_id):
            await message.answer("🚫 Operation canceled by user.")
            return

        # اگر چند لینک ارسال شده (پروفایل یا چند پین)
        if 2 <= len(pinterest_urls) <= 5:
            await handle_multiple_pinterest_links(message, pinterest_urls)
            return
        elif len(pinterest_urls) != 1:
            await message.answer("⚠️ Please send exactly 1 or 2–5 Pinterest links.")
            return

        url = pinterest_urls[0]
        loading_message = await message.answer("We are processing your request...")

        try:
            loop = asyncio.get_running_loop()

            # بررسی دوباره فلگ قبل از اجرای عملیات سنگین
            if getattr(task_manager, "cancel_flags", {}).get(user_id):
                await message.bot.edit_message_text(
                    chat_id=message.chat.id,
                    message_id=loading_message.message_id,
                    text="🚫 Operation canceled before processing started."
                )
                return

            # حالت: Pin مستقیم (عکس / ویدیو)
            if "/pin/" in url or "pin.it" in url:
                content_type, file_url, caption = await loop.run_in_executor(
                    executor, fetch_pinterest_content, url
                )

                # چک لغو حین یا بعد از دریافت داده از executor
                if getattr(task_manager, "cancel_flags", {}).get(user_id):
                    await message.bot.edit_message_text(
                        chat_id=message.chat.id,
                        message_id=loading_message.message_id,
                        text="🚫 Operation canceled during processing."
                    )
                    return

                if not file_url:
                    await message.bot.edit_message_text(
                        chat_id=message.chat.id,
                        message_id=loading_message.message_id,
                        text="❌ Failed to retrieve file URL. Please try again later."
                    )
                    return

                if content_type == "image":
                    local_file = await pinterest_download(file_url, ".jpg")
                    sent_msg = await message.bot.edit_message_media(
                        chat_id=message.chat.id,
                        message_id=loading_message.message_id,
                        media=types.InputMediaPhoto(
                            media=types.FSInputFile(local_file), caption=caption, parse_mode="Markdown"
                        )
                    )
                    # Record download history
                    if sent_msg:
                        try:
                            from bot import record_download
                            file_size = os.path.getsize(local_file) if os.path.exists(local_file) else None
                            await record_download(
                                user_id, "pinterest", url,
                                file_type="image",
                                file_size=file_size
                            )
                        except Exception as hist_e:
                            logger.debug(f"Failed to record download history: {hist_e}")
                elif content_type == "video":
                    local_file = await pinterest_download(file_url, ".mp4")
                    sent_msg = await message.bot.edit_message_media(
                        chat_id=message.chat.id,
                        message_id=loading_message.message_id,
                        media=types.InputMediaVideo(
                            media=types.FSInputFile(local_file), caption=caption, parse_mode="Markdown"
                        )
                    )
                    # Record download history
                    if sent_msg:
                        try:
                            from bot import record_download
                            file_size = os.path.getsize(local_file) if os.path.exists(local_file) else None
                            await record_download(
                                user_id, "pinterest", url,
                                file_type="video",
                                file_size=file_size
                            )
                        except Exception as hist_e:
                            logger.debug(f"Failed to record download history: {hist_e}")
                else:
                    await message.bot.edit_message_text(
                        chat_id=message.chat.id,
                        message_id=loading_message.message_id,
                        text="❌ Sorry, couldn't detect the content type."
                    )

            # حالت: پروفایل
            else:
                profile_info = await loop.run_in_executor(
                    executor, fetch_pinterest_profile, url
                )

                # بررسی فلگ بعد از دریافت پروفایل
                if getattr(task_manager, "cancel_flags", {}).get(user_id):
                    await message.bot.edit_message_text(
                        chat_id=message.chat.id,
                        message_id=loading_message.message_id,
                        text="🚫 Operation canceled by user."
                    )
                    return

                caption = (
                    f"Profile: [{profile_info['username']}]({profile_info['profile_url']})\n"
                    f"Pins: {profile_info['pins_count']}\n"
                    f"Description: {profile_info['description']}\n\n"
                    "Download by <a href='https://t.me/Faryseneaidownloder_bot'>Faryseneaidownloderbot</a>"
                )

                if profile_info.get('profile_image'):
                    local_file = await pinterest_download(profile_info['profile_image'], ".jpg")
                    await message.bot.edit_message_media(
                        chat_id=message.chat.id,
                        message_id=loading_message.message_id,
                        media=types.InputMediaPhoto(
                            media=types.FSInputFile(local_file), caption=caption, parse_mode="Markdown"
                        )
                    )
                else:
                    await message.bot.edit_message_text(
                        chat_id=message.chat.id,
                        message_id=loading_message.message_id,
                        text=caption,
                        parse_mode="Markdown"
                    )

            # تلاش برای حذف پیام کاربر (اختیاری)
            try:
                await message.delete()
            except Exception:
                pass

        except asyncio.CancelledError:
            # Cancel was requested; update UI if possible, then re-raise to let TaskManager handle cleanup
            try:
                await message.bot.edit_message_text(
                    chat_id=message.chat.id,
                    message_id=loading_message.message_id,
                    text="🚫 Operation canceled by user."
                )
            except Exception:
                pass
            raise
        except Exception as e:
            logger.error(f"[Pinterest] Error: {e}", exc_info=True)
            await message.bot.edit_message_text(
                chat_id=message.chat.id,
                message_id=loading_message.message_id,
                text="⚠️ An unexpected error occurred. Please try again later."
            )

    # اجرای کل منطق داخل TaskManager (بدون پرانتز برای تابع)
    await task_manager.start_task(message.from_user.id, process_pinterest_download)



# تابع دریافت محتوای پینترست (بهینه‌شده)
def fetch_pinterest_content(pin_url: str):
    start_time = time.time()
    session = requests.Session()  # حفظ اتصال برای سرعت بیشتر
    headers = {"User-Agent": ua.random}

    # استفاده از پروکسی‌های چرخشی
    for attempt in range(len(WORKING_PROXIES)):
        proxy = get_next_proxy()
        try:
            r = session.get(pin_url, headers=headers, proxies={"http": proxy, "https": proxy}, timeout=10)
            r.raise_for_status()
            break
        except Exception as e:
            logger.warning(f"Proxy failed (attempt {attempt + 1}): {proxy} - {e}")
            continue
    else:
        raise Exception("All proxies failed")

    soup = BeautifulSoup(r.text, "lxml")

    # توضیحات با selector سریع‌تر
    desc_tag = soup.select_one('meta[name="description"]')
    description = desc_tag.get("content") if desc_tag else None

    # تعداد لایک، کامنت، ID پین‌کننده
    likes_count = ""
    comments_count = "نامشخص"
    username = "نامشخص"
    full_name = "نامشخص"

    # استخراج اطلاعات از اسکریپت‌های JSON (محدود به اسکریپت‌های خاص)
    def find_in_json(data, key, default="نامشخص"):
        if isinstance(data, dict):
            if key in data:
                return str(data[key])
            for value in data.values():
                result = find_in_json(value, key, default)
                if result != default:
                    return result
        elif isinstance(data, list):
            for item in data:
                result = find_in_json(item, key, default)
                if result != default:
                    return result
        return default

    # فقط اسکریپت‌های JSON رو select کنید (سریع‌تر از find_all)
    scripts = soup.select('script[type="application/ld+json"], script[type="application/json"]')
    for script in scripts:
        if script.string:
            try:
                data = json.loads(script.string)
                likes_count = find_in_json(data, "save_count", likes_count) if likes_count == "نامشخص" else likes_count
                likes_count = find_in_json(data, "aggregated_save_count", likes_count) if likes_count == "نامشخص" else likes_count
                comments_count = find_in_json(data, "commentCount", comments_count)
                username = find_in_json(data, "username", username)
                full_name = find_in_json(data, "full_name", full_name)
            except json.JSONDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error parsing script JSON: {e}")

    # تلاش برای استخراج از درخواست‌های شبکه (با session)
    if likes_count == "نامشخص" or comments_count == "نامشخص" or username == "نامشخص" or full_name == "نامشخص":
        try:
            pin_id = pin_url.split("/")[-2] if "/pin/" in pin_url else None
            if pin_id:
                api_url = f"https://www.pinterest.com/resource/PinResource/get/?data={{\"options\":{{\"id\":\"{pin_id}\",\"field_set_key\":\"detailed\"}}}}"
                # Get fresh proxy for API request
                api_proxy = get_next_proxy()
                api_response = session.get(api_url, headers=headers, proxies={"http": api_proxy, "https": api_proxy}, timeout=10)
                api_response.raise_for_status()
                api_data = api_response.json()
                resource_data = api_data.get("resource_response", {}).get("data", {})
                
                aggregated_data = resource_data.get("aggregated_pin_data", {})
                likes_count = str(aggregated_data.get("aggregated_save_count", aggregated_data.get("save_count", likes_count)))
                comments_count = str(resource_data.get("comment_count", comments_count))
                pinner = resource_data.get("pinner", {})
                username = pinner.get("username", username)
                full_name = pinner.get("full_name", full_name)
        except Exception as e:
            logger.error(f"Error fetching PinResource: {e}")

    caption = f"Pin: [pin]({pin_url})\nLikes: {likes_count}\nComments: {comments_count}\nDescription: {description or 'No description'}\nSource: Pinterest\n\nDownload by Faryseneaidownloder_bot (https://t.me/Faryseneaidownloder_bot)"

    # بررسی نوع محتوا با selector سریع‌تر
    video_tag = soup.select_one('meta[property="og:video"]')
    if video_tag and video_tag.get("content"):
        logger.info(f"Fetch time: {time.time() - start_time:.2f} seconds")
        return "video", video_tag["content"], caption

    # جستجو داخل JSON برای ویدیو
    for script_tag in soup.select('script[type="application/ld+json"]'):
        try:
            data = json.loads(script_tag.string)
            if isinstance(data, dict) and "contentUrl" in data:
                logger.info(f"Fetch time: {time.time() - start_time:.2f} seconds")
                return "video", data["contentUrl"], caption
        except json.JSONDecodeError:
            logger.error("Error decoding ld+json script")
            continue
        except Exception as e:
            logger.error(f"Error processing ld+json: {e}")
            continue

    img_tag = soup.select_one('meta[property="og:image"]')
    if img_tag and img_tag.get("content"):
        img_url = img_tag["content"]
        if "i.pinimg.com" in img_url:
            img_url = img_url.replace("/236x/", "/originals/").replace("/474x/", "/originals/").replace("/736x/", "/originals/")
        logger.info(f"Fetch time: {time.time() - start_time:.2f} seconds")
        return "image", img_url, caption

    raise ValueError("Sorry, I couldn't detect the content of this pin. Please try again or contact support [@FaryseneAI_Support](https://t.me/FaryseneAI_Support).")

# تابع دریافت اطلاعات پروفایل (بهینه‌شده)
def fetch_pinterest_profile(profile_url: str):
    start_time = time.time()
    session = requests.Session()  # حفظ اتصال
    headers = {"User-Agent": ua.random}

    # استفاده از پروکسی‌های چرخشی
    for attempt in range(len(WORKING_PROXIES)):
        proxy = get_next_proxy()
        try:
            r = session.get(profile_url, headers=headers, proxies={"http": proxy, "https": proxy}, timeout=10)
            r.raise_for_status()
            break
        except Exception as e:
            logger.warning(f"Profile proxy failed (attempt {attempt + 1}): {proxy} - {e}")
            continue
    else:
        raise Exception("All proxies failed")

    soup = BeautifulSoup(r.text, "lxml")

    # بررسی اینکه آیا URL واقعاً پروفایل است
    if "/pin/" in profile_url.lower():
        raise ValueError("This link appears to be a pin, not a profile. Please send a profile link.")

    username = profile_url.strip("/").split("/")[-1]
    profile_image = None
    pins_count = "نامشخص"
    description = "بدون توضیح"

    img_tag = soup.select_one('meta[property="og:image"]')
    if img_tag:
        profile_image = img_tag.get("content")

    desc_tag = soup.select_one('meta[name="description"]')
    if desc_tag:
        description = desc_tag.get("content")

    # استخراج تعداد پین‌ها از متن صفحه (بهینه: فقط بخشی از متن)
    pins_text = soup.body.get_text() if soup.body else ""  # محدود به body برای سرعت
    if "pins" in pins_text.lower():
        try:
            pins_count = [int(s) for s in re.findall(r'\d+', pins_text) if s.isdigit()][0]  # با regex سریع‌تر
        except Exception:
            pass

    profile_info = {
        "username": username,
        "profile_image": profile_image,
        "pins_count": pins_count,
        "description": description,
        "profile_url": profile_url
    }
    logger.info(f"Profile fetch time: {time.time() - start_time:.2f} seconds")
    return profile_info
# تابع هندل کردن 5 لینک پینترست به صورت گروهی
async def handle_multiple_pinterest_links(message: types.Message, urls: list[str]):
    loading_message = await message.answer(f"We are processing your {len(urls)} Pinterest links...")

    try:
        loop = asyncio.get_running_loop()

        # Cancel check before starting heavy work
        user_id = message.from_user.id
        if getattr(task_manager, "cancel_flags", {}).get(user_id):
            await message.bot.edit_message_text(
                chat_id=message.chat.id,
                message_id=loading_message.message_id,
                text="🚫 Operation canceled by user."
            )
            return

        # Fetch all contents concurrently
        tasks = [loop.run_in_executor(executor, fetch_pinterest_content, url) for url in urls]
        results = await asyncio.gather(*tasks)

        # Cancel check after heavy work
        if getattr(task_manager, "cancel_flags", {}).get(user_id):
            await message.bot.edit_message_text(
                chat_id=message.chat.id,
                message_id=loading_message.message_id,
                text="🚫 Operation canceled by user."
            )
            return

        # Build media group
        media_group = []
        descriptions = []
        links_line = " ".join([f"<a href='{url}'>pin {i+1}</a>" for i, url in enumerate(urls)])

        for i, (content_type, file_url, single_caption) in enumerate(results):
            # Extract description from single_caption
            desc_match = re.search(r'Description: (.+?)\nSource:', single_caption)
            desc = desc_match.group(1) if desc_match else 'No description'
            descriptions.append(desc)

            if content_type == "image":
                local_file = await pinterest_download(file_url, ".jpg")
                media = types.InputMediaPhoto(media=types.FSInputFile(local_file))
            elif content_type == "video":
                local_file = await pinterest_download(file_url, ".mp4")
                media = types.InputMediaVideo(media=types.FSInputFile(local_file))
            else:
                continue  # Skip if unknown

            media_group.append(media)

        # Set caption on the first media after collecting all descriptions
        if media_group:
            descriptions_text = "\n".join([f"<blockquote>{desc}</blockquote>" for desc in descriptions])
            media_group[0].caption = f"{links_line}\n\nDescriptions:\n{descriptions_text}\n\nDownload by <a href='https://t.me/Faryseneaidownloderbot'>Faryseneaidownloderbot</a>"
            media_group[0].parse_mode = "HTML"

        # Send media group
        await message.bot.send_media_group(chat_id=message.chat.id, media=media_group)
        await message.bot.delete_message(chat_id=message.chat.id, message_id=loading_message.message_id)
        await message.delete()

    except asyncio.CancelledError:
        try:
            await message.bot.edit_message_text(
                chat_id=message.chat.id,
                message_id=loading_message.message_id,
                text="🚫 Operation canceled by user."
            )
        except Exception:
            pass
        raise
    except Exception as e:
        logger.error(f"Multiple Pinterest Error: {e}")
        await message.bot.edit_message_text(
            chat_id=message.chat.id,
            message_id=loading_message.message_id,
            text="Sorry, an error occurred while processing your 5 links. Please try again later."
        )
