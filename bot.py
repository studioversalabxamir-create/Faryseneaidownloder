# bot.py (updated)
import os
import asyncio
import re
from aiogram import Dispatcher, Router, Bot
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.client.default import DefaultBotProperties
from aiogram.types import FSInputFile, CallbackQuery
from aiogram.fsm.storage.memory import MemoryStorage
from db_manager import init_db, migrate_db, get_active_daily_users, is_daily_songs_enabled, disable_daily_songs, record_daily_send, record_unsubscribe, add_download_history
import requests
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz
import logging
import time
from task_manager import task_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- تنظیمات و هندلرها ---
from config import BOT_TOKEN, PROXY, ALLOWED_USERS, FFMPEG_PATH
import config
from handlers import detector
from handlers import tiktok, pinterest, youtube, spotify, instagram, twitter, threads

from handlers.youtube import youtube_download_handler
from handlers.spotify import spotify_download_handler
from handlers.soundcloud import soundcloud_download_handler
from handlers.pinterest import pinterest_download_handler, handle_multiple_pinterest_links
from handlers.twitter import twitter_download_handler
from handlers.threads import threads_download_handler
from handlers.tiktok import tiktok_download_handler
from handlers.instagram import instagram_download_handler

from handlers.tiktok import router as tiktok_router
from handlers.twitter import router as twitter_router
from handlers.threads import router as threads_router
from handlers.instagram import router as instagram_router


from handlers.detector import detect_platform


# Import utility functions
from utils import extract_urls, validate_url, format_file_size
from error_handler import handle_errors

def is_simultaneous_links(urls: list[str]) -> bool:
    """Check if exactly 2 or 3 Spotify links"""
    if len(urls) not in [2, 3]:
        return False
    return all(detect_platform(url) == "spotify" for url in urls)

# --- Helper Functions ---
async def record_download(user_id: int, platform: str, url: str, title: str = None, 
                        artist: str = None, file_type: str = None, file_size: int = None):
    """Record a successful download to history"""
    try:
        add_download_history(user_id, platform, url, title, artist, file_type, file_size, 'completed')
        logger.info(f"Recorded download: user={user_id}, platform={platform}, url={url[:50]}")
    except Exception as e:
        logger.error(f"Failed to record download history: {e}")

def record_download_sync(user_id: int, platform: str, url: str, title: str = None, 
                         artist: str = None, file_type: str = None, file_size: int = None):
    """Synchronous version for use in non-async contexts"""
    try:
        add_download_history(user_id, platform, url, title, artist, file_type, file_size, 'completed')
        logger.info(f"Recorded download: user={user_id}, platform={platform}, url={url[:50]}")
    except Exception as e:
        logger.error(f"Failed to record download history: {e}")

# --- توابع روزانه ---
DAILY_SONGS_FOLDER = "daily_songs"

async def send_daily_song(bot: Bot, time_slot: str):
    """ارسال آهنگ روزانه به کاربران فعال"""
    # Find the file for this time slot
    daily_files = [f for f in os.listdir(DAILY_SONGS_FOLDER) if f.startswith(f"daily_{time_slot}-") and f.endswith(".mp3")]
    if not daily_files:
        # Warn admin
        admin_id = ALLOWED_USERS[0] if ALLOWED_USERS else None
        if admin_id:
            await bot.send_message(admin_id, f"⚠️ فایل روزانه برای {time_slot} در پوشه {DAILY_SONGS_FOLDER} یافت نشد. لطفاً فایل را قرار دهید.")
        return

    file_name = daily_files[0]  # Take the first matching file
    file_path = os.path.join(DAILY_SONGS_FOLDER, file_name)

    # Parse filename to extract title and artist
    # Format: daily_{time_slot}-{title} - {artist}.mp3
    base_name = file_name.replace(f"daily_{time_slot}-", "").replace(".mp3", "")
    if " - " in base_name:
        title, artist = base_name.split(" - ", 1)
    else:
        title = base_name
        artist = "Unknown Artist"

    # Try to extract Spotify link from filename or use placeholder
    spotify_link = "https://open.spotify.com"  # Default placeholder
    # If filename contains artist and title, could search Spotify API here

    # Get active users
    active_users = get_active_daily_users()
    if not active_users:
        logging.info("No active users for daily songs")
        return

    caption = f"🎧 Daily bot suggestion\n✨ Today's selected song for you\n\n{artist} – {title}\n🔗 Spotify: {spotify_link}"

    keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="❌ Unsubscribe", callback_data="unsubscribe_daily")]
    ])

    # Send in batches to avoid rate limits
    batch_size = 20  # Increased for better speed
    for i in range(0, len(active_users), batch_size):
        batch = active_users[i:i+batch_size]
        tasks = []
        for user_id in batch:
            task = bot.send_audio(
                chat_id=user_id,
                audio=FSInputFile(file_path),
                caption=caption,
                reply_markup=keyboard,
                parse_mode="HTML"
            )
            tasks.append(task)

        # Send concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for j, result in enumerate(results):
            if isinstance(result, Exception):
                logging.error(f"Failed to send daily song to {batch[j]}: {result}")

        await asyncio.sleep(0.5)  # Shorter delay between batches

    # Record stats
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    record_daily_send(today, time_slot, len(active_users), len(active_users))  # Assuming all sent successfully

    logging.info(f"Sent {time_slot} daily song to {len(active_users)} users")

# تسک ترکینگ
# Use the shared global instance from task_manager

# --- روت اصلی برای شناسایی لینک‌ها ---
router = Router()

# --- هندلر برای لینک‌های همزمان ---
async def handle_simultaneous_links(message: Message, urls: list[str]):
    """Handle 2 or 3 Spotify links simultaneously"""
    user_id = message.from_user.id
    if user_id not in ALLOWED_USERS:
        await message.answer("بات در حال توسعه میباشد...بزودی رونمایی خواهد شد ")
        return

    # Send initial message as reply to user's message
    num_links = len(urls)
    progress_msg = await message.reply(f"🎵 Received {num_links} Spotify links. Processing simultaneously...")

    # Import the simultaneous handler from spotify
    from handlers.spotify import handle_simultaneous_spotify_links
    await handle_simultaneous_spotify_links(message, urls, progress_msg)

        
@router.message()
async def detect_all_links(message: Message):
    text = (message.text or "").strip()
    if not text:
        return

    urls = extract_urls(text)
    
    # Validate all URLs
    valid_urls = [url for url in urls if validate_url(url)]
    if not valid_urls:
        await message.answer("❌ No valid links found in your message.")
        return
    
    urls = valid_urls

    # Check for simultaneous links (2 or 3 Spotify links)
    if is_simultaneous_links(urls):
        await handle_simultaneous_links(message, urls)
        return

    # Check for multiple Pinterest links (2 to 5)
    if 2 <= len(urls) <= 5 and all(detect_platform(url) == "pinterest" for url in urls):
        await handle_multiple_pinterest_links(message, urls)
        return

    # Check for too many links (more than 5)
    if len(urls) > 5:
        await message.answer("❌ Maximum 5 links supported at once. Please send fewer links.")
        return

    # Single link processing
    if len(urls) == 1:
        url = urls[0]
        platform = detect_platform(url)

        if platform == "youtube":
            await youtube_download_handler(message)
        elif platform == "spotify":
            await spotify_download_handler(message)
        #elif platform == "soundcloud":
            #await soundcloud_download_handler(message)
        elif platform == "pinterest":
            await pinterest_download_handler(message)
        elif platform == "twitter":
            await twitter_download_handler(message)
        elif platform == "threads":
            await threads_download_handler(message)
        elif platform == "tiktok":
            await tiktok_download_handler(message)
        elif platform == "instagram":
            await instagram_download_handler(message)
        else:
            await message.answer(f"❌ Unsupported platform: {platform}. This platform will be added soon.")
    else:
        await message.answer("❌ Please send one of: 1 link, 2-3 Spotify links, 2-5 Pinterest links, or 2-5 TikTok links.")


# --- تابع اصلی ---
async def main():
    ffmpeg_path = os.getenv("FFMPEG_PATH") or config.FFMPEG_PATH
    if ffmpeg_path and ffmpeg_path != "ffmpeg":
        os.environ["FFMPEG_PATH"] = ffmpeg_path

    proxy_url = PROXY

    bot_properties = DefaultBotProperties(parse_mode=ParseMode.HTML)

    bot = Bot(token=BOT_TOKEN, proxy=proxy_url, default_bot_properties=bot_properties)

    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    

    # Scheduler for daily songs
    scheduler = AsyncIOScheduler(timezone=pytz.timezone('Asia/Tehran'))

    # Add daily jobs in Tehran timezone
    scheduler.add_job(send_daily_song, CronTrigger(hour=9, minute=0), args=[bot, "morning"])  # 09:00 Tehran
    scheduler.add_job(send_daily_song, CronTrigger(hour=14, minute=0), args=[bot, "noon"])    # 14:00 Tehran
    scheduler.add_job(send_daily_song, CronTrigger(hour=22, minute=0), args=[bot, "evening"]) # 22:00 Tehran

    # --- روت استارت ---
    main_router = Router()

    @main_router.message(Command("start"))
    async def start_handler(message: Message):
        if message.from_user.id not in ALLOWED_USERS:
            await message.answer("The bot is in development. Access is limited.")
            return
        # Auto-enable daily songs for allowed users
        from db_manager import enable_daily_songs
        enable_daily_songs(message.from_user.id)
        await message.answer(
            "🎉 Welcome to Farysene AI v1.0\n\n"
            "Supported platforms (Beta):\n"
            "• Spotify • Pinterest • Instagram • TikTok • YouTube • Twitter (X) • Threads\n\n"
            "How it works:\n"
            "• Send a link from any supported platform\n"
            "• For Spotify: you can send 2-3 links to process simultaneously\n"
            "• For Pinterest: send up to 5 links in one message\n\n"
            "Daily content:\n"
            "• 3 curated songs every day\n"
            "• Short books (coming soon)\n\n"
            "Account: Full Pro access — no limits (3 months free during beta)\n\n"
            "Help: /help — Updates: @Faryseneaibot — Support: @FaryseneSupport\n\n"
            "Let’s get started ✈️"
        )
    @main_router.message(Command("help"))
    async def help_handler(message: Message):
        if message.from_user.id not in ALLOWED_USERS:
            await message.answer("The bot is in development. Access is limited.")
            return
        await message.answer(
            "🆘 Farysene AI v1.0 — Help (Beta)\n\n"
            "How to use:\n"
            "• Send a direct link from Spotify, Pinterest, Instagram, TikTok, YouTube, Twitter (X), Threads\n"
            "• Search is supported for Spotify (send track/artist keywords)\n\n"
            "Free plan:\n"
            "• 13 downloads/day\n"
            "• Standard quality\n\n"
            "Commands:\n"
            "• /plans — Subscriptions\n"
            "• /upgrade — Upgrade\n"
            "• /status — Platform status\n"
            "• /history — Your last downloads\n"
            "• /stats — Your stats\n"
            "• /retry — Retry last process\n\n"
            "Limits & formats:\n"
            "• Processing time: 30s–5min\n"
            "• Formats: MP3, MP4, JPG, PNG\n"
            "• Size: up to 2GB\n\n"
            "Privacy & Safety:\n"
            "• We only use your Telegram ID and temporary links\n"
            "• No phone/chat data stored\n"
            "• API encryption, temporary data, token control\n\n"
            "Policy:\n"
            "• Illegal usage is prohibited — violations may be suspended\n\n"
            "Support: @Farysenesupport — Guide: @Faryseneaibot\n"
            "🚀 Beta in active development — feedback is welcome!"
        )
    @main_router.message(Command("playlist"))
    async def playlist_handler(message: Message):
        if message.from_user.id not in ALLOWED_USERS:
           await message.answer("The bot is in development. Access is limited.")
           return
        await message.answer(
           "🎧This section is under development..\n"
           "🎧 This section is currently under development."
        )
    
    @main_router.message(Command("history"))
    async def history_handler(message: Message):
        if message.from_user.id not in ALLOWED_USERS:
            await message.answer("The bot is in development. Access is limited.")
            return
        
        from db_manager import get_user_download_history
        history = get_user_download_history(message.from_user.id, limit=10)
        
        if not history:
            await message.answer(
                "📜 Your download history is empty.\n\n"
                "💡 After you download files, your history will appear here."
            )
            return
        
        history_text = "📜 Your download history:\n📜 Your Download History:\n\n"
        for i, item in enumerate(history, 1):
            platform = item[0]
            url = item[1]
            title = item[2] or "بدون عنوان"
            artist = item[3] or ""
            file_type = item[4] or ""
            created_at = item[6]
            
            history_text += f"{i}. {platform.upper()}\n"
            if artist:
                history_text += f"   🎵 {artist} - {title}\n"
            else:
                history_text += f"   🎵 {title}\n"
            if file_type:
                history_text += f"   📁 {file_type}\n"
            history_text += f"   📅 {created_at}\n\n"
            
            if len(history_text) > 3500:  # Telegram message limit
                history_text += "...\n(نمایش ۱۰ مورد اخیر)"
                break
        
        await message.answer(history_text)
    
    @main_router.message(Command("stats"))
    async def stats_handler(message: Message):
        if message.from_user.id not in ALLOWED_USERS:
            await message.answer("The bot is in development. Access is limited.")
            return
        
        from db_manager import get_user_stats
        from utils import format_file_size
        from task_manager import task_manager
        
        stats = get_user_stats(message.from_user.id)
        rate_limit_status = task_manager.get_rate_limit_status(message.from_user.id)
        
        stats_text = "📊 Your Statistics:\n\n"
        stats_text += f"📥 Total downloads: {stats['total_downloads']}\n\n"
        
        if stats['platform_stats']:
            stats_text += "📱 By platform:\n"
            for platform, count in stats['platform_stats']:
                stats_text += f"   • {platform.upper()}: {count}\n"
            stats_text += "\n"
        
        if stats['total_size'] > 0:
            stats_text += f"💾 Total size: {format_file_size(stats['total_size'])}\n\n"
        
        stats_text += f"🕐 Last 7 days: {stats['recent_downloads']}\n\n"
        
        stats_text += f"⚡ Remaining requests: {rate_limit_status['remaining']}/{rate_limit_status['limit']}"
        
        await message.answer(stats_text)
    
    @main_router.message(Command("status"))
    async def status_handler(message: Message):
        """Check platform availability status"""
        if message.from_user.id not in ALLOWED_USERS:
            await message.answer("The bot is in development. Access is limited.")
            return
        
        from platform_checker import check_all_platforms
        
        status_msg = await message.answer("🔍 Checking platform status…")
        
        platforms_status = await check_all_platforms()
        
        status_text = "🌐 Platform status:\n\n"
        for platform, is_available in platforms_status.items():
            status_icon = "✅" if is_available else "❌"
            status_text += f"{status_icon} {platform.upper()}: {'Available' if is_available else 'Unavailable'}\n\n"
        
        await status_msg.edit_text(status_text)
    @main_router.message(Command("cancel"))
    async def cancel_handler(message: Message):
        await task_manager.cancel_task(message.from_user.id)
        # Stop current task or process.


    @main_router.message(Command("retry"))
    async def retry_handler(message: Message):
        if message.from_user.id not in ALLOWED_USERS:
            await message.answer("The bot is in development. Access is limited.")
            return
        
        success = await task_manager.retry_last(message.from_user.id)
        if success:
            retry_msg = await message.answer(
               "🔁 Retrying the last process…"
        )
            # Auto-delete the message after 7 seconds
            await asyncio.sleep(7)
            try:
                await retry_msg.delete()
            except Exception:
                pass
        else:
            await message.answer(
                "❌ No previous operation found to retry.\n\n"
                "💡 Please send a link first."
            )



    @main_router.message(Command("daily_on"))
    async def daily_on_handler(message: Message):
        user_id = message.from_user.id
        if user_id not in ALLOWED_USERS:
            await message.answer("The bot is in development. Access is limited.")
            return

        from db_manager import enable_daily_songs
        if enable_daily_songs(user_id):
            await message.answer("✅ Daily songs enabled! You will receive three songs every morning, noon and evening.")
        else:
            await message.answer("❌ Failed to enable daily songs.")

    @main_router.callback_query(lambda q: q.data == "unsubscribe_daily")
    async def unsubscribe_daily_callback(query: CallbackQuery):
        user_id = query.from_user.id
        if disable_daily_songs(user_id):
            # Record unsubscribe
            from datetime import datetime
            today = datetime.now().strftime("%Y-%m-%d")
            record_unsubscribe(today)
            await query.answer("✅ You unsubscribed from daily songs.")
            await query.message.answer("❌ Daily songs disabled. Use /daily_on to re-enable.")
        else:
            await query.answer("❌ Error while disabling.")

    # --- اضافه کردن روترها ---
    dp.include_router(main_router)
    dp.include_router(router)            
    dp.include_router(youtube.router)
    dp.include_router(spotify.router)
    # dp.include_router(soundcloud.router)  # Disabled until dependencies are ready
    dp.include_router(pinterest.router)
    dp.include_router(twitter_router)
    dp.include_router(threads_router)
    dp.include_router(tiktok.router)
    dp.include_router(instagram_router)

    # اجرای بات
    scheduler.start()
    logging.info("Scheduler started for daily songs")
    await dp.start_polling(bot)


if __name__ == "__main__":
    # Initialize database
    init_db()
    migrate_db()

    # Log startup
    logger.info("Starting Farysene AI Bot v1.0...")
    logger.info(f"Allowed users: {ALLOWED_USERS}")

    # Run bot
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
