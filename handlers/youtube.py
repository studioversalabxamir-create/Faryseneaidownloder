import os
import asyncio
import yt_dlp
from aiogram import Router
from aiogram.types import Message, FSInputFile
from concurrent.futures import ThreadPoolExecutor
from handlers.detector import detect_platform
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, InputMediaPhoto
from aiogram.types import CallbackQuery
from yt_dlp.utils import sanitize_filename
import subprocess
from html import escape
import whisper
from fpdf import FPDF

router = Router()
executor = ThreadPoolExecutor(max_workers=2)

DOWNLOAD_DIR = "downloads"
COOKIES_FILE = "cookies.txt"
# Import proxy from config
try:
    from config import PROXY, FFMPEG_PATH
except ImportError:
    PROXY = "http://174.136.204.40:80"
    FFMPEG_PATH = "ffmpeg"

def download_youtube_video(url: str) -> str:
    DOWNLOAD_DIR_ABS = os.path.abspath(DOWNLOAD_DIR)
    os.makedirs(DOWNLOAD_DIR_ABS, exist_ok=True)

    ydl_opts = {
        'outtmpl': f'{DOWNLOAD_DIR_ABS}/%(title).100s.%(ext)s',
        'format': 'bestvideo+bestaudio/best',
        'cookiefile': 'cookies.txt',
        'noplaylist': True,
        'socket_timeout': 120,
        'quiet': False,
        'no_warnings': False,
        'ffmpeg_location': FFMPEG_PATH if FFMPEG_PATH != "ffmpeg" else None,  # Only set if custom path
    }

    if PROXY:
        ydl_opts['proxy'] = PROXY
    if os.path.exists(COOKIES_FILE):
        ydl_opts['cookiefile'] = COOKIES_FILE

    print("شروع دانلود...")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            if os.path.exists(filename):
                print(f"دانلود موفق: {filename}")
                return filename
            else:
                raise FileNotFoundError("فایل دانلود نشده است.")
    except yt_dlp.utils.DownloadError as de:
        print(f"DownloadError: {de}, تلاش fallback")
        ydl_opts['format'] = 'best'
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            if os.path.exists(filename):
                print(f"دانلود fallback موفق: {filename}")
                return filename
            else:
                raise FileNotFoundError("فایل دانلود نشده است بعد از fallback.")
    except Exception as e:
        print(f"خطا در دانلود: {e}")
        raise e

def download_specific_format(url: str, kind: str, quality: str, loop, message_to_update):
    import os
    import asyncio
    from yt_dlp import YoutubeDL

    DOWNLOAD_DIR = globals().get("DOWNLOAD_DIR", "downloads")
    PROXY = globals().get("PROXY", None)
    COOKIES_FILE = globals().get("COOKIES_FILE", None)

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)

    if kind == "audio":
        format_code = f"bestaudio[abr<={quality}]"
        ext = "mp3"
        postprocessors = [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': quality
        }]
    else:
        format_code = f"bestvideo[height<={quality}]+bestaudio/best[height<={quality}]"
        ext = "mp4"
        postprocessors = []

    def progress_hook(d):
        try:
            status = d.get('status')
            if status == 'downloading':
                percent = d.get('_percent_str', '0%').strip()
                text = f"⏬ در حال دانلود... {percent}"
                coro = getattr(message_to_update, "edit_text", None)
                if callable(coro):
                    future = asyncio.run_coroutine_threadsafe(coro(text), loop)
                    future.result()

            elif status == 'finished':
                coro = getattr(message_to_update, "edit_text", None)
                if callable(coro):
                    future = asyncio.run_coroutine_threadsafe(
                        coro("✅ دانلود کامل شد! در حال ارسال فایل..."), loop
                    )
                    future.result()

        except Exception as e:
            print(f"[progress_hook error]: {e}")

    ydl_opts = {
        'format': format_code,
        'outtmpl': f'{DOWNLOAD_DIR}/%(title).100s.%(ext)s',
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
        'merge_output_format': ext,
        'ffmpeg_location': FFMPEG_PATH if FFMPEG_PATH != "ffmpeg" else None,  # Only set if custom path
        'prefer_ffmpeg': True,
        'postprocessors': postprocessors,
        'progress_hooks': [progress_hook],
    }

    if PROXY:
        ydl_opts['proxy'] = PROXY
    if COOKIES_FILE and os.path.exists(COOKIES_FILE):
        ydl_opts['cookiefile'] = COOKIES_FILE

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)

            if not filename:
                raise Exception("❌ مسیر فایل ساخته نشد (filename = None)")

            if kind == "audio":
                filename = filename.rsplit(".", 1)[0] + ".mp3"

            if os.path.exists(filename):
                return filename
            else:
                raise FileNotFoundError(f"دانلود انجام نشد. فایل یافت نشد: {filename}")

    except Exception as e:
        raise Exception(f"خطا در دانلود: {str(e)}")

def extract_video_info(url: str) -> dict:
    from yt_dlp import YoutubeDL

    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'skip_download': True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        return ydl.extract_info(url, download=False)

async def is_premium_user(user_id: int) -> bool:
    premium_users = []  # آیدی‌های کاربران اشتراکی را اینجا بگذار
    return user_id in premium_users

@router.message()
async def youtube_download_handler(message: Message):
    if not message.text:
        await message.answer("❌ متن پیام خالی است.")
        return

    url = message.text.strip()
    if detect_platform(url) != "youtube":
        return

    loading_msg = await message.answer("⏳ در حال دریافت اطلاعات ویدیو...")

    try:
        loop = asyncio.get_event_loop()
        info = await loop.run_in_executor(executor, extract_video_info, url)


        title = info.get("title", "ویدیو")
        duration = round(info.get("duration", 0) / 60, 1)  # تبدیل ثانیه به دقیقه
        thumbnail = info.get("thumbnail")
        video_id = info.get("id")


        # ساخت کیبورد انتخاب کیفیت
        keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [
                InlineKeyboardButton(text="🎷 صدا 128kbps", callback_data=f"audio|128|{video_id}"),
                InlineKeyboardButton(text="🎷 صدا 320kbps", callback_data=f"audio|320|{video_id}")
            ],
            [
                InlineKeyboardButton(text="🎥 360p", callback_data=f"video|360|{video_id}"),
                InlineKeyboardButton(text="🎥 480p", callback_data=f"video|480|{video_id}")
            ],
            [
                InlineKeyboardButton(text="🎥 720p", callback_data=f"video|720|{video_id}"),
                InlineKeyboardButton(text="🎥 1080p", callback_data=f"video|1080|{video_id}")
            ],
            [
                InlineKeyboardButton(text="🔒 4K (نیازمند اشتراک پایه)", callback_data=f"video|2160|{video_id}"),
                InlineKeyboardButton(text="🔒 8K (نیازمند اشتراک پایه)", callback_data=f"video|4320|{video_id}")
            ],
            [
                InlineKeyboardButton(text="ℹ️ مشاهده توضیحات", callback_data=f"desc|{video_id}")
            ]
        ])

        caption = f"<b>{title[:1000]}</b>\n⏱ مدت: {duration} دقیقه\n\n👇 کیفیت مورد نظر رو انتخاب کن:"
        caption = (caption[:1020] + "...") if len(caption) > 1024 else caption
        if thumbnail:
            await message.answer_photo(photo=thumbnail, caption=caption, reply_markup=keyboard)
        else:
            await message.answer(caption, reply_markup=keyboard)

    except Exception as e:
        await message.answer(f"🚫 خطا در دریافت اطلاعات:\n<code>{str(e)}</code>")
     # بررسی اشتراک کاربر (برای مثال ساده)
   
@router.callback_query()
async def handle_callback(query: CallbackQuery):
    try:
        data = query.data.split("|")
    except AttributeError:
        await query.message.answer("🚫 خطا: داده‌های callback نامعتبر است.")
        return

    # بررسی DOWNLOAD_DIR
    DOWNLOAD_DIR = globals().get("DOWNLOAD_DIR", "downloads")
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    # بررسی executor
    executor = globals().get("executor", None)
    if executor is None:
        await query.message.answer("🚫 خطا: ThreadPoolExecutor تنظیم نشده است.")
        return
       
    elif data[0] == "videoqualitymenu":
            video_id = data[1]
            url = f"https://www.youtube.com/watch?v={video_id}"
            info = await asyncio.get_event_loop().run_in_executor(executor, extract_video_info, url)
            title = info.get("title", "ویدیو")
            thumbnail = info.get("thumbnail")
            duration = round(info.get("duration", 0) / 60, 1)

            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="🎥 360p", callback_data=f"video|360|{video_id}"),
                InlineKeyboardButton(text="🎥 480p", callback_data=f"video|480|{video_id}")],
                [InlineKeyboardButton(text="🎥 720p", callback_data=f"video|720|{video_id}"),
                InlineKeyboardButton(text="🎥 1080p", callback_data=f"video|1080|{video_id}")],
                [InlineKeyboardButton(text="🔒 4K", callback_data=f"video|2160|{video_id}"),
                InlineKeyboardButton(text="🔒 8K", callback_data=f"video|4320|{video_id}")]
            ])

            if thumbnail:
                await query.message.reply_photo(
                    photo=thumbnail,
                    caption=f"<b>{title}</b>\n⏱ مدت: {duration} دقیقه\n\n👇 کیفیت مورد نظر رو انتخاب کن:",
                    reply_markup=keyboard,
                    parse_mode="HTML"
                )
            else:
                await query.message.reply(
                    f"<b>{title}</b>\n⏱ مدت: {duration} دقیقه\n\n👇 کیفیت مورد نظر رو انتخاب کن:",
                    reply_markup=keyboard,
                    parse_mode="HTML"
                )

    if data[0] == "convert":
        if len(data) < 4:
            await query.message.answer("🚫 خطا: داده‌های ناکافی برای تبدیل.")
            return
        format_type, kind, video_id = data[1], data[2], data[3]
        url = f"https://www.youtube.com/watch?v={video_id}"

        try:
            info = await asyncio.get_event_loop().run_in_executor(executor, extract_video_info, url)
            description = info.get("description", "❌ توضیحاتی موجود نیست.")
            thumbnail = info.get("thumbnail")
            title = info.get("title", "عنوان نامشخص")

            # حذف کیبورد قبلی
            await query.message.edit_reply_markup(reply_markup=None)

            short_desc = description[:1024]
            has_more = len(description) > 1024

            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [InlineKeyboardButton(text="More ⬇️", callback_data=f"moredesc|{video_id}")]
            ]) if has_more else None

            if thumbnail:
                await query.message.answer_photo(
                    photo=thumbnail,
                    caption=f"<b>{title}</b>\n\n📝 {short_desc}",
                    reply_to_message_id=query.message.message_id,
                    reply_markup=keyboard,
                    parse_mode="HTML"
                )
            else:
                await query.message.answer(
                    f"<b>{title}</b>\n\n📝 {short_desc}",
                    reply_to_message_id=query.message.message_id,
                    reply_markup=keyboard,
                    parse_mode="HTML"
                )

        except Exception as e:
            await query.message.answer(f"🚫 خطا در دریافت توضیحات:\n<code>{str(e)}</code>")

    elif data[0] == "moredesc":
        if len(data) < 2:
            await query.message.answer("🚫 خطا: داده‌های ناکافی برای ادامه توضیحات.")
            return
        video_id = data[1]
        url = f"https://www.youtube.com/watch?v={video_id}"

        try:
            info = await asyncio.get_event_loop().run_in_executor(executor, extract_video_info, url)
            description = info.get("description", "❌ توضیحاتی نیست.")
            full = escape(description[1024:2024]) if len(description) > 1024 else ""

            await query.message.answer(f"📝 ادامه توضیحات:\n\n<code>{full}</code>")    

        except Exception as e:
            await query.message.answer(f"🚫 خطا در ادامه توضیحات:\n<code>{str(e)}</code>")

    elif data[0] == "desc":
        if len(data) < 2:
            await query.message.answer("🚫 خطا: شناسه ویدیو ناقص است.")
            return

        video_id = data[1]
        url = f"https://www.youtube.com/watch?v={video_id}"

        try:
            info = await asyncio.get_event_loop().run_in_executor(executor, extract_video_info, url)
            title = info.get("title", "بدون عنوان")
            description = info.get("description", "❌ توضیحی برای ویدیو موجود نیست.")
            thumbnail = info.get("thumbnail")

            short_desc = escape(description[:950])
            has_more = len(description) > 1000

        # ارسال پیام توضیحات
            if thumbnail:
                desc_msg = await query.message.answer_photo(
                   photo=thumbnail,
                   caption=f"<b>{title}</b>\n\n📝 {short_desc}",
                   parse_mode="HTML"
                )
            else:
                desc_msg = await query.message.answer(
                    f"<b>{title}</b>\n\n📝 {short_desc}",
                    parse_mode="HTML"
                )

        # ارسال کیبورد تبدیل به فایل روی ریپلای پیام توضیحات
            await query.message.bot.send_message(
                chat_id=query.message.chat.id,
                text="🎯 آیا مایل به تبدیل این توضیحات به فرمت فایل هستید؟",
                reply_to_message_id=desc_msg.message_id,
                reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                   [
                        InlineKeyboardButton(text="📄 PDF", callback_data=f"descconvertfile|pdf|{video_id}"),
                        InlineKeyboardButton(text="📜 TXT", callback_data=f"descconvertfile|txt|{video_id}")
                    ],
                    [
                        InlineKeyboardButton(text="❌ نه لازم نیست", callback_data=f"descconvertfile|cancel|{video_id}")
                    ]
              ])
            )

        except Exception as e:
            await query.message.answer(f"🚫 خطا در دریافت توضیحات:\n<code>{str(e)}</code>")
 

    elif data[0] == "transcribe":
        video_id = data[1]
        audio_file_path = f"downloads/audio_{video_id}.mp3"

        if not os.path.exists(audio_file_path):
            await query.message.answer("❌ فایل صوتی یافت نشد.")
            return

        from whisper import load_model
        model = whisper.load_model("base")
        result = model.transcribe("path/to/audio.mp3")

        text = result.get("text", "❌ متنی یافت نشد.")
        await query.message.reply(f"🧠 متن استخراج‌شده:\n\n{text}")


    elif data[0] in ("audio", "video"):
        if len(data) < 3:
            await query.message.answer("🚫 خطا: داده‌های ناکافی برای دانلود.")
            return
        kind, quality, video_id = data[0], data[1], data[2]
        user_id = query.from_user.id
        
        # بررسی دسترسی به کیفیت‌های بالا برای ویدیو
        if kind == "video":
            quality_int = int(quality) if quality.isdigit() else 0
            if quality_int >= 2160 and not await is_premium_user(user_id):
                await query.answer("⛔️ کیفیت‌های 4K و 8K فقط برای کاربران اشتراکی فعال است.", show_alert=True)
                return
        
        url = f"https://www.youtube.com/watch?v={video_id}"

        try:
            # حذف دکمه‌های پیام قبلی
            await query.message.edit_reply_markup(reply_markup=None)

            # دریافت اطلاعات و نمایش تامنیل + عنوان + تاریخ آپلود
            info = await asyncio.get_event_loop().run_in_executor(executor, extract_video_info, url)
            thumbnail = info.get("thumbnail")
            title = info.get("title", "ویدیو")
            upload_date = info.get("upload_date", "نامشخص")
            if upload_date != "نامشخص" and len(upload_date) >= 8:
                upload_date = f"{upload_date[:4]}-{upload_date[4:6]}-{upload_date[6:8]}"  # فرمت YYYY-MM-DD

            if thumbnail:
                media = InputMediaPhoto(
                    media=thumbnail,
                    caption=f"<b>{title}</b>\n📅 تاریخ آپلود: {upload_date}",
                    parse_mode="HTML"
                )
                await query.message.edit_media(media)
            else:
                await query.message.edit_caption(
                    f"<b>{title}</b>\n📅 تاریخ آپلود: {upload_date}",
                    parse_mode="HTML"
                )

            # پیام پیشرفت دانلود
            progress_msg = await query.message.answer("⏳ در حال دانلود... 0%")

            loop = asyncio.get_running_loop()
            filename = await loop.run_in_executor(
                executor,
                download_specific_format,
                url,
                kind,
                quality,
                loop,
                progress_msg  # پیام پیشرفت دانلود
                ) 

            if os.path.exists(filename):
                try:
                    abs_path = os.path.abspath(filename)
                    file = FSInputFile(abs_path, filename=os.path.basename(abs_path))

                    # ساخت دکمه‌ها
                    buttons = [[InlineKeyboardButton(text="🔍 توضیحات بیشتر", callback_data=f"desc|{video_id}")]]
                    if kind == "audio":
                        buttons.append([InlineKeyboardButton(text="📥 استخراج Voice", callback_data=f"voice|{video_id}")])
                    markup = InlineKeyboardMarkup(inline_keyboard=buttons)

                    # Get file size for history
                    file_size = os.path.getsize(filename) if os.path.exists(filename) else None
                    
                    # ارسال فایل با توجه به نوع
                    if kind == "video":
                        sent_message = await query.bot.send_video(
                            chat_id=query.message.chat.id,
                            video=file,
                            caption=f"<b>🎥 {title}</b>\n🔗 <a href='{url}'>مشاهده در یوتیوب</a>",
                            parse_mode="HTML",
                            supports_streaming=True,
                            reply_markup=markup
                        )
                        # Record download history
                        try:
                            from bot import record_download
                            await record_download(
                                query.from_user.id, "youtube", url, title, 
                                file_type="video", file_size=file_size
                            )
                        except Exception:
                            pass
                    else:
                        buttons = [
                            [InlineKeyboardButton(text="📁 دریافت نسخه ویدیویی", callback_data=f"videoqualitymenu|{video_id}")],
                            [InlineKeyboardButton(text="🧠 استخراج متن صدا", callback_data=f"transcribe|{video_id}")]
                        ]

                        markup = InlineKeyboardMarkup(inline_keyboard=buttons)

                        # ارسال فایل صوتی با ذخیره‌سازی پیام برای ریپلای بعدی
                        sent_message = await query.bot.send_audio(
                            chat_id=query.message.chat.id,
                            audio=file,
                            caption=(
                                f"<b>{title}</b>\n"
                                f"🎧 کیفیت: {quality}kbps\n"
                                f"🔗 <a href='{url}'>مشاهده در یوتیوب</a>\n\n"
                                f"📥 <b>دانلود نسخه ویدیویی</b>\n"
                                f"@YourBotUsername"
                            ),
                            parse_mode="HTML",
                            supports_streaming=True,
                            reply_markup=markup
                        )
                        # Record download history for audio
                        try:
                            from bot import record_download
                            await record_download(
                                query.from_user.id, "youtube", url, title,
                                file_type="audio", file_size=file_size
                            )
                        except Exception:
                            pass

# پاک کردن پیام پیشرفت
                    await progress_msg.delete()

# نمایش گزینه تبدیل فرمت، با ریپلای روی پیام فایل ارسال شده
                    if sent_message:
                        await query.bot.send_message(
                            chat_id=query.message.chat.id,
                            text="🎯 آیا مایلید فایل به یکی از فرمت‌های زیر تبدیل شود؟",
                            reply_markup=InlineKeyboardMarkup(inline_keyboard=[
                                [
                                    InlineKeyboardButton(text="📄 PDF", callback_data=f"descconvert|pdf|{video_id}"),
                                    InlineKeyboardButton(text="📜 TXT", callback_data=f"descconvert|txt|{video_id}")
                                ],
                                [
                                    InlineKeyboardButton(text="🧾 SRT (در صورت وجود)", callback_data=f"descconvert|srt|{video_id}"),
                                    InlineKeyboardButton(text="❌ نه لازم نیست", callback_data=f"descconvert|cancel|{video_id}")
                                ]
                            ]), 
                        reply_to_message_id=sent_message.message_id
                    )

                except Exception as e:
                    await query.message.answer(f"🚫 خطا در ارسال فایل:\n<code>{e}</code>")
                finally:
                    try:
                        if os.path.exists(filename):
                            os.remove(filename)
                    except Exception:
                        pass
            else:
                await query.message.answer("❌ فایل دانلود شده پیدا نشد.")

        except Exception as e:
            await query.message.answer(f"🚫 خطا در دانلود فایل:\n<code>{str(e)}</code>")

    elif data[0] == "descconvert":
        if len(data) < 3:
            await query.message.answer("🚫 خطا: داده‌های ناکافی برای تبدیل توضیحات.")
            return

        format_type, video_id = data[1], data[2]

    if format_type == "cancel":
        await query.message.delete()
        await query.message.edit_reply_markup(reply_markup=None)  # پاک کردن کیبورد
        await query.message.reply("Download complete.")  # پیام ساده
        return

    url = f"https://www.youtube.com/watch?v={video_id}"
    output_file = None  # تعریف اولیه برای استفاده در finally

    try:
        # استخراج اطلاعات ویدیو
        info = await asyncio.get_event_loop().run_in_executor(executor, extract_video_info, url)
        description = info.get("description", "❌ توضیحاتی یافت نشد.")
        title = info.get("title", f"desc_{video_id}")
        subtitles = info.get("subtitles", {})

        # امن‌سازی نام فایل
        safe_title = "".join(c if c.isalnum() else "_" for c in title.strip())[:50]
        output_file = os.path.join(DOWNLOAD_DIR, f"{safe_title}.{format_type}")

        if format_type == "txt":
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(description)

        elif format_type == "pdf":
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)  # فونت پیش‌فرض استاندارد FPDF
            for line in description.split("\n"):
                pdf.multi_cell(0, 10, line)
            pdf.output(output_file)

        elif format_type == "srt":
            import requests
            en_subs = subtitles.get("en", [])
            if en_subs and "url" in en_subs[0]:
                srt_url = en_subs[0]["url"]
                srt_text = requests.get(srt_url).text
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(srt_text)
            
                await query.message.answer("❌ زیرنویس انگلیسی برای این ویدیو موجود نیست.")
                return

        else:
            if format_type not in ("txt", "pdf", "srt"):
                await query.message.answer("❌ فرمت درخواستی پشتیبانی نمی‌شود.")
                return
# بعد از این بررسی، بقیه کد ادامه پیدا می‌کند.

        # ارسال فایل
        if os.path.exists(output_file):
            file = FSInputFile(output_file, filename=os.path.basename(output_file))
            await query.message.answer_document(
                document=file,
                caption=f"📄 توضیحات ویدیو در قالب <b>{format_type.upper()}</b> آماده شد.",
                parse_mode="HTML"
            )
        else:
            await query.message.answer("❌ فایل نهایی پیدا نشد.")

    except Exception as e:
        await query.message.answer(f"🚫 خطا در تبدیل توضیحات:\n<code>{e}</code>")

    finally:
        if output_file and os.path.exists(output_file):
            try:
                os.remove(output_file)
            except Exception:
                pass

        if data[0] == "convert":
            if len(data) < 4:
                await query.message.answer("🚫 خطا: داده‌های ناکافی برای تبدیل.")
                return
            format_type, kind, video_id = data[1], data[2], data[3]

    if format_type == "cancel":
        # حذف پیام سوال تبدیل فرمت
        await query.message.delete()
        return

    # نمایش وضعیت در حال تبدیل
    await query.message.edit_text("⏳ در حال تبدیل فرمت هستیم...")

    # مسیر فایل قبلی را پیدا کن
    video_title = "output"
    for f in os.listdir(DOWNLOAD_DIR):
        if video_id in f:
            video_title = f
            break

    filepath = os.path.join(DOWNLOAD_DIR, video_title)

    if not os.path.exists(filepath):
        await query.message.edit_text("❌ فایل پیدا نشد برای تبدیل.")
        return

    try:
        # مسیر خروجی جدید
        output_file = None

        if format_type == "pdf":
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)  # استفاده از فونت پیش‌فرض
            pdf.multi_cell(0, 10, f"تبدیل از: {video_title}")
            output_file = os.path.join(DOWNLOAD_DIR, video_title + ".pdf")
            pdf.output(output_file)


        elif format_type == "txt":
            output_file = os.path.join(DOWNLOAD_DIR, video_title + ".txt")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(f"متن تبدیل شده از فایل {video_title}")

        elif format_type == "voice":
            output_file = os.path.join(DOWNLOAD_DIR, video_title + ".ogg")
            result = subprocess.run(
                ["ffmpeg", "-i", filepath, "-vn", "-acodec", "libopus", output_file],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                await query.message.edit_text(f"🚫 خطا در تبدیل به voice:\n<code>{result.stderr}</code>")
                return

        elif format_type == "srt":
            import requests
            info = {}  # فرض بر اینکه info قبلاً به‌درستی تعریف شده است
            subtitles = info.get("subtitles", {})
            if "en" in subtitles and subtitles["en"]:
                srt_url = subtitles["en"][0]["url"]
                srt_text = requests.get(srt_url).text
                output_file = os.path.join(DOWNLOAD_DIR, video_title + ".srt")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(srt_text)
            else:
                await query.message.answer("❌ زیرنویس انگلیسی برای این ویدیو موجود نیست.")
                return
        else:
            await query.message.answer("❌ فرمت نامعتبر است.")
            return

        if output_file and os.path.exists(output_file):
            final_file = FSInputFile(output_file, filename=os.path.basename(output_file))
            await query.message.delete()
            await query.message.answer_document(
                document=final_file,
                caption=f"✅ فایل تبدیل شده در قالب <b>{format_type.upper()}</b> آماده است.",
                parse_mode="HTML"
            )
        else:
            await query.message.edit_text("🚫 تبدیل موفق نبود یا فایل تولید نشد.")

    except Exception as e:
        await query.message.edit_text(f"🚫 خطا در تبدیل:\n<code>{e}</code>")

    finally:
        if output_file and os.path.exists(output_file):
            try:
                os.remove(output_file)
            except Exception:
                pass
