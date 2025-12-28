import os
import asyncio
import logging
from aiogram import Router, types
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, FSInputFile, CallbackQuery
from aiogram.filters import CommandStart
from concurrent.futures import ThreadPoolExecutor
from handlers.detector import detect_platform  # باید برای SoundCloud به‌روزرسانی شود
from yt_dlp.utils import sanitize_filename  # برای دانلود SoundCloud قابل استفاده است
from subprocess import run, CalledProcessError
import time
from urllib.parse import urlparse
from shutil import which
import subprocess
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, APIC  # برای تگ‌گذاری SoundCloud
import requests  # برای درخواست به API SoundCloud
from pydub import AudioSegment  # برای پردازش فایل‌های صوتی SoundCloud
import tempfile
from typing import Tuple, Optional, Dict, List, Union
from filelock import FileLock
from requests.exceptions import RequestException
import lyricsgenius  # برای متن آهنگ SoundCloud
from tempfile import NamedTemporaryFile

# --- SoundCloud-specific imports ---
import soundcloud
import scdl  
from demucs.separate import main as demucs_separate 
from audio_utils import separate_vocals
from openai import OpenaiError
import openai 

# --- ساخت بات ---
router = Router()
executor = ThreadPoolExecutor(max_workers=2)

# --- تنظیمات ---
BOT_TOKEN = os.getenv("BOT_TOKEN") or "8041920673:AAFhScBujoQx-48mLi7D-JnvfH9Z-bBxLNw"
SUPPORT_CHAT_ID = int(os.getenv("SUPPORT_CHAT_ID") or "8196909396")
DOWNLOAD_DIR = os.path.abspath("musics_download")
PROXY = "http://174.136.204.40:80"
FFMPEG_PATH = r"G:\zAll data (All Mine)\Codeing\ffmpeg\bin\ffmpeg.exe"
GENIUS_API_TOKEN = os.getenv("GENIUS_API_TOKEN")  # باید ست شده باشد

# --- SoundCloud Client ID ---
SOUNDCLOUD_CLIENT_ID = os.getenv("SOUNDCLOUD_CLIENT_ID") or "14ohwm8W6qaxxnP9HjCxZpu6FcKGACBJ"

# --- تنظیم لاگ ---
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


lock_path = os.path.join(DOWNLOAD_DIR, "directory.lock")
file_lock = FileLock(lock_path, timeout=10)

# --- دایرکتوری دانلود ---
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
output_path = os.path.join(DOWNLOAD_DIR)


# --- بررسی نصب FFmpeg ---
if not os.path.exists(FFMPEG_PATH):
    logger.error(f"FFmpeg در مسیر {FFMPEG_PATH} یافت نشد.")
    raise FileNotFoundError(f"FFmpeg در مسیر {FFMPEG_PATH} یافت نشد.")

# --- تابع بررسی لینک سانکلود ---
def validate_soundcloud_url(url: str) -> Tuple[bool, Optional[str], Optional[str]]:
    try:
        # بررسی خالی نبودن و نوع URL
        if not url or not isinstance(url, str):
            logger.error("لینک ورودی خالی یا نامعتبر است.")
            raise ValueError("لینک ورودی خالی یا نامعتبر است.")

        # تجزیه URL
        parsed = urlparse(url.strip())
        
        # بررسی دامنه SoundCloud
        if parsed.netloc != "soundcloud.com":
            logger.warning(f"دامنه غیرمجاز: {parsed.netloc}")
            return False, None, None

        # استخراج مسیرها
        parts = parsed.path.strip("/").split("/")
        valid_types = {"tracks", "sets", "playlists", "users"}  # انواع محتوا در SoundCloud

        # بررسی ساختار مسیر
        if len(parts) >= 2 and parts[0] in valid_types:
            logger.debug(f"لینک معتبر: نوع={parts[0]}, شناسه={parts[1]}")
            return True, parts[0], parts[1]
        elif len(parts) >= 1 and parts[0] in {"tracks", "sets"}:  # ساختار ساده‌تر برای آهنگ‌ها
            logger.debug(f"لینک معتبر: نوع={parts[0]}, شناسه={parts[0]}")
            return True, parts[0], parts[0]
        
        logger.warning(f"ساختار مسیر نامعتبر: {parsed.path}")
        return False, None, None

    except Exception as e:
        logger.error(f"خطا در اعتبارسنجی URL: {e}")
        return False, None, None

# فرمت زمان، مورد نیاز نمایش اطلاعات
def format_soundcloud_duration(seconds: Union[float, int]) -> str:
    try:
        # بررسی ورودی
        if not isinstance(seconds, (float, int)):
            logger.error("مقدار ورودی نامعتبر: {seconds}")
            raise ValueError("مقدار ورودی باید عدد باشد.")

        seconds = int(float(seconds))
        if seconds < 0:
            logger.error("مقدار ثانیه منفی است.")
            raise ValueError("مقدار ثانیه نمی‌تواند منفی باشد.")

        # محاسبه دقیقه و ثانیه
        minutes = seconds // 60
        secs = seconds % 60
        logger.debug(f"تبدیل {seconds} ثانیه به {minutes}:{secs:02d} برای SoundCloud")
        return f"{minutes}:{secs:02d}"

    except ValueError as e:
        logger.error(f"خطا در فرمت زمان برای SoundCloud: {e}")
        raise ValueError(f"خطا در فرمت زمان: {e}")
    except Exception as e:
        logger.error(f"خطای ناشناخته در فرمت زمان برای SoundCloud: {e}")
        raise RuntimeError(f"خطای ناشناخته: {e}")

#تابع استخراج اطلاعات آهنگ
def extract_soundcloud_track_info(track_id: str) -> Dict[str, Optional[str]]:
    try:
        # بررسی معتبر بودن track_id
        if not track_id or not isinstance(track_id, str):
            logger.error(f"شناسه ترک SoundCloud نامعتبر: {track_id}")
            raise ValueError("شناسه ترک نامعتبر است.")

        # ایجاد کلاینت SoundCloud (فرض بر استفاده از soundcloud-lib یا API)
        client = soundcloud.Client(client_id=SOUNDCLOUD_CLIENT_ID)
        track = client.get('/tracks/' + track_id)

        title = track.get("title", "بدون عنوان")
        if not title:
            logger.warning(f"عنوان ترک SoundCloud {track_id} یافت نشد.")

        artist = track.get("user", {}).get("username", "نامشخص")
        album_name = track.get("playlist", {}).get("title", "نامشخص") if track.get("playlist") else "نامشخص"
        thumbnail = track.get("artwork_url") or track.get("waveform_url")
        release_date = track.get("created_at", "نامشخص")
        duration_ms = track.get("duration", 0)
        duration_minutes = max(round(duration_ms / 1000 / 60, 1), 0.1)

        return {
            "title": title,
            "artist": artist,
            "album": album_name,
            "thumbnail": thumbnail,
            "release_date": release_date,
            "duration": duration_minutes
        }

    except RequestException as e:
        api_errors.inc()
        errors_total.inc()
        logger.error(f"خطا در درخواست API SoundCloud برای ترک {track_id}: {e}")
        return {
            "title": "نامشخص",
            "artist": "نامشخص",
            "album": "نامشخص",
            "thumbnail": None,
            "release_date": "نامشخص",
            "duration": 0
        }
    except ValueError as e:
        errors_total.inc()
        logger.error(f"خطای اعتبارسنجی برای ترک SoundCloud {track_id}: {e}")
        return {
            "title": "نامشخص",
            "artist": "نامشخص",
            "album": "نامشخص",
            "thumbnail": None,
            "release_date": "نامشخص",
            "duration": 0
        }
    except Exception as e:
        errors_total.inc()
        logger.error(f"خطای ناشناخته برای ترک SoundCloud {track_id}: {e}")
        return {
            "title": "نامشخص",
            "artist": "نامشخص",
            "album": "نامشخص",
            "thumbnail": None,
            "release_date": "نامشخص",
            "duration": 0
        }

# --- تابع دانلود با spotdl ---
def download_soundcloud(url: str, content_type: str, content_id: str) -> List[str]:
    # بررسی وجود scdl
    if not which("scdl"):
        logger.error("scdl در PATH یافت نشد.")
        raise FileNotFoundError("❌ scdl در محیط جاری پیدا نشد. لطفاً آن را نصب کنید.")

    # بررسی وجود FFmpeg
    if not FFMPEG_PATH or not os.path.exists(FFMPEG_PATH):
        logger.error(f"FFmpeg در مسیر {FFMPEG_PATH} یافت نشد.")
        raise FileNotFoundError(f"❌ FFmpeg یافت نشد: {FFMPEG_PATH}")

    # تنظیم مسیر خروجی
    output_path = os.path.abspath(os.path.join(DOWNLOAD_DIR, "%(title)s.%(ext)s"))

    # ساخت دستور scdl
    cmd = [
        "scdl",
        "-l", url,  # لینک SoundCloud
        "-o", output_path,
        "--format", "mp3",
        "--no-metadata",  # متادیتا بعداً اضافه می‌شود
    ]

    # افزودن FFmpeg به دستور
    cmd += ["--ffmpeg-path", FFMPEG_PATH]

    # افزودن پروکسی اگر وجود داشته باشد
    if PROXY and PROXY.strip():
        cmd += ["--proxy", PROXY]

    # تنظیم محیط
    env = os.environ.copy()
    env["PATH"] = f"{os.path.dirname(FFMPEG_PATH)};{env.get('PATH', '')}"

    # ایجاد دایرکتوری دانلود
    try:
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    except OSError as e:
        logger.error(f"خطا در ایجاد دایرکتوری {DOWNLOAD_DIR}: {e}")
        raise RuntimeError(f"⛔️ خطا در ایجاد دایرکتوری دانلود: {e}")

    # پاکسازی فایل‌های mp3 قدیمی با قفل
    with file_lock:
        for f in os.listdir(DOWNLOAD_DIR):
            if f.endswith(".mp3"):
                file_path = os.path.join(DOWNLOAD_DIR, f)
                try:
                    os.remove(file_path)
                    logger.debug(f"فایل قدیمی {f} حذف شد.")
                except (OSError, PermissionError) as e:
                    logger.warning(f"⚠️ خطا در حذف فایل قدیمی {f}: {e}")

    # بررسی نسخه scdl
    try:
        result = run(
            ["scdl", "--version"],
            capture_output=True,
            text=True,
            env=env,
            timeout=30
        )
        if result.returncode != 0:
            logger.error(f"خطا در بررسی نسخه scdl: {result.stderr.strip()}")
            raise RuntimeError("⛔️ خطا در بررسی نسخه scdl.")
    except TimeoutError:
        logger.error("زمان‌بندی بررسی نسخه scdl به اتمام رسید.")
        raise RuntimeError("⏳ زمان بررسی نسخه scdl به پایان رسید.")
    except CalledProcessError as e:
        logger.error(f"خطا در اجرای scdl --version: {e}")
        raise RuntimeError(f"⛔️ خطا در بررسی scdl: {e}")

    # اجرای دانلود
    start_time = time.time()
    try:
        process = run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=600
        )
        if process.returncode != 0:
            logger.error(f"اجرای scdl با خطا: {process.stderr.strip()}")
            raise RuntimeError(f"⛔️ خطا در اجرای scdl: {process.stderr.strip()}")

        logger.debug(f"خروجی scdl: {process.stdout.strip()}")

    except TimeoutError:
        errors_total.inc()
        logger.error("زمان اجرای scdl به اتمام رسید.")
        raise RuntimeError("⏳ زمان اجرای scdl به پایان رسید.")
    except CalledProcessError as e:
        errors_total.inc()
        logger.error(f"خطا در اجرای scdl: {e}")
        raise RuntimeError(f"⛔️ خطا در دانلود: {e}")
    except Exception as e:
        errors_total.inc()
        logger.error(f"خطای ناشناخته در اجرای scdl: {e}")
        raise RuntimeError(f"⛔️ خطای ناشناخته: {e}")

    # جمع‌آوری فایل‌های mp3
    files = []
    with file_lock:
        for f in os.listdir(DOWNLOAD_DIR):
            file_path = os.path.join(DOWNLOAD_DIR, f)
            if f.endswith(".mp3"):
                try:
                    # بررسی سلامت فایل
                    file_size = os.path.getsize(file_path)
                    if file_size < 50 * 1024:  # حداقل 50KB برای جلوگیری از فایل‌های خراب
                        logger.warning(f"فایل {f} خیلی کوچک است ({file_size} بایت).")
                        continue
                    files.append(file_path)
                    logger.info(f"فایل آماده: {file_path}")
                except OSError as e:
                    logger.warning(f"خطا در بررسی فایل {f}: {e}")

    if not files:
        logger.warning("هیچ فایل mp3 معتبر یافت نشد.")
        return []

    logger.info(f"دانلود با موفقیت در {time.time() - start_time:.2f} ثانیه تکمیل شد.")
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
        response = requests.get(cover_url, timeout=10)
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

# تعریف استثنای سفارشی برای Genius
class GeniusError(Exception):
    pass

def extract_lyrics_from_api(title: str, artist: str, api_token: str) -> str:
    # بررسی ورودی‌ها
    if not api_token or not isinstance(api_token, str):
        logger.error("توکن API Genius نامعتبر یا خالی است.")
        raise ValueError("توکن API Genius نامعتبر است.")
    
    if not title or not artist or not isinstance(title, str) or not isinstance(artist, str):
        logger.error(f"عنوان یا هنرمند نامعتبر است: title={title}, artist={artist}")
        raise ValueError("عنوان یا هنرمند نامعتبر است.")

    try:
        # مقداردهی اولیه Genius
        genius = lyricsgenius.Genius(api_token, timeout=10, retries=1)
        genius.verbose = False  # کاهش لاگ‌های غیرضروری
        logger.debug(f"جستجوی متن آهنگ: {title} توسط {artist}")

        # جستجوی آهنگ
        song = genius.search_song(title, artist)
        if song and hasattr(song, 'lyrics') and song.lyrics:
            logger.info(f"متن آهنگ برای {title} توسط {artist} یافت شد.")
            return song.lyrics.strip()
        
        logger.warning(f"متن آهنگ برای {title} توسط {artist} یافت نشد.")
        return ""

    except RequestException as e:
        api_errors.inc()
        logger.error(f"خطای شبکه در API Genius: {e}")
        raise GeniusError(f"خطای شبکه در API Genius: {e}")
    except Exception as e:
        errors_total.inc()
        logger.error(f"خطای ناشناخته در استخراج متن از Genius: {e}")
        raise RuntimeError(f"خطای ناشناخته در API Genius: {e}")

# استخراج متن، وابسته به فایل
def transcribe_lyrics_from_file(mp3_path: str) -> str:
    # بررسی وجود فایل MP3
    if not os.path.exists(mp3_path):
        logger.error(f"فایل MP3 در مسیر {mp3_path} یافت نشد.")
        raise FileNotFoundError(f"فایل MP3 یافت نشد: {mp3_path}")

    # بررسی وجود FFmpeg
    if not FFMPEG_PATH or not os.path.exists(FFMPEG_PATH):
        logger.error(f"FFmpeg در مسیر {FFMPEG_PATH} یافت نشد.")
        raise FileNotFoundError(f"FFmpeg یافت نشد: {FFMPEG_PATH}")

    temp_wav: Optional[str] = None
    try:
        # تبدیل MP3 به WAV
        try:
            audio = AudioSegment.from_mp3(mp3_path)
            temp_wav = mp3_path.replace(".mp3", "_speech.wav")
            audio.export(temp_wav, format="wav", parameters=["-ar", "16000"], ffmpeg=FFMPEG_PATH)
            logger.debug(f"فایل WAV در {temp_wav} ایجاد شد.")
        except Exception as e:
            logger.error(f"خطا در تبدیل MP3 به WAV: {e}")
            raise RuntimeError(f"خطا در تبدیل فایل به WAV: {e}")

        # بررسی وجود فایل WAV
        if not os.path.exists(temp_wav):
            logger.error(f"فایل WAV در {temp_wav} ایجاد نشد.")
            raise FileNotFoundError(f"فایل WAV ایجاد نشد: {temp_wav}")

        # استخراج متن با OpenAI Whisper
        try:
            with open(temp_wav, "rb") as f:
                transcript = openai.Audio.transcribe(
                    model="whisper-1",
                    file=f,
                    language="en"  # فرض زبان انگلیسی، قابل تنظیم
                )
                logger.info(f"متن آهنگ از {mp3_path} استخراج شد.")
                return transcript.get("text", "متن استخراج نشد.")
        except OpenaiError as e:
            logger.error(f"خطا در API OpenAI: {e}")
            raise RuntimeError(f"خطا در استخراج متن با Whisper: {e}")
        except Exception as e:
            logger.error(f"خطای ناشناخته در استخراج متن: {e}")
            raise RuntimeError(f"خطای ناشناخته در Whisper: {e}")

    except Exception as e:
        errors_total.inc()
        logger.error(f"خطا در فرآیند استخراج متن: {e}")
        return f"خطا در استخراج متن: {str(e)}"
    finally:
        # پاکسازی فایل موقت
        if temp_wav and os.path.exists(temp_wav):
            try:
                os.remove(temp_wav)
                logger.debug(f"فایل موقت {temp_wav} حذف شد.")
            except OSError as e:
                logger.warning(f"خطا در حذف فایل موقت {temp_wav}: {e}")

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
            [InlineKeyboardButton(text="🖐 خرید اکانت SoundCloud", callback_data="buy_account")]
        ])
        await message.answer(
            "به ربات SoundCloud خوش آمدید. یکی از گزینه‌ها را انتخاب کنید:",
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
            "🔗 لطفاً لینک آهنگ، آلبوم یا پلی‌لیست SoundCloud را ارسال کنید:",
            parse_mode="HTML"
        )
        await query.answer()
        logger.debug(f"کاربر {query.from_user.id}: درخواست لینک ارسال شد.")
    except Exception as e:
        logger.error(f"کاربر {query.from_user.id}: خطا در ارسال پیام درخواست لینک: {e}")
        await query.answer(f"❌ خطا: لطفاً دوباره تلاش کنید.", show_alert=True)
        raise RuntimeError(f"خطا در ارسال پیام: {e}")

@router.callback_query(lambda q: q.data == "info")
async def ask_for_info(query: CallbackQuery) -> None:
    try:
        await query.message.answer(
            "🔍 لینک هنرمند یا پلی‌لیست SoundCloud را ارسال کنید:",
            parse_mode="HTML"
        )
        await query.answer()
        logger.debug(f"درخواست لینک اطلاعات از کاربر {query.from_user.id} ارسال شد.")
    except Exception as e:
        logger.error(f"خطا در ارسال پیام درخواست اطلاعات: {e}")
        await query.answer(f"❌ خطا در پردازش درخواست: {str(e)}", show_alert=True)
        raise RuntimeError(f"خطا در ارسال پیام: {e}")

# --- هندلر اصلی دانلود اسپاتیفای ---
@router.message(lambda m: m.text and "soundcloud.com" in m.text.lower())
async def soundcloud_download_handler(message: types.Message) -> None:
    try:
        # بررسی وجود متن
        if not message.text:
            logger.warning(f"کاربر {message.from_user.id}: پیام غیرمتنی دریافت شد.")
            await message.answer("❌ لطفاً یک لینک متنی SoundCloud ارسال کنید.", parse_mode="HTML")
            return

        url = message.text.strip()
        logger.info(f"کاربر {message.from_user.id}: لینک دریافت‌شده: {url}")

        # بررسی پلتفرم
        platform = detect_platform(url)
        logger.info(f"کاربر {message.from_user.id}: پلتفرم تشخیص‌داده‌شده: {platform}")
        if platform != "soundcloud":
            await message.answer("❌ این لینک برای SoundCloud نیست.", parse_mode="HTML")
            return

        # اعتبارسنجی URL
        is_valid, content_type, content_id = validate_soundcloud_url(url)
        if not is_valid or not content_type or not content_id:
            logger.warning(f"کاربر {message.from_user.id}: لینک SoundCloud نامعتبر است: {url}")
            await message.answer("❌ لینک SoundCloud نامعتبر است.", parse_mode="HTML")
            return

        # پیام موقت
        msg = await message.reply("🔍 در حال دریافت اطلاعات...")
        try:
            await asyncio.sleep(1)
            await msg.delete()
        except Exception as e:
            logger.warning(f"کاربر {message.from_user.id}: خطا در حذف پیام موقت: {e}")
            await asyncio.sleep(0.5)  # تأخیر برای جلوگیری از نرخ بالای درخواست

        # پردازش بر اساس نوع محتوا
        if content_type in ("users", "playlists", "sets"):
            await show_info(message, content_type, content_id)
        else:
            await handle_soundcloud_download(message, url, content_type, content_id)

    except ValueError as e:
        logger.error(f"کاربر {message.from_user.id}: خطای اعتبارسنجی: {e}")
        await message.answer(f"❌ خطا: {str(e)}", parse_mode="HTML")
    except RequestException as e:
        logger.error(f"کاربر {message.from_user.id}: خطا در API SoundCloud: {e}")
        await message.answer(f"❌ خطا در پردازش لینک: {str(e)}", parse_mode="HTML")
    except RuntimeError as e:
        logger.error(f"کاربر {message.from_user.id}: خطا در پردازش لینک: {e}")
        await message.answer(f"❌ خطا در پردازش: {str(e)}", parse_mode="HTML")
    except Exception as e:
        logger.error(f"کاربر {message.from_user.id}: خطای ناشناخته: {e}")
        await message.answer(f"❌ خطای ناشناخته: {str(e)}", parse_mode="HTML")

# --- دانلود موزیک با spotdl ---
@router.message(lambda m: m.text and "soundcloud.com" in m.text.lower())
async def handle_soundcloud_download(message: types.Message, url: str, content_type: str, content_id: str) -> None:
    user_id = message.from_user.id
    try:
        info: Optional[dict] = None
        cover_msg: Optional[types.Message] = None

        inline_buttons = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="🎵 اضافه به پلی‌لیست", callback_data="add_to_playlist"),
                InlineKeyboardButton(text="📝 دریافت متن آهنگ", callback_data="lyrics")
            ],
            [
                InlineKeyboardButton(text="🎙 جدا سازی وکال", callback_data="separate_vocal"),
                InlineKeyboardButton(text="🛒 خرید اکانت SoundCloud", callback_data="buy_account")
            ]
        ])

        if content_type == "tracks":
            info = await asyncio.get_event_loop().run_in_executor(executor, extract_soundcloud_track_info, content_id)
            if not info or not info.get("title"):
                logger.error(f"کاربر {user_id}: اطلاعات آهنگ برای شناسه {content_id} ناقص است.")
                raise ValueError("اطلاعات آهنگ ناقص است.")

            total_seconds = int(info["duration"] * 60)
            minutes = total_seconds // 60
            seconds = total_seconds % 60
            duration_str = f"{minutes}:{seconds:02d}"

            caption = (
                f"<b>{info['artist']} – {info['title']}</b>\n"
                f"Album: {info['album']}\n"
                f"Duration: {duration_str}\n"
                f"Release date: {info['release_date']}"
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
                await asyncio.sleep(0.5)  # تأخیر برای جلوگیری از نرخ بالا
            except Exception as e:
                logger.error(f"کاربر {user_id}: خطا در ارسال کاور یا کپشن: {e}")
                cover_msg = await message.answer(
                    text=caption,
                    reply_markup=inline_buttons,
                    parse_mode="HTML"
                )

        progress_msg = await message.answer("⏳ در حال دانلود آهنگ از SoundCloud... لطفاً صبر کنید", parse_mode="HTML")
        try:
            files = await asyncio.get_event_loop().run_in_executor(
                executor, download_soundcloud, url, content_type, content_id
            )
        except RequestException as e:
            await progress_msg.delete()
            logger.error(f"کاربر {user_id}: خطا در API SoundCloud: {e}")
            raise RuntimeError(f"خطا در دانلود: {e}")
        except Exception as e:
            await progress_msg.delete()
            logger.error(f"کاربر {user_id}: خطا در دانلود فایل‌ها: {e}")
            raise RuntimeError(f"خطا در دانلود: {e}")

        try:
            await progress_msg.delete()
        except Exception as e:
            logger.warning(f"کاربر {user_id}: خطا در حذف پیام موقت: {e}")
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
                logger.warning(f"کاربر {user_id}: فایل {path} یافت نشد.")
                continue

            try:
                safe_filename = sanitize_filename(os.path.basename(path))
                safe_path = os.path.join(DOWNLOAD_DIR, safe_filename)

                try:
                    os.rename(path, safe_path)
                except (OSError, FileExistsError) as e:
                    logger.warning(f"کاربر {user_id}: خطا در تغییر نام فایل {path}: {e}")
                    safe_path = path

                if info and info.get("thumbnail"):
                    try:
                        embed_cover(safe_path, info["thumbnail"])
                    except Exception as e:
                        logger.warning(f"کاربر {user_id}: خطا در افزودن کاور به {safe_path}: {e}")

                try:
                    await message.answer_document(
                        document=FSInputFile(safe_path, filename=safe_filename),
                        disable_notification=True
                    )
                    sent_success = True
                    await asyncio.sleep(0.5)  # تأخیر برای مدیریت نرخ
                except Exception as e:
                    logger.error(f"کاربر {user_id}: خطا در ارسال فایل {safe_path}: {e}")
                    await message.answer(
                        f"❌ خطا در ارسال فایل: {safe_filename}",
                        parse_mode="HTML"
                    )

                try:
                    os.remove(safe_path)
                except OSError as e:
                    logger.warning(f"کاربر {user_id}: خطا در حذف فایل {safe_path}: {e}")

            except Exception as e:
                logger.error(f"کاربر {user_id}: خطا در پردازش فایل {path}: {e}")
                await message.answer(
                    f"❌ خطا در پردازش فایل {os.path.basename(path)}",
                    parse_mode="HTML"
                )

        if sent_success and cover_msg:
            await message.answer(
                "📥 دانلود با موفقیت انجام شد.\nآیا امکانات بیشتری نیاز دارید؟",
                reply_to_message_id=cover_msg.message_id,
                parse_mode="HTML"
            )

    except ValueError as e:
        logger.error(f"کاربر {user_id}: خطای اعتبارسنجی: {e}")
        await message.answer(f"❌ خطا: اطلاعات آهنگ ناقص است.", parse_mode="HTML")
    except RequestException as e:
        logger.error(f"کاربر {user_id}: خطا در API SoundCloud: {e}")
        await message.answer(f"❌ خطا در دانلود: {str(e)}", parse_mode="HTML")
    except RuntimeError as e:
        logger.error(f"کاربر {user_id}: خطا در دانلود: {e}")
        await message.answer(f"❌ خطا در دانلود: {str(e)}", parse_mode="HTML")
    except Exception as e:
        logger.error(f"کاربر {user_id}: خطای ناشناخته: {e}")
        await message.answer(f"❌ خطای ناشناخته: {str(e)}", parse_mode="HTML")

# --- اطلاعات پلی‌لیست یا هنرمند ---
async def show_info(message: types.Message, content_type: str, content_id: str) -> None:
    # بررسی ورودی‌ها
    if content_type not in ("users", "playlists", "sets"):
        logger.error(f"نوع محتوای نامعتبر: {content_type}")
        raise ValueError(f"نوع محتوای نامعتبر: {content_type}")
    if not content_id or not isinstance(content_id, str):
        logger.error(f"شناسه محتوای نامعتبر: {content_id}")
        raise ValueError(f"شناسه محتوای نامعتبر: {content_id}")

    try:
        if content_type == "users":
            await message.answer("🎨 در حال دریافت اطلاعات هنرمند...")

            client = soundcloud.Client(client_id=SOUNDCLOUD_CLIENT_ID)
            artist = client.get(f'/users/{content_id}')
            name = artist.get("username", "نامشخص")
            followers = artist.get("followers_count", 0)
            bio = artist.get("description", "موجود نیست")
            image = artist.get("avatar_url") or artist.get("visuals", {}).get("visual", [{}])[0].get("entry", {}).get("url")
            track_count = artist.get("track_count", 0)
            wiki_link = f"https://en.wikipedia.org/wiki/{name.replace(' ', '_')}"
            playlist_name = f"This is {name}"
            playlist_url = f"https://soundcloud.com/{name}/sets"

            caption = (
                f"<b>👤 {name}</b>\n"
                f"🔊 تعداد ترک‌ها: {track_count}\n"
                f"❤️ دنبال‌کنندگان: {followers:,}\n"
                f"📝 بیوگرافی: {bio}\n"
                f"🌍 کشور: نامشخص\n"  # SoundCloud اطلاعات کشور را به‌طور مستقیم ارائه نمی‌دهد
                f"📚 <a href='{wiki_link}'>ویکی‌پدیا</a>\n"
                f"🎧 <a href='{playlist_url}'>پلی‌لیست: {playlist_name}</a>\n"
                f"🔗 <a href='https://soundcloud.com/{name}'>مشاهده در SoundCloud</a>"
            )

            try:
                if image:
                    await message.answer_photo(photo=image, caption=caption, parse_mode="HTML")
                else:
                    await message.answer(caption, parse_mode="HTML")
            except Exception as e:
                logger.error(f"خطا در ارسال پیام هنرمند: {e}")
                await message.answer(caption, parse_mode="HTML")

        elif content_type in ("playlists", "sets"):
            await message.answer("📦 در حال دریافت اطلاعات پلی‌لیست...")

            client = soundcloud.Client(client_id=SOUNDCLOUD_CLIENT_ID)
            data = client.get(f'/playlists/{content_id}')
            name = data.get("title", "نامشخص")
            owner = data.get("user", {}).get("username", "نامشخص")
            release_date = data.get("created_at", "نامشخص")[:10] if data.get("created_at") else "نامشخص"
            total_tracks = data.get("track_count", 0)
            tracks = data.get("tracks", [])
            top_track = (
                max(tracks, key=lambda x: x.get("playback_count", 0)).get("title", "نامشخص")
                if tracks else "نامشخص"
            )
            image = data.get("artwork_url") or data.get("tracks", [{}])[0].get("artwork_url")

            user_rating = "⭐️ 8.7 / 10"  # مقدار پیش‌فرض

            caption = (
                f"<b>🎶 {name}</b>\n"
                f"👤 {owner}\n"
                f"📅 تاریخ انتشار: {release_date}\n"
                f"🔢 تعداد ترک‌ها: {total_tracks}\n"
                f"{user_rating}\n"
                f"🔝 ترک برتر: {top_track}\n"
                f"🔗 <a href='https://soundcloud.com/playlists/{content_id}'>مشاهده در SoundCloud</a>"
            )

            try:
                if image:
                    await message.answer_photo(photo=image, caption=caption, parse_mode="HTML")
                else:
                    await message.answer(caption, parse_mode="HTML")
            except Exception as e:
                logger.error(f"خطا در ارسال پیام پلی‌لیست: {e}")
                await message.answer(caption, parse_mode="HTML")

    except RequestException as e:
        api_errors.inc()
        errors_total.inc()
        logger.error(f"خطا در درخواست API SoundCloud: {e}")
        await message.answer(f"❌ خطا در دریافت اطلاعات: {str(e)}", parse_mode="HTML")
    except ValueError as e:
        errors_total.inc()
        logger.error(f"خطای اعتبارسنجی: {e}")
        await message.answer(f"❌ خطا: {str(e)}", parse_mode="HTML")
    except Exception as e:
        errors_total.inc()
        logger.error(f"خطای ناشناخته در پردازش اطلاعات: {e}")
        await message.answer(f"❌ خطای ناشناخته: {str(e)}", parse_mode="HTML")

@router.callback_query(lambda q: q.data == "separate_vocal")
async def handle_separate_vocal(query: CallbackQuery) -> None:
    message = query.message
    user_id = query.from_user.id

    try:
        await query.answer()
        await message.reply("🔧 در حال جداسازی وکال و بیت، لطفاً چند لحظه صبر کنید...", parse_mode="HTML")

        # دریافت فایل موسیقی از تاریخچه
        documents = [m for m in await message.chat.get_history(limit=5) if m.document]
        if not documents:
            logger.warning(f"کاربر {user_id}: فایل موسیقی در تاریخچه یافت نشد.")
            await message.reply("❌ فایل موسیقی یافت نشد.", parse_mode="HTML")
            raise ValueError("فایل موسیقی یافت نشد.")

        doc = documents[0]
        file_path = os.path.join(DOWNLOAD_DIR, f"{doc.document.file_id}.mp3")
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)

        # دانلود فایل
        try:
            with open(file_path, 'wb') as f:
                await doc.document.download(destination_file=f)
            logger.debug(f"کاربر {user_id}: فایل موسیقی در {file_path} دانلود شد.")
        except Exception as e:
            logger.error(f"کاربر {user_id}: خطا در دانلود فایل موسیقی: {e}")
            raise RuntimeError(f"خطا در دانلود فایل: {e}")

        # جداسازی وکال و بیت
        try:
            vocal_mp3, instrumental_mp3 = await asyncio.get_event_loop().run_in_executor(
                None, separate_vocals, file_path
            )
            logger.info(f"کاربر {user_id}: وکال و بیت از {file_path} جدا شدند.")
        except Exception as e:
            logger.error(f"کاربر {user_id}: خطا در جداسازی وکال: {e}")
            raise RuntimeError(f"خطا در جداسازی وکال: {e}")

        # ارسال فایل‌ها
        try:
            with open(vocal_mp3, 'rb') as vocal_file, open(instrumental_mp3, 'rb') as instrumental_file:
                await message.reply_document(
                    document=types.FSInputFile(vocal_mp3, filename=os.path.basename(vocal_mp3)),
                    caption="🎙 وکال",
                    parse_mode="HTML"
                )
                await message.reply_document(
                    document=types.FSInputFile(instrumental_mp3, filename=os.path.basename(instrumental_mp3)),
                    caption="🎸 بیت (Instrumental)",
                    parse_mode="HTML"
                )
            logger.info(f"کاربر {user_id}: فایل‌های وکال و بیت ارسال شدند.")
        except Exception as e:
            logger.error(f"کاربر {user_id}: خطا در ارسال فایل‌ها: {e}")
            raise RuntimeError(f"خطا در ارسال فایل‌ها: {e}")

    except ValueError as e:
        errors_total.inc()
        logger.error(f"کاربر {user_id}: خطای اعتبارسنجی: {e}")
        await message.reply(f"❌ خطا: {str(e)}", parse_mode="HTML")
    except RuntimeError as e:
        errors_total.inc()
        logger.error(f"کاربر {user_id}: خطا در پردازش: {e}")
        await message.reply(f"❌ خطا در عملیات: {str(e)}", parse_mode="HTML")
    except Exception as e:
        errors_total.inc()
        logger.error(f"کاربر {user_id}: خطای ناشناخته: {e}")
        await message.reply(f"❌ خطای ناشناخته: {str(e)}", parse_mode="HTML")
    finally:
        # پاکسازی فایل‌ها
        for file in [file_path, vocal_mp3, instrumental_mp3]:
            if 'file' in locals() and file and os.path.exists(file):
                try:
                    os.remove(file)
                    logger.debug(f"کاربر {user_id}: فایل موقت {file} حذف شد.")
                except OSError as e:
                    logger.warning(f"کاربر {user_id}: خطا در حذف فایل موقت {file}: {e}")

@router.callback_query(lambda q: q.data == "lyrics")
async def handle_lyrics(query: CallbackQuery) -> None:
    message = query.message
    await query.answer()

    await message.reply("🔍 در حال بررسی متن موسیقی...")

    try:
        # بررسی وجود توکن Genius
        if not GENIUS_API_TOKEN:
            logger.error("توکن API Genius تنظیم نشده است.")
            raise ValueError("توکن API Genius تنظیم نشده است.")

        # دریافت فایل موسیقی از تاریخچه چت
        documents = [m for m in await message.chat.get_history(limit=10) if m.document]
        if not documents:
            logger.warning("فایل موسیقی در تاریخچه یافت نشد.")
            await message.reply("❌ فایل موسیقی برای استخراج متن یافت نشد.")
            return

        doc = documents[0]
        file_path = os.path.join(DOWNLOAD_DIR, f"{doc.document.file_id}.mp3")
        os.makedirs(DOWNLOAD_DIR, exist_ok=True)

        # دانلود فایل
        try:
            with open(file_path, 'wb') as f:
                await doc.document.download(destination_file=f)
            logger.debug(f"فایل موسیقی در {file_path} دانلود شد.")
        except Exception as e:
            logger.error(f"خطا در دانلود فایل موسیقی: {e}")
            raise RuntimeError(f"خطا در دانلود فایل: {e}")

        mp3_path = file_path
        caption = message.caption or ""
        title, artist = "نامشخص", "نامشخص"

        # استخراج عنوان و هنرمند از کپشن
        if "–" in caption:
            try:
                artist, title = [x.strip() for x in caption.split("–", 1)]
                if not title or not artist:
                    logger.warning("عنوان یا هنرمند در کپشن نامعتبر است.")
                    title, artist = "نامشخص", "نامشخص"
            except ValueError:
                logger.warning("خطا در تفکیک کپشن برای عنوان و هنرمند.")
                title, artist = "نامشخص", "نامشخص"

        # تلاش برای دریافت متن از API Genius
        try:
            lyrics = extract_lyrics_from_api(title, artist, GENIUS_API_TOKEN)
            if lyrics:
                await message.reply(
                    f"🎶 متن آهنگ یافت شد:\n\n<code>{lyrics[:4000]}</code>",  # محدود کردن طول برای تلگرام
                    parse_mode="HTML"
                )
                logger.info(f"متن آهنگ برای {title} توسط {artist} از Genius دریافت شد.")
            else:
                await message.reply("🎧 متن در Genius یافت نشد. در حال استخراج از فایل صوتی...")
                text = transcribe_lyrics_from_file(mp3_path)
                await message.reply(
                    f"📝 متن استخراج‌شده:\n\n<code>{text[:4000]}</code>",  # محدود کردن طول
                    parse_mode="HTML"
                )
                logger.info(f"متن آهنگ از فایل {mp3_path} استخراج شد.")
        except Exception as e:
            logger.error(f"خطا در استخراج متن آهنگ: {e}")
            await message.reply(f"❌ خطا در دریافت متن: {str(e)}", parse_mode="HTML")

    except ValueError as e:
        errors_total.inc()
        logger.error(f"خطای اعتبارسنجی: {e}")
        await message.reply(f"❌ خطا: {str(e)}", parse_mode="HTML")
    except RuntimeError as e:
        errors_total.inc()
        logger.error(f"خطا در فرآیند استخراج: {e}")
        await message.reply(f"❌ خطا در دریافت متن: {str(e)}", parse_mode="HTML")
    except Exception as e:
        errors_total.inc()
        logger.error(f"خطای ناشناخته در دریافت متن: {e}")
        await message.reply(f"❌ خطای ناشناخته: {str(e)}", parse_mode="HTML")
    finally:
        # پاکسازی فایل
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.debug(f"فایل موقت {file_path} حذف شد.")
            except OSError as e:
                logger.warning(f"خطا در حذف فایل موقت {file_path}: {e}")

@router.callback_query(lambda c: c.data == "add_to_playlist")
async def add_to_playlist_callback(query: CallbackQuery) -> None:
    try:
        await query.answer("✅ آهنگ به پلی‌لیست شما اضافه شد.", show_alert=True)
        logger.info(f"کاربر {query.from_user.id} درخواست افزودن به پلی‌لیست را اجرا کرد.")
    except Exception as e:
        errors_total.inc()
        logger.error(f"خطا در پردازش درخواست افزودن به پلی‌لیست: {e}")
        await query.answer(f"❌ خطا در افزودن به پلی‌لیست: {str(e)}", show_alert=True)
        raise RuntimeError(f"خطا در پردازش درخواست: {e}")

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
        errors_total.inc()
        logger.error(f"خطای اعتبارسنجی: {e}")
        await query.answer(f"❌ خطا: لینک نامعتبر است.", show_alert=True)
    except Exception as e:
        errors_total.inc()
        logger.error(f"خطای ناشناخته در پردازش درخواست پیشنهاد: {e}")
        await query.answer(f"❌ خطای ناشناخته: {str(e)}", show_alert=True)
        raise RuntimeError(f"خطا در پردازش درخواست: {e}")

@router.callback_query()
async def handle_unhandled_callback(query: types.CallbackQuery) -> None:
    logger.warning(f"کاربر {query.from_user.id}: کال‌بک ناشناخته: {query.data}")
    await query.answer("❌ درخواست ناشناخته است.", show_alert=True)

# جداسازی وکال، ویژگی اضافی
def separate_soundcloud_vocals(mp3_path: str) -> Tuple[str, str]:
    # بررسی وجود Demucs
    if not which("demucs"):
        logger.error("Demucs در PATH یافت نشد.")
        raise FileNotFoundError("❌ Demucs یافت نشد. لطفاً آن را نصب کنید.")

    # بررسی وجود FFmpeg
    if not FFMPEG_PATH or not os.path.exists(FFMPEG_PATH):
        logger.error(f"FFmpeg در مسیر {FFMPEG_PATH} یافت نشد.")
        raise FileNotFoundError(f"❌ FFmpeg یافت نشد: {FFMPEG_PATH}")

    # بررسی وجود فایل MP3
    if not os.path.exists(mp3_path):
        logger.error(f"فایل MP3 در مسیر {mp3_path} یافت نشد.")
        raise FileNotFoundError(f"فایل MP3 یافت نشد: {mp3_path}")

    # ایجاد دایرکتوری موقت
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()
        logger.debug(f"دایرکتوری موقت ایجاد شد برای SoundCloud: {temp_dir}")

        # اجرای جداسازی با Demucs
        cmd = ["demucs", "--two-stems=vocals", mp3_path, "-o", temp_dir]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            logger.debug(f"خروجی Demucs برای SoundCloud: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"خطا در اجرای Demucs برای SoundCloud: {e.stderr}")
            raise RuntimeError(f"خطا در جداسازی وکال از SoundCloud: {e.stderr}")

        # مسیر فایل‌های خروجی
        stem_dir = os.path.join(temp_dir, "htdemucs", os.path.basename(mp3_path)[:-4])
        vocal_path = os.path.join(stem_dir, "vocals.wav")
        instrumental_path = os.path.join(stem_dir, "no_vocals.wav")

        # بررسی وجود فایل‌های خروجی
        for path in [vocal_path, instrumental_path]:
            if not os.path.exists(path):
                logger.error(f"فایل خروجی Demucs برای SoundCloud یافت نشد: {path}")
                raise FileNotFoundError(f"فایل خروجی Demucs یافت نشد: {path}")

        # تبدیل WAV به MP3
        vocal_mp3 = vocal_path.replace(".wav", ".mp3")
        instrumental_mp3 = instrumental_path.replace(".wav", ".mp3")

        try:
            AudioSegment.from_wav(vocal_path).export(
                vocal_mp3,
                format="mp3",
                parameters=["-q:a", "2"],
                ffmpeg=FFMPEG_PATH
            )
            AudioSegment.from_wav(instrumental_path).export(
                instrumental_mp3,
                format="mp3",
                parameters=["-q:a", "2"],
                ffmpeg=FFMPEG_PATH
            )
            logger.info(f"فایل‌های MP3 برای SoundCloud ایجاد شدند: {vocal_mp3}, {instrumental_mp3}")
        except Exception as e:
            logger.error(f"خطا در تبدیل WAV به MP3 برای SoundCloud: {e}")
            raise RuntimeError(f"خطا در تبدیل فایل‌ها به MP3: {e}")

        # پاکسازی فایل‌های WAV
        for file in [vocal_path, instrumental_path]:
            try:
                if os.path.exists(file):
                    os.remove(file)
                    logger.debug(f"فایل موقت برای SoundCloud {file} حذف شد.")
            except OSError as e:
                logger.warning(f"خطا در حذف فایل موقت برای SoundCloud {file}: {e}")

        return vocal_mp3, instrumental_mp3

    except Exception as e:
        errors_total.inc()
        logger.error(f"خطای ناشناخته در جداسازی وکال برای SoundCloud: {e}")
        raise RuntimeError(f"خطای ناشناخته: {e}")
    finally:
        # پاکسازی دایرکتوری موقت
        if temp_dir and os.path.exists(temp_dir):
            try:
                for root, _, files in os.walk(temp_dir, topdown=False):
                    for file in files:
                        os.remove(os.path.join(root, file))
                    os.rmdir(root)
                os.rmdir(temp_dir)
                logger.debug(f"دایرکتوری موقت برای SoundCloud {temp_dir} حذف شد.")
            except OSError as e:
                logger.warning(f"خطا در حذف دایرکتوری موقت برای SoundCloud {temp_dir}: {e}")

def get_track_inline_buttons(track_url: str) -> InlineKeyboardMarkup:
    try:
        # بررسی ورودی
        if not track_url or not isinstance(track_url, str):
            logger.error(f"لینک ترک نامعتبر: {track_url}")
            raise ValueError("لینک ترک نامعتبر است.")

        # بررسی ساده برای اطمینان از فرمت لینک
        if "soundcloud.com" not in track_url.lower():
            logger.warning(f"لینک غیرمرتبط با SoundCloud: {track_url}")
            raise ValueError("لینک باید مربوط به SoundCloud باشد.")

        keyboard = InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="🎵 اضافه به پلی‌لیست", callback_data="add_to_playlist"),
                InlineKeyboardButton(text="🛒 خرید اکانت SoundCloud", url="https://t.me/YOUR_SUPPORT_BOT")
            ],
            [
                InlineKeyboardButton(text="🎙 جدا سازی وکال / بیت", callback_data="split_vocal"),
                InlineKeyboardButton(text="📝 دریافت متن آهنگ", callback_data="lyrics")
            ],
            [
                InlineKeyboardButton(text="🎧 پیشنهاد آهنگ مشابه", callback_data=f"suggest|{track_url}")
            ]
        ])
        logger.debug(f"دکمه‌های اینلاین برای {track_url} ایجاد شد.")
        return keyboard

    except ValueError as e:
        errors_total.inc()
        logger.error(f"خطا در ایجاد دکمه‌های اینلاین: {e}")
        raise ValueError(f"خطا در ایجاد دکمه‌ها: {e}")
    except Exception as e:
        errors_total.inc()
        logger.error(f"خطای ناشناخته در ایجاد دکمه‌های اینلاین: {e}")
        raise RuntimeError(f"خطای ناشناخته: {e}")

@router.callback_query(lambda c: c.data == "split_vocal")
async def split_vocal_callback(query: CallbackQuery) -> None:
    try:
        await query.answer("🎙 جداسازی وکال هنوز فعال نشده است.", show_alert=True)
        logger.info(f"کاربر {query.from_user.id} درخواست جداسازی وکال را اجرا کرد (غیرفعال).")
    except Exception as e:
        errors_total.inc()
        logger.error(f"خطا در پردازش درخواست جداسازی وکال: {e}")
        await query.answer(f"❌ خطا در پردازش درخواست: {str(e)}", show_alert=True)
        raise RuntimeError(f"خطا در پردازش درخواست: {e}")

@router.callback_query(lambda c: c.data == "lyrics")
async def lyrics_callback(query: CallbackQuery):
    await query.answer("📝 متن آهنگ موجود نیست یا در حال توسعه است.", show_alert=True)
