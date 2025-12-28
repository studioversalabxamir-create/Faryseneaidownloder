#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sqlite3
import os
import re
import shutil
import logging
import requests
from datetime import datetime, timedelta
from contextlib import contextmanager
from typing import Optional, Tuple, List
from flask import Flask, jsonify, render_template
from typing import Optional, Dict
import sys

# -----------------------
# کانفیگ
# -----------------------
DB_PATH = "bot_cache.db"
CACHE_DIR = "cache"
MAX_CACHE_SIZE = 1024 * 1024 * 1024  # 1GB
CACHE_EXPIRY_DAYS = 7
from config import BOT_TOKEN, PROXY as PROXY_URL
# مسیر فایل دیتابیس (مکان نسبی به همین فایل)
DB_FILENAME = "bot_cache.db"
DB_PATH = os.path.join(os.path.dirname(__file__), DB_FILENAME)
# تنظیمات پروکسی از config (خالی باشد یعنی بدون پروکسی)

# -----------------------
# لاگر
# -----------------------
logger = logging.getLogger("db_manager")
if not logger.handlers:
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(sh)
    fh = logging.FileHandler("db_manager.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)
logger.setLevel(logging.DEBUG)


# -----------------------
# Flask app
# -----------------------
app = Flask(__name__)

# -----------------------
# کمک‌کننده‌ها
# -----------------------
def sanitize_filename(s: str) -> str:
    """فایل‌نیم امن بساز."""
    s = str(s)
    s = re.sub(r"[^\w\-_\.]", "_", s)
    return s[:250]

def row_get(row, key, default=None):
    try:
        return row[key]
    except (KeyError, IndexError):
        # sqlite3.Row در صورت نبودن کلید KeyError می‌دهد
        return default


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def get_last_file(user_id: int) -> Optional[Dict]:
    """
    آخرین فایلی که کاربر ذخیره کرده رو برمی‌گردونه.
    اگر فایل روی دیسک موجود نباشه، cache_path = None داده میشه.
    خروجی یک دیکشنری با تمام اطلاعات فایل یا None در صورت نبود رکورد.
    """

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("""
            SELECT id, file_id, artist, title, url, thumbnail,
                   path, cache_path, file_size, status, duration,
                   created_at, last_accessed
            FROM files
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (user_id,))

        row = cur.fetchone()
        if not row:
            return None

        # اطلاعات پایه
        file_data = dict(row)

        # انتخاب مسیر برای چک
        cache_path = row["cache_path"] or row["path"]

        if cache_path and os.path.exists(cache_path):
            # فایل موجوده → به‌روزرسانی last_accessed
            cur.execute("UPDATE files SET last_accessed = datetime('now') WHERE id = ?", (row["id"],))
            conn.commit()
            file_data["cache_path"] = cache_path
        else:
            # فایل وجود نداره → مقدار None بذاریم
            file_data["cache_path"] = None

        return file_data

    except Exception as e:
        logger.error(f"Error in get_last_file: {e}")
        return None
    finally:
        conn.close()


@contextmanager
def get_db_conn():
    """Context manager برای ارتباط با دیتابیس"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        # فعال کردن foreign key اگر نیاز شد (اینجا معمولا کاربردی ندارد اما خوب است)
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
    finally:
        conn.close()


def _columns_of_table(conn: sqlite3.Connection, table_name: str):
    """برمی‌گرداند مجموعه‌ای از نام ستون‌ها در جدول مشخص"""
    cur = conn.execute(f"PRAGMA table_info({table_name})")
    return {row["name"] for row in cur.fetchall()}


def _add_column_if_missing(conn: sqlite3.Connection, table_name: str, column_name: str, column_type: str, default_sql: str = None):
    """اگر ستون وجود نداشت به جدول اضافه می‌کند."""
    cols = _columns_of_table(conn, table_name)
    if column_name in cols:
        return False
    sql = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
    # اگر مقدار پیش‌فرض SQL می‌خواهیم اضافه کنیم، می‌توانیم default_sql را الحاق کنیم
    if default_sql:
        sql += f" DEFAULT ({default_sql})"
    logger.info(f"Adding column `{column_name}` to `{table_name}`")
    conn.execute(sql)
    return True


def init_db():
    """
    ایجاد یا بروزرسانی اسکیمای دیتابیس.
    - اگر جدول files وجود نداشته باشد آن را ایجاد می‌کند با ستون‌های کامل.
    - اگر جدول وجود داشته باشد اما ستون‌هایی کم داشته باشد آن‌ها را اضافه می‌کند.
    """
    try:
        # مطمئن شو پوشه دیتابیس وجود دارد
        os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)

        with get_db_conn() as conn:
            cur = conn.cursor()

            # ایجاد جدول کامل (اگر وجود ندارد)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    file_id TEXT UNIQUE NOT NULL,
                    content_type TEXT,
                    artist TEXT,
                    title TEXT,
                    url TEXT,
                    thumbnail TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    last_accessed TEXT DEFAULT (datetime('now')),
                    path TEXT,
                    cache_path TEXT,
                    file_size INTEGER,
                    status TEXT,
                    duration INTEGER,
                    tg_file_id TEXT,
                    album TEXT,
                    release_date TEXT,
                    popularity INTEGER,
                    genre TEXT,
                    total_tracks INTEGER,
                    label TEXT,
                    top_track TEXT
                )
            """)

            # ایجاد جدول کاربران برای ویژگی‌های روزانه
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY,
                    daily_songs_enabled BOOLEAN DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now')),
                    last_active TEXT DEFAULT (datetime('now'))
                )
            """)

            # ایندکس‌ها
            cur.execute("CREATE INDEX IF NOT EXISTS idx_files_user ON files(user_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_files_fileid ON files(file_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_files_created ON files(created_at)")

            # اطمینان از وجود همه ستون‌های ضروری (برای دیتابیس‌های قدیمی)
            required_columns = {
                "content_type": "TEXT",
                "thumbnail": "TEXT",
                "path": "TEXT",
                "cache_path": "TEXT",
                "file_size": "INTEGER",
                "status": "TEXT",
                "last_accessed": "TEXT",
                "created_at": "TEXT",
                "duration": "INTEGER",
                "tg_file_id": "TEXT",  # برای ذخیره file_id تلگرام جهت آپلود سریع
                "album": "TEXT",
                "release_date": "TEXT",
                "popularity": "INTEGER",
                "genre": "TEXT",
                "total_tracks": "INTEGER",
                "label": "TEXT",
                "top_track": "TEXT"
            }

            for col, coltype in required_columns.items():
                try:
                    _add_column_if_missing(conn, "files", col, coltype)
                except sqlite3.OperationalError as e:
                    # در مواردی که ستون از قبل وجود داشته باشه یا SQLite اجازه نده
                    logger.warning(f"Could not add column {col}: {e}")

            conn.commit()

        logger.info(f"✅ Database initialized/updated successfully at: {DB_PATH}")

    except sqlite3.Error as e:
        logger.exception(f"❌ SQLite error during init_db: {e}")
        raise
    except Exception as e:
        logger.exception(f"❌ Unexpected error during init_db: {e}")
        raise



def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def parse_sqlite_ts(ts) -> Optional[datetime]:
    """سریعا انواع فرمت‌های زمان sqlite رو تبدیل به datetime می‌کنه."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts)
    if isinstance(ts, str):
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(ts, fmt)
            except Exception:
                continue
    return None

# -----------------------
# ایجاد/مهاجرت دیتابیس
# -----------------------
def init_db_schema():
    """
    در صورت نبود جدول کامل، آن را ایجاد می‌کند با تمام ستون‌های لازم.
    """
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                file_id TEXT UNIQUE NOT NULL,
                artist TEXT,
                title TEXT,
                url TEXT,
                thumbnail TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                last_accessed TEXT DEFAULT (datetime('now')),
                path TEXT,
                cache_path TEXT,
                file_size INTEGER,
                status TEXT,
                duration INTEGER
            )
        """)

        # ایجاد جدول کاربران برای ویژگی‌های روزانه
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                daily_songs_enabled BOOLEAN DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now')),
                last_active TEXT DEFAULT (datetime('now'))
            )
        """)

        # جدول آمار روزانه
        cur.execute("""
            CREATE TABLE IF NOT EXISTS daily_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                time_slot TEXT,
                active_users INTEGER DEFAULT 0,
                songs_sent INTEGER DEFAULT 0,
                unsubscribes INTEGER DEFAULT 0
            )
        """)

        # ایندکس‌ها
        cur.execute("CREATE INDEX IF NOT EXISTS idx_files_user ON files(user_id)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_files_created ON files(created_at)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_files_fileid ON files(file_id)")

    logger.info("✅ Schema ensured (init_db_schema)")


def migrate_db():
    """
    اگر دیتابیس قدیمی ستون‌هایی نداشت، اضافه‌شان می‌کند.
    """
    required = {
        "content_type": "TEXT",
        "thumbnail": "TEXT",
        "path": "TEXT",
        "cache_path": "TEXT",
        "file_size": "INTEGER",
        "status": "TEXT",
        "last_accessed": "TEXT",
        "created_at": "TEXT",
        "duration": "INTEGER",   # ← اضافه شد
        "tg_file_id": "TEXT",    # برای ذخیره file_id تلگرام جهت آپلود سریع
        "album": "TEXT",
        "release_date": "TEXT",
        "popularity": "INTEGER",
        "genre": "TEXT",
        "total_tracks": "INTEGER",
        "label": "TEXT",
        "top_track": "TEXT"
    }

    with get_db() as conn:
        cur = conn.cursor()

        # اگر جدول وجود ندارد → ایجاد کامل
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='files'")
        if cur.fetchone() is None:
            init_db_schema()
            return

        # بررسی ستون‌های موجود
        cur.execute("PRAGMA table_info(files)")
        existing = {row["name"] for row in cur.fetchall()}

        for col, coltype in required.items():
            if col not in existing:
                try:
                    logger.info(f"Adding missing column '{col}' to files table")
                    cur.execute(f"ALTER TABLE files ADD COLUMN {col} {coltype}")
                except Exception as e:
                    logger.warning(f"Could not add column {col}: {e}")

        # بررسی وجود جدول users
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
        if cur.fetchone() is None:
            logger.info("Creating users table")
            cur.execute("""
                CREATE TABLE users (
                    user_id INTEGER PRIMARY KEY,
                    daily_songs_enabled BOOLEAN DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now')),
                    last_active TEXT DEFAULT (datetime('now'))
                )
            """)

        # بررسی وجود جدول daily_stats
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='daily_stats'")
        if cur.fetchone() is None:
            logger.info("Creating daily_stats table")
            cur.execute("""
                CREATE TABLE daily_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    time_slot TEXT,
                    active_users INTEGER DEFAULT 0,
                    songs_sent INTEGER DEFAULT 0,
                    unsubscribes INTEGER DEFAULT 0
                )
            """)

        # Create download_history table if it doesn't exist
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='download_history'")
        if cur.fetchone() is None:
            logger.info("Creating download_history table")
            cur.execute("""
                CREATE TABLE download_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    platform TEXT NOT NULL,
                    url TEXT NOT NULL,
                    title TEXT,
                    artist TEXT,
                    file_type TEXT,
                    file_size INTEGER,
                    created_at TEXT DEFAULT (datetime('now')),
                    status TEXT DEFAULT 'completed',
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_history_user ON download_history(user_id)")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_history_created ON download_history(created_at)")

        logger.info("✅ Migration finished (migrate_db)")



# -----------------------
# دانلود از تلگرام (در صورت داشتن file_id)
# -----------------------
def download_file_from_telegram(file_id: str, dest_dir: str = CACHE_DIR) -> Optional[str]:
    """فایل را از Telegram Bot API دریافت و در dest_dir ذخیره می‌کند. مسیر فایل ساخته شده را برمی‌گرداند."""
    if BOT_TOKEN is None or BOT_TOKEN == "PUT_YOUR_TOKEN_HERE":
        logger.error("Telegram bot token not set. Set TELEGRAM_BOT_TOKEN env or constant.")
        return None

    # تنظیم پروکسی
    proxies = {"http": PROXY_URL, "https": PROXY_URL} if PROXY_URL and PROXY_URL != "http://your_proxy_host:port" else None

    try:
        r = requests.get(f"https://api.telegram.org/bot{BOT_TOKEN}/getFile", params={"file_id": file_id}, timeout=15, proxies=proxies)
        r.raise_for_status()
        j = r.json()
        if not j.get("ok"):
            logger.error(f"Telegram getFile error: {j}")
            return None
        file_path = j["result"]["file_path"]
        file_name = os.path.basename(file_path)
        file_name = sanitize_filename(file_name)
        url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
        ensure_cache_dir()
        local_path = os.path.join(dest_dir, f"{sanitize_filename(file_id)}_{file_name}")
        # stream download
        with requests.get(url, stream=True, timeout=60, proxies=proxies) as rr:
            rr.raise_for_status()
            with open(local_path, "wb") as fw:
                for chunk in rr.iter_content(chunk_size=8192):
                    if chunk:
                        fw.write(chunk)
        logger.info(f"Downloaded telegram file {file_id} -> {local_path}")
        return local_path
    except requests.RequestException as e:
        logger.exception(f"Failed to download file from telegram ({file_id}) with proxy {PROXY_URL if proxies else 'None'}: {e}")
        return None
    except Exception as e:
        logger.exception(f"Failed to download file from telegram ({file_id}): {e}")
        return None

# -----------------------
# ذخیره فایل و متادیتا
# -----------------------
def save_file(user_id: int,
              file_id: str,
              artist: Optional[str] = None,
              title: Optional[str] = None,
              url: Optional[str] = None,
              thumbnail: Optional[str] = None,
              source_path: Optional[str] = None,
              source_bytes: Optional[bytes] = None,
              duration: Optional[int] = None,
              tg_file_id: Optional[str] = None,
              album: Optional[str] = None,
              release_date: Optional[str] = None,
              popularity: Optional[int] = None,
              genre: Optional[str] = None,
              content_type: Optional[str] = None,
              total_tracks: Optional[int] = None,
              label: Optional[str] = None,
              top_track: Optional[str] = None) -> Optional[str]:
    """
    فایل را در پوشه‌ی cache ذخیره می‌کند (اگر داده شده) و رکورد DB را درج/آپدیت می‌کند.
    اگر موفق شد مسیر محلی فایل را برمی‌گرداند، در غیر این صورت None.
    - source_path: مسیر فایلی که می‌خواهی کپی شود به کش (مثلاً فایل دانلود شده از اسپاتیفای)
    - source_bytes: بایت‌های فایل (اگر داری) — یکی از این دو کافیست.
    - duration: مدت زمان آهنگ (به ثانیه)
    - content_type: نوع محتوا ("track" / "album" / ...)
    - total_tracks, label, top_track: فیلدهای اضافه برای آلبوم
    """
    try:
        ensure_cache_dir()
        # آماده کردن نام فایل امن
        base_name = f"{user_id}_{sanitize_filename(file_id)}.mp3"
        dest_path = os.path.join(CACHE_DIR, base_name)

        # اگر source_path داده شده و وجود داره، کپی کن
        if source_path:
            if os.path.exists(source_path):
                shutil.copy2(source_path, dest_path)
                logger.info(f"Copied local source {source_path} -> {dest_path}")
            else:
                logger.warning(f"Given source_path does not exist: {source_path}")
                source_path = None

        # اگر source_bytes داده شده، بنویس
        if source_bytes:
            with open(dest_path, "wb") as fw:
                fw.write(source_bytes)
            logger.info(f"Wrote bytes to {dest_path}")

        # بررسی مسیر و حجم فایل
        final_path = dest_path if (dest_path and os.path.exists(dest_path)) else None
        file_size = None
        if final_path:
            try:
                file_size = os.path.getsize(final_path)
            except OSError as e:
                logger.warning(f"Could not get size for {final_path}: {e}")
                file_size = None

        # درج در دیتابیس
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO files (
                    user_id, file_id, content_type, artist, title, url, thumbnail,
                    path, cache_path, file_size, status, duration, tg_file_id, album, release_date, popularity, genre,
                    total_tracks, label, top_track, last_accessed
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(file_id) DO UPDATE SET
                    user_id=excluded.user_id,
                    content_type=COALESCE(excluded.content_type, files.content_type),
                    artist=excluded.artist,
                    title=excluded.title,
                    url=excluded.url,
                    thumbnail=excluded.thumbnail,
                    path=COALESCE(excluded.path, files.path),
                    cache_path=COALESCE(excluded.cache_path, files.cache_path),
                    file_size=COALESCE(excluded.file_size, files.file_size),
                    duration=COALESCE(excluded.duration, files.duration),
                    tg_file_id=COALESCE(excluded.tg_file_id, files.tg_file_id),
                    album=COALESCE(excluded.album, files.album),
                    release_date=COALESCE(excluded.release_date, files.release_date),
                    popularity=COALESCE(excluded.popularity, files.popularity),
                    genre=COALESCE(excluded.genre, files.genre),
                    total_tracks=COALESCE(excluded.total_tracks, files.total_tracks),
                    label=COALESCE(excluded.label, files.label),
                    top_track=COALESCE(excluded.top_track, files.top_track),
                    status=COALESCE(excluded.status, files.status),
                    last_accessed=datetime('now')
            """, (
                user_id, file_id, content_type, artist, title, url, thumbnail,
                final_path, final_path, file_size, "cached" if final_path else "no-file",
                duration, tg_file_id, album, release_date, popularity, genre,
                total_tracks, label, top_track
            ))
        logger.info(f"Saved DB record for file_id={file_id} (type={content_type}, path={final_path}, duration={duration})")
        cleanup_cache()
        return final_path

    except sqlite3.Error as e:
        logger.exception(f"SQLite error when saving file {file_id}: {e}")
        return None
    except Exception as e:
        logger.exception(f"Error when saving file {file_id}: {e}")
        return None


# -----------------------
# اطمینان از وجود فایل محلی (دانلود از تلگرام در صورت نیاز)
# -----------------------
def ensure_local_file_by_row(row: sqlite3.Row) -> Optional[str]:
    """
    اگر path موجود و فایل هست برمی‌گرداند.
    در غیر این صورت اگر file_id موجود باشد تلاش به دانلود از تلگرام می‌کند و DB را آپدیت می‌کند.
    """
    file_id = row["file_id"]
    path = row["path"]
    if path and os.path.exists(path):
        # update last_accessed
        with get_db() as conn:
            conn.execute("UPDATE files SET last_accessed = datetime('now') WHERE file_id = ?", (file_id,))
        return path

    # اگر local path نیست، تلاش برای دانلود از تلگرام
    if file_id:
        local = download_file_from_telegram(file_id)
        if local:
            size = os.path.getsize(local)
            with get_db() as conn:
                conn.execute("UPDATE files SET path = ?, file_size = ?, last_accessed = datetime('now'), status = ? WHERE file_id = ?",
                             (local, size, "cached", file_id))
            return local
        else:
            logger.warning(f"Could not restore file {file_id} from telegram.")
            return None
    logger.warning("Row has no path and no file_id to recover from.")
    return None

# -----------------------
# گرفتن آخرین فایل یک کاربر
# -----------------------
def get_last_file(user_id: int) -> Optional[Tuple]:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT id, user_id, file_id, artist, title, url, thumbnail, created_at, last_accessed, path, file_size, status
            FROM files WHERE user_id = ? ORDER BY created_at DESC LIMIT 1
        """, (user_id,))
        row = cur.fetchone()
        if not row:
            logger.debug(f"No files for user {user_id}")
            return None
        # اگر فایل محلی نیست سعی کن بازیابی کنی
        if not row["path"] or not os.path.exists(row["path"]):
            logger.info(f"Local file missing for {row['file_id']}, trying to restore from telegram if possible.")
            local = ensure_local_file_by_row(row)
            return (row["file_id"], row["artist"], row["title"], row["url"], local)
        # در غیر این صورت مسیر را برمی‌گردانیم
        # آپدیت last_accessed
    with get_db() as conn:
        conn.execute("UPDATE files SET last_accessed = datetime('now') WHERE file_id = ?", (row["file_id"],))
        return (row["file_id"], row["artist"], row["title"], row["url"], row["path"])

# -----------------------
# Telegram File ID functions for fast-path optimization
# -----------------------

def get_tg_file_id(content_id: str, user_id: int) -> Optional[str]:
    """Get Telegram file_id for fast-path uploads"""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT tg_file_id FROM files WHERE file_id = ? AND user_id = ? AND tg_file_id IS NOT NULL", (content_id, user_id))
            row = cur.fetchone()
            if row and row[0]:
                logger.debug(f"Found tg_file_id for {content_id}: {row[0]}")
                return row[0]
        return None
    except Exception as e:
        logger.error(f"Error getting tg_file_id for {content_id}: {e}")
        return None

def update_tg_file_id(content_id: str, user_id: int, tg_file_id: str) -> bool:
    """Update Telegram file_id for fast-path uploads"""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("UPDATE files SET tg_file_id = ?, last_accessed = CURRENT_TIMESTAMP WHERE file_id = ? AND user_id = ?", (tg_file_id, content_id, user_id))
            if cur.rowcount > 0:
                logger.debug(f"Updated tg_file_id for {content_id}: {tg_file_id}")
                return True
            else:
                logger.warning(f"No rows updated for tg_file_id {content_id}")
                return False
    except Exception as e:
        logger.error(f"Error updating tg_file_id for {content_id}: {e}")
        return False

def get_tg_file_id(content_id: str, user_id: int) -> Optional[str]:
    """
    Get Telegram file_id for a given content_id and user_id for fast upload.
    Returns tg_file_id if exists, None otherwise.
    """
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT tg_file_id FROM files WHERE file_id = ? AND user_id = ?", (content_id, user_id))
            row = cur.fetchone()
            if row and row["tg_file_id"]:
                # Update last_accessed
                conn.execute("UPDATE files SET last_accessed = datetime('now') WHERE file_id = ?", (content_id,))
                logger.info(f"Found tg_file_id for {content_id}: {row['tg_file_id']}")
                return row["tg_file_id"]
        return None
    except Exception as e:
        logger.error(f"Error getting tg_file_id for {content_id}: {e}")
        return None

def update_tg_file_id(content_id: str, user_id: int, tg_file_id: str) -> bool:
    """
    Update the tg_file_id for a given content_id and user_id.
    Returns True if successful, False otherwise.
    """
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("UPDATE files SET tg_file_id = ?, last_accessed = datetime('now') WHERE file_id = ? AND user_id = ?",
                       (tg_file_id, content_id, user_id))
            if cur.rowcount > 0:
                logger.info(f"Updated tg_file_id for {content_id}: {tg_file_id}")
                return True
            else:
                logger.warning(f"No rows updated for tg_file_id update: {content_id}")
                return False
    except Exception as e:
        logger.error(f"Error updating tg_file_id for {content_id}: {e}")
        return False

# -----------------------
# پاکسازی کش بر اساس تاریخ و حجم
# -----------------------
def cleanup_cache():
    """
    حذف فایل‌های منقضی و در صورت بالا بودن مجموع حجم، حذف قدیمی‌ترین تا زیر حد.
    به جای حذف رکورد کامل، path را NULL می‌کنیم تا metadata باقی بماند.
    """
    try:
        ensure_cache_dir()
        total_size = 0
        rows_to_check: List[Tuple[str, str]] = []  # (path, created_at)
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT path, created_at FROM files WHERE path IS NOT NULL")
            for row in cur.fetchall():
                p = row["path"]
                c = row["created_at"]
                if p and os.path.exists(p):
                    try:
                        size = os.path.getsize(p)
                        total_size += size
                        rows_to_check.append((p, c))
                    except Exception:
                        logger.exception(f"Error getting size for {p}")

            # حذف فایل‌های منقضی بر اساس created_at
            expired = []
            for p, c in rows_to_check:
                created_dt = parse_sqlite_ts(c) or datetime.now()
                if datetime.now() > created_dt + timedelta(days=CACHE_EXPIRY_DAYS):
                    expired.append(p)

            for p in expired:
                try:
                    if os.path.exists(p):
                        sz = os.path.getsize(p)
                        os.remove(p)
                        total_size -= sz
                    conn.execute("UPDATE files SET path = NULL, file_size = NULL, status = 'expired' WHERE path = ?", (p,))
                    logger.info(f"Removed expired cache file {p}")
                except Exception:
                    logger.exception(f"Failed to remove expired cache file {p}")

            # اگر هنوز حجم بیشتر از حد است، حذف قدیمی‌ترین بر اساس last_accessed یا created_at
            if total_size > MAX_CACHE_SIZE:
                cur.execute("""
                    SELECT path FROM files
                    WHERE path IS NOT NULL
                    ORDER BY COALESCE(last_accessed, created_at) ASC
                """)
                for (p,) in cur.fetchall():
                    if total_size <= MAX_CACHE_SIZE:
                        break
                    if p and os.path.exists(p):
                        try:
                            sz = os.path.getsize(p)
                            os.remove(p)
                            total_size -= sz
                            conn.execute("UPDATE files SET path = NULL, file_size = NULL, status = 'removed_by_size' WHERE path = ?", (p,))
                            logger.info(f"Removed {p} to reduce cache size (freed {sz} bytes)")
                        except Exception:
                            logger.exception(f"Failed to remove cache file {p}")
    except Exception as e:
        logger.exception(f"cleanup_cache error: {e}")

def get_or_restore_local_path(file_id) -> str:
    with get_db() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM files WHERE file_id = ? LIMIT 1", (file_id,))
        row = cur.fetchone()
    if not row:
        raise FileNotFoundError(f"{file_id} not in DB")

    # اگر path معتبر و موجود است، برگردون
    p = row["path"]
    if p and os.path.exists(p):
        return p

    # تلاش به بازیابی از تلگرام
    local = ensure_local_file_by_row(row)
    if local:
        return local

    raise FileNotFoundError(f"Could not get local file for {file_id}")

# -----------------------
# Flask routes
# -----------------------
@app.route("/")
def index():
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, user_id, file_id, artist, title, thumbnail, created_at, last_accessed, path, file_size, status FROM files ORDER BY created_at DESC")
            files = cur.fetchall()
        # اگر template داری از render_template استفاده کن، وگرنه JSON برگردون.
        # return render_template('index.html', files=files)
        # برای سادگی JSON:
        return jsonify([dict(row) for row in files])
    except Exception as e:
        logger.exception("Failed to load index")
        return jsonify({"error": str(e)}), 500


@app.route("/clear_cache", methods=["POST"])
def clear_cache_route():
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT path FROM files WHERE path IS NOT NULL")
            for (p,) in cur.fetchall():
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                        logger.info(f"Cleared cache file {p}")
                    except Exception:
                        logger.exception(f"Failed to remove {p}")
            cur.execute("UPDATE files SET path = NULL, file_size = NULL, status = 'cleared'")
        return jsonify({"message": "Cache cleared"})
    except Exception as e:
        logger.exception("Failed clear_cache")
        return jsonify({"error": str(e)}), 500


@app.route("/redownload/<file_id>", methods=["POST"])
def redownload_route(file_id):
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM files WHERE file_id = ? LIMIT 1", (file_id,))
            row = cur.fetchone()
            if not row:
                return jsonify({"error": "not found"}), 404
            local = ensure_local_file_by_row(row)
            if not local:
                return jsonify({"error": "could not redownload"}), 500
            return jsonify({"file_id": file_id, "path": local})
    except Exception as e:
        logger.exception("redownload failed")
        return jsonify({"error": str(e)}), 500

# -----------------------
# توابع مدیریت کاربران برای ویژگی روزانه
# -----------------------
def enable_daily_songs(user_id: int) -> bool:
    """فعال کردن ارسال روزانه آهنگ‌ها برای کاربر"""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO users (user_id, daily_songs_enabled, last_active)
                VALUES (?, 1, datetime('now'))
            """, (user_id,))
        logger.info(f"Daily songs enabled for user {user_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to enable daily songs for user {user_id}: {e}")
        return False

def disable_daily_songs(user_id: int) -> bool:
    """غیرفعال کردن ارسال روزانه آهنگ‌ها برای کاربر"""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE users SET daily_songs_enabled = 0, last_active = datetime('now')
                WHERE user_id = ?
            """, (user_id,))
        logger.info(f"Daily songs disabled for user {user_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to disable daily songs for user {user_id}: {e}")
        return False

def is_daily_songs_enabled(user_id: int) -> bool:
    """بررسی فعال بودن ارسال روزانه آهنگ‌ها برای کاربر"""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT daily_songs_enabled FROM users WHERE user_id = ?", (user_id,))
            row = cur.fetchone()
            return bool(row[0]) if row else False
    except Exception as e:
        logger.error(f"Failed to check daily songs status for user {user_id}: {e}")
        return False

def get_active_daily_users() -> list[int]:
    """دریافت لیست کاربرانی که ارسال روزانه فعال دارند"""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("SELECT user_id FROM users WHERE daily_songs_enabled = 1")
            return [row[0] for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"Failed to get active daily users: {e}")
        return []

def update_user_activity(user_id: int):
    """به‌روزرسانی زمان آخرین فعالیت کاربر"""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT OR REPLACE INTO users (user_id, last_active)
                VALUES (?, datetime('now'))
            """, (user_id,))
    except Exception as e:
        logger.error(f"Failed to update user activity for {user_id}: {e}")

def record_daily_send(date: str, time_slot: str, active_users: int, songs_sent: int):
    """ثبت آمار ارسال روزانه"""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO daily_stats (date, time_slot, active_users, songs_sent)
                VALUES (?, ?, ?, ?)
            """, (date, time_slot, active_users, songs_sent))
    except Exception as e:
        logger.error(f"Failed to record daily send stats: {e}")

def record_unsubscribe(date: str):
    """ثبت انصراف روزانه"""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""
                UPDATE daily_stats SET unsubscribes = unsubscribes + 1
                WHERE date = ? AND time_slot = (SELECT MAX(time_slot) FROM daily_stats WHERE date = ?)
            """, (date, date))
    except Exception as e:
        logger.error(f"Failed to record unsubscribe: {e}")

def get_daily_stats():
    """دریافت آمار روزانه"""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT date, time_slot, active_users, songs_sent, unsubscribes
                FROM daily_stats
                ORDER BY date DESC, time_slot DESC
                LIMIT 10
            """)
            return cur.fetchall()
    except Exception as e:
        logger.error(f"Failed to get daily stats: {e}")
        return []


# -----------------------
# Download History Functions
# -----------------------
def add_download_history(user_id: int, platform: str, url: str, title: str = None, 
                        artist: str = None, file_type: str = None, file_size: int = None, 
                        status: str = 'completed'):
    """Add a download to history"""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO download_history 
                (user_id, platform, url, title, artist, file_type, file_size, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, platform, url, title, artist, file_type, file_size, status))
            return True
    except Exception as e:
        logger.error(f"Failed to add download history: {e}")
        return False


def get_user_download_history(user_id: int, limit: int = 20):
    """Get user's download history"""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            cur.execute("""
                SELECT platform, url, title, artist, file_type, file_size, created_at, status
                FROM download_history
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (user_id, limit))
            return cur.fetchall()
    except Exception as e:
        logger.error(f"Failed to get download history: {e}")
        return []


def get_user_stats(user_id: int):
    """Get user statistics"""
    try:
        with get_db() as conn:
            cur = conn.cursor()
            
            # Total downloads
            cur.execute("SELECT COUNT(*) FROM download_history WHERE user_id = ?", (user_id,))
            total_downloads = cur.fetchone()[0]
            
            # Downloads by platform
            cur.execute("""
                SELECT platform, COUNT(*) as count
                FROM download_history
                WHERE user_id = ?
                GROUP BY platform
                ORDER BY count DESC
            """, (user_id,))
            platform_stats = cur.fetchall()
            
            # Total file size
            cur.execute("""
                SELECT SUM(file_size) FROM download_history 
                WHERE user_id = ? AND file_size IS NOT NULL
            """, (user_id,))
            total_size = cur.fetchone()[0] or 0
            
            # Recent activity (last 7 days)
            cur.execute("""
                SELECT COUNT(*) FROM download_history
                WHERE user_id = ? AND created_at >= datetime('now', '-7 days')
            """, (user_id,))
            recent_downloads = cur.fetchone()[0]
            
            return {
                'total_downloads': total_downloads,
                'platform_stats': platform_stats,
                'total_size': total_size,
                'recent_downloads': recent_downloads
            }
    except Exception as e:
        logger.error(f"Failed to get user stats: {e}")
        return {
            'total_downloads': 0,
            'platform_stats': [],
            'total_size': 0,
            'recent_downloads': 0
        }

# -----------------------
# اجرای اصلی
# -----------------------
if __name__ == "__main__":
    # مهاجرت و اطمینان از اسکیمای کامل
    migrate_db()         # اگر جدول نبود خودِ migrate آن را خواهد ساخت یا ستون‌های لازم را اضافه می‌کند
    # init_db_schema()   # می‌توانید صریح صدا بزنید اما migrate_db کافیست

    # یکبار cleanup اجرا کن (اختیاری ولی خوب است)
    cleanup_cache()

    # اجرا
    app.run(host="0.0.0.0", port=5000, debug=False)