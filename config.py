from dotenv import load_dotenv
import os

# Load .env
load_dotenv()

# Telegram bot
BOT_TOKEN = os.getenv("BOT_TOKEN")

# Network / proxy
PROXY = os.getenv("PROXY", "http://174.136.204.40:80")

# FFmpeg paths (use environment variable or default system path)
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")  # Default to system PATH
FFPROBE_PATH = os.getenv("FFPROBE_PATH", "ffprobe")  # Default to system PATH

# Spotify API
SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID", "")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET", "")

# Genius API token (only used when explicitly enabled)
GENIUS_API_TOKEN = os.getenv("GENIUS_API_TOKEN", "")

# Business/feature defaults
# Restrict features to these users (align with bot.py & handlers)
ALLOWED_USERS = [1418268157]

# Preferred Spotify market for search relevance (default to US for widest catalog)
MARKET = os.getenv("SPOTIFY_MARKET", "US").upper()

# Inline search tuning
INLINE_MIN_QUERY_LEN = int(os.getenv("INLINE_MIN_QUERY_LEN", "3"))
# Increase default page size for richer results while keeping Telegram's 50 limit in code
INLINE_PAGE_SIZE = int(os.getenv("INLINE_PAGE_SIZE", "30"))
INLINE_CACHE_TTL = int(os.getenv("INLINE_CACHE_TTL", "30"))  # seconds
INLINE_CACHE_SIZE = int(os.getenv("INLINE_CACHE_SIZE", "200"))
INLINE_THROTTLE_PERMITS = int(os.getenv("INLINE_THROTTLE_PERMITS", "3"))  # requests
INLINE_THROTTLE_WINDOW = float(os.getenv("INLINE_THROTTLE_WINDOW", "2.0"))  # seconds

# Multi-market augmentation for broader global results
# Comma-separated ISO country codes prioritized for fallback searches
_DEFAULT_SEARCH_MARKETS = "US,GB,DE,FR,CA,BR,MX,IN,JP,KR,TR"
SEARCH_MARKETS = [
    m.strip().upper()
    for m in (os.getenv("SEARCH_MARKETS") or _DEFAULT_SEARCH_MARKETS).split(",")
    if m.strip()
]
# When True, perform additional market searches if initial results are too few
MULTI_MARKET_ON_SPARSE = os.getenv("MULTI_MARKET_ON_SPARSE", "1") == "1"
# Minimum acceptable results before triggering multi-market augmentation
MIN_INLINE_RESULTS = int(os.getenv("MIN_INLINE_RESULTS", "10"))
# Max number of extra markets to try beyond the primary MARKET (speed vs breadth)
MULTI_MARKET_MAX_EXTRA = int(os.getenv("MULTI_MARKET_MAX_EXTRA", "4"))