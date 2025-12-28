from enum import Enum

class Platform(Enum):
    YOUTUBE = "youtube"
    INSTAGRAM = "instagram"
    TIKTOK = "tiktok"
    TWITTER = "twitter"
    FACEBOOK = "facebook"
    PINTEREST = "pinterest"
    THREADS = "threads"
    LINKEDIN = "linkedin"
    SPOTIFY = "spotify"
    SOUNDCLOUD = "soundcloud"
    CASTBOX = "castbox"
    UNKNOWN = "unknown"


def detect_platform(url: str) -> str:
    """Detect platform from URL in a unified, structured way"""
    if not url:
        return Platform.UNKNOWN.value

    u = url.lower()

    patterns = {
        Platform.YOUTUBE: ["youtube.com", "youtu.be"],
        Platform.INSTAGRAM: ["instagram.com", "instagr.am"],
        Platform.TIKTOK: ["tiktok.com"],
        Platform.TWITTER: ["twitter.com", "x.com"],
        Platform.FACEBOOK: ["facebook.com"],
        Platform.PINTEREST: ["pinterest.com", "pin.it"],
        Platform.THREADS: ["threads.net", "threads.com"],
        Platform.LINKEDIN: ["linkedin.com"],
        Platform.SPOTIFY: ["spotify.com", "open.spotify.com"],
        Platform.SOUNDCLOUD: ["soundcloud.com"],
        Platform.CASTBOX: ["castbox.fm"],
    }

    for platform, keys in patterns.items():
        if any(k in u for k in keys):
            return platform.value

    return Platform.UNKNOWN.value

