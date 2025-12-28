"""
Utility functions for the downloader bot
"""
import re
import logging
import os
from typing import Optional, List
from urllib.parse import urlparse
from config import PROXY

logger = logging.getLogger(__name__)


def validate_url(url: str) -> bool:
    """
    Validate if a string is a valid URL
    """
    if not url or not isinstance(url, str):
        return False
    
    url = url.strip()
    if not url:
        return False
    
    try:
        result = urlparse(url)
        return all([result.scheme in ['http', 'https'], result.netloc])
    except Exception:
        return False


def extract_urls(text: str) -> List[str]:
    """
    Extract all valid URLs from text using regex
    Returns list of valid URLs
    """
    if not text:
        return []
    
    url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
    matches = re.findall(url_pattern, text)
    
    # Validate each URL
    valid_urls = [url for url in matches if validate_url(url)]
    return valid_urls


def sanitize_filename(filename: str, max_length: int = 200) -> str:
    """
    Sanitize filename for filesystem safety
    """
    if not filename:
        return "download"
    
    # Remove invalid characters
    invalid_chars = r'[<>:"/\\|?*\x00-\x1f]'
    sanitized = re.sub(invalid_chars, '_', filename)
    
    # Remove leading/trailing spaces and dots
    sanitized = sanitized.strip(' .')
    
    # Limit length
    if len(sanitized) > max_length:
        name, ext = os.path.splitext(sanitized)
        sanitized = name[:max_length - len(ext)] + ext
    
    return sanitized or "download"


def get_proxy() -> Optional[str]:
    """
    Get proxy from config
    """
    return PROXY if PROXY else None


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    """
    if size_bytes == 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    unit_index = 0
    size = float(size_bytes)
    
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024
        unit_index += 1
    
    return f"{size:.2f} {units[unit_index]}"


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


def safe_remove_file(filepath: str) -> bool:
    """
    Safely remove a file, handling errors gracefully
    """
    if not filepath or not os.path.exists(filepath):
        return False
    
    try:
        os.remove(filepath)
        logger.debug(f"Removed file: {filepath}")
        return True
    except Exception as e:
        logger.warning(f"Failed to remove file {filepath}: {e}")
        return False


def ensure_directory(path: str) -> bool:
    """
    Ensure directory exists, create if it doesn't
    """
    try:
        os.makedirs(path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        return False


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to max length
    """
    if not text:
        return ""
    
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

