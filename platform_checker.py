"""
Platform availability checker
"""
import asyncio
import logging
import aiohttp
from typing import Dict, Optional
from utils import get_proxy

logger = logging.getLogger(__name__)

# Cache for platform status (5 minute TTL)
_platform_status_cache: Dict[str, tuple[bool, float]] = {}
CACHE_TTL = 300  # 5 minutes


async def check_platform_availability(platform: str) -> bool:
    """
    Check if a platform is currently available/accessible
    Returns True if platform is accessible, False otherwise
    """
    import time
    current_time = time.time()
    
    # Check cache first
    if platform in _platform_status_cache:
        is_available, cached_time = _platform_status_cache[platform]
        if current_time - cached_time < CACHE_TTL:
            return is_available
    
    # Test URLs for each platform
    test_urls = {
        "youtube": "https://www.youtube.com",
        "spotify": "https://open.spotify.com",
        "instagram": "https://www.instagram.com",
        "tiktok": "https://www.tiktok.com",
        "pinterest": "https://www.pinterest.com",
        "twitter": "https://twitter.com",
        "facebook": "https://www.facebook.com"
    }
    
    if platform not in test_urls:
        return True  # Unknown platform, assume available
    
    test_url = test_urls[platform]
    proxy = get_proxy()
    
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            proxies = {"http": proxy, "https": proxy} if proxy else None
            async with session.get(test_url, proxy=proxy, allow_redirects=True) as response:
                is_available = response.status < 500
                _platform_status_cache[platform] = (is_available, current_time)
                return is_available
    except Exception as e:
        logger.warning(f"Failed to check {platform} availability: {e}")
        # On error, assume available (don't block users)
        _platform_status_cache[platform] = (True, current_time)
        return True


async def check_all_platforms() -> Dict[str, bool]:
    """
    Check availability of all platforms
    Returns dict mapping platform name to availability status
    """
    platforms = ["youtube", "spotify", "instagram", "tiktok", "pinterest"]
    results = {}
    
    tasks = [check_platform_availability(platform) for platform in platforms]
    availability_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for platform, result in zip(platforms, availability_results):
        if isinstance(result, Exception):
            logger.error(f"Error checking {platform}: {result}")
            results[platform] = True  # Assume available on error
        else:
            results[platform] = result
    
    return results

