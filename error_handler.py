"""
Error handling utilities for the downloader bot
"""
import logging
from typing import Callable, Any
from functools import wraps
from aiogram.types import Message
from utils import truncate_text

logger = logging.getLogger(__name__)


def handle_errors(user_message: str = None, log_error: bool = True):
    """
    Decorator for handling errors in async functions
    Provides user-friendly error messages
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_msg = str(e)
                
                if log_error:
                    logger.error(f"Error in {func.__name__}: {error_msg}", exc_info=True)
                
                # Try to send error message to user if Message is in args
                message = None
                for arg in args:
                    if isinstance(arg, Message):
                        message = arg
                        break
                
                if message:
                    # User-friendly error messages
                    friendly_msg = _get_friendly_error(error_msg)
                    if user_message:
                        friendly_msg = f"{user_message}\n\n{friendly_msg}"
                    
                    try:
                        await message.answer(friendly_msg)
                    except Exception:
                        logger.error("Failed to send error message to user")
                
                return None
        
        return wrapper
    return decorator


def _get_friendly_error(error_msg: str) -> str:
    """
    Convert technical error messages to user-friendly ones
    """
    error_lower = error_msg.lower()
    
    # Network errors
    if "timeout" in error_lower or "timed out" in error_lower:
        return (
            "⏱️ خطا: زمان اتصال به سرور به پایان رسید.\n"
            "⏱️ Error: Connection timeout.\n\n"
            "💡 لطفاً دوباره تلاش کنید یا از /retry استفاده کنید."
        )
    
    if "connection" in error_lower or "network" in error_lower:
        return (
            "🌐 خطا: مشکل در اتصال به اینترنت.\n"
            "🌐 Error: Network connection issue.\n\n"
            "💡 لطفاً اتصال اینترنت خود را بررسی کنید."
        )
    
    # File errors
    if "file not found" in error_lower or "no such file" in error_lower:
        return (
            "📁 خطا: فایل یافت نشد.\n"
            "📁 Error: File not found.\n\n"
            "💡 ممکن است لینک نامعتبر باشد یا محتوا حذف شده باشد."
        )
    
    # Permission errors
    if "permission" in error_lower or "access denied" in error_lower:
        return (
            "🔒 خطا: دسترسی به محتوا محدود است.\n"
            "🔒 Error: Access denied.\n\n"
            "💡 ممکن است محتوا خصوصی باشد یا نیاز به احراز هویت داشته باشد."
        )
    
    # Rate limit errors
    if "rate limit" in error_lower or "too many requests" in error_lower:
        return (
            "⏳ خطا: تعداد درخواست‌ها بیش از حد مجاز است.\n"
            "⏳ Error: Rate limit exceeded.\n\n"
            "💡 لطفاً چند لحظه صبر کنید و دوباره تلاش کنید."
        )
    
    # Platform-specific errors
    if "spotify" in error_lower:
        return (
            "🎵 خطا در دانلود از Spotify.\n"
            "🎵 Error downloading from Spotify.\n\n"
            "💡 لطفاً لینک را بررسی کنید و دوباره تلاش کنید."
        )
    
    if "youtube" in error_lower or "yt" in error_lower:
        return (
            "📺 خطا در دانلود از YouTube.\n"
            "📺 Error downloading from YouTube.\n\n"
            "💡 لطفاً لینک را بررسی کنید و دوباره تلاش کنید."
        )
    
    if "instagram" in error_lower:
        return (
            "📷 خطا در دانلود از Instagram.\n"
            "📷 Error downloading from Instagram.\n\n"
            "💡 لطفاً لینک را بررسی کنید و دوباره تلاش کنید."
        )
    
    if "tiktok" in error_lower:
        return (
            "🎬 خطا در دانلود از TikTok.\n"
            "🎬 Error downloading from TikTok.\n\n"
            "💡 لطفاً لینک را بررسی کنید و دوباره تلاش کنید."
        )
    
    # Generic error
    error_display = truncate_text(error_msg, 200)
    return (
        f"❌ خطا: {error_display}\n"
        f"❌ Error: {error_display}\n\n"
        "💡 لطفاً دوباره تلاش کنید یا از /retry استفاده کنید.\n"
        "💡 If the problem persists, contact support: @Farysenesupport"
    )


async def safe_execute(func: Callable, *args, **kwargs) -> tuple[bool, Any]:
    """
    Safely execute a function and return (success, result)
    """
    try:
        result = await func(*args, **kwargs) if callable(func) else func
        return True, result
    except Exception as e:
        logger.error(f"Error in safe_execute: {e}", exc_info=True)
        return False, str(e)

