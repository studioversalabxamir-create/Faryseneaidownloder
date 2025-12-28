"""
Resource management utilities for the downloader bot
"""
import os
import logging
import asyncio
from functools import wraps
from typing import Callable, Any, List
from utils import safe_remove_file

logger = logging.getLogger(__name__)


def cleanup_files(*file_paths: str):
    """
    Decorator to automatically clean up files after function execution
    Usage:
        @cleanup_files("file1.mp3", "file2.mp4")
        async def my_function():
            # files will be cleaned up after function completes or on error
            pass
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            finally:
                for file_path in file_paths:
                    if file_path and os.path.exists(file_path):
                        safe_remove_file(file_path)
                        logger.debug(f"Cleaned up file: {file_path}")
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            finally:
                for file_path in file_paths:
                    if file_path and os.path.exists(file_path):
                        safe_remove_file(file_path)
                        logger.debug(f"Cleaned up file: {file_path}")
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class TempFileManager:
    """
    Context manager for temporary files that need cleanup
    """
    def __init__(self):
        self.files: List[str] = []
    
    def add(self, file_path: str):
        """Add a file to be cleaned up"""
        if file_path and file_path not in self.files:
            self.files.append(file_path)
    
    def cleanup(self):
        """Clean up all registered files"""
        for file_path in self.files:
            safe_remove_file(file_path)
        self.files.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False


async def cleanup_temp_files(file_paths: List[str]):
    """
    Clean up a list of temporary files
    """
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            safe_remove_file(file_path)

