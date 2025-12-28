import asyncio
import logging
from typing import Any, Callable, Dict, Optional


class TaskManager:
    """
    🎯 سیستم مدیریت تسک کاربران
    ------------------------
    • مدیریت Taskهای فعال هر کاربر
    • لغو عملیات در حال اجرا (cancel)
    • اجرای مجدد آخرین عملیات (retry)
    • کنترل نرخ ارسال درخواست‌ها (cooldown)
    """

    def __init__(self, cooldown: int = 5):
        # هر کاربر یک Task فعال دارد
        self.active_tasks: Dict[int, asyncio.Task] = {}
        # ذخیره‌ی آخرین تابع و آرگومان‌ها برای retry
        self.last_jobs: Dict[int, tuple[Callable[..., Any], tuple, dict]] = {}
        # کنترل نرخ درخواست (۵ ثانیه)
        self.cooldown_users: set[int] = set()
        self.cooldown_time = cooldown
        # فلگ برای تشخیص لغو عملیات در حال اجرا
        self.cancel_flags: Dict[int, bool] = {}
        # قفل همزمانی برای جلوگیری از race condition
        self._lock = asyncio.Lock()
        # Rate limiting per user (requests per minute)
        self.user_request_counts: Dict[int, list[float]] = {}
        self.max_requests_per_minute = 10  # Maximum requests per minute per user

    # ------------------------------------------------------------
    async def start_task(
        self, user_id: int, coro_func: Callable[..., Any], *args, **kwargs
    ) -> Optional[Any]:
        """
        اجرای تابع async در قالب Task با ثبت در لیست فعال‌ها
        """

        async with self._lock:
            # Rate limiting check
            if not self._check_rate_limit(user_id):
                logging.warning(f"[TaskManager] User {user_id} exceeded rate limit.")
                return None
            
            # جلوگیری از اسپم (در حالت cooldown)
            if user_id in self.cooldown_users:
                logging.info(f"[TaskManager] User {user_id} is in cooldown.")
                return None

            # جلوگیری از اجرای همزمان چند تسک برای یک کاربر
            if user_id in self.active_tasks:
                logging.info(f"[TaskManager] User {user_id} already has an active task.")
                return None

            # بازنشانی فلگ لغو
            self.cancel_flags[user_id] = False

            # ایجاد Task جدید
            task = asyncio.create_task(coro_func(*args, **kwargs))
            self.active_tasks[user_id] = task
            self.last_jobs[user_id] = (coro_func, args, kwargs)

        # فعال‌سازی cooldown جداگانه (بدون قفل)
        asyncio.create_task(self._cooldown_timer(user_id))

        try:
            result = await task
            return result
        except asyncio.CancelledError:
            logging.info(f"[TaskManager] Task for user {user_id} canceled.")
            return None
        except Exception as e:
            logging.error(f"[TaskManager] Error in task for {user_id}: {e}", exc_info=True)
            return None
        finally:
            # پاکسازی بعد از پایان یا خطا
            async with self._lock:
                self.active_tasks.pop(user_id, None)
                self.cancel_flags.pop(user_id, None)

    # ------------------------------------------------------------
    async def cancel_task(self, user_id: int) -> bool:
        """
        لغو Task فعال کاربر (در صورت وجود)
        """
        async with self._lock:
            # فعال کردن فلگ لغو برای اطلاع به تابع
            self.cancel_flags[user_id] = True

            task = self.active_tasks.get(user_id)
            if not task:
                logging.info(f"[TaskManager] No active task to cancel for {user_id}")
                return False

            task.cancel()
            self.active_tasks.pop(user_id, None)
            logging.info(f"[TaskManager] Task for user {user_id} canceled manually.")
            return True

    # ------------------------------------------------------------
    async def retry_last(self, user_id: int) -> bool:
        """
        اجرای مجدد آخرین Task کاربر (retry)
        """
        job = self.last_jobs.get(user_id)
        if not job:
            logging.info(f"[TaskManager] No previous task found for retry ({user_id})")
            return False

        coro_func, args, kwargs = job
        logging.info(f"[TaskManager] Retrying last job for user {user_id}")

        # اجرای مجدد همان Task
        asyncio.create_task(self.start_task(user_id, coro_func, *args, **kwargs))
        return True

    # ------------------------------------------------------------
    async def _cooldown_timer(self, user_id: int):
        """
        تایمر محدودیت ارسال درخواست‌ها (rate limit)
        """
        self.cooldown_users.add(user_id)
        try:
            await asyncio.sleep(self.cooldown_time)
        finally:
            self.cooldown_users.discard(user_id)
            logging.debug(f"[TaskManager] Cooldown expired for {user_id}")
    
    # ------------------------------------------------------------
    def _check_rate_limit(self, user_id: int) -> bool:
        """
        Check if user has exceeded rate limit (requests per minute)
        Returns True if request is allowed, False if rate limit exceeded
        """
        import time
        current_time = time.time()
        
        # Clean old requests (older than 1 minute)
        if user_id in self.user_request_counts:
            self.user_request_counts[user_id] = [
                req_time for req_time in self.user_request_counts[user_id]
                if current_time - req_time < 60
            ]
        else:
            self.user_request_counts[user_id] = []
        
        # Check if limit exceeded
        if len(self.user_request_counts[user_id]) >= self.max_requests_per_minute:
            return False
        
        # Record this request
        self.user_request_counts[user_id].append(current_time)
        return True
    
    # ------------------------------------------------------------
    def get_rate_limit_status(self, user_id: int) -> dict:
        """
        Get rate limit status for a user
        """
        import time
        current_time = time.time()
        
        if user_id in self.user_request_counts:
            # Clean old requests
            self.user_request_counts[user_id] = [
                req_time for req_time in self.user_request_counts[user_id]
                if current_time - req_time < 60
            ]
            remaining = self.max_requests_per_minute - len(self.user_request_counts[user_id])
        else:
            remaining = self.max_requests_per_minute
        
        return {
            'remaining': remaining,
            'limit': self.max_requests_per_minute,
            'reset_in': 60 - (current_time - self.user_request_counts[user_id][0]) if user_id in self.user_request_counts and self.user_request_counts[user_id] else 0
        }

    # ------------------------------------------------------------
    def get_status(self, user_id: int) -> str:
        """
        گزارش وضعیت کاربر: active | cooldown | idle
        """
        if user_id in self.active_tasks:
            return "active"
        elif user_id in self.cooldown_users:
            return "cooldown"
        else:
            return "idle"

    # ------------------------------------------------------------
    async def shutdown(self):
        """
        لغو تمام Taskهای فعال (در هنگام خاموش شدن بات)
        """
        async with self._lock:
            for uid, task in list(self.active_tasks.items()):
                task.cancel()
            self.active_tasks.clear()
            self.cancel_flags.clear()
            logging.info("[TaskManager] All active tasks canceled.")


# ------------------------------------------------------------
# Global Instance - برای import در هندلرها
# ------------------------------------------------------------
task_manager: TaskManager = TaskManager(cooldown=5)
