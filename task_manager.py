import asyncio
import logging
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


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
        # کنترل نرخ درخواست و تنظیمات مرتبط (گروه‌بندی برای کاهش تعداد attributes)
        self._config = {
            "cooldown_time": cooldown,
            "rate_limit": {
                "max_requests_per_minute": 10,
                "user_request_counts": {},  # type: Dict[int, list[float]]
            },
        }
        # فلگ برای تشخیص لغو عملیات در حال اجرا
        self.cancel_flags: Dict[int, bool] = {}
        # قفل همزمانی برای جلوگیری از race condition
        self._lock = asyncio.Lock()

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
                logger.warning("[TaskManager] User %s exceeded rate limit.", user_id)
                return None

            # جلوگیری از اسپم (در حالت cooldown)
            if user_id in getattr(self, "cooldown_users", set()):
                # Note: cooldown_users is intentionally not an attribute anymore;
                # we track cooldown by adding to a transient set below.
                logger.info("[TaskManager] User %s is in cooldown.", user_id)
                return None

            # جلوگیری از اجرای همزمان چند تسک برای یک کاربر
            if user_id in self.active_tasks:
                logger.info("[TaskManager] User %s already has an active task.", user_id)
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
            logger.info("[TaskManager] Task for user %s canceled.", user_id)
            return None
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.error("Error in task for %s: %s", user_id, e, exc_info=True)
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
                logger.info("[TaskManager] No active task to cancel for %s", user_id)
                return False

            task.cancel()
            self.active_tasks.pop(user_id, None)
            logger.info("[TaskManager] Task for user %s canceled manually.", user_id)
            return True

    # ------------------------------------------------------------
    async def retry_last(self, user_id: int) -> bool:
        """
        اجرای مجدد آخرین Task کاربر (retry)
        """
        job = self.last_jobs.get(user_id)
        if not job:
            logger.info("[TaskManager] No previous task found for retry (%s)", user_id)
            return False

        coro_func, args, kwargs = job
        logger.info("[TaskManager] Retrying last job for user %s", user_id)

        # اجرای مجدد همان Task
        asyncio.create_task(self.start_task(user_id, coro_func, *args, **kwargs))
        return True

    # ------------------------------------------------------------
    async def _cooldown_timer(self, user_id: int):
        """
        تایمر محدودیت ارسال درخواست‌ها (rate limit)
        """
        # Use a transient set local to method to avoid another instance attribute
        # but we still want cooldown behavior per user: use an in-memory set stored on the instance lazily
        cooldown_set = getattr(self, "_cooldown_users", None)
        if cooldown_set is None:
            cooldown_set = set()
            setattr(self, "_cooldown_users", cooldown_set)

        cooldown_set.add(user_id)
        try:
            await asyncio.sleep(self._config["cooldown_time"])
        finally:
            cooldown_set.discard(user_id)
            logger.debug("[TaskManager] Cooldown expired for %s", user_id)

    # ------------------------------------------------------------
    def _check_rate_limit(self, user_id: int) -> bool:
        """
        Check if user has exceeded rate limit (requests per minute)
        Returns True if request is allowed, False if rate limit exceeded
        """
        import time

        current_time = time.time()
        cfg = self._config["rate_limit"]
        counts = cfg["user_request_counts"]

        # Clean old requests (older than 1 minute)
        if user_id in counts:
            counts[user_id] = [req_time for req_time in counts[user_id] if current_time - req_time < 60]
        else:
            counts[user_id] = []

        # Check if limit exceeded
        if len(counts[user_id]) >= cfg["max_requests_per_minute"]:
            return False

        # Record this request
        counts[user_id].append(current_time)
        return True

    # ------------------------------------------------------------
    def get_rate_limit_status(self, user_id: int) -> dict:
        """
        Get rate limit status for a user
        """
        import time

        current_time = time.time()
        cfg = self._config["rate_limit"]
        counts = cfg["user_request_counts"]

        if user_id in counts:
            # Clean old requests
            counts[user_id] = [req_time for req_time in counts[user_id] if current_time - req_time < 60]
            remaining = cfg["max_requests_per_minute"] - len(counts[user_id])
            reset_in = 60 - (current_time - counts[user_id][0]) if counts[user_id] else 0
        else:
            remaining = cfg["max_requests_per_minute"]
            reset_in = 0

        return {
            "remaining": remaining,
            "limit": cfg["max_requests_per_minute"],
            "reset_in": int(reset_in),
        }

    # ------------------------------------------------------------
    def get_status(self, user_id: int) -> str:
        """
        گزارش وضعیت کاربر: active | cooldown | idle
        """
        if user_id in self.active_tasks:
            return "active"

        cooldown_set = getattr(self, "_cooldown_users", set())
        if user_id in cooldown_set:
            return "cooldown"

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
            logger.info("[TaskManager] All active tasks canceled.")


# ------------------------------------------------------------
# Global Instance - برای import در هندلرها
# ------------------------------------------------------------
task_manager: TaskManager = TaskManager(cooldown=5)
