from playwright.sync_api import sync_playwright

def save_twitter_session_state():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()

        page = context.new_page()
        page.goto("https://x.com/login", wait_until="networkidle")
        print("\n🔹 لطفاً در مرورگر بازشده لاگین کن (با اکانتی که کوکی‌های cookies.txt ازش گرفتی)")
        print("وقتی کاملاً لاگین شدی، Enter رو در ترمینال بزن...\n")
        input("⏳ منتظرم... بعد از لاگین و باز شدن صفحه اصلی X، Enter بزن: ")

        # ذخیره session در فایل JSON
        context.storage_state(path="twitter_state.json")
        print("\n✅ فایل twitter_state.json با موفقیت ساخته شد!")
        print("📁 مسیر ذخیره: در همین پوشه فعلی (پوش کنار این اسکریپت)")
        browser.close()

if __name__ == "__main__":
    save_twitter_session_state()
