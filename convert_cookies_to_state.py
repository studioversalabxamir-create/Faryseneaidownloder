import json
from http.cookiejar import MozillaCookieJar

def convert_cookies_to_state(cookies_txt_path="xcookies.txt", output_path="twitter_state.json"):
    jar = MozillaCookieJar()
    jar.load(cookies_txt_path, ignore_discard=True, ignore_expires=True)

    cookies = []
    for c in jar:
        cookies.append({
            "name": c.name,
            "value": c.value,
            "domain": c.domain,
            "path": c.path,
            "expires": c.expires,
            "httpOnly": c.get_nonstandard_attr('HttpOnly') or False,
            "secure": c.secure,
            "sameSite": "Lax"
        })

    state = {"cookies": cookies, "origins": []}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    print(f"✅ فایل {output_path} با موفقیت ساخته شد! ({len(cookies)} کوکی ذخیره شد)")

if __name__ == "__main__":
    convert_cookies_to_state()
