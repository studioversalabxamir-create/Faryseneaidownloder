#!/usr/bin/env python3
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# List of proxies to test
proxies = [
    "http://104.199.219.13:3128",
    "http://115.239.234.43:7302", 
    "http://122.175.58.131:80",
    "http://128.199.202.122:8080",
    "http://134.209.29.120:3128",
    "http://138.68.60.8:3128",
    "http://139.59.1.14:8080",
    "http://159.203.61.169:3128"
]

def test_proxy(proxy):
    """Test a single proxy"""
    try:
        proxies_dict = {
            "http": proxy,
            "https": proxy
        }
        
        # Test with a simple request
        start_time = time.time()
        response = requests.get(
            "http://httpbin.org/ip", 
            proxies=proxies_dict, 
            timeout=10,
            verify=False
        )
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            ip_info = response.json()
            return {
                "proxy": proxy,
                "status": "WORKING",
                "response_time": round(response_time, 2),
                "ip": ip_info.get("origin", "Unknown")
            }
        else:
            return {
                "proxy": proxy,
                "status": "FAILED",
                "response_time": round(response_time, 2),
                "error": f"Status code: {response.status_code}"
            }
    except Exception as e:
        return {
            "proxy": proxy,
            "status": "FAILED", 
            "response_time": 0,
            "error": str(e)
        }

def main():
    print("Testing proxies...")
    print("=" * 80)
    
    working_proxies = []
    
    # Test proxies concurrently
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_proxy = {executor.submit(test_proxy, proxy): proxy for proxy in proxies}
        
        for future in as_completed(future_to_proxy):
            result = future.result()
            status_symbol = "[OK]" if result["status"] == "WORKING" else "[FAIL]"
            print(f"{status_symbol} {result['proxy']}")
            print(f"   Status: {result['status']}")
            if result["status"] == "WORKING":
                print(f"   Response Time: {result['response_time']}s")
                print(f"   IP: {result['ip']}")
                working_proxies.append(result['proxy'])
            else:
                print(f"   Error: {result.get('error', 'Unknown error')}")
            print("-" * 40)
    
    print(f"\nSUMMARY:")
    print(f"Total proxies tested: {len(proxies)}")
    print(f"Working proxies: {len(working_proxies)}")
    
    if working_proxies:
        print(f"\nWorking proxies:")
        for proxy in working_proxies:
            print(f"  - {proxy}")
    else:
        print("\nNo working proxies found!")
    
    return working_proxies

if __name__ == "__main__":
    working = main()
    
    # Save working proxies to file for easy access
    if working:
        with open("working_proxies.txt", "w") as f:
            for proxy in working:
                f.write(f"{proxy}\n")
        print(f"\nWorking proxies saved to working_proxies.txt")
    else:
        print("\nNo working proxies to save.")