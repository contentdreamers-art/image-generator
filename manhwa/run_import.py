import sys, threading
sys.path.insert(0, "/home/runner/workspace/manhwa")
import character_library as cl
from concurrent.futures import ThreadPoolExecutor, as_completed

with open("/tmp/pinterest_to_import.txt") as f:
    urls = [l.strip() for l in f if l.strip()]

total = len(urls)
done, errors = 0, 0
lock = threading.Lock()

def import_one(url):
    try:
        tid, _ = cl.add_from_url(url, category="other", img_type="face", auto_analyze=False)
        return ("ok", tid)
    except Exception as e:
        return ("err", str(e)[:80])

print(f"Starting parallel import of {total} URLs (10 workers, no auto-analyze)...", flush=True)

with ThreadPoolExecutor(max_workers=10) as pool:
    futures = {pool.submit(import_one, u): u for u in urls}
    for i, fut in enumerate(as_completed(futures), 1):
        status, val = fut.result()
        with lock:
            if status == "ok":
                done += 1
            else:
                errors += 1
        if i % 50 == 0 or i == total:
            print(f"[{i}/{total}] imported={done} errors={errors}", flush=True)

print(f"DONE: {done} imported, {errors} errors", flush=True)
