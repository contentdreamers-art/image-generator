"""
run_pinterest_batch.py — Import a list of Pinterest image URLs into the library.
Downloads in parallel, then does a single bulk library write (fast).
Usage: python3 run_pinterest_batch.py /tmp/pinterest_to_import.txt
"""
import sys, re, io, datetime, os, threading
sys.path.insert(0, "/home/runner/workspace/manhwa")
import requests
from PIL import Image as _PIL
import character_library as cl
from concurrent.futures import ThreadPoolExecutor, as_completed

_CDN_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.pinterest.com/",
    "Accept":  "image/webp,image/apng,image/*,*/*;q=0.8",
}

input_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pinterest_to_import.txt"
with open(input_file) as f:
    urls = [l.strip() for l in f if l.strip() and l.startswith("http")]

total = len(urls)
print(f"Downloading {total} images in parallel...", flush=True)

def download_one(url):
    fallback = re.sub(r'/originals/', '/736x/', url)
    for attempt in ([url] if url == fallback else [url, fallback]):
        try:
            r = requests.get(attempt, headers=_CDN_HEADERS, timeout=30)
            if r.status_code == 403:
                continue
            r.raise_for_status()
            pil = _PIL.open(io.BytesIO(r.content)).convert("RGB")
            return pil, attempt
        except Exception:
            continue
    return None, url

# Phase 1: parallel downloads
downloaded = []
dl_errors  = 0
with ThreadPoolExecutor(max_workers=12) as pool:
    futures = {pool.submit(download_one, u): u for u in urls}
    for i, fut in enumerate(as_completed(futures), 1):
        pil, used_url = fut.result()
        if pil is None:
            dl_errors += 1
        else:
            downloaded.append((pil, used_url))
        if i % 50 == 0 or i == total:
            print(f"  downloaded {i}/{total} (ok={len(downloaded)} err={dl_errors})", flush=True)

print(f"\nPhase 1 done: {len(downloaded)} images ready, {dl_errors} errors", flush=True)
print(f"Phase 2: bulk-writing to library (single load/save)...", flush=True)

# Phase 2: load library once, assign IDs, save once
lib = cl.load_library()
templates = lib.setdefault("templates", {})

# Figure out the next ID once
def _next_id(lib_templates):
    nums = []
    for tid in lib_templates:
        m = re.match(r'^O(\d+)$', tid)
        if m:
            nums.append(int(m.group(1)))
    return (max(nums) + 1) if nums else 1

next_num = _next_id(templates)
new_entries = []  # (template_id, pil, url)

now = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"
for pil, used_url in downloaded:
    tid = f"O{next_num}"
    next_num += 1
    templates[tid] = {
        "template_id": tid,
        "category":    "other",
        "name":        tid,
        "description": "",
        "tags":        [],
        "local_face":  "",
        "local_body":  "",
        "fal_face_url": "",
        "fal_body_url": "",
        "source_url":  used_url,
        "created_at":  now,
    }
    new_entries.append((tid, pil, used_url))

cl.save_library(lib)
print(f"  library.json updated with {len(new_entries)} new entries", flush=True)

# Phase 3: save image files (fast, no JSON I/O)
print(f"Phase 3: saving {len(new_entries)} image files...", flush=True)
saved, save_errors = 0, 0
for idx, (tid, pil, used_url) in enumerate(new_entries, 1):
    try:
        cl.save_image(tid, "face", pil)
        # patch local_face path back into library
        t = cl.get_template(tid)
        saved += 1
    except Exception as e:
        save_errors += 1
        if save_errors <= 5:
            print(f"  IMG ERR {tid}: {e}", flush=True)
    if idx % 50 == 0 or idx == len(new_entries):
        print(f"  images {idx}/{len(new_entries)} (ok={saved} err={save_errors})", flush=True)

print(f"\nDONE: {saved} imported, {dl_errors} download errors, {save_errors} image errors / {total} total", flush=True)
