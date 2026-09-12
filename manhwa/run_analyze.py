"""
Fast bulk analysis: all AI calls in parallel, ONE save at the end.
Safe to run alongside the app (no lock contention during API phase).
"""
import sys, os, io, base64, json, threading
sys.path.insert(0, os.path.dirname(__file__))
import character_library as cl
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image as _PILImage
import anthropic

MAX_WORKERS = 10
MAX_EDGE    = 768
JPEG_Q      = 75
MODEL       = cl._DEFAULT_ANALYSIS_MODEL

def _encode_image(path: str) -> str:
    img = _PILImage.open(path).convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_EDGE:
        scale = MAX_EDGE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), _PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_Q)
    return base64.standard_b64encode(buf.getvalue()).decode()

def _call_api(b64: str) -> dict:
    client = anthropic.Anthropic(timeout=90.0)
    msg = client.messages.create(
        model=MODEL,
        max_tokens=1536,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64",
             "media_type": "image/jpeg", "data": b64}},
            {"type": "text", "text": cl._ANALYSIS_PROMPT},
        ]}],
    )
    import re
    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw, re.IGNORECASE).rstrip("`").strip()
    return json.loads(raw)

def _analyze_one(t: dict) -> tuple:
    tid = t["template_id"]
    img_path = t.get("local_face") or t.get("local_body") or ""
    if not img_path or not os.path.exists(img_path):
        return tid, {"error": "no local image"}
    try:
        b64 = _encode_image(img_path)
        analysis = _call_api(b64)
        return tid, analysis
    except Exception as e:
        return tid, {"error": str(e)[:120]}

ANALYSIS_FIELDS = [
    "perceived_gender", "age_group", "apparent_age_range",
    "body_type", "archetype", "scene_suitability", "auto_name",
    "summary", "pose", "setting", "lighting", "art_style",
    "reusability", "generation_prompt", "subject_count", "mood",
]

def _save_batch(results: dict):
    """Atomically merge a batch of results into library.json."""
    import fcntl
    lock_path = cl._LIB_JSON + ".lock"
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            lib = cl.load_library()
            t_map = lib.get("templates", {})
            saved = 0
            for tid, analysis in results.items():
                if "error" in analysis or tid not in t_map:
                    continue
                t = t_map[tid]
                t["ai_analysis"] = analysis
                if not t.get("name") or t.get("name") == tid:
                    t["name"] = analysis.get("auto_name", tid)
                if t.get("category", "other") in ("other", "", None):
                    cat_map = {"female": "female", "male": "male"}
                    t["category"] = cat_map.get(
                        analysis.get("suggested_category", "other"), "other")
                existing_tags = set(t.get("tags") or [])
                ai_tags = set(analysis.get("tags") or [])
                t["tags"] = sorted(existing_tags | ai_tags)
                for field in ANALYSIS_FIELDS:
                    if field in analysis:
                        t[field] = analysis[field]
                saved += 1
            cl._atomic_save(cl._LIB_JSON, lib)
            return saved
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def main(batch_size: int = 200):
    lib = cl.load_library()
    templates = lib.get("templates", {})
    pending = [
        t for t in templates.values()
        if not t.get("ai_analysis")
        and (os.path.exists(t.get("local_face", ""))
             or os.path.exists(t.get("local_body", "")))
    ]
    total = len(pending)
    print(f"To analyze: {total}  Model: {MODEL}  Batch size: {batch_size}", flush=True)
    if total == 0:
        print("Nothing to do.")
        return

    grand_done = grand_err = 0

    # Process in chunks — each chunk is fully analyzed then saved before the next
    for batch_start in range(0, total, batch_size):
        batch = pending[batch_start:batch_start + batch_size]
        batch_end = batch_start + len(batch)
        print(f"\n── Batch {batch_start+1}–{batch_end}/{total} ──", flush=True)

        results: dict = {}
        done = err = 0
        lock = threading.Lock()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_analyze_one, t): t for t in batch}
            for i, fut in enumerate(as_completed(futures), 1):
                tid, analysis = fut.result()
                with lock:
                    results[tid] = analysis
                    if "error" in analysis:
                        err += 1
                        print(f"  [{i}/{len(batch)}] ERR {tid}: {analysis['error'][:60]}", flush=True)
                    else:
                        done += 1
                        if i % 25 == 0 or i == len(batch):
                            print(f"  [{i}/{len(batch)}] ok={done} err={err}", flush=True)

        saved = _save_batch(results)
        grand_done += done
        grand_err  += err
        print(f"  → Saved {saved}. Grand total: ok={grand_done} err={grand_err}", flush=True)

    # Final verification
    lib2 = cl.load_library()
    t2 = lib2.get("templates", {})
    analyzed = sum(1 for x in t2.values() if x.get("ai_analysis"))
    print(f"\nDONE. Library: {len(t2)} total, {analyzed} analyzed", flush=True)

if __name__ == "__main__":
    main()
