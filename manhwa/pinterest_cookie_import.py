"""
pinterest_cookie_import.py — Import Pinterest saved pins using a browser session cookie.

How to get your cookie (30 seconds):
  1. Open pinterest.com in your browser — make sure you're logged in.
  2. Press F12 (or Cmd+Opt+I on Mac) to open DevTools.
  3. Go to  Application → Cookies → https://www.pinterest.com
  4. Find the row named  _pinterest_sess
  5. Copy the entire Value column (it's a long string starting with TWc...)
  6. Paste it in the Cookie field and click Import.

This uses gallery-dl under the hood — no API approval needed, works immediately.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

import character_library as lib_mod


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_pinterest_username(pinterest_sess: str) -> Optional[str]:
    """Resolve the username for the logged-in account via the Pinterest API.

    gallery-dl 1.32+ dropped the 'pinterest://me' shorthand; we need the real
    username to build  https://www.pinterest.com/<user>/_saved/  instead.
    """
    cookies = {"_pinterest_sess": pinterest_sess}
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "X-Requested-With": "XMLHttpRequest",
    }
    # Pinterest's internal resource endpoint — no API key needed, just the cookie.
    url = (
        "https://www.pinterest.com/resource/UserResource/get/"
        "?source_url=%2F&data=%7B%22options%22%3A%7B%7D%2C%22context%22%3A%7B%7D%7D"
    )
    try:
        resp = requests.get(url, cookies=cookies, headers=headers, timeout=15)
        data = resp.json()
        username = (
            data.get("resource_response", {})
                .get("data", {})
                .get("username")
        )
        return username or None
    except Exception:
        pass
    # Fallback: hit the homepage and parse the username from the embedded JSON.
    try:
        resp = requests.get(
            "https://www.pinterest.com/",
            cookies=cookies, headers=headers, timeout=15,
        )
        import re
        m = re.search(r'"username"\s*:\s*"([^"]+)"', resp.text)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None


def _pinterest_headers(referer: str = "https://www.pinterest.com/") -> Dict[str, str]:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/javascript, */*, q=0.01",
        "Accept-Language": "en-US,en;q=0.9",
        "X-Requested-With": "XMLHttpRequest",
        "X-APP-VERSION": "7b4e983",
        "Referer": referer,
    }


def _get_all_boards(sess: str, username: str) -> List[Dict]:
    """Return all boards for the user, trying several Pinterest API approaches."""
    cookies = {"_pinterest_sess": sess}
    import re as _re

    # ── Approach 1: BoardsResource API (paginated) ────────────────────────────
    boards: List[Dict] = []
    bookmark: Optional[str] = None
    for endpoint in (
        "https://www.pinterest.com/resource/BoardsResource/get/",
        "https://www.pinterest.com/resource/ProfileBoardsResource/get/",
    ):
        boards = []
        bookmark = None
        for _ in range(50):   # max 50 pages × 100 boards = 5 000 boards
            options: Dict[str, Any] = {
                "username": username,
                "field_set_key": "profile_grid_item",
                "sort": "last_pinned_to",
                "page_size": 100,
            }
            if bookmark:
                options["bookmarks"] = [bookmark]
            params = {
                "source_url": f"/{username}/",
                "data": json.dumps({"options": options, "context": {}}),
                "_": str(int(time.time() * 1000)),
            }
            try:
                resp = requests.get(
                    endpoint, params=params, cookies=cookies,
                    headers=_pinterest_headers(f"https://www.pinterest.com/{username}/"),
                    timeout=20,
                )
                data = resp.json()
            except Exception:
                break
            resource = data.get("resource_response", {})
            batch = resource.get("data") or []
            boards.extend(batch)
            new_bm = resource.get("bookmark")
            if not new_bm or new_bm == "-end-" or new_bm == bookmark:
                break
            bookmark = new_bm
        if boards:
            return boards

    # ── Approach 2: scrape the profile page HTML for board links ──────────────
    try:
        resp = requests.get(
            f"https://www.pinterest.com/{username}/",
            cookies=cookies,
            headers=_pinterest_headers(f"https://www.pinterest.com/{username}/"),
            timeout=20,
        )
        # Pinterest embeds page state as JSON in a script tag
        for pattern in (
            r'<script id="__PWS_DATA__"[^>]*>(.*?)</script>',
            r'<script id="initial-state"[^>]*>(.*?)</script>',
            r'window\.__PWS_DATA__\s*=\s*(\{.*?\});\s*</script>',
        ):
            m = _re.search(pattern, resp.text, _re.DOTALL)
            if not m:
                continue
            try:
                pws = json.loads(m.group(1))
                # Walk common paths where board data lives
                for path in (
                    ["props", "initialReduxState", "boards"],
                    ["props", "pageProps", "boards"],
                    ["initialReduxState", "boards"],
                ):
                    node = pws
                    for key in path:
                        node = node.get(key, {}) if isinstance(node, dict) else {}
                    if isinstance(node, dict) and node:
                        boards = [b for b in node.values()
                                  if isinstance(b, dict) and b.get("id")]
                        if boards:
                            return boards
            except Exception:
                continue
        # Fallback: extract board URLs from raw HTML href attributes
        board_slugs = _re.findall(
            rf'href="/{_re.escape(username)}/([^/"]+)/"', resp.text
        )
        if board_slugs:
            return [{"id": None, "slug": s, "name": s} for s in dict.fromkeys(board_slugs)]
    except Exception:
        pass

    return []


def _get_board_image_urls(
    sess: str,
    username: str,
    board_id: str,
    board_slug: str,
    max_pins: int = 100_000,
) -> List[str]:
    """Return all image URLs for a board using Pinterest's BoardFeedResource API."""
    cookies = {"_pinterest_sess": sess}
    image_urls: List[str] = []
    bookmark: Optional[str] = None

    while len(image_urls) < max_pins:
        options: Dict[str, Any] = {
            "board_id": board_id,
            "page_size": 25,
            "prepend": False,
        }
        if bookmark:
            options["bookmarks"] = [bookmark]

        params = {
            "source_url": f"/{username}/{board_slug}/",
            "data": json.dumps({"options": options, "context": {}}),
            "_": str(int(time.time() * 1000)),
        }
        try:
            resp = requests.get(
                "https://www.pinterest.com/resource/BoardFeedResource/get/",
                params=params,
                cookies=cookies,
                headers=_pinterest_headers(f"https://www.pinterest.com/{username}/{board_slug}/"),
                timeout=20,
            )
            data = resp.json()
        except Exception:
            break

        resource = data.get("resource_response", {})
        pins = resource.get("data") or []
        if not pins:
            break

        for pin in pins:
            # Walk image size keys largest-first to get best quality
            images = pin.get("images") or {}
            url = None
            for size_key in ("orig", "1200x", "736x", "474x"):
                img = images.get(size_key)
                if img and img.get("url"):
                    url = img["url"]
                    break
            if url:
                image_urls.append(url)

        new_bookmark = resource.get("bookmark")
        if not new_bookmark or new_bookmark == "-end-" or new_bookmark == bookmark:
            break
        bookmark = new_bookmark

    return image_urls


def _download_image(url: str, dest_path: str) -> bool:
    """Download a single image URL to dest_path. Returns True on success."""
    try:
        resp = requests.get(url, timeout=30, stream=True)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in resp.iter_content(65536):
                f.write(chunk)
        return True
    except Exception:
        return False


def _write_gdl_config(tmpdir: str, pinterest_sess: str) -> str:
    """Write a gallery-dl config file that authenticates with the session cookie."""
    cfg = {
        "extractor": {
            "pinterest": {
                "cookies": {
                    "_pinterest_sess": pinterest_sess.strip(),
                },
                # Download the best available image size
                "images": True,
                # Flatten board sub-dirs into one directory
                "directory": ["{category}"],
                "filename":  "{pin_id}.{extension}",
            },
            # Global download directory
            "base-directory": tmpdir,
        },
        "downloader": {
            "http": {
                "retries": 3,
                "timeout":  30,
            },
        },
    }
    cfg_path = os.path.join(tmpdir, "gdl_config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)
    return cfg_path


def _run_gallery_dl(
    url: str,
    cfg_path: str,
    download_dir: str,
    max_count: int = 100_000,
    progress_cb: Optional[Callable] = None,
) -> Tuple[List[str], List[str]]:
    """
    Run gallery-dl and return (downloaded_file_paths, error_messages).
    gallery-dl is called as a subprocess so we can stream its output.
    """
    gdl_bin = shutil.which("gallery-dl") or sys.executable + " -m gallery_dl"

    cmd = [
        "gallery-dl",
        "--config", cfg_path,
        "--dest",   download_dir,
        "--range",  f"1-{max_count}",
        "--no-skip",
        url,
    ]

    downloaded: List[str] = []
    errors:     List[str] = []

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        count = 0
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.rstrip()
            if not line:
                continue
            # gallery-dl prints the destination path when it saves a file
            if line.startswith("#") or "Downloading" in line:
                continue
            # Lines that look like file paths
            if os.sep in line and any(
                line.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif")
            ):
                downloaded.append(line)
                count += 1
                if progress_cb:
                    progress_cb(count, max_count, Path(line).stem[:40])
            elif "error" in line.lower() or "warning" in line.lower():
                errors.append(line)

        proc.wait()
        if proc.returncode not in (0, 1):  # 1 = some skipped, not fatal
            errors.append(f"gallery-dl exited with code {proc.returncode}")

    except FileNotFoundError:
        # Fallback: run via python -m gallery_dl
        try:
            result = subprocess.run(
                [sys.executable, "-m", "gallery_dl",
                 "--config", cfg_path,
                 "--dest",   download_dir,
                 "--range",  f"1-{max_count}",
                 url],
                capture_output=True, text=True, timeout=3600,
            )
            # Parse downloaded paths from output
            for line in result.stdout.splitlines():
                line = line.strip()
                if os.sep in line and any(
                    line.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")
                ):
                    downloaded.append(line)
            if result.returncode not in (0, 1):
                errors.append(result.stderr[-500:] if result.stderr else "Unknown error")
        except Exception as e:
            errors.append(f"gallery-dl launch failed: {e}")

    # Fallback: scan download_dir for any images gallery-dl placed there
    if not downloaded:
        for root, _, files in os.walk(download_dir):
            for fname in files:
                if any(fname.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")):
                    downloaded.append(os.path.join(root, fname))

    return downloaded, errors


# ── Public API ────────────────────────────────────────────────────────────────

def import_saved_pins_via_cookie(
    pinterest_sess:  str,
    max_count:       int  = 100_000,
    auto_analyze:    bool = False,
    progress_cb:     Optional[Callable] = None,   # (current, total, label)
) -> Dict[str, Any]:
    """
    Import all saved Pinterest pins using a browser session cookie.

    pinterest_sess   — value of the _pinterest_sess cookie from your browser.
    max_count        — cap on how many pins to import in one run.
    auto_analyze     — run Claude Sonnet on each image (do in bulk later for speed).
    progress_cb      — callable(current, total, label_str).

    Returns {"imported": [...template_ids], "skipped": N, "errors": [...], "total": N}
    """
    from PIL import Image as _PIL
    import io

    sess = (pinterest_sess or "").strip()
    if not sess:
        return {"imported": [], "skipped": 0,
                "errors": [("cookie", "No _pinterest_sess cookie provided.")],
                "total": 0}

    # Validate cookie looks plausible
    if len(sess) < 20:
        return {"imported": [], "skipped": 0,
                "errors": [("cookie", "Cookie value looks too short — make sure you copied the full Value.")],
                "total": 0}

    from PIL import Image as _PIL
    import io as _io

    if progress_cb:
        progress_cb(0, max_count, "Connecting to Pinterest…")

    # Step 1 — resolve username
    username = _get_pinterest_username(sess)
    if not username:
        return {
            "imported": [], "skipped": 0, "total": 0,
            "errors": [("cookie", (
                "Could not resolve your Pinterest username from the cookie. "
                "Make sure you copied the full _pinterest_sess value and are still logged in."
            ))],
        }

    if progress_cb:
        progress_cb(0, max_count, f"Loading boards for @{username}…")

    # Step 2 — enumerate ALL boards via internal API (handles pagination)
    if progress_cb:
        progress_cb(0, max_count, f"Loading boards for @{username}…")
    boards = _get_all_boards(sess, username)

    # Step 3 — collect all image URLs
    all_image_urls: List[str] = []

    if boards:
        # API / HTML scrape returned board list — pull pins from each board
        for bi, board in enumerate(boards):
            if len(all_image_urls) >= max_count:
                break
            board_id   = board.get("id") or ""
            board_slug = (
                board.get("slug")
                or board.get("url", "").rstrip("/").split("/")[-1]
                or board.get("name", "")
            )
            board_name = board.get("name") or board_slug
            if not board_slug:
                continue
            if progress_cb:
                progress_cb(
                    len(all_image_urls), max_count,
                    f"Scanning board {bi+1}/{len(boards)}: {board_name[:30]}…",
                )
            # If we have a board_id use the API; otherwise use gallery-dl on board URL
            if board_id:
                urls = _get_board_image_urls(
                    sess, username, board_id, board_slug,
                    max_pins=max_count - len(all_image_urls),
                )
                all_image_urls.extend(urls)
            else:
                # HTML-scrape fallback: run gallery-dl on the board URL
                tmpdir_b = tempfile.mkdtemp(prefix="pinterest_board_")
                try:
                    cfg_b = _write_gdl_config(tmpdir_b, sess)
                    dl_dir = os.path.join(tmpdir_b, "imgs")
                    os.makedirs(dl_dir, exist_ok=True)
                    board_url = f"https://www.pinterest.com/{username}/{board_slug}/"
                    dl_files, _ = _run_gallery_dl(
                        board_url, cfg_path=cfg_b, download_dir=dl_dir,
                        max_count=max_count - len(all_image_urls),
                    )
                    all_image_urls.extend(["file://" + f for f in dl_files])
                finally:
                    pass   # cleaned up per-board below after download loop
    else:
        # Last resort: gallery-dl on the full _saved/ URL
        if progress_cb:
            progress_cb(0, max_count, f"Falling back to gallery-dl for @{username}…")
        tmpdir_gdl = tempfile.mkdtemp(prefix="pinterest_gdl_fb_")
        try:
            cfg_gdl  = _write_gdl_config(tmpdir_gdl, sess)
            dl_dir   = os.path.join(tmpdir_gdl, "imgs")
            os.makedirs(dl_dir, exist_ok=True)
            dl_files, dl_errs = _run_gallery_dl(
                f"https://www.pinterest.com/{username}/_saved/",
                cfg_path=cfg_gdl, download_dir=dl_dir, max_count=max_count,
                progress_cb=progress_cb,
            )
            if not dl_files:
                return {
                    "imported": [], "skipped": 0, "total": 0,
                    "errors": [("boards", (
                        "Could not find any pins. "
                        "The cookie may have expired — re-copy _pinterest_sess from DevTools. "
                        f"Details: {'; '.join(dl_errs[:3]) if dl_errs else 'no images found'}"
                    ))],
                }
            all_image_urls = ["file://" + f for f in dl_files]
        finally:
            pass   # tmpdir_gdl cleaned up later

    total_found = len(all_image_urls)
    if progress_cb and total_found:
        progress_cb(0, total_found, f"Found {total_found} pins. Downloading…")

    # Step 4 — download and add to library
    existing_urls = lib_mod.get_existing_source_urls()   # set of already-imported source URLs
    imported: List[str]             = []
    errors:   List[Tuple[str, str]] = []
    skipped   = 0

    for i, img_url in enumerate(all_image_urls):
        if len(imported) + skipped >= max_count:
            break

        if img_url in existing_urls:
            skipped += 1
            if progress_cb and i % 10 == 0:
                progress_cb(i + 1, total_found, f"Skipping already-imported pins… ({skipped} skipped)")
            continue

        if progress_cb and i % 5 == 0:
            progress_cb(i + 1, total_found, f"Importing {i+1}/{total_found}…")

        try:
            if img_url.startswith("file://"):
                # Local file from gallery-dl fallback
                local_path = img_url[len("file://"):]
                pil = _PIL.open(local_path).convert("RGB")
            else:
                resp = requests.get(img_url, timeout=30, stream=True)
                resp.raise_for_status()
                img_bytes = resp.content
                pil = _PIL.open(_io.BytesIO(img_bytes)).convert("RGB")

            template_id, _ = lib_mod.add_template("other")
            lib_mod.save_image(template_id, "face", pil)

            base: Dict[str, Any] = {
                "source_url": img_url,
                "name":       template_id,
            }

            if auto_analyze:
                try:
                    analysis = lib_mod.analyze_image_with_ai(pil)
                    cat_map  = {"female": "female", "male": "male",
                                "setting": "setting", "creature": "creature"}
                    updates  = {
                        **base,
                        "ai_analysis": analysis,
                        "name":        analysis.get("auto_name", base["name"]),
                        "category":    cat_map.get(
                            analysis.get("suggested_category", "other"), "other"),
                        "tags":        sorted(set(analysis.get("tags") or [])),
                    }
                    for field in ("perceived_gender", "age_group", "body_type",
                                  "archetype", "mood", "scene_suitability"):
                        if field in analysis:
                            updates[field] = analysis[field]
                    lib_mod.update_template(template_id, **updates)
                except Exception:
                    lib_mod.update_template(template_id, **base)
            else:
                lib_mod.update_template(template_id, **base)

            imported.append(template_id)
            existing_urls.add(img_url)

        except Exception as ex:
            errors.append((img_url, str(ex)))

    return {
        "imported": imported,
        "skipped":  skipped,
        "errors":   errors,
        "total":    total_found,
    }


def import_board_via_cookie(
    board_url:    str,
    pinterest_sess: str,
    max_count:    int  = 10_000,
    auto_analyze: bool = False,
    progress_cb:  Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Import a specific Pinterest board using a browser session cookie.
    board_url — full URL like https://www.pinterest.com/username/board-name/
    """
    from PIL import Image as _PIL

    sess = (pinterest_sess or "").strip()
    if not sess or not board_url:
        return {"imported": [], "skipped": 0,
                "errors": [("input", "Provide both a board URL and your session cookie.")],
                "total": 0}

    tmpdir = tempfile.mkdtemp(prefix="pinterest_gdl_board_")
    try:
        cfg_path     = _write_gdl_config(tmpdir, sess)
        download_dir = os.path.join(tmpdir, "images")
        os.makedirs(download_dir, exist_ok=True)

        dl_files, dl_errors = _run_gallery_dl(
            board_url.strip(),
            cfg_path=cfg_path,
            download_dir=download_dir,
            max_count=max_count,
            progress_cb=progress_cb,
        )

        existing_ids = lib_mod.get_existing_pinterest_ids()
        imported: List[str] = []
        errors: List[Tuple[str, str]] = []
        skipped = 0

        for i, fpath in enumerate(dl_files):
            pin_id = Path(fpath).stem
            if pin_id in existing_ids:
                skipped += 1
                continue
            try:
                pil = _PIL.open(fpath).convert("RGB")
                template_id, _ = lib_mod.add_template("other")
                lib_mod.save_image(template_id, "face", pil)
                base = {
                    "source_url":       f"https://www.pinterest.com/pin/{pin_id}/",
                    "pinterest_pin_id": pin_id,
                    "name":             template_id,
                }
                if auto_analyze:
                    try:
                        analysis = lib_mod.analyze_image_with_ai(pil)
                        cat_map  = {"female": "female", "male": "male",
                                    "setting": "setting", "creature": "creature"}
                        lib_mod.update_template(template_id, **{
                            **base,
                            "ai_analysis": analysis,
                            "name":        analysis.get("auto_name", base["name"]),
                            "category":    cat_map.get(
                                analysis.get("suggested_category", "other"), "other"),
                            "tags":        sorted(set(analysis.get("tags") or [])),
                        })
                    except Exception:
                        lib_mod.update_template(template_id, **base)
                else:
                    lib_mod.update_template(template_id, **base)
                imported.append(template_id)
                existing_ids.add(pin_id)
            except Exception as ex:
                errors.append((fpath, str(ex)))

        return {"imported": imported, "skipped": skipped,
                "errors": errors, "total": len(dl_files)}

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
