"""
pinterest_import.py — Import Pinterest saved pins into the character library.

Two modes:

1. DATA EXPORT (recommended — no API key needed):
   - User requests their data at pinterest.com/settings/privacy → "Download your data"
   - Pinterest emails a ZIP file (a few hours)
   - User uploads the ZIP here → import_from_export_zip(zip_path) handles it
   - The ZIP contains user_data/saves.json or pins.json with every saved pin

2. API MODE (requires PINTEREST_ACCESS_TOKEN secret):
   - Only works once Pinterest approves the developer app
   - import_all_pins() uses GET /v5/pins to stream every saved pin
"""

import csv
import io
import json
import os
import time
import zipfile
from typing import Any, Dict, Generator, List, Optional, Tuple

import requests

_API_BASE     = "https://api.pinterest.com/v5"
_IMAGE_EXTS   = {".jpg", ".jpeg", ".png", ".webp"}
_PAGE_SIZE    = 100   # max pins per API page
_DOWNLOAD_TBF = 0.1  # seconds between image downloads (be a good citizen)

# Pinterest CDN requires Referer header — "originals" returns 403 without it;
# use 736x which is publicly accessible with these headers.
_CDN_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer":  "https://www.pinterest.com/",
    "Accept":   "image/webp,image/apng,image/*,*/*;q=0.8",
}


# ── Auth helpers ──────────────────────────────────────────────────────────────

def _token() -> str:
    tok = os.environ.get("PINTEREST_ACCESS_TOKEN", "")
    if not tok:
        raise RuntimeError(
            "PINTEREST_ACCESS_TOKEN not found. "
            "Connect your Pinterest account in Replit Settings → Connectors → Pinterest."
        )
    return tok


def _headers() -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {_token()}",
        "Content-Type":  "application/json",
    }


def _get(path: str, params: Optional[Dict] = None) -> Dict[str, Any]:
    url = f"{_API_BASE}{path}"
    r   = requests.get(url, headers=_headers(), params=params or {}, timeout=20)
    r.raise_for_status()
    return r.json()


# ── Board listing ─────────────────────────────────────────────────────────────

def list_boards() -> List[Dict[str, Any]]:
    """
    Return all boards for the authenticated user.
    Each dict has: id, name, description, pin_count, privacy.
    """
    boards: List[Dict] = []
    bookmark: Optional[str] = None

    while True:
        params: Dict[str, Any] = {"page_size": 100}
        if bookmark:
            params["bookmark"] = bookmark
        data     = _get("/boards", params)
        items    = data.get("items") or []
        boards.extend(items)
        bookmark = data.get("bookmark")
        if not items or not bookmark:
            break

    return boards


def board_choices() -> List[str]:
    """
    Return human-readable choice strings: "Board Name (123 pins) [id]"
    Sorted by name. Prepend the special "Saved" / "All Pins" virtual entries.
    """
    try:
        boards = list_boards()
    except Exception as e:
        return [f"⚠️ Could not load boards: {e}"]

    items = []
    for b in sorted(boards, key=lambda x: x.get("name", "").lower()):
        name  = b.get("name", "Untitled")
        count = b.get("pin_count", "?")
        bid   = b.get("id", "")
        items.append(f"{name} ({count} pins)  [{bid}]")
    return items or ["No boards found"]


def _parse_board_id(choice: str) -> str:
    """Extract board ID from "Board Name (123 pins)  [board_id]" string."""
    if "[" in choice and choice.endswith("]"):
        return choice.rsplit("[", 1)[1].rstrip("]").strip()
    return choice.strip()


# ── Pin fetching ──────────────────────────────────────────────────────────────

def _pins_from_board(board_id: str, max_pins: int = 1000) -> Generator[Dict, None, None]:
    """Yield pin dicts from a board, up to max_pins, using cursor pagination."""
    fetched:  int           = 0
    bookmark: Optional[str] = None

    while fetched < max_pins:
        batch = min(_PAGE_SIZE, max_pins - fetched)
        params: Dict[str, Any] = {"page_size": batch}
        if bookmark:
            params["bookmark"] = bookmark

        try:
            data = _get(f"/boards/{board_id}/pins", params)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return   # board not found / no access
            raise

        items = data.get("items") or []
        for pin in items:
            yield pin
            fetched += 1

        bookmark = data.get("bookmark")
        if not items or not bookmark or fetched >= max_pins:
            break


def _best_image_url(pin: Dict) -> Optional[str]:
    """
    Extract the best available image URL from a pin dict.
    Prefers original > 1200x > 736x > 600x > 400x.
    """
    media  = pin.get("media") or {}
    images = media.get("images") or {}

    for key in ("original", "1200x", "736x", "600x", "400x", "150x150"):
        entry = images.get(key) or {}
        url   = entry.get("url", "")
        if url:
            return url

    # Fallback: alt_text link or direct url field
    return pin.get("link") or None


# ── Main import entry point ───────────────────────────────────────────────────

def import_board_images(
    board_choice: str,
    max_count:    int  = 200,
    img_type:     str  = "face",
    auto_analyze: bool = True,
    progress_cb         = None,
) -> Dict[str, Any]:
    """
    Fetch pins from a board, download each image, and add it to the character
    library. AI analysis runs per image to auto-tag every dimension.

    board_choice — string from board_choices() or a raw board_id.
    max_count    — maximum number of images to import.
    img_type     — "face" or "body" (which slot to fill in the template).
    auto_analyze — run Claude Vision on each image (recommended).
    progress_cb  — optional callable(current, total, label) for progress updates.

    Returns {"imported": [...], "skipped": n, "errors": [...], "total": n}.
    """
    import character_library as lib_mod
    from PIL import Image as _PIL

    board_id = _parse_board_id(board_choice)
    if not board_id:
        return {"imported": [], "skipped": 0, "errors": [("", "No board ID")], "total": 0}

    imported:  List[str]        = []
    errors:    List[Tuple[str, str]] = []
    skipped:   int              = 0
    pin_list   = list(_pins_from_board(board_id, max_count))
    total      = len(pin_list)

    for i, pin in enumerate(pin_list):
        pin_id = pin.get("id", f"pin_{i}")
        label  = pin.get("title") or pin.get("alt_text") or pin_id

        if progress_cb:
            progress_cb(i + 1, total, f"{label[:40]}")

        image_url = _best_image_url(pin)
        if not image_url:
            skipped += 1
            continue

        try:
            r = requests.get(image_url, headers=_CDN_HEADERS, timeout=30)
            r.raise_for_status()
            pil = _PIL.open(io.BytesIO(r.content)).convert("RGB")

            # Create template (AI will decide category if auto_analyze)
            template_id, _ = lib_mod.add_template("other")
            lib_mod.save_image(template_id, img_type, pil)

            if auto_analyze:
                try:
                    analysis = lib_mod.analyze_image_with_ai(pil)
                    cat_map  = {"female": "female", "male": "male",
                                "setting": "setting", "creature": "creature"}
                    detected_cat = cat_map.get(
                        analysis.get("suggested_category", "other"), "other"
                    )
                    updates: Dict[str, Any] = {
                        "ai_analysis":  analysis,
                        "name":         analysis.get("auto_name", template_id),
                        "category":     detected_cat,
                        "tags":         sorted(set(analysis.get("tags") or [])),
                        "source_url":   image_url,
                        "pinterest_pin_id": pin_id,
                    }
                    for field in ("perceived_gender", "age_group", "body_type",
                                  "archetype", "mood", "scene_suitability"):
                        if field in analysis:
                            updates[field] = analysis[field]
                    lib_mod.update_template(template_id, **updates)
                except Exception:
                    # Keep the template even if analysis fails
                    lib_mod.update_template(
                        template_id,
                        source_url=image_url,
                        pinterest_pin_id=pin_id,
                    )
            else:
                lib_mod.update_template(
                    template_id,
                    source_url=image_url,
                    pinterest_pin_id=pin_id,
                )

            imported.append(template_id)
            time.sleep(_DOWNLOAD_TBF)

        except Exception as e:
            errors.append((pin_id, str(e)))

    return {
        "imported": imported,
        "skipped":  skipped,
        "errors":   errors,
        "total":    total,
    }


# ── All saved pins (primary entry point) ─────────────────────────────────────

def _all_pins_generator(max_pins: int = 10000) -> Generator[Dict, None, None]:
    """
    Yield every pin the authenticated user has saved, using GET /v5/pins.
    This is the canonical "all my pins" endpoint — not filtered by board.
    Handles cursor-based pagination automatically.
    """
    fetched:  int           = 0
    bookmark: Optional[str] = None

    while fetched < max_pins:
        batch  = min(_PAGE_SIZE, max_pins - fetched)
        params: Dict[str, Any] = {"page_size": batch}
        if bookmark:
            params["bookmark"] = bookmark

        try:
            data = _get("/pins", params)
        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code in (401, 403):
                raise RuntimeError(
                    "Pinterest access denied. Reconnect your account in Replit Connectors."
                ) from e
            raise

        items = data.get("items") or []
        for pin in items:
            yield pin
            fetched += 1

        bookmark = data.get("bookmark")
        if not items or not bookmark or fetched >= max_pins:
            break


def import_all_pins(
    max_count:    int  = 10000,
    auto_analyze: bool = True,
    progress_cb         = None,
) -> Dict[str, Any]:
    """
    Import EVERY pin the user has ever saved on Pinterest.

    Uses GET /v5/pins — streams all saved pins across all boards, no board
    selection needed.  Already-imported pins are skipped (deduplication by
    pinterest_pin_id).  Image type (face vs body) is auto-detected from AI
    analysis: face-only images → face slot, full-body → body slot.

    max_count    — cap (default 10 000 = effectively unlimited).
    auto_analyze — run Claude Sonnet on each image for full trope + tag detection.
                   Disable for speed; run bulk_analyze_unanalyzed() later.
    progress_cb  — callable(current, total_so_far, label).

    Returns {"imported": [...], "skipped": n, "errors": [...], "total": n}.
    """
    import character_library as lib_mod
    from PIL import Image as _PIL

    # Build dedup set once upfront
    existing_pin_ids = lib_mod.get_existing_pinterest_ids()

    imported:  List[str]             = []
    errors:    List[Tuple[str, str]] = []
    skipped:   int                   = 0
    processed: int                   = 0

    for pin in _all_pins_generator(max_count):
        processed += 1
        pin_id = pin.get("id", f"pin_{processed}")
        label  = pin.get("title") or pin.get("alt_text") or pin_id

        if progress_cb:
            progress_cb(processed, max_count, f"{label[:40]}")

        # ── Deduplication ──────────────────────────────────────────────────
        if str(pin_id) in existing_pin_ids:
            skipped += 1
            continue

        image_url = _best_image_url(pin)
        if not image_url:
            skipped += 1
            continue

        try:
            r = requests.get(image_url, headers=_CDN_HEADERS, timeout=30)
            r.raise_for_status()
            pil = _PIL.open(io.BytesIO(r.content)).convert("RGB")

            template_id, _ = lib_mod.add_template("other")

            base_updates: Dict[str, Any] = {
                "source_url":       image_url,
                "pinterest_pin_id": pin_id,
            }

            if auto_analyze:
                try:
                    analysis     = lib_mod.analyze_image_with_ai(pil)
                    # Auto-detect face vs body slot from AI result
                    if analysis.get("is_face_only"):
                        img_type = "face"
                    elif analysis.get("is_full_body"):
                        img_type = "body"
                    else:
                        img_type = "face"  # default — face refs are most useful
                    lib_mod.save_image(template_id, img_type, pil)

                    cat_map = {"female": "female", "male": "male",
                               "setting": "setting", "creature": "creature"}
                    detected_cat = cat_map.get(
                        analysis.get("suggested_category", "other"), "other"
                    )
                    updates = {
                        **base_updates,
                        "ai_analysis": analysis,
                        "name":        analysis.get("auto_name", template_id),
                        "category":    detected_cat,
                        "tags":        sorted(set(analysis.get("tags") or [])),
                    }
                    for field in ("perceived_gender", "age_group", "body_type",
                                  "archetype", "mood", "scene_suitability",
                                  "is_face_only", "is_full_body"):
                        if field in analysis:
                            updates[field] = analysis[field]
                    lib_mod.update_template(template_id, **updates)
                except Exception:
                    # Analysis failed — save as face, keep basic metadata
                    lib_mod.save_image(template_id, "face", pil)
                    lib_mod.update_template(template_id, **base_updates)
            else:
                # No analysis — save to face slot by default
                lib_mod.save_image(template_id, "face", pil)
                lib_mod.update_template(template_id, **base_updates)

            imported.append(template_id)
            existing_pin_ids.add(str(pin_id))  # prevent duplicates within same run
            time.sleep(_DOWNLOAD_TBF)

        except Exception as e:
            errors.append((pin_id, str(e)))

    return {
        "imported": imported,
        "skipped":  skipped,
        "errors":   errors,
        "total":    processed,
    }


def import_saved_pins(
    max_count:    int  = 10000,
    auto_analyze: bool = True,
    progress_cb         = None,
) -> Dict[str, Any]:
    """Alias for import_all_pins() — kept for backwards compatibility."""
    return import_all_pins(
        max_count=max_count,
        auto_analyze=auto_analyze,
        progress_cb=progress_cb,
    )


# ── Pinterest Data Export importer ────────────────────────────────────────────

def _extract_image_url_from_export_pin(pin: Dict) -> Optional[str]:
    """
    Extract the best image URL from a pin record in Pinterest's data export.
    Handles both the regular export format and the GDPR Subject Access Request format.
    """
    # ── GDPR / Subject Access Request format ──────────────────────────────
    # Keys use title case with spaces: "Image URL", "Pin ID", etc.
    sar_url = pin.get("Image URL") or pin.get("image url") or pin.get("Image Url")
    if sar_url and isinstance(sar_url, str) and sar_url.startswith("http"):
        return sar_url

    # ── Regular / modern export (2023+): nested images dict ───────────────
    images = pin.get("images") or pin.get("media") or {}
    if isinstance(images, dict):
        for key in ("original", "1200x", "736x", "600x", "474x", "236x"):
            entry = images.get(key) or {}
            url = entry.get("url") if isinstance(entry, dict) else None
            if url:
                return url

    # ── Flat fields used in older exports ─────────────────────────────────
    for field in ("image_url", "image", "dominant_color"):
        val = pin.get(field)
        if val and isinstance(val, str) and val.startswith("http"):
            if any(val.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")):
                return val

    # ── Pinterest CDN URL constructed from pin id ──────────────────────────
    pin_id = (
        pin.get("Pin ID") or pin.get("id") or pin.get("pin_id") or pin.get("pin id")
    )
    if pin_id:
        pid = str(pin_id).strip()
        if pid.isdigit() and len(pid) >= 6:
            return (f"https://i.pinimg.com/736x/"
                    f"{pid[:2]}/{pid[2:4]}/{pid[4:6]}/{pid}.jpg")

    return None


def _parse_pins_from_sar_html(html_content: str) -> List[Dict]:
    """
    Parse Pinterest SAR (Subject Access Request) HTML export files.

    Each pin block looks like:
        <a href="https://www.pinterest.com/pin/1146729123902508087/">...</a>
        Image: a59ba0930571f4fdbd15d9e1a67c85e6 <br>
        Board Name: Quick Saves <br>
        ...

    The image hash maps to CDN URL:
        https://i.pinimg.com/originals/a5/9b/a0/a59ba0930571f4fdbd15d9e1a67c85e6.jpg
    """
    import re

    pins: List[Dict] = []

    # Extract all pin URLs — each one starts a new pin block
    pin_url_pattern = re.compile(
        r'href="(https://www\.pinterest\.com/pin/(\d+)/)"'
    )
    # Image hash on its own line: "Image: <32-char hex>"
    image_line_pattern = re.compile(r'Image:\s*([a-f0-9]{32})', re.IGNORECASE)
    # Image hash inside JSON: "image": "..."
    image_json_pattern = re.compile(r'"image"\s*:\s*"([a-f0-9]{32})"')
    # Board name
    board_pattern  = re.compile(r'Board Name:\s*(.+?)\s*<br', re.IGNORECASE)
    title_pattern  = re.compile(r'Title:\s*(.+?)\s*<br', re.IGNORECASE)

    # Split by pin URL anchors to get one block per pin
    blocks = pin_url_pattern.split(html_content)
    # blocks = [pre, full_url, pin_id, content, full_url, pin_id, content, ...]
    i = 1
    while i + 2 < len(blocks):
        full_url = blocks[i]
        pin_id   = blocks[i + 1]
        content  = blocks[i + 2]
        i += 3

        # Extract image hash — prefer the plain "Image:" line, fall back to JSON
        img_match = image_line_pattern.search(content) or image_json_pattern.search(content)
        if not img_match:
            continue
        img_hash = img_match.group(1).lower()

        # Build CDN URL — 736x is publicly accessible; originals returns 403
        image_url = (
            f"https://i.pinimg.com/736x/"
            f"{img_hash[:2]}/{img_hash[2:4]}/{img_hash[4:6]}/{img_hash}.jpg"
        )

        board_m = board_pattern.search(content)
        title_m = title_pattern.search(content)
        board   = board_m.group(1).strip() if board_m else ""
        title   = title_m.group(1).strip() if title_m else ""
        if title.lower() in ("no data", ""):
            title = ""

        pins.append({
            "Pin ID":    pin_id,
            "Image URL": image_url,
            "img_hash":  img_hash,
            "pin_url":   full_url,
            "Board Name": board,
            "Title":     title,
        })

    return pins


def _load_pins_from_export_zip(zip_path: str) -> Tuple[List[Dict], str]:
    """
    Open a Pinterest data export ZIP and return (pins, debug_info).

    Handles all known export format variations:
      - GDPR SAR: paginated HTML files in pins/ folder  ← Pinterest's current format
      - JSON files (older exports)
      - CSV fallback
    """
    pins: List[Dict] = []
    debug_lines: List[str] = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        debug_lines.append(
            f"ZIP has {len(names)} files: " +
            ", ".join(sorted(names)[:40]) + ("…" if len(names) > 40 else "")
        )

        # ── Priority 1: SAR HTML files in pins/ folder ─────────────────────
        # Pinterest's current GDPR export uses paginated HTML: pins/0001.html, etc.
        html_files = sorted([
            n for n in names
            if n.lower().endswith(".html") and "pin" in n.lower()
        ])
        if not html_files:
            # Also try any HTML file at all
            html_files = sorted([n for n in names if n.lower().endswith(".html")
                                  and "start_here" not in n.lower()])

        for hname in html_files:
            try:
                with zf.open(hname) as f:
                    content = f.read().decode("utf-8", errors="replace")
                found = _parse_pins_from_sar_html(content)
                debug_lines.append(f"  HTML {hname}: {len(found)} pins")
                pins.extend(found)
            except Exception as e:
                debug_lines.append(f"  HTML {hname} failed: {e}")

        if pins:
            debug_lines.append(f"Total from HTML: {len(pins)} pins")
            return pins, "\n".join(debug_lines)

        # ── Priority 2: JSON files ─────────────────────────────────────────
        json_files = sorted(
            [n for n in names if n.lower().endswith(".json")],
            key=lambda x: (0 if "pin" in x.lower() else 1, x.count("/"), len(x))
        )
        for jname in json_files:
            try:
                with zf.open(jname) as f:
                    data = json.load(f)
            except Exception as e:
                debug_lines.append(f"  JSON {jname} failed: {e}")
                continue

            found: List[Dict] = []
            if isinstance(data, list):
                found = [x for x in data if isinstance(x, dict)]
            elif isinstance(data, dict):
                for key in ("pins", "Pins", "saves", "items", "data", "results"):
                    val = data.get(key)
                    if isinstance(val, list):
                        found = [x for x in val if isinstance(x, dict)]
                        break
                if not found:
                    for key in ("boards", "board_saves"):
                        for board in (data.get(key) or []):
                            found.extend(board.get("pins") or board.get("saves") or [])
            debug_lines.append(f"  JSON {jname}: {len(found)} records")
            pins.extend(found)

        if pins:
            return pins, "\n".join(debug_lines)

        # ── Priority 3: CSV fallback ───────────────────────────────────────
        for cname in sorted(n for n in names if n.lower().endswith(".csv")):
            try:
                with zf.open(cname) as f:
                    text = f.read().decode("utf-8", errors="replace").splitlines()
                rows = list(csv.DictReader(iter(text)))
                debug_lines.append(f"  CSV {cname}: {len(rows)} rows")
                if rows:
                    pins.extend(rows)
            except Exception as e:
                debug_lines.append(f"  CSV {cname} failed: {e}")

    return pins, "\n".join(debug_lines)


def import_from_export_zip(
    zip_path:     str,
    max_count:    int  = 100_000,
    auto_analyze: bool = False,
    progress_cb         = None,
) -> Dict[str, Any]:
    """
    Import pins from a Pinterest data export ZIP.

    How to get the export:
      pinterest.com → Settings → Privacy and data → Download your data → Request

    Pinterest emails a download link within a few hours.  Upload the ZIP here
    and this function reads it, downloads every saved pin image from Pinterest's
    CDN, and adds them to the library.  No API token needed.

    auto_analyze  — run Claude Sonnet per image (recommended to do later in bulk).
    progress_cb   — callable(current, total, label).
    """
    import character_library as lib_mod
    from PIL import Image as _PIL

    try:
        all_pins, debug_info = _load_pins_from_export_zip(zip_path)
    except Exception as e:
        return {"imported": [], "skipped": 0, "errors": [(zip_path, str(e))], "total": 0,
                "debug": str(e)}

    if not all_pins:
        return {
            "imported": [], "skipped": 0,
            "errors":   [("parse", f"No pins found.\n{debug_info}")],
            "total": 0,
            "debug": debug_info,
        }

    # Cap and deduplicate by pin id
    existing_ids = lib_mod.get_existing_pinterest_ids()
    pins_to_import = []
    seen: set = set()
    for p in all_pins:
        pid = str(
            p.get("Pin ID") or p.get("pin id") or
            p.get("id") or p.get("pin_id") or id(p)
        ).strip()
        if pid not in existing_ids and pid not in seen:
            pins_to_import.append((pid, p))
            seen.add(pid)
        if len(pins_to_import) >= max_count:
            break

    total     = len(pins_to_import)
    imported: List[str]             = []
    errors:   List[Tuple[str, str]] = []
    skipped   = len(all_pins) - total

    for i, (pin_id, pin) in enumerate(pins_to_import):
        label = pin.get("title") or pin.get("note") or pin_id
        if progress_cb:
            progress_cb(i + 1, total, str(label)[:40])

        image_url = _extract_image_url_from_export_pin(pin)
        if not image_url:
            skipped += 1
            continue

        try:
            r = requests.get(
                image_url,
                headers=_CDN_HEADERS,
                timeout=30,
            )
            r.raise_for_status()
            pil = _PIL.open(io.BytesIO(r.content)).convert("RGB")

            template_id, _ = lib_mod.add_template("other")
            lib_mod.save_image(template_id, "face", pil)

            base: Dict[str, Any] = {
                "source_url":       image_url,
                "pinterest_pin_id": pin_id,
                "name":             str(label)[:60] if label != pin_id else template_id,
            }

            if auto_analyze:
                try:
                    analysis = lib_mod.analyze_image_with_ai(pil)
                    cat_map = {"female": "female", "male": "male",
                               "setting": "setting", "creature": "creature"}
                    updates = {
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
            existing_ids.add(pin_id)
            time.sleep(_DOWNLOAD_TBF)

        except Exception as e:
            errors.append((pin_id, str(e)))

    return {"imported": imported, "skipped": skipped, "errors": errors, "total": total}


def import_from_export_files(
    file_paths:   List[str],
    max_count:    int  = 100_000,
    auto_analyze: bool = False,
    progress_cb         = None,
) -> Dict[str, Any]:
    """
    Import pins from raw Pinterest export files (HTML, JSON, or CSV).
    Use this when macOS auto-unzipped the Pinterest export — just select all
    files from the pins/ folder and pass their paths here.
    """
    from PIL import Image as _PIL

    all_pins: List[Dict] = []
    debug_lines: List[str] = []

    for fpath in file_paths:
        try:
            lower = fpath.lower()
            if lower.endswith((".html", ".htm")):
                with open(fpath, encoding="utf-8", errors="replace") as f:
                    content = f.read()
                found = _parse_pins_from_sar_html(content)
                debug_lines.append(f"{os.path.basename(fpath)}: {len(found)} pins")
                all_pins.extend(found)
            elif lower.endswith(".json"):
                with open(fpath, encoding="utf-8", errors="replace") as f:
                    data = json.load(f)
                found = data if isinstance(data, list) else []
                debug_lines.append(f"{os.path.basename(fpath)}: {len(found)} records")
                all_pins.extend(found)
            elif lower.endswith(".csv"):
                with open(fpath, encoding="utf-8", errors="replace") as f:
                    rows = list(csv.DictReader(f))
                debug_lines.append(f"{os.path.basename(fpath)}: {len(rows)} rows")
                all_pins.extend(rows)
        except Exception as e:
            debug_lines.append(f"{os.path.basename(fpath)}: error — {e}")

    debug_info = "\n".join(debug_lines)

    if not all_pins:
        return {
            "imported": [], "skipped": 0,
            "errors":   [("parse", f"No pins found in the uploaded files.\n{debug_info}")],
            "total": 0, "debug": debug_info,
        }

    # Reuse the same download logic as import_from_export_zip
    import character_library as lib_mod

    existing_ids = lib_mod.get_existing_pinterest_ids()
    pins_to_import = []
    seen: set = set()
    for p in all_pins:
        pid = str(
            p.get("Pin ID") or p.get("pin id") or
            p.get("id") or p.get("pin_id") or id(p)
        ).strip()
        if pid not in existing_ids and pid not in seen:
            pins_to_import.append((pid, p))
            seen.add(pid)
        if len(pins_to_import) >= max_count:
            break

    total   = len(pins_to_import)
    skipped = len(all_pins) - total
    imported: List[str]             = []
    errors:   List[Tuple[str, str]] = []

    for i, (pin_id, pin) in enumerate(pins_to_import):
        label = pin.get("Title") or pin.get("title") or pin.get("Board Name") or pin_id
        if progress_cb:
            progress_cb(i + 1, total, str(label)[:40])

        image_url = _extract_image_url_from_export_pin(pin)
        if not image_url:
            skipped += 1
            continue

        try:
            r = requests.get(image_url, headers=_CDN_HEADERS, timeout=30)
            r.raise_for_status()
            pil = _PIL.open(io.BytesIO(r.content)).convert("RGB")

            template_id, _ = lib_mod.add_template("other")
            lib_mod.save_image(template_id, "face", pil)

            base: Dict[str, Any] = {
                "source_url":       image_url,
                "pinterest_pin_id": pin_id,
                "name":             str(label)[:60] if str(label) != pin_id else template_id,
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
            time.sleep(_DOWNLOAD_TBF)

        except Exception as e:
            errors.append((pin_id, str(e)))

    return {"imported": imported, "skipped": skipped, "errors": errors,
            "total": total, "debug": debug_info}
