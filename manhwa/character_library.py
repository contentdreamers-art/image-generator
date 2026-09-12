"""
character_library.py — Server-wide character template library for NB2 Edit.

Library lives at:  manhwa/character_library/
  library.json   — JSON index of all templates
  images/        — local PNG reference images (face + body per template)

Templates are shared across all projects on this server instance.
Each project stores its own cast (story_char → template_id) in ProjectState.

AI ANALYSIS
-----------
When an image is saved to the library, analyze_image_with_ai() calls Claude Vision
to extract: gender, age group, body type, archetype, mood, auto-name, scene suitability,
distinctive features, and suggested tags. These are stored on the template and
power automatic category inference, smart search, and gap detection.

CATEGORIES
----------
Categories are free-form strings. The default set is [female, male, other], but any
AI-detected attribute or user-defined slug can become a category. Templates belong to
exactly ONE category (their perceived_gender bucket by default), but their tags span
any dimension (age, archetype, mood, body type, scene type, etc.).

GAP DETECTION
-------------
detect_library_gaps() scans all saved project beat_plans to find visual scene types
that appear frequently in stories but have few (or no) reference images covering them.
"""

import base64
import io
import json
import math
import os
import re
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Paths ────────────────────────────────────────────────────────────────────
_MODULE_DIR   = os.path.dirname(os.path.abspath(__file__))
_LIB_ROOT     = os.path.join(_MODULE_DIR, "character_library")
_IMG_DIR      = os.path.join(_LIB_ROOT, "images")
_LIB_JSON     = os.path.join(_LIB_ROOT, "library.json")
_USAGE_JSON   = os.path.join(_LIB_ROOT, "url_usage.json")   # global, persistent, cross-project

# ── Write lock (prevents concurrent load→modify→save corruption) ──────────────
import threading as _threading
_LIB_LOCK   = _threading.RLock()  # in-process lock for library.json (RLock: update_template→save_library re-entry)
_USAGE_LOCK = _threading.Lock()   # in-process lock for url_usage.json
_LIB_FLOCK_PATH = None            # set lazily to _LIB_JSON + ".lock"

DEFAULT_CATEGORIES = ["female", "male", "other"]

# ── Helpers ──────────────────────────────────────────────────────────────────

def _ensure_dirs() -> None:
    os.makedirs(_LIB_ROOT, exist_ok=True)
    os.makedirs(_IMG_DIR, exist_ok=True)


def _atomic_save(path: str, data: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ── Library I/O ──────────────────────────────────────────────────────────────

_LIB_CACHE: Dict[str, Any] = {}
_LIB_CACHE_TIME: float = 0.0
_LIB_CACHE_TTL: float = 30.0  # seconds — fast enough for edits to show up quickly


def _invalidate_library_cache() -> None:
    global _LIB_CACHE_TIME
    _LIB_CACHE_TIME = 0.0


def load_library(force: bool = False) -> Dict[str, Any]:
    global _LIB_CACHE, _LIB_CACHE_TIME
    import time as _time
    now = _time.monotonic()
    if not force and _LIB_CACHE and (now - _LIB_CACHE_TIME) < _LIB_CACHE_TTL:
        return _LIB_CACHE

    _ensure_dirs()
    if not os.path.exists(_LIB_JSON):
        lib: Dict[str, Any] = {
            "version": 1,
            "categories": list(DEFAULT_CATEGORIES),
            "templates": {},
        }
        _atomic_save(_LIB_JSON, lib)
        _LIB_CACHE = lib
        _LIB_CACHE_TIME = now
        return lib
    try:
        with open(_LIB_JSON, "r", encoding="utf-8") as f:
            lib = json.load(f)
        _LIB_CACHE = lib
        _LIB_CACHE_TIME = now
        return lib
    except Exception:
        return {"version": 1, "categories": list(DEFAULT_CATEGORIES), "templates": {}}


def save_library(lib: Dict[str, Any]) -> None:
    """Save library atomically with both in-process and cross-process file locking."""
    import fcntl as _fcntl
    _ensure_dirs()
    lock_path = _LIB_JSON + ".lock"
    with _LIB_LOCK:                          # in-process serialisation
        with open(lock_path, "w") as _lf:    # cross-process serialisation
            _fcntl.flock(_lf, _fcntl.LOCK_EX)
            try:
                _atomic_save(_LIB_JSON, lib)
            finally:
                _fcntl.flock(_lf, _fcntl.LOCK_UN)
    _invalidate_library_cache()  # next load_library() call will re-read from disk


# ── Global URL-usage counter ──────────────────────────────────────────────────
# url_usage.json lives in character_library/ alongside library.json.
# It persists FOREVER across all projects, all users, all generations.
# Every time a reference image URL is sent to the model it gets +1 here.
# resolve_scene_refs reads this at call-time so newer generations always
# prefer images that have been used least across the entire history.

def load_global_url_usage() -> Dict[str, int]:
    """Return {url: count} from the persistent global usage database."""
    _ensure_dirs()
    if not os.path.exists(_USAGE_JSON):
        return {}
    try:
        with open(_USAGE_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: int(v) for k, v in data.items() if isinstance(v, (int, float))}
    except Exception:
        return {}


def increment_global_url_usage(urls: List[str]) -> None:
    """Atomically increment the global usage count for each URL in *urls*."""
    if not urls:
        return
    import fcntl as _fcntl
    _ensure_dirs()
    lock_path = _USAGE_JSON + ".lock"
    with _USAGE_LOCK:
        with open(lock_path, "w") as _lf:
            _fcntl.flock(_lf, _fcntl.LOCK_EX)
            try:
                usage = load_global_url_usage()
                for u in urls:
                    if u:
                        usage[u] = usage.get(u, 0) + 1
                _atomic_save(_USAGE_JSON, usage)
            finally:
                _fcntl.flock(_lf, _fcntl.LOCK_UN)


# ── Auto-tag inference maps (no API cost — reads existing ai_analysis) ────────

_AGE_TAG_MAP: Dict[str, str] = {
    "child":       "child",
    "teen":        "teen",
    "young adult": "young adult",
    "adult":       "adult",
    "mature":      "middle aged",
    "elderly":     "elderly",
}

# (keywords to check in combined text, tag to add)
_ERA_KEYWORD_MAP: List[Tuple[List[str], str]] = [
    (["medieval", "feudal", "chainmail", "tunic", "peasant", "serf", "kingdom",
      "dungeon", "tavern", "castle", "manor", "medieval fantasy"], "medieval"),
    (["magic", "mage", "wizard", "sorcerer", "warlock", "witch", "elf", "dragon",
      "arcane", "rune", "spell", "enchanted", "mystical", "fantasy", "royal fantasy",
      "isekai", "supernatural"], "fantasy"),
    (["modern", "jacket", "jeans", "t-shirt", "hoodie", "sneakers", "suit",
      "business", "office", "school uniform", "street", "slice of life",
      "contemporary", "sci fi", "cyberpunk"], "modern"),
    (["historical", "ancient", "roman", "greek", "dynasty", "imperial",
      "samurai", "edo", "shogunate", "feudal japan"], "historical"),
]

_CLASS_KEYWORD_MAP: List[Tuple[List[str], str]] = [
    (["king", "queen", "emperor", "empress", "monarch", "throne", "crown",
      "sovereign"], "royalty"),
    (["prince", "princess"], "royalty"),
    (["noble", "nobleman", "aristocrat", "lord", "lady", "duke", "duchess",
      "count", "baron", "silk", "velvet", "embroidered gown", "ornate"], "noble"),
    (["knight", "armor", "armour", "chainmail", "gauntlet", "vambrace",
      "lance", "shield", "sword and shield", "paladin"], "knight"),
    (["soldier", "guard", "warrior", "mercenary", "fighter"], "warrior"),
    (["mage", "wizard", "sorcerer", "warlock", "witch", "spellcaster",
      "arcane robe", "staff", "wand"], "mage"),
    (["priest", "monk", "nun", "cleric", "holy robe", "temple"], "cleric"),
    (["farmer", "peasant", "serf", "farmhand", "harvest", "plow",
      "rough linen", "field worker"], "farmer"),
    (["villager", "commoner", "townsfolk", "townsperson"], "villager"),
    (["merchant", "trader", "shopkeeper", "vendor"], "merchant"),
    (["scholar", "scribe", "academic", "professor", "librarian",
      "scroll", "ink-stained"], "scholar"),
]


def enrich_tags_from_analysis(template_id: str) -> Dict[str, Any]:
    """Infer age / era / social-class tags from the already-stored ai_analysis
    fields — zero extra API calls.  Returns {"tid": ..., "added": [...]}."""
    lib = load_library()
    t   = lib.get("templates", {}).get(template_id)
    if not t:
        return {"tid": template_id, "added": []}

    ai           = t.get("ai_analysis") or {}
    existing     = set(t.get("tags") or [])
    new_tags: set = set()

    # ── Age group ──────────────────────────────────────────────────────────────
    age_raw = (ai.get("age_group") or "").strip().lower()
    age_tag = _AGE_TAG_MAP.get(age_raw)
    if age_tag:
        new_tags.add(age_tag)

    # ── Era ────────────────────────────────────────────────────────────────────
    setting    = ai.get("setting") or {}
    genre_raw  = (setting.get("genre_world") or "").strip().lower()
    cloth_raw  = (ai.get("clothing_description") or "").strip().lower()
    tags_raw   = " ".join(existing).lower()
    combined   = f"{genre_raw} {cloth_raw} {tags_raw}"
    for keywords, era_tag in _ERA_KEYWORD_MAP:
        if any(kw in combined for kw in keywords):
            new_tags.add(era_tag)

    # ── Social class ───────────────────────────────────────────────────────────
    archetype_raw = (ai.get("archetype") or "").strip().lower()
    class_text    = f"{archetype_raw} {cloth_raw} {tags_raw}"
    for keywords, class_tag in _CLASS_KEYWORD_MAP:
        if any(kw in class_text for kw in keywords):
            new_tags.add(class_tag)

    added = sorted(new_tags - existing)
    if added:
        update_template(template_id, tags=sorted(existing | new_tags))
    return {"tid": template_id, "added": added}


def enrich_all_tags() -> Dict[str, Any]:
    """Run enrich_tags_from_analysis on every template. Returns summary."""
    lib   = load_library()
    total = 0
    enriched = 0
    added_all: List[str] = []
    for tid in list(lib.get("templates", {}).keys()):
        r = enrich_tags_from_analysis(tid)
        total += 1
        if r.get("added"):
            enriched += 1
            added_all.extend(r["added"])
    return {"total": total, "enriched": enriched, "new_tags": sorted(set(added_all))}


# ── Clothing Classes ──────────────────────────────────────────────────────────
# ── Clothing Classes ─────────────────────────────────────────────────────────
#
# Stored in library.json under "clothing_classes":
#   {
#     "<class_name>": {
#       "variations": [
#         {"tid": "<template_id>", "clothing_dna": "<prompt text>"},
#         ...up to 10
#       ]
#     }
#   }
#
# Legacy flat format { "tid": ..., "clothing_dna": ... } is auto-migrated on read.

DEFAULT_CLOTHING_CLASSES: List[str] = [
    # Medieval ↔ Modern pairs (medieval first, modern second)
    "noble",          "modern-noble",
    "royalty",        "modern-royalty",
    "warrior",        "modern-warrior",
    "knight",         "modern-knight",
    "mage",           "modern-mage",
    "farmer",         "modern-farmer",
    "merchant",       "modern-merchant",
    "villager",       "modern-villager",
    "servant",        "modern-servant",
    "commoner",       "modern-commoner",
    "guard",          "modern-guard",
]

# Default clothing DNA for medieval classes that currently ship empty —
# used immediately even without a library image. (Classes the user has already
# populated with image variations are never touched by these defaults.)
DEFAULT_MEDIEVAL_CLOTHING_DNA: Dict[str, str] = {
    "warrior": (
        "hardened leather armor over a padded gambeson, iron shoulder guards, "
        "sword scabbard at the hip, bracers, worn leather boots — battle-ready medieval fighter"
    ),
    "knight": (
        "polished steel plate armor with chainmail underneath, surcoat bearing a heraldic emblem, "
        "sword belt, armored gauntlets and sabatons — noble medieval knight"
    ),
    "villager": (
        "rough-spun linen tunic with a simple leather belt, patched woolen trousers, "
        "simple cloth shoes or sandals, humble earth-toned medieval peasant clothing"
    ),
}

# Default clothing DNA text for modern classes — used immediately even without a library image.
# Keys must match the class names above exactly.
DEFAULT_MODERN_CLOTHING_DNA: Dict[str, str] = {
    "modern-noble": (
        "tailored charcoal or navy suit with fine lapels and a silk tie, polished leather Oxford shoes, "
        "subtle luxury watch, crisp white dress shirt — business elite aesthetic"
    ),
    "modern-royalty": (
        "haute couture designer outfit — structured blazer or floor-length gown, precious jewelry "
        "(pearl earrings or platinum chain), flawless styling, poised celebrity or public-figure look"
    ),
    "modern-warrior": (
        "military tactical uniform — camouflage or OD-green fatigues, body-armor plate carrier, "
        "combat boots, utility belt, dog tags"
    ),
    "modern-knight": (
        "dark police or detective uniform — crisp navy or black button-up shirt, tactical pants, "
        "badge clipped to chest, utility belt with radio holster, steel-toed boots"
    ),
    "modern-mage": (
        "white lab coat over a professional collared shirt, ID-badge lanyard, practical dark slacks, "
        "analytical and authoritative scientific-professional look"
    ),
    "modern-farmer": (
        "denim overalls over a plaid flannel shirt, weathered wide-brim work hat, "
        "heavy work boots, sturdy canvas work gloves tucked into pocket"
    ),
    "modern-merchant": (
        "business-casual — pressed button-up shirt (no tie), dark slim chinos, "
        "clean leather loafers or oxfords, slim wristwatch, practical professional look"
    ),
    "modern-villager": (
        "plain everyday clothes — well-worn jeans, simple T-shirt or hoodie, "
        "clean sneakers, unremarkable casual civilian look"
    ),
    "modern-servant": (
        "service uniform — crisp white button-up shirt, pressed black slacks, "
        "black apron or vest, polished shoes, minimal accessories, professional hospitality look"
    ),
    "modern-commoner": (
        "casual street clothes — jeans and a graphic tee or zip-up hoodie, "
        "sneakers or canvas shoes, ordinary everyday modern person"
    ),
    "modern-guard": (
        "dark navy or black security uniform — guard badge on chest, "
        "earpiece, utility belt, steel-toed boots, authoritative but civilian-security look"
    ),
}


# ── Shot recipes — reusable cinematic prompt blocks ──────────────────────────
# Each recipe encodes a proven angle + lighting + production-value formula.
# "when" tells Claude which story beats fit; "prompt" is injected into the panel block.
DEFAULT_SHOT_RECIPES: Dict[str, Dict[str, str]] = {
    "menace-closeup": {
        "when": "villain reveal, threat, sinister power display, intimidation, demonic presence",
        "prompt": ("EXTREME CLOSE-UP — face fills the entire panel, dramatic under-lighting, "
                   "glowing pupil-less eyes, deep shadows carving the cheekbones and brow, "
                   "single-hue light wash background, rim light tracing the hair silhouette, oppressive atmosphere"),
    },
    "power-pose-aura": {
        "when": "protagonist power moment, transformation, entering battle, strength reveal",
        "prompt": ("FULL-BODY LOW ANGLE — character stands towering over the camera, "
                   "engulfed in a blazing energy aura with rising particles, harsh rim light on every edge, "
                   "dark moody background swallowed by the glow, cape or clothing whipped by power wind"),
    },
    "silhouette-authority": {
        "when": "oppressors, ruling class, faceless authority, crowds bowing, institutional power",
        "prompt": ("BACKLIT SILHOUETTES — figures rendered as pure black shapes against intense "
                   "vertical light streaks, no facial detail, imposing scale difference between figures, "
                   "monochrome single-color background, cathedral-like vertical composition"),
    },
    "aerial-establishing": {
        "when": "location introduction, time skip, world-building, showing scale of a place or event",
        "prompt": ("HIGH-ANGLE AERIAL — bird's-eye view looking down on the scene, "
                   "tiny figures dwarfed by environment, atmospheric haze, muted desaturated palette "
                   "with one accent color, cinematic depth with foreground elements framing the view"),
    },
    "impact-action": {
        "when": "punch, hit, explosion, sudden violence, combat impact, crash",
        "prompt": ("IMPACT FRAME — dutch angle tilted 15-30 degrees, radial speed lines exploding "
                   "from the point of impact, motion blur on the moving limb, debris and spit particles frozen mid-air, "
                   "bold manhwa SFX lettering integrated into the art, high contrast flash lighting"),
    },
    "emotional-closeup": {
        "when": "grief, tears, love, tenderness, heartbreak, vulnerable moment, child innocence",
        "prompt": ("SOFT CLOSE-UP — face at three-quarter angle filling most of the frame, "
                   "soft diffused warm light, glistening tear highlights or gentle blush, shallow depth of field "
                   "with dreamy bokeh background, delicate line work on eyes and lips"),
    },
    "pov-hands": {
        "when": "giving a ring, handshake, reaching out, receiving an item, first-person perspective moment",
        "prompt": ("FIRST-PERSON POV — camera sees through the character's eyes, their hands enter frame "
                   "from the bottom edge, focus locked on the hands and what they touch, "
                   "background softly blurred, intimate framing"),
    },
    "carry-rescue": {
        "when": "carrying someone, rescue, protective embrace, princess carry, aftermath of saving",
        "prompt": ("TWO-CHARACTER CARRY — one character holds the other in their arms, "
                   "low three-quarter angle emphasizing the protector's strength, "
                   "the carried figure limp or clinging, dramatic directional light from behind, "
                   "environmental particles (rain, embers, dust) drifting through the beam"),
    },
    "betrayal-reveal": {
        "when": "betrayal, shocking revelation, knife in the back, trust broken, cruel confession",
        "prompt": ("OVER-SHOULDER REVEAL — foreground shoulder/head in shadow framing the betrayer's "
                   "coldly smiling face, split lighting (half lit, half black), venomous eye highlight, "
                   "cold desaturated palette with one blood-red accent"),
    },
    "wide-confrontation": {
        "when": "standoff, two forces facing each other, before a fight, arrival of a challenger",
        "prompt": ("WIDE SHOT CONFRONTATION — two figures at opposite panel edges facing each other, "
                   "vast tension-filled empty space between them, ground-level camera, "
                   "wind-blown dust or leaves crossing the gap, long shadows, duel-at-dawn atmosphere"),
    },
    "quiet-moment": {
        "when": "calm scene, daily life, conversation, peaceful interlude, domestic warmth",
        "prompt": ("EYE-LEVEL MEDIUM SHOT — relaxed natural framing, warm ambient light "
                   "(window light or golden hour), soft shadows, cozy environmental detail, "
                   "balanced composition with breathing room"),
    },
    "despair-highangle": {
        "when": "defeat, humiliation, character at their lowest, abandonment, falling",
        "prompt": ("HIGH ANGLE LOOKING DOWN — camera towers above the crushed character making them "
                   "small and helpless in the frame, cold blue-grey palette, harsh single overhead light, "
                   "large oppressive negative space pressing down on the figure"),
    },
}


def get_shot_recipes() -> Dict[str, Dict[str, str]]:
    """Return {recipe_name: {when, prompt}} — library overrides merged over defaults."""
    lib = load_library()
    saved = lib.get("shot_recipes") or {}
    merged = dict(DEFAULT_SHOT_RECIPES)
    for k, v in saved.items():
        if isinstance(v, dict) and (v.get("prompt") or "").strip():
            merged[k] = v
    return merged


def save_shot_recipe(name: str, when: str, prompt: str) -> str:
    """Add or update a shot recipe in the library (survives code updates)."""
    name = re.sub(r"[^a-z0-9_\-]", "-", name.strip().lower()).strip("-")
    if not name or not prompt.strip():
        return "Recipe needs a name and a prompt."
    with _LIB_LOCK:
        lib = load_library()
        recipes = lib.setdefault("shot_recipes", {})
        # Cap length — each recipe prompt is repeated once per panel in the page prompt.
        recipes[name] = {"when": when.strip()[:200], "prompt": prompt.strip()[:450]}
        save_library(lib)
    return f"✅ Recipe '{name}' saved."


def _normalize_cls(cls_data: Dict) -> Dict:
    """Migrate old flat {tid, clothing_dna} format → {variations: [{tid, clothing_dna}]}."""
    if "variations" in cls_data:
        return cls_data
    tid = (cls_data.get("tid") or "").strip()
    dna = (cls_data.get("clothing_dna") or "").strip()
    return {"variations": [{"tid": tid, "clothing_dna": dna}] if (tid or dna) else []}


def get_clothing_classes() -> Dict[str, Dict]:
    """Return {class_name: {variations: [...]}} dict (normalized)."""
    raw = load_library().get("clothing_classes") or {}
    return {k: _normalize_cls(v) for k, v in raw.items()}


def get_clothing_class_variations(class_name: str) -> List[Dict]:
    """Return list of {tid, clothing_dna} for a class."""
    cls = get_clothing_classes().get((class_name or "").strip().lower())
    return (cls or {}).get("variations") or []


def ensure_default_classes() -> None:
    """Create placeholder entries for standard class names if absent.
    Modern classes are seeded with default DNA text so they work immediately,
    even before the user assigns library images to them."""
    with _LIB_LOCK:
        lib = load_library()
        classes = lib.setdefault("clothing_classes", {})
        changed = False
        for name in DEFAULT_CLOTHING_CLASSES:
            if name not in classes:
                default_dna = DEFAULT_MODERN_CLOTHING_DNA.get(name, "") or DEFAULT_MEDIEVAL_CLOTHING_DNA.get(name, "")
                if default_dna:
                    classes[name] = {"variations": [{"tid": "", "clothing_dna": default_dna}]}
                else:
                    classes[name] = {"variations": []}
                changed = True
        if changed:
            save_library(lib)


def set_clothing_class(name: str, tid: str = "", clothing_dna: str = "") -> str:
    """Ensure a class exists. If it has no variations and data is given, creates one.
    Called automatically during story parse to register newly detected class names."""
    name = name.strip().lower()
    if not name:
        return "Class name cannot be empty."
    with _LIB_LOCK:
        lib = load_library()
        classes = lib.setdefault("clothing_classes", {})
        existing = _normalize_cls(classes.get(name) or {})
        variations = existing.get("variations") or []
        if not variations and ((tid or "").strip() or (clothing_dna or "").strip()):
            variations = [{"tid": (tid or "").strip(), "clothing_dna": (clothing_dna or "").strip()}]
        classes[name] = {"variations": variations}
        save_library(lib)
    return ""


def add_clothing_class_variation(name: str, tid: str = "", clothing_dna: str = "") -> str:
    """Append a new variation to a class (max 10). Returns error string or ''."""
    name = (name or "").strip().lower()
    if not name:
        return "Class name cannot be empty."
    tid          = (tid or "").strip()
    clothing_dna = (clothing_dna or "").strip()
    if not clothing_dna and not tid:
        return "Provide a clothing description or select a reference image."
    with _LIB_LOCK:
        lib = load_library()
        classes = lib.setdefault("clothing_classes", {})
        existing = _normalize_cls(classes.get(name) or {})
        variations = existing.get("variations") or []
        if len(variations) >= 10:
            return "Maximum 10 variations per class. Delete one to add another."
        variations.append({"tid": tid, "clothing_dna": clothing_dna})
        classes[name] = {"variations": variations}
        save_library(lib)
    return ""


def update_clothing_class_variation(name: str, idx: int, tid: str = "", clothing_dna: str = "") -> str:
    """Update an existing variation by index. Returns error string or ''."""
    name = (name or "").strip().lower()
    clothing_dna = (clothing_dna or "").strip()
    if not clothing_dna:
        return "Clothing description cannot be empty."
    with _LIB_LOCK:
        lib = load_library()
        classes = lib.setdefault("clothing_classes", {})
        existing = _normalize_cls(classes.get(name) or {})
        variations = existing.get("variations") or []
        if idx < 0 or idx >= len(variations):
            return f"Variation {idx+1} does not exist."
        variations[idx] = {"tid": (tid or "").strip(), "clothing_dna": clothing_dna}
        classes[name] = {"variations": variations}
        save_library(lib)
    return ""


def delete_clothing_class_variation(name: str, idx: int) -> str:
    """Remove a variation by index. Returns status message."""
    name = (name or "").strip().lower()
    with _LIB_LOCK:
        lib = load_library()
        classes = lib.setdefault("clothing_classes", {})
        existing = _normalize_cls(classes.get(name) or {})
        variations = existing.get("variations") or []
        if idx < 0 or idx >= len(variations):
            return f"Variation {idx+1} does not exist."
        variations.pop(idx)
        classes[name] = {"variations": variations}
        save_library(lib)
    return f"✅ Deleted variation {idx+1}."


def delete_clothing_class(name: str) -> None:
    """Delete an entire clothing class."""
    name = (name or "").strip().lower()
    with _LIB_LOCK:
        lib = load_library()
        lib.setdefault("clothing_classes", {}).pop(name, None)
        save_library(lib)


def get_class_clothing_dna(class_name: str, seed_key: str = "") -> str:
    """Return a clothing description from the class's variations.

    If seed_key is given (e.g. the character's name), the SAME variation is
    deterministically returned every call — so one character keeps one outfit
    across all panels and pages, while different characters sharing a class
    still get different variations. Without seed_key, picks randomly."""
    variations = get_clothing_class_variations(class_name)
    with_dna = [v["clothing_dna"] for v in variations if (v.get("clothing_dna") or "").strip()]
    if not with_dna:
        return ""
    if seed_key:
        import hashlib as _hl
        idx = int(_hl.md5(f"{class_name}|{seed_key}".encode()).hexdigest(), 16) % len(with_dna)
        return with_dna[idx]
    import random as _rnd
    return _rnd.choice(with_dna)


# ── Category management ───────────────────────────────────────────────────────

def get_categories(lib: Optional[Dict] = None) -> List[str]:
    if lib is None:
        lib = load_library()
    cats = lib.get("categories") or []
    # Always ensure defaults exist
    for d in DEFAULT_CATEGORIES:
        if d not in cats:
            cats.append(d)
    return cats


def add_category(name: str) -> Tuple[str, List[str]]:
    """Add a new category slug. Returns (slug, updated_list)."""
    slug = re.sub(r"[^a-z0-9_\-]", "_", name.strip().lower()).strip("_")
    if not slug:
        return "", get_categories()
    lib = load_library()
    cats = get_categories(lib)
    if slug not in cats:
        cats.append(slug)
        lib["categories"] = cats
        save_library(lib)
    return slug, cats


def remove_category(name: str) -> List[str]:
    """Remove a category (only if empty). Returns updated list."""
    lib = load_library()
    cats = get_categories(lib)
    slug = name.strip().lower()
    if slug in DEFAULT_CATEGORIES:
        return cats  # never remove defaults
    templates = [t for t in lib.get("templates", {}).values() if t.get("category") == slug]
    if templates:
        return cats  # non-empty — refuse
    if slug in cats:
        cats.remove(slug)
    lib["categories"] = cats
    save_library(lib)
    return cats


# ── Template ID generation ────────────────────────────────────────────────────

def _next_template_id(lib: Dict[str, Any], category: str) -> str:
    prefix = (category or "x")[:1].upper()
    existing_nums = []
    for tid in lib.get("templates", {}):
        if tid.startswith(prefix) and tid[1:].isdigit():
            existing_nums.append(int(tid[1:]))
    n = max(existing_nums, default=0) + 1
    return f"{prefix}{n:03d}"


# ── Template CRUD ─────────────────────────────────────────────────────────────

def add_template(
    category: str,
    name: str = "",
    description: str = "",
    tags: Optional[List[str]] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Create a new empty template in category. Returns (template_id, template_dict)."""
    import datetime as _dt
    lib = load_library()
    template_id = _next_template_id(lib, category)
    template: Dict[str, Any] = {
        "template_id": template_id,
        "category":    category,
        "name":        name or template_id,
        "description": description,
        "tags":        tags or [],
        "local_face":  "",
        "local_body":  "",
        "fal_face_url": "",
        "fal_body_url": "",
        "created_at":  _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    lib.setdefault("templates", {})[template_id] = template
    save_library(lib)
    return template_id, template


def get_library_stats() -> Dict[str, Any]:
    """
    Return counts and a time-bucketed activity log.
    Timestamps: use template['created_at'] if present; fall back to mtime of
    local_face (or local_body) for pre-timestamp images.
    Returns:
        total          – total templates
        analyzed       – templates with ai_analysis
        needs_analysis – templates with a local image but no ai_analysis
        activity_log   – list of {"label": "Aug 3 7 PM", "count": n, "ts": epoch}
                         sorted newest-first, grouped by calendar-hour bucket
    """
    import datetime as _dt
    import os as _os

    lib = load_library()
    templates = list(lib.get("templates", {}).values())
    total = len(templates)
    analyzed = sum(1 for t in templates if t.get("ai_analysis"))
    # Use field presence — no os.path.exists calls (fast)
    needs = sum(
        1 for t in templates
        if not t.get("ai_analysis")
        and (t.get("local_face") or t.get("local_body"))
    )

    # Build time buckets — only use created_at (no mtime fallback to avoid fs calls)
    buckets: Dict[str, int] = {}  # "YYYY-MM-DD HH" → count
    for t in templates:
        ts_str = t.get("created_at")
        epoch: Optional[float] = None
        if ts_str:
            try:
                epoch = _dt.datetime.fromisoformat(ts_str.rstrip("Z")).replace(
                    tzinfo=_dt.timezone.utc
                ).timestamp()
            except Exception:
                pass
        if epoch is None:
            continue
        dt_utc = _dt.datetime.utcfromtimestamp(epoch)
        bucket_key = dt_utc.strftime("%Y-%m-%d %H")
        buckets[bucket_key] = buckets.get(bucket_key, 0) + 1

    # Format for display — convert UTC hour bucket to a readable label
    log_entries = []
    for bucket_key, count in sorted(buckets.items(), reverse=True):
        try:
            dt = _dt.datetime.strptime(bucket_key, "%Y-%m-%d %H")
            label = dt.strftime("%b %-d, %-I %p (UTC)")
        except Exception:
            label = bucket_key
        log_entries.append({"label": label, "count": count,
                             "ts": _dt.datetime.strptime(bucket_key, "%Y-%m-%d %H").timestamp()})

    return {
        "total":          total,
        "analyzed":       analyzed,
        "needs_analysis": needs,
        "activity_log":   log_entries,
    }


def get_template(template_id: str) -> Optional[Dict[str, Any]]:
    lib = load_library()
    return lib.get("templates", {}).get(template_id)


def list_templates(category: Optional[str] = None) -> List[Dict[str, Any]]:
    lib = load_library()
    templates = list(lib.get("templates", {}).values())
    if category:
        templates = [t for t in templates if t.get("category", "").lower() == category.lower()]
    return sorted(templates, key=lambda t: t.get("template_id", ""))


def update_template(template_id: str, **kwargs) -> Optional[Dict[str, Any]]:
    with _LIB_LOCK:
        lib = load_library()
        t = lib.get("templates", {}).get(template_id)
        if t is None:
            return None
        for k, v in kwargs.items():
            t[k] = v
        save_library(lib)
        return t


def delete_template(template_id: str) -> bool:
    lib = load_library()
    if template_id not in lib.get("templates", {}):
        return False
    del lib["templates"][template_id]
    save_library(lib)
    for img_type in ("face", "body"):
        p = os.path.join(_IMG_DIR, f"{template_id}_{img_type}.png")
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass
    return True


# ── Image storage ─────────────────────────────────────────────────────────────

def image_path(template_id: str, img_type: str) -> str:
    """Local path for a template image (face/body). May not exist yet."""
    return os.path.join(_IMG_DIR, f"{template_id}_{img_type}.png")


def save_image(template_id: str, img_type: str, pil_img) -> str:
    """Save a PIL image locally and update the template record. Returns local path."""
    _ensure_dirs()
    path = image_path(template_id, img_type)
    pil_img.save(path, format="PNG")
    update_template(template_id, **{f"local_{img_type}": path})
    return path


def get_face_image(template_id: str):
    """Return PIL image for the face reference, or None."""
    try:
        from PIL import Image
        p = image_path(template_id, "face")
        if os.path.exists(p):
            return Image.open(p).convert("RGB")
    except Exception:
        pass
    return None


def get_body_image(template_id: str):
    """Return PIL image for the body reference, or None."""
    try:
        from PIL import Image
        p = image_path(template_id, "body")
        if os.path.exists(p):
            return Image.open(p).convert("RGB")
    except Exception:
        pass
    return None


# ── FAL upload ────────────────────────────────────────────────────────────────

def upload_template_to_fal(template_id: str) -> Dict[str, str]:
    """
    Upload local face + body images to FAL storage, cache URLs in library.json.
    Returns {"face_url": ..., "body_url": ..., "status": "ok" | error msg}.
    """
    from build import upload_pil_to_fal
    from PIL import Image

    lib = load_library()
    t = lib.get("templates", {}).get(template_id)
    if t is None:
        return {"status": f"Template {template_id!r} not found."}

    result: Dict[str, str] = {"status": "ok"}
    for img_type in ("face", "body"):
        local_path = t.get(f"local_{img_type}", "")
        if not local_path or not os.path.exists(local_path):
            result[f"{img_type}_url"] = t.get(f"fal_{img_type}_url", "")
            continue
        try:
            pil = Image.open(local_path).convert("RGB")
            url = upload_pil_to_fal(pil)
            t[f"fal_{img_type}_url"] = url
            result[f"{img_type}_url"] = url
        except Exception as e:
            result["status"] = f"Upload failed for {img_type}: {e}"
            result[f"{img_type}_url"] = t.get(f"fal_{img_type}_url", "")

    save_library(lib)
    return result


def bulk_upload_all_to_fal(
    skip_already_uploaded: bool = True,
    workers: int = 6,
    progress_cb=None,
) -> Dict[str, Any]:
    """
    Upload local face (and body) images for every template that is missing FAL URLs.
    Uses a thread pool for speed — 6 workers typically uploads 180 images in ~2-3 min.

    progress_cb: optional callable(done, total, tid, status) called from worker threads.
    Returns {"ok": N, "skipped": N, "errors": N, "error_ids": [...]}
    """
    from build import upload_pil_to_fal
    from PIL import Image as _PILImg
    import concurrent.futures as _cf

    lib = load_library()
    templates = lib.get("templates", {})
    todo = []
    for tid, t in templates.items():
        needs_face = not t.get("fal_face_url") and os.path.exists(t.get("local_face", ""))
        needs_body = not t.get("fal_body_url") and os.path.exists(t.get("local_body", ""))
        if needs_face or needs_body:
            todo.append(tid)
        # Also re-upload if told to force (skip_already_uploaded=False)
        elif not skip_already_uploaded:
            todo.append(tid)

    total = len(todo)
    result = {"ok": 0, "skipped": len(templates) - total, "errors": 0, "error_ids": []}

    if total == 0:
        return result

    _lock = _LIB_LOCK   # re-use module-level write lock

    def _upload_one(tid: str):
        """Upload face+body for one template, save directly into lib dict, returns status."""
        t = templates.get(tid)
        if not t:
            return tid, "missing"
        uploaded_any = False
        for img_type in ("face", "body"):
            local_path = t.get(f"local_{img_type}", "")
            if not local_path or not os.path.exists(local_path):
                continue
            if skip_already_uploaded and t.get(f"fal_{img_type}_url"):
                continue
            try:
                pil = _PILImg.open(local_path).convert("RGB")
                url = upload_pil_to_fal(pil)
                with _lock:
                    t[f"fal_{img_type}_url"] = url
                    # Flush to disk immediately so progress survives interrupts
                    save_library(lib)
                uploaded_any = True
            except Exception as e:
                return tid, f"error:{e}"
        return tid, "ok" if uploaded_any else "skipped"

    done_count = 0
    with _cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_upload_one, tid): tid for tid in todo}
        for fut in _cf.as_completed(futs):
            tid, status = fut.result()
            done_count += 1
            if status == "ok":
                result["ok"] += 1
            elif status.startswith("error"):
                result["errors"] += 1
                result["error_ids"].append(tid)
            if progress_cb:
                try:
                    progress_cb(done_count, total, tid, status)
                except Exception:
                    pass

    return result


def get_fal_urls(template_id: str) -> Tuple[str, str]:
    """Return (fal_face_url, fal_body_url). Empty strings if not uploaded."""
    t = get_template(template_id)
    if not t:
        return "", ""
    return t.get("fal_face_url", ""), t.get("fal_body_url", "")


# ── AI Image Analysis ─────────────────────────────────────────────────────────

_ANALYSIS_PROMPT = """\
Analyze this image for a manhwa/webtoon reference library. Be thorough — one image \
can serve as a character reference, pose reference, setting reference, and mood \
reference simultaneously. Extract EVERY field below.

Return ONLY a valid JSON object — no markdown fences, no extra text.

{
  "summary": "2-3 sentence prose description of the full image — who, what pose, what setting, what mood. Be vivid and specific.",

  "has_character": true/false,
  "is_group_shot": true/false,
  "subject_count": 1,
  "is_face_only": true/false,
  "is_full_body": true/false,

  "perceived_gender": "female" | "male" | "other" | "unclear" | "none",
  "age_group": "child" | "teen" | "young adult" | "adult" | "mature" | "elderly" | "none",
  "apparent_age_range": "e.g. 19-25 or none",
  "body_type": "slim" | "athletic" | "muscular" | "curvy" | "average" | "heavyset" | "none",
  "archetype": "protagonist" | "antagonist" | "warrior" | "scholar" | "noble" | "villain" | "mentor" | "romantic lead" | "support" | "child character" | "monster" | "crowd" | "none",
  "hair": "detailed: length + color + style, e.g. 'medium golden orange layered with white headband'",
  "skin_tone": "fair" | "light" | "medium" | "tan" | "dark" | "very dark" | "none",
  "eye_description": "color + shape, e.g. 'amber gold, large, confident'",
  "clothing_description": "specific outfit description including colors, style, notable items",
  "distinctive_features": ["glasses", "scar", "tattoo", "weapon at hip", etc],

  "pose": {
    "orientation": "e.g. three quarter front / full front / side / back",
    "framing": "e.g. face only / bust / waist up / three quarter body / full body",
    "camera_angle": "eye level / slight high angle / low angle / bird eye / worm eye",
    "body_stance": "e.g. standing, sitting, crouching, lying down",
    "weight_distribution": "e.g. weight on left leg, leaning right",
    "right_arm": "e.g. raised near face / behind back / at side",
    "left_arm": "e.g. bent at waist / extended forward",
    "head_tilt": "e.g. tilted slightly right / straight",
    "gaze": "e.g. direct at viewer / slightly past viewer / downward",
    "pose_mood": "e.g. relaxed confidence, elegant authority, tense readiness"
  },

  "setting": {
    "location_type": "e.g. warm wooden castle corridor / urban rooftop / forest clearing",
    "indoor_outdoor": "indoor" | "outdoor" | "unclear",
    "genre_world": "modern" | "medieval fantasy" | "royal fantasy" | "sci fi" | "historical" | "academy" | "supernatural" | "isekai" | "post apocalyptic" | "slice of life" | "none",
    "time_of_day": "day" | "night" | "sunset" | "dawn" | "unclear",
    "environment_complexity": "simple" | "moderate" | "complex"
  },

  "lighting": {
    "primary_light": "e.g. warm golden sunlight from upper right",
    "overall_mood": "warm" | "cool" | "dramatic" | "soft" | "dark" | "bright"
  },

  "art_style": "premium Korean manhwa" | "standard manhwa" | "Japanese manga" | "semi realistic" | "chibi" | "painterly" | "sketch",

  "reusability": {
    "pose_score": 0.0-1.0,
    "character_design_score": 0.0-1.0,
    "environment_score": 0.0-1.0,
    "lighting_score": 0.0-1.0
  },

  "generation_prompt": "A ready-to-use image generation prompt that faithfully recreates this exact image — include character appearance, outfit, pose, camera angle, setting, lighting, and art style. Should be 3-5 sentences.",

  "tags": [
    "OUTPUT ONLY real tag strings that apply. Use spaces not hyphens. Include 15-35 tags.",

    "-- CHARACTER IDENTITY --",
    "female", "male", "nonbinary", "child character", "teen", "young adult", "adult", "mature", "elderly",
    "solo", "duo", "group", "couple",
    "slim", "athletic", "muscular", "curvy", "heavyset",
    "protagonist", "antagonist", "warrior", "scholar", "noble", "villain", "mentor", "romantic lead", "support", "monster",
    "long hair", "short hair", "bald", "white hair", "dark hair", "blonde hair", "red hair", "orange hair", "silver hair",
    "fair skin", "medium skin", "tan skin", "dark skin",
    "light eyes", "dark eyes", "glasses", "glowing eyes",

    "-- EMOTION / EXPRESSION --",
    "fierce", "gentle", "cold", "warm", "mysterious", "cheerful", "serious", "sad",
    "confident", "tired", "shocked", "scared", "angry", "determined", "crying",
    "laughing", "smiling", "in pain", "exhausted", "desperate", "calm", "blushing", "smirking",

    "-- SETTING --",
    "indoor", "outdoor",
    "ocean", "beach", "underwater", "rain", "snow", "ice",
    "forest", "nature", "mountains", "cliff", "cave",
    "urban", "city", "street", "rooftop", "alley",
    "school", "classroom", "library", "office",
    "bedroom", "home", "kitchen", "living room", "dining room",
    "hospital", "prison", "church", "market", "restaurant", "cafe",
    "inside car", "inside carriage", "inside vehicle", "on road", "train", "ship",
    "battlefield", "arena", "temple", "ruins", "dungeon", "castle", "palace", "tavern",
    "night", "day", "sunset", "dawn",
    "dark environment", "bright environment", "foggy", "fire",

    "-- COMPANIONS / OBJECTS --",
    "pet present", "dog", "cat", "horse", "magical creature", "monster present",
    "weapon present", "sword", "gun", "magic spell", "shield", "bow",

    "-- ACTION / SITUATION --",
    "fighting", "combat", "battle", "training",
    "running", "jumping", "falling", "flying",
    "sleeping", "resting", "sitting", "standing", "walking",
    "driving", "riding", "on horseback",
    "arguing", "talking", "whispering",
    "eating", "drinking",
    "swimming", "diving",
    "embracing", "kissing", "holding hands", "romance scene",
    "injured", "bleeding", "wounded",
    "reading", "studying", "writing", "teaching", "working",
    "bullying", "being bullied", "protecting",
    "alone", "with others", "crowd scene",
    "hiding", "sneaking", "confrontation", "revelation",

    "-- POSE / CAMERA --",
    "face only", "bust shot", "waist up", "three quarter body", "full body",
    "front view", "three quarter view", "side view", "back view",
    "eye level", "high angle", "low angle",
    "action pose", "standing pose", "sitting pose", "dynamic pose", "crossed legs", "leaning",

    "-- ATMOSPHERE / TONE --",
    "dramatic", "peaceful", "intense", "melancholic", "epic", "tense", "serene",
    "romantic", "horror", "dark tone", "hopeful", "tragic", "comedic",
    "action moment", "quiet moment", "emotional moment", "power moment",
    "warm lighting", "cool lighting", "golden hour", "night scene",

    "-- MANHWA TROPES --",
    "truck kun", "isekai setup", "reincarnation", "summoned to another world",
    "power awakening", "aura release", "power up", "transformation sequence", "new form",
    "system ui", "status screen", "level up", "stat window", "skill notification",
    "villain reveal", "sinister smile", "shadow menace",
    "protagonist rage", "broken expression", "silent fury", "grief burst",
    "romance confession", "almost kiss", "love interest moment",
    "betrayal moment", "trust broken", "backstab", "plot twist", "shocking revelation",
    "training arc", "sparring", "power scaling",
    "boss fight", "dungeon crawl", "final clash", "last stand",
    "team gathering", "nakama power",
    "sacrifice moment", "heroic death", "character death",
    "comedic beat", "chibi moment", "reaction face",
    "flashback", "origin story", "memory scene",
    "tournament arc", "arena fight",
    "misunderstanding", "rom com moment",
    "hero speech", "resolve moment",
    "cliffhanger", "iconic pose", "chapter cover pose", "silhouette reveal"
  ],

  "suggested_category": "female" | "male" | "other" | "setting" | "creature",
  "auto_name": "Short vivid label — 3-6 words. Examples: 'Noble Female Knight Doorway', 'Power Awakening Blue Aura', 'Two Friends Laughing Cafe'. Be specific.",
  "scene_suitability": ["action","romance","drama","comedy","thriller","fantasy","slice of life","horror","supernatural","isekai","tournament","training"]
}

RULES:
- Tags use SPACES not hyphens. Write "young adult" not "young-adult", "dark tone" not "dark-tone".
- Do NOT copy the category labels (-- CHARACTER IDENTITY -- etc) into the tags array.
- Output 15-35 real tags that genuinely describe the image.
- For tropes: only apply if clearly visible (glowing status window = "system ui", truck hit = "truck kun").
- generation_prompt should be self-contained — someone using it without the image should get the same scene."""


_DEFAULT_ANALYSIS_MODEL = "claude-haiku-4-5-20251001"


def analyze_image_with_ai(
    pil_img,
    model: str = _DEFAULT_ANALYSIS_MODEL,
) -> Dict[str, Any]:
    """
    Use Claude Vision to extract structured metadata from a reference image.
    Defaults to claude-3-5-sonnet for rich trope + expression detection.
    Pass model="claude-3-5-haiku-20241022" for faster/cheaper bulk runs.

    Returns a dict with gender, age, archetype, manhwa tropes, tags, etc.
    Raises on API failure — callers should handle exceptions gracefully.
    """
    import anthropic

    from PIL import Image as _PILImage
    img = pil_img.convert("RGB")
    # Keep images compact — 768px max edge is plenty for face/style analysis
    # and avoids multi-MB base64 payloads that cause API timeouts
    max_edge = 768
    w, h = img.size
    if max(w, h) > max_edge:
        scale = max_edge / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), _PILImage.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")

    client = anthropic.Anthropic(timeout=60.0)
    msg = client.messages.create(
        model=model,
        max_tokens=1536,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
                {"type": "text", "text": _ANALYSIS_PROMPT},
            ],
        }],
    )
    raw = msg.content[0].text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw, flags=re.IGNORECASE).rstrip("`").strip()
    return json.loads(raw)


def get_existing_pinterest_ids() -> set:
    """Return the set of all pinterest_pin_id values already in the library."""
    lib = load_library()
    ids = set()
    for t in lib.get("templates", {}).values():
        pid = t.get("pinterest_pin_id")
        if pid:
            ids.add(str(pid))
    return ids


def get_existing_source_urls() -> set:
    """Return the set of all source_url values already in the library.

    Used by the cookie importer to skip images already imported by URL
    regardless of how they were originally added.
    """
    lib = load_library()
    urls = set()
    for t in lib.get("templates", {}).values():
        url = t.get("source_url")
        if url:
            urls.add(str(url))
        # Also treat uploaded FAL URLs as known so we don't re-import them
        for key in ("fal_face_url", "fal_body_url"):
            u = t.get(key)
            if u:
                urls.add(str(u))
    return urls


def analyze_and_update_template(
    template_id: str,
    img_type: str = "auto",
    model: str = _DEFAULT_ANALYSIS_MODEL,
) -> Dict[str, Any]:
    """
    Run AI analysis on a template image and write results back into the record.
    img_type="auto" tries face first, then body.
    Returns the analysis dict (or {"error": ...}).
    """
    from PIL import Image as _PILImage

    t = get_template(template_id)
    if t is None:
        return {"error": f"Template {template_id!r} not found."}

    # Resolve which image to analyse
    if img_type == "auto":
        img_path = t.get("local_face", "") or t.get("local_body", "")
        img_path = img_path if img_path and os.path.exists(img_path) else ""
    else:
        img_path = t.get(f"local_{img_type}", "")
        if not img_path or not os.path.exists(img_path):
            alt = "body" if img_type == "face" else "face"
            img_path = t.get(f"local_{alt}", "")
    if not img_path or not os.path.exists(img_path):
        return {"error": "No local image found to analyze."}

    try:
        pil = _PILImage.open(img_path).convert("RGB")
        analysis = analyze_image_with_ai(pil, model=model)
    except Exception as e:
        return {"error": str(e)}

    # Write analysis fields back to template
    updates: Dict[str, Any] = {"ai_analysis": analysis}

    # Auto-name if template still uses default ID name
    if not t.get("name") or t.get("name") == template_id:
        updates["name"] = analysis.get("auto_name", template_id)

    # Auto-category (only update if category is default "other" or blank)
    if t.get("category", "other") in ("other", "", None):
        cat_map = {"female": "female", "male": "male"}
        new_cat = cat_map.get(analysis.get("suggested_category", "other"), "other")
        updates["category"] = new_cat

    # Merge AI-suggested tags with existing user tags (no duplicates)
    existing_tags = set(t.get("tags") or [])
    ai_tags = set(analysis.get("tags") or [])
    updates["tags"] = sorted(existing_tags | ai_tags)

    # Store key attributes at top level for easy filtering/search
    for field in ("perceived_gender", "age_group", "apparent_age_range",
                  "body_type", "archetype", "scene_suitability", "auto_name",
                  "summary", "pose", "setting", "lighting", "art_style",
                  "reusability", "generation_prompt", "subject_count"):
        if field in analysis:
            updates[field] = analysis[field]

    update_template(template_id, **updates)
    return analysis


def bulk_analyze_unanalyzed(
    progress_cb=None,
    model: str = _DEFAULT_ANALYSIS_MODEL,
    max_workers: int = 10,
) -> Dict[str, Any]:
    """
    Run AI analysis on every template that doesn't have an ai_analysis yet.
    Uses a thread pool (default 10 workers) so multiple Claude calls run in
    parallel — ~10× faster than serial for large batches.
    progress_cb(current, total, template_id) is called as each job completes.
    Returns {"done": n, "skipped": n, "errors": [...]}.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    lib = load_library()
    templates = [
        t for t in lib.get("templates", {}).values()
        if not t.get("ai_analysis")
        and (os.path.exists(t.get("local_face", "")) or os.path.exists(t.get("local_body", "")))
    ]
    total = len(templates)
    done_count = 0
    errors: List[str] = []
    counter_lock = threading.Lock()

    def _analyze_one(t: Dict) -> Tuple[str, Dict]:
        tid = t["template_id"]
        result = analyze_and_update_template(tid, model=model)
        return tid, result

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_analyze_one, t): t for t in templates}
        for i, future in enumerate(as_completed(futures)):
            tid, result = future.result()
            with counter_lock:
                if "error" in result:
                    errors.append(f"{tid}: {result['error']}")
                else:
                    done_count += 1
                completed = done_count + len(errors)
            if progress_cb:
                progress_cb(completed, total, tid)

    return {"done": done_count, "skipped": total - done_count - len(errors), "errors": errors}


# ── URL / file import ─────────────────────────────────────────────────────────

def add_from_url(
    url: str,
    category: str = "other",
    img_type: str = "face",
    auto_analyze: bool = True,
) -> Tuple[str, Dict[str, Any]]:
    """
    Download an image from a URL, save it to the library as a new template, and
    optionally run AI analysis.  Returns (template_id, template_dict).
    Raises on download failure.
    """
    import requests
    from PIL import Image as _PILImage

    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    pil = _PILImage.open(io.BytesIO(r.content)).convert("RGB")

    template_id, t = add_template(category)
    save_image(template_id, img_type, pil)
    # Always persist the source URL so dedup works across sessions
    update_template(template_id, source_url=url)

    if auto_analyze:
        try:
            analysis = analyze_image_with_ai(pil)
            # Overwrite category with AI suggestion
            cat_map = {"female": "female", "male": "male"}
            detected_cat = cat_map.get(analysis.get("suggested_category", "other"), category)
            updates = {
                "ai_analysis": analysis,
                "name": analysis.get("auto_name", template_id),
                "category": detected_cat,
                "tags": sorted(set(analysis.get("tags") or [])),
            }
            for field in ("perceived_gender", "age_group", "body_type", "archetype",
                          "mood", "scene_suitability"):
                if field in analysis:
                    updates[field] = analysis[field]
            update_template(template_id, **updates)
            # Move to detected category's ID-space if different
            t = get_template(template_id) or t
        except Exception:
            pass  # analysis failure never blocks import

    return template_id, get_template(template_id) or t


def bulk_import_urls(
    urls: List[str],
    category: str = "other",
    img_type: str = "face",
    auto_analyze: bool = True,
    progress_cb=None,
) -> Dict[str, Any]:
    """
    Download and import a list of image URLs.
    Returns {"imported": [...template_ids], "errors": [...(url, msg)]}.
    """
    imported, errors = [], []
    for i, url in enumerate(urls):
        if progress_cb:
            progress_cb(i + 1, len(urls), url)
        try:
            tid, _ = add_from_url(url, category=category, img_type=img_type, auto_analyze=auto_analyze)
            imported.append(tid)
        except Exception as e:
            errors.append((url, str(e)))
    return {"imported": imported, "errors": errors}


def bulk_import_urls_fast(
    urls: List[str],
    category: str = "other",
    img_type: str = "face",
    workers: int = 10,
    progress_cb=None,
) -> Dict[str, Any]:
    """
    Fast bulk URL import: assigns all IDs upfront, downloads images in parallel,
    does ONE atomic library save at the end instead of one per image.
    Returns {"imported": n, "errors": n, "skipped": n}.
    """
    import datetime as _dt
    import threading as _thr
    import requests as _req
    from PIL import Image as _PILImage
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # ── 1. Pre-assign all IDs under the write lock (one read, no saves yet) ───
    with _LIB_LOCK:
        import fcntl as _fcntl
        lock_path = _LIB_JSON + ".lock"
        with open(lock_path, "w") as _lf:
            _fcntl.flock(_lf, _fcntl.LOCK_EX)
            try:
                lib = load_library()
                prefix = (category or "x")[:1].upper()
                existing_nums = [
                    int(tid[1:]) for tid in lib.get("templates", {})
                    if tid.startswith(prefix) and tid[1:].isdigit()
                ]
                next_n = max(existing_nums, default=0) + 1
                # Pre-assign one ID per URL
                pre_ids = [f"{prefix}{(next_n + i):03d}" for i in range(len(urls))]
            finally:
                _fcntl.flock(_lf, _fcntl.LOCK_UN)

    # ── 2. Download + save images in parallel (pure file I/O, no lib writes) ──
    now_iso = _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"
    results: Dict[int, Dict] = {}   # index → {"ok": True, "tid": ..., "local": ...} or {"err": ...}
    counter_lock = _thr.Lock()
    done_count = [0]

    def _download(idx_url):
        idx, url = idx_url
        attempt_url = url
        for try_url in [url, url.replace("/originals/", "/736x/").replace("/1200x/", "/736x/")]:
            try:
                r = _req.get(try_url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code == 403:
                    continue
                r.raise_for_status()
                pil = _PILImage.open(io.BytesIO(r.content)).convert("RGB")
                tid = pre_ids[idx]
                path = image_path(tid, img_type)
                pil.save(path, format="PNG")
                with counter_lock:
                    done_count[0] += 1
                    n = done_count[0]
                if progress_cb:
                    progress_cb(n, len(urls), tid)
                return idx, {"ok": True, "tid": tid, "local": path, "source_url": url}
            except Exception as e:
                if "403" not in str(e):
                    return idx, {"err": str(e)[:80]}
        return idx, {"err": "403 on all resolutions"}

    _ensure_dirs()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_download, (i, u)): i for i, u in enumerate(urls)}
        for fut in as_completed(futures):
            idx, res = fut.result()
            results[idx] = res

    # ── 3. One atomic library save with all new templates ─────────────────────
    ok_results = {i: r for i, r in results.items() if r.get("ok")}
    err_count  = sum(1 for r in results.values() if "err" in r)

    if ok_results:
        with _LIB_LOCK:
            import fcntl as _fcntl2
            lock_path2 = _LIB_JSON + ".lock"
            with open(lock_path2, "w") as _lf2:
                _fcntl2.flock(_lf2, _fcntl2.LOCK_EX)
                try:
                    lib = load_library()      # fresh read inside the lock
                    for i, r in ok_results.items():
                        tid = r["tid"]
                        lib.setdefault("templates", {})[tid] = {
                            "template_id": tid,
                            "category":    category,
                            "name":        tid,
                            "description": "",
                            "tags":        [],
                            "local_face":  r["local"] if img_type == "face" else "",
                            "local_body":  r["local"] if img_type == "body" else "",
                            "fal_face_url": "",
                            "fal_body_url": "",
                            "source_url":  r["source_url"],
                            "ai_analysis": None,
                            "created_at":  now_iso,
                        }
                    _atomic_save(_LIB_JSON, lib)
                finally:
                    _fcntl2.flock(_lf2, _fcntl2.LOCK_UN)

    return {"imported": len(ok_results), "errors": err_count, "skipped": 0,
            "total": len(urls)}


def bulk_import_files(
    file_paths: List[str],
    category: str = "auto",
    img_type: str = "face",
    auto_analyze: bool = False,
    progress_cb=None,
) -> Dict[str, Any]:
    """
    Import images from a list of local file paths (e.g. from Gradio multi-file upload).

    category="auto"  →  AI decides the category per image (requires auto_analyze=True);
                        falls back to "other" if analysis fails or is disabled.
    auto_analyze     →  Run Claude Vision on each image.  Slow (~1-2s/image) but gives
                        full multi-dimensional tagging.  Recommended to do later with
                        bulk_analyze_unanalyzed() if batch is large.

    Returns {"imported": [...template_ids], "errors": [...(path, msg)], "total": n}.
    """
    from PIL import Image as _PILImage

    imported, errors = [], []
    total = len(file_paths)

    for i, path in enumerate(file_paths):
        if progress_cb:
            progress_cb(i + 1, total, os.path.basename(path))
        try:
            pil = _PILImage.open(path).convert("RGB")

            # Assign initial category; AI may override it after analysis
            init_cat = "other" if category == "auto" else category
            template_id, _ = add_template(init_cat)
            save_image(template_id, img_type, pil)

            if auto_analyze:
                try:
                    analysis = analyze_image_with_ai(pil)
                    cat_map  = {"female": "female", "male": "male",
                                "setting": "setting", "creature": "creature"}
                    if category == "auto":
                        detected_cat = cat_map.get(
                            analysis.get("suggested_category", "other"), "other"
                        )
                    else:
                        detected_cat = init_cat

                    updates: Dict[str, Any] = {
                        "ai_analysis": analysis,
                        "name":     analysis.get("auto_name", template_id),
                        "category": detected_cat,
                        "tags":     sorted(set(analysis.get("tags") or [])),
                    }
                    for field in ("perceived_gender", "age_group", "body_type",
                                  "archetype", "mood", "scene_suitability"):
                        if field in analysis:
                            updates[field] = analysis[field]
                    update_template(template_id, **updates)
                except Exception:
                    pass  # keep the template even if analysis fails

            imported.append(template_id)
        except Exception as e:
            errors.append((os.path.basename(path), str(e)))

    return {"imported": imported, "errors": errors, "total": total}


def bulk_import_zip(
    zip_path: str,
    category: str = "auto",
    img_type: str = "face",
    auto_analyze: bool = False,
    progress_cb=None,
) -> Dict[str, Any]:
    """
    Extract a ZIP file and import every image found inside it.
    Returns the same dict shape as bulk_import_files.
    """
    import zipfile
    import tempfile

    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

    extracted_paths: List[str] = []
    tmpdir = tempfile.mkdtemp(prefix="lib_zip_")

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if os.path.splitext(name.lower())[1] in IMAGE_EXTS:
                    # Flatten folder structure — keep only the filename
                    dest_name = os.path.basename(name)
                    if not dest_name:
                        continue
                    dest = os.path.join(tmpdir, dest_name)
                    # Avoid collisions
                    if os.path.exists(dest):
                        base, ext = os.path.splitext(dest_name)
                        dest = os.path.join(tmpdir, f"{base}_{len(extracted_paths)}{ext}")
                    with zf.open(name) as src, open(dest, "wb") as dst:
                        dst.write(src.read())
                    extracted_paths.append(dest)

        if not extracted_paths:
            return {"imported": [], "errors": [("zip", "No image files found inside ZIP.")], "total": 0}

        result = bulk_import_files(
            extracted_paths,
            category=category,
            img_type=img_type,
            auto_analyze=auto_analyze,
            progress_cb=progress_cb,
        )
    finally:
        # Clean up extracted temp files
        import shutil
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except Exception:
            pass

    return result


# ── Reference resolution helpers ──────────────────────────────────────────────

# ── Unified beat-plan → library tag extractor ─────────────────────────────────
# Camera type → library tags
_CAMERA_TAG_MAP: Dict[str, List[str]] = {
    "dramatic medium character shot": ["dramatic", "character shot", "medium shot", "intense"],
    "over-the-shoulder shot":         ["over the shoulder", "dialogue", "conversation", "indoor"],
    "tight combat close-up":          ["close-up", "intense", "confrontation", "action moment"],
    "low-angle attack shot":          ["low angle", "power moment", "action moment", "iconic pose"],
    "wide chaos shot":                ["wide shot", "action moment", "chaos", "outdoor"],
    "extreme impact close-up":        ["extreme close-up", "impact", "action moment", "intense"],
    # bird's eye / overhead — user's example
    "birds eye":                      ["overhead", "birds eye", "aerial", "wide shot"],
    "overhead":                       ["overhead", "birds eye", "aerial", "wide shot"],
    "aerial":                         ["overhead", "birds eye", "aerial", "outdoor"],
}

# Scene type → library tags
_SCENE_TYPE_TAG_MAP: Dict[str, List[str]] = {
    "ACTION":    ["action moment", "confrontation", "intense", "power moment"],
    "EMOTION":   ["emotional moment", "dramatic", "intense", "sad"],
    "DIALOGUE":  ["dialogue", "conversation", "calm", "indoor"],
    "MEMORY":    ["memory", "soft lighting", "quiet moment", "melancholy"],
    "AWAKENING": ["power moment", "awakening", "iconic pose", "dramatic"],
    "AFTERMATH": ["aftermath", "quiet moment", "dramatic", "emotional moment"],
}

# Action keyword → library tags (scanned against the suggested_action free text)
_ACTION_KEYWORD_TAG_MAP: List[tuple] = [
    # Food / meals
    (["eat", "meal", "dinner", "lunch", "breakfast", "food", "drink", "cafe", "restaurant",
      "cook", "kitchen"],
     ["dining room", "indoor", "warm lighting"]),
    # Combat / fights
    (["fight", "combat", "battle", "punch", "kick", "slash", "stab", "shoot", "attack",
      "strike", "smash", "clash", "duel"],
     ["action moment", "confrontation", "arena fight", "intense"]),
    # Chase / movement
    (["run", "chase", "flee", "escape", "rush", "sprint", "jump", "leap"],
     ["action moment", "outdoor", "dramatic"]),
    # Nature / outdoors
    (["garden", "flower", "tree", "forest", "park", "field", "meadow", "nature",
      "grass", "leaf", "petals", "blossom"],
     ["outdoor", "natural setting", "blue sky"]),
    # Rain / weather
    (["rain", "storm", "thunder", "lightning", "snow", "fog", "mist"],
     ["dramatic", "outdoor", "dark environment"]),
    # Night
    (["night", "darkness", "dark", "moonlight", "stars"],
     ["night scene", "dark environment", "night"]),
    # Crowd / public
    (["crowd", "street", "market", "public", "city", "town", "alley"],
     ["outdoor", "city", "street", "urban"]),
    # School / study
    (["school", "class", "study", "book", "library", "campus", "lecture"],
     ["school", "classroom", "indoor", "bright environment"]),
    # Confrontation / tension
    (["confront", "face", "stare", "challenge", "standoff", "reveal", "shock", "surprise"],
     ["confrontation", "intense", "dramatic"]),
    # Emotional / breakdown
    (["cry", "sob", "weep", "tears", "grief", "mourn", "despair", "break down"],
     ["emotional moment", "sad", "crying", "melancholy"]),
    # Celebration / joy
    (["celebrate", "laugh", "smile", "cheer", "happy", "joy", "party"],
     ["cheerful", "bright environment", "happy"]),
    # Rooftop / height
    (["rooftop", "roof", "tower", "cliff", "ledge", "skyscraper", "high"],
     ["outdoor", "cityscape", "dramatic"]),
    # Underground / dungeon
    (["dungeon", "cave", "underground", "basement", "prison", "cell"],
     ["dark environment", "indoor"]),
    # Power / awakening
    (["power", "energy", "glow", "aura", "awaken", "transform", "unlock", "unleash"],
     ["power moment", "awakening", "dramatic", "iconic pose"]),
]


def _beat_to_search_tags(bp: Dict) -> List[str]:
    """Convert ALL signals in a beat plan into a single unified list of library search tags.

    Pulls from: camera_type, scene_type, suggested_action text, suggested_location,
    suggested_sub_location, character_emotions — everything the panel needs visually.
    Returns a deduplicated flat list ready for _find_top_n_templates.
    """
    tags: List[str] = []
    seen: set = set()

    def _add(new_tags: List[str]) -> None:
        for t in new_tags:
            tl = t.lower()
            if tl not in seen:
                seen.add(tl)
                tags.append(tl)

    # Camera type
    cam = str(bp.get("camera_type") or bp.get("suggested_perspective") or "").lower()
    for key, cam_tags in _CAMERA_TAG_MAP.items():
        if key in cam:
            _add(cam_tags)
            break

    # Scene type
    stype = str(bp.get("scene_type") or "").upper()
    if stype in _SCENE_TYPE_TAG_MAP:
        _add(_SCENE_TYPE_TAG_MAP[stype])

    # Character emotions
    raw_emo = bp.get("character_emotions") or {}
    if isinstance(raw_emo, dict):
        for emo_str in raw_emo.values():
            emo_lower = str(emo_str).lower()
            for emo_key, emo_tags in _EMOTION_TAG_MAP.items():
                if emo_key in emo_lower:
                    _add(emo_tags)

    # Suggested action — scan for keyword clusters
    action_text = str(bp.get("suggested_action") or "").lower()
    if action_text:
        for keywords, action_tags in _ACTION_KEYWORD_TAG_MAP:
            if any(k in action_text for k in keywords):
                _add(action_tags)

    # Location + sub-location
    loc  = str(bp.get("suggested_location") or "").lower()
    sub  = str(bp.get("suggested_sub_location") or "").lower()
    for loc_text in [loc, sub]:
        if not loc_text or loc_text == "none":
            continue
        for key, loc_tags in _LOCATION_TAG_MAP.items():
            if key in loc_text:
                _add(loc_tags)
        # Also scan action text for location keywords
        for key, loc_tags in _LOCATION_TAG_MAP.items():
            if key in action_text:
                _add(loc_tags)

    return tags


# Emotion word → library tags to search
_EMOTION_TAG_MAP: Dict[str, List[str]] = {
    "angry":      ["angry", "intense", "confrontation", "rage", "furious"],
    "furious":    ["angry", "furious", "intense", "confrontation"],
    "rage":       ["angry", "intense", "confrontation", "furious"],
    "sad":        ["sad", "crying", "emotional moment", "melancholy", "grief"],
    "crying":     ["crying", "sad", "emotional moment"],
    "grief":      ["sad", "emotional moment", "grief"],
    "happy":      ["cheerful", "happy", "bright environment", "smile"],
    "cheerful":   ["cheerful", "happy", "bright environment"],
    "fearful":    ["fearful", "shocked", "terrified", "tense"],
    "shocked":    ["shocked", "surprised", "fearful", "intense"],
    "tense":      ["tense", "intense", "dramatic"],
    "calm":       ["calm", "quiet moment", "peaceful"],
    "cold":       ["calm", "intense", "cold"],
    "confident":  ["confident", "power moment", "iconic pose"],
    "determined": ["confident", "determined", "power moment"],
    "romantic":   ["romantic", "warm lighting", "tender"],
    "mysterious": ["mysterious", "dark tone"],
    "desperate":  ["desperate", "emotional moment", "intense"],
    "pain":       ["intense", "emotional moment"],
    "excited":    ["cheerful", "power moment", "energetic"],
    "serious":    ["serious", "intense", "calm"],
    "smug":       ["confident", "calm", "smug"],
    "worried":    ["tense", "emotional moment", "worried"],
    "surprised":  ["shocked", "surprised"],
    "heroic":     ["power moment", "confident", "iconic pose"],
    "menacing":   ["dark tone", "intense", "confrontation"],
}

# Location keyword → library tags to search
_LOCATION_TAG_MAP: Dict[str, List[str]] = {
    # Use ACTUAL tags that exist on templates (check library tags, not abstract concepts)
    "school":       ["school", "academy", "academy setting", "indoor", "bright environment",
                     "elite school", "japanese school", "campus"],
    "classroom":    ["classroom", "school", "indoor", "bright environment"],
    "corridor":     ["corridor", "hallway", "indoor"],
    "hallway":      ["hallway", "corridor", "indoor"],
    "hospital":     ["indoor", "bright environment", "hallway"],
    "street":       ["street", "outdoor", "city", "tree lined street", "cobblestone"],
    "alley":        ["alley", "outdoor", "night scene", "dark environment"],
    "forest":       ["forest", "outdoor"],
    "city":         ["city", "cityscape", "outdoor", "european city"],
    "european":     ["european city", "outdoor", "street"],
    "village":      ["village setting", "outdoor"],
    "room":         ["indoor", "bedroom", "dining room"],
    "bedroom":      ["bedroom", "indoor"],
    "dining":       ["dining room", "indoor"],
    "office":       ["indoor"],
    "luxury":       ["indoor", "elegant"],
    "mansion":      ["castle", "indoor", "elegant"],
    "penthouse":    ["cityscape", "indoor"],
    "rooftop":      ["outdoor", "city", "cityscape"],
    "night":        ["night", "night scene", "dark environment"],
    "outdoor":      ["outdoor", "blue sky"],
    "indoor":       ["indoor"],
    "dungeon":      ["dark environment", "indoor"],
    "fantasy":      ["castle", "outdoor", "rpg world"],
    "castle":       ["castle", "indoor", "elegant"],
    "arena":        ["arena", "arena fight", "battle"],
    "cafe":         ["indoor"],
    "restaurant":   ["dining room", "indoor"],
    "lab":          ["indoor"],
    "gym":          ["indoor"],
    "park":         ["outdoor", "blue sky", "bench"],
    "bench":        ["bench", "outdoor"],
    "campus":       ["campus", "academy", "outdoor"],
    "car":          ["indoor"],
    "train":        ["indoor"],
    "underground":  ["dark environment", "indoor"],
    "sky":          ["blue sky", "outdoor", "cloudy sky"],
    "water":        ["outdoor"],
    "club":         ["indoor", "night"],
    "gala":         ["indoor", "elegant"],
    "clock tower":  ["clock tower", "castle", "academy"],
}

# Setting-related tags used in label display
_SETTING_DISPLAY_TAGS = frozenset([
    "indoor", "outdoor", "urban", "luxury", "academic", "dark tone",
    "bright environment", "night", "day", "natural setting", "professional",
    "domestic", "warm lighting", "cool lighting", "elegant", "dramatic",
])


_BG_TEMPLATE_TAGS = frozenset([
    "background art", "background scene", "visual novel background",
    "anime background", "environment art", "background",
])


def _score_template(
    t: Dict,
    search_set: set,
    prefer_background: bool = False,
) -> float:
    """Return raw tag-match score for a template (no url_usage penalty)."""
    t_tags = {tg.lower() for tg in (t.get("tags") or [])}
    score: float = len(search_set & t_tags)
    if prefer_background and (t_tags & _BG_TEMPLATE_TAGS):
        score += 3.0
    if score < 1.0:
        for s in search_set:
            if len(s) < 3:
                continue
            for tg in t_tags:
                if s in tg or tg in s:
                    score += 0.5
                    break
        if prefer_background and (t_tags & _BG_TEMPLATE_TAGS):
            score += 1.0
    return score


def _find_top_n_templates(
    search_tags: List[str],
    lib: Dict,
    used_urls: Optional[set] = None,
    prefer_key: str = "fal_body_url",
    fallback_key: str = "fal_face_url",
    prefer_background: bool = False,
    n: int = 5,
) -> List[Dict]:
    """Return up to N templates with the highest tag-match scores.

    url_usage is intentionally NOT applied here — the caller picks the
    least-used candidate from the returned list so usage tracking happens
    at the call site, not buried inside scoring.
    """
    search_set = {t.lower() for t in search_tags}
    scored: List[tuple] = []
    for t in lib.get("templates", {}).values():
        url = t.get(prefer_key) or t.get(fallback_key)
        if not url:
            continue
        if used_urls and url in used_urls:
            continue
        sc = _score_template(t, search_set, prefer_background=prefer_background)
        if sc > 0:
            scored.append((sc, t))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:n]]


def _pick_least_used(
    candidates: List[Dict],
    prefer_key: str = "fal_body_url",
    fallback_key: str = "fal_face_url",
    url_usage: Optional[Dict[str, int]] = None,
) -> Optional[Dict]:
    """From a list of candidates, return the one whose URL has been used least."""
    if not candidates:
        return None
    if not url_usage:
        return candidates[0]
    def _uses(t: Dict) -> int:
        url = t.get(prefer_key) or t.get(fallback_key) or ""
        return url_usage.get(url, 0)
    return min(candidates, key=_uses)


def _find_best_template(
    search_tags: List[str],
    lib: Dict,
    used_urls: Optional[set] = None,
    prefer_key: str = "fal_face_url",
    fallback_key: str = "fal_body_url",
    prefer_background: bool = False,
    url_usage: Optional[Dict[str, int]] = None,
) -> Optional[Dict]:
    """Find best tag-match template whose preferred URL hasn't been used yet.

    Keeps backward compatibility. For the top-N / least-used pattern use
    _find_top_n_templates + _pick_least_used instead.
    """
    search_set = {t.lower() for t in search_tags}
    best_score, best_t = -1.0, None
    for t in lib.get("templates", {}).values():
        url = t.get(prefer_key) or t.get(fallback_key)
        if not url:
            continue
        if used_urls and url in used_urls:
            continue
        score = _score_template(t, search_set, prefer_background=prefer_background)
        if url_usage and url in url_usage:
            score -= math.log2(url_usage[url] + 1) * 0.5
        if score > best_score:
            best_score, best_t = score, t
    return best_t if best_score > 0 else None


def get_template_appearance_prompt(template_id: str) -> str:
    """Build a concise visual-appearance string from a template's AI analysis.

    Used to inject real character looks (from the library cast image) into
    the generation prompt so the model knows what each cast member looks like.
    Falls back to template name if AI analysis is absent.
    """
    lib = load_library()
    t = lib.get("templates", {}).get(template_id)
    if not t:
        return ""
    ai = t.get("ai_analysis") or {}
    # generation_prompt is the most complete single-field option
    gen = (ai.get("generation_prompt") or "").strip()
    if gen:
        return gen
    if not ai:
        return (t.get("name") or "").strip()
    parts: List[str] = []
    gender  = (ai.get("perceived_gender") or "").strip()
    age     = (ai.get("age_group") or "").strip()
    body    = (ai.get("body_type") or "").strip()
    arch    = (ai.get("archetype") or "").strip()
    hair    = (ai.get("hair") or "").strip()
    skin    = (ai.get("skin_tone") or "").strip()
    eye     = (ai.get("eye_description") or "").strip()
    cloth   = (ai.get("clothing_description") or "").strip()
    feat    = (ai.get("distinctive_features") or "").strip()
    if age or gender:
        parts.append(f"{age} {gender}".strip())
    if arch:
        parts.append(arch)
    if body:
        parts.append(body)
    if hair:
        parts.append(f"{hair} hair" if "hair" not in hair.lower() else hair)
    if skin:
        parts.append(f"{skin} skin" if "skin" not in skin.lower() else skin)
    if eye:
        parts.append(eye)
    if cloth:
        parts.append(cloth)
    if feat:
        parts.append(feat)
    result = ", ".join(p for p in parts if p)
    return result or (t.get("name") or "").strip()


def resolve_scene_refs(
    char_names: List[str],
    character_cast: Dict[str, str],
    page_beat_plans: Optional[List[Dict]] = None,
    used_urls: Optional[set] = None,
    max_total: int = 14,
    max_chars: int = 5,
    url_usage: Optional[Dict[str, int]] = None,
    char_usage: Optional[Dict[str, int]] = None,
    char_types: Optional[Dict[str, str]] = None,
    char_fields: Optional[Dict[str, Dict]] = None,
) -> Tuple[List[Dict[str, str]], str, List[Dict[str, Any]]]:
    """
    Resolve ALL reference images for a page from the character library.

    Three reference types (all sourced from the library):
      1. Identity   — character face/body URLs (who each person IS)
      2. Emotion    — template tagged with matching emotion (pose/expression only)
      3. Setting    — template tagged with matching location (background/env only)

    Args:
        char_names:       characters in this beat (primary list)
        character_cast:   {story_char_name: template_id}
        page_beat_plans:  all beat_plan dicts for the current page
        used_urls:        URLs already used earlier in this story (no reuse)
        max_total:        hard cap on total refs
        max_chars:        max character identity refs

    Returns:
        refs:   [{url, tag}] — for NB2 Edit image_urls / reference_images
        prefix: structured prompt text describing each ref's role
        gaps:   [{type, description, needed}] for missing refs
    """
    lib = load_library()
    # ── Global URL usage — load from persistent database ─────────────────────
    # url_usage.json accumulates across ALL projects and ALL generations ever.
    # Any per-project counter passed in via the url_usage arg is merged on top
    # so that very-recent same-session uses also get penalised.
    _global_usage: Dict[str, int] = load_global_url_usage()
    if url_usage:
        for _u, _c in url_usage.items():
            _global_usage[_u] = _global_usage.get(_u, 0) + _c
    url_usage = _global_usage

    # _setting_used: blacklist for SETTING refs only — prevents the same env image
    # appearing on every page.  Character face/body refs are NOT blacklisted
    # because identity consistency requires the same face ref every time the
    # character appears.
    _setting_used: set = set(used_urls or [])
    # _char_url_used: within-this-call dedup so the same URL isn't added twice.
    _char_url_used: set = set()
    refs: List[Dict[str, str]] = []
    gaps: List[Dict[str, Any]] = []
    pos = 1

    identity_lines: List[str] = []
    emotion_lines:  List[str] = []
    setting_lines:  List[str] = []

    # ── Phase 1: Character identity refs ──────────────────────────────────────
    # Collect unique chars from all page beat plans + the direct char_names arg
    all_chars: List[str] = list(dict.fromkeys(char_names))
    if page_beat_plans:
        for bp in page_beat_plans:
            for cn in (bp.get("suggested_characters") or []):
                if cn not in all_chars:
                    all_chars.append(cn)

    # Sort by char_usage ascending — least-referenced characters get priority
    # in the max_chars slot budget so heavily-featured chars don't crowd out rare ones.
    if char_usage and len(all_chars) > 1:
        all_chars = sorted(all_chars, key=lambda c: char_usage.get(c, 0))

    # Compute the minimum usage count among eligible chars so we can detect
    # chars that appear 3× more than the least-featured (body ref skipped for them).
    _cu_vals = [char_usage.get(c, 0) for c in all_chars[:max_chars]] if char_usage else []
    _min_cu  = min(_cu_vals) if _cu_vals else 0

    chars_with_body = 0
    appearance_overrides: Dict[str, str] = {}

    def _creature_text_identity(char_name: str) -> Optional[str]:
        """
        Build a text-only appearance anchor for a creature/animal character
        when no library template exists.  Returns None if not enough data.
        """
        _ctype = (char_types or {}).get(char_name, "").lower()
        if _ctype not in ("animal", "beast", "creature"):
            return None
        _fields = (char_fields or {}).get(char_name) or {}
        skin   = _fields.get("skin") or _fields.get("hair") or ""
        eyes   = _fields.get("eyes") or ""
        build  = _fields.get("build") or ""
        anchor = _fields.get("anchor") or ""
        species = _fields.get("species") or _ctype
        parts = [p for p in [species, skin, eyes, build, anchor] if p and str(p).strip()]
        if not parts:
            return None
        return (
            f"IDENTITY for {char_name!r} (text description — no reference image): "
            f"{', '.join(parts)}. "
            "This character is NON-HUMAN — always draw them as a creature with this exact appearance. "
            "Do NOT draw a human body, human face, or human clothing for this character."
        )

    for char_name in all_chars[:max_chars]:
        if pos > max_total:
            break
        template_id = character_cast.get(char_name)
        if not template_id:
            # For creature/animal characters with no library template, synthesise
            # a text-only identity anchor from story bible fields rather than
            # firing a gap — creatures don't need a photo ref, just a description.
            _text_id = _creature_text_identity(char_name)
            if _text_id:
                identity_lines.append(f"  {_text_id}")
                # Also populate appearance_overrides so the prompt builder uses it
                _f = (char_fields or {}).get(char_name) or {}
                _desc_parts = [p for p in [
                    _f.get("skin") or _f.get("hair"), _f.get("eyes"),
                    _f.get("build"), _f.get("anchor")
                ] if p]
                if _desc_parts and char_name not in appearance_overrides:
                    appearance_overrides[char_name] = ", ".join(_desc_parts)
            else:
                gaps.append({"type": "character", "needed": char_name,
                             "description": f"No cast template assigned for '{char_name}'"})
            continue
        t = lib.get("templates", {}).get(template_id)
        if not t:
            _text_id = _creature_text_identity(char_name)
            if _text_id:
                identity_lines.append(f"  {_text_id}")
                _f = (char_fields or {}).get(char_name) or {}
                _desc_parts = [p for p in [
                    _f.get("skin") or _f.get("hair"), _f.get("eyes"),
                    _f.get("build"), _f.get("anchor")
                ] if p]
                if _desc_parts and char_name not in appearance_overrides:
                    appearance_overrides[char_name] = ", ".join(_desc_parts)
            else:
                gaps.append({"type": "character", "needed": char_name,
                             "description": f"Template {template_id!r} missing from library"})
            continue

        face_url = t.get("fal_face_url", "")
        body_url = t.get("fal_body_url", "")
        char_label = t.get("name") or template_id

        # Detect character species type for appropriate identity language.
        # char_types (story-level) is authoritative; fall back to library category.
        _char_field_type = ""
        if char_types:
            _char_field_type = (char_types.get(char_name) or "").lower()
        if not _char_field_type:
            _char_field_type = (t.get("category") or "").lower()
        _is_nonhuman = _char_field_type in ("animal", "beast", "creature", "non-human")

        # Get the appearance description from the template AI analysis
        appearance = get_template_appearance_prompt(template_id)
        if appearance and char_name not in appearance_overrides:
            appearance_overrides[char_name] = appearance

        # Face ref: always include whenever the character appears — identity
        # requires the same face URL every time.  Only skip for same-page dedup.
        if face_url and face_url not in _char_url_used and pos <= max_total:
            refs.append({"url": face_url, "tag": "character"})
            if _is_nonhuman:
                if appearance:
                    identity_lines.append(
                        f"  Image {pos} — {char_name} identity only (NON-HUMAN creature). "
                        f"{appearance[:400]}. "
                        "Lock the exact creature body shape, scale/fur/skin pattern, eye color, and species anatomy in every panel. "
                        "Never draw a human body, human face, or human limbs for this character."
                    )
                else:
                    identity_lines.append(
                        f"  Image {pos} — {char_name} identity only (NON-HUMAN creature). "
                        "Lock exact creature body shape, scale/fur/skin pattern, and eye color across all panels. "
                        "Never substitute a human body."
                    )
            else:
                if appearance:
                    identity_lines.append(
                        f"  Image {pos} — {char_name} identity only (face). "
                        f"{appearance[:400]}. "
                        "Lock this exact face shape, hair color/style, eye color, and skin tone in every panel. "
                        "Do not redesign or age this character between panels."
                    )
                else:
                    identity_lines.append(
                        f"  Image {pos} — {char_name} identity only (face). "
                        "Lock exact face shape, eye color, hair color/style, and skin tone across all panels."
                    )
            _char_url_used.add(face_url)
            pos += 1

        # Body ref: skip if this character has been in the story 3× more pages
        # than the least-featured character in this scene — give rare chars the slot.
        _this_cu = char_usage.get(char_name, 0) if char_usage else 0
        _body_overused = _min_cu > 0 and _this_cu > _min_cu * 3
        if body_url and body_url not in _char_url_used and chars_with_body < 2 and pos <= max_total and not _body_overused:
            refs.append({"url": body_url, "tag": "character"})
            if _is_nonhuman:
                identity_lines.append(
                    f"  Image {pos} — {char_name} identity only (full creature body). "
                    "Lock full creature form, body proportions, species anatomy, and signature markings to match this image. "
                    "Never apply human anatomy."
                )
            else:
                identity_lines.append(
                    f"  Image {pos} — {char_name} identity only (body/outfit). "
                    "Lock the outfit design, body proportions, and signature costume details to match this image exactly."
                )
            _char_url_used.add(body_url)
            chars_with_body += 1
            pos += 1

    # ── Phase 2: Setting / environment refs ──────────────────────────────────
    # ── Phase 2: Per-panel, per-signal refs ──────────────────────────────────
    # Each panel has multiple distinct visual needs: a camera angle, an emotion,
    # an action, a location.  Each need is its OWN separate search → top-5
    # candidates → pick the least-used image not yet on this page.
    # So a panel with a bird's-eye camera + sad emotion + garden gets 3 refs.
    #
    # Within-page uniqueness is ALWAYS enforced.
    # Cross-page blacklist is relaxed (level 1) if ideal search finds nothing.
    # Initialise outside the if-block so they're always defined for the return value
    # panel_ref_map: {panel_0idx: {"camera": [num,...], "mood": [num,...], "action": [num,...]}}
    panel_ref_map: Dict[int, Dict[str, List[int]]] = {}
    page_loc_refs: List[int] = []              # ref nums shared across all panels (location)

    if page_beat_plans:
        _page_panel_urls: set = set()    # within-this-page dedup
        _gap_logged: set = set()

        def _search_and_add(signal_tags: List[str], signal_hint: str,
                            panel_idx: Optional[int] = None,
                            is_location: bool = False,
                            signal_type: str = "mood") -> bool:
            """
            Run a two-level search for one visual signal, add the best ref.
            panel_idx:   which panel this ref belongs to (0-indexed). None = page-level.
            is_location: True for the single page-level setting ref.
            signal_type: category label stored in panel_ref_map — "camera", "mood", "action".
            Returns True if a ref was added, False otherwise.
            """
            nonlocal pos
            if pos >= max_total or not signal_tags:
                return False
            for _cross_page_avoid in [_setting_used, set()]:
                _avoid_this = _cross_page_avoid | _page_panel_urls | _char_url_used
                _candidates = _find_top_n_templates(
                    signal_tags, lib,
                    used_urls=_avoid_this,
                    prefer_key="fal_body_url",
                    fallback_key="fal_face_url",
                    n=5,
                )
                _picked = _pick_least_used(
                    _candidates,
                    prefer_key="fal_body_url",
                    fallback_key="fal_face_url",
                    url_usage=url_usage,
                )
                if _picked:
                    _u = _picked.get("fal_body_url") or _picked.get("fal_face_url")
                    if _u and _u not in _avoid_this:
                        refs.append({"url": _u, "tag": "style"})
                        _label = _picked.get("name") or _picked.get("template_id", "")
                        matched = [tg for tg in (_picked.get("tags") or [])
                                   if tg in signal_tags or tg in _SETTING_DISPLAY_TAGS]
                        tag_hint = ", ".join(matched[:3]) if matched else signal_hint
                        # Build scoped role description — one narrow job per ref
                        if is_location:
                            _role_line = (
                                f"  Image {pos} — ALL PANELS setting/architecture only. "
                                f"Use ONLY the background environment, architecture, and ambient lighting "
                                f"[{tag_hint}]. "
                                "All people in this image are invisible — never copy their face, clothing, or identity."
                            )
                        elif panel_idx is not None:
                            _panel_n = panel_idx + 1
                            _role_map = {
                                "camera": (
                                    f"Panel {_panel_n} camera/composition only. "
                                    f"Copy the shot angle, depth, and character framing [{tag_hint}]. "
                                ),
                                "mood": (
                                    f"Panel {_panel_n} mood/atmosphere only. "
                                    f"Borrow the lighting quality, color palette, and emotional tone [{tag_hint}]. "
                                ),
                                "action": (
                                    f"Panel {_panel_n} action/scene energy only. "
                                    f"Match the compositional energy and visual intensity [{tag_hint}]. "
                                ),
                            }
                            _scope = _role_map.get(signal_type, f"Panel {_panel_n} visual reference [{tag_hint}]. ")
                            _role_line = (
                                f"  Image {pos} — {_scope}"
                                "All people are invisible mannequins — never copy their face, hair, clothing, skin tone, or identity."
                            )
                        else:
                            _role_line = (
                                f"  Image {pos} — visual reference [{tag_hint}]. "
                                "People are invisible."
                            )
                        emotion_lines.append(_role_line)
                        _setting_used.add(_u)
                        _page_panel_urls.add(_u)
                        # Record which panel this ref number belongs to, keyed by signal type
                        if is_location:
                            page_loc_refs.append(pos)
                        elif panel_idx is not None:
                            _panel_bucket = panel_ref_map.setdefault(panel_idx, {})
                            _panel_bucket.setdefault(signal_type, []).append(pos)
                        pos += 1
                        return True
            return False

        # ── Per-unique-location refs (ONE ref per distinct location on the page) ─
        # Deduplicate locations first, then search once per unique place.
        # • 6 bathroom panels  → 1 ref  (no contradictory lighting)
        # • 6 panels, 6 places → 6 refs (each location gets its own ref)
        # • 3 park + 2 kitchen → 2 refs (one per unique setting)
        _seen_locs: List[str] = []       # ordered-unique location strings
        for _bp in page_beat_plans:
            _loc = (_bp.get("suggested_location") or "").strip()
            _sub = (_bp.get("suggested_sub_location") or "").strip()
            _lf  = f"{_loc} {_sub}".strip() if _sub else _loc
            if _lf and _lf.lower() not in ("", "none") and _lf not in _seen_locs:
                _seen_locs.append(_lf)
        for _uloc in _seen_locs:
            if pos >= max_total:
                break
            _page_loc_tags: List[str] = []
            for _lk, _lv in _LOCATION_TAG_MAP.items():
                if _lk in _uloc.lower():
                    _page_loc_tags.extend(_lv)
            if not _page_loc_tags:
                _page_loc_tags = [w for w in _uloc.lower().split() if len(w) > 3]
            _page_loc_tags = list(dict.fromkeys(_page_loc_tags))
            if _page_loc_tags:
                _search_and_add(_page_loc_tags, f"location: {_uloc}", is_location=True)

        # ── Per-panel signals: camera, emotion, action (NOT location) ─────────
        for _pidx, bp in enumerate(page_beat_plans):
            if pos >= max_total:
                break

            # Signal 1: Camera / composition
            cam = str(bp.get("camera_type") or bp.get("suggested_perspective") or "").lower()
            cam_tags: List[str] = []
            for _cam_key, _cam_vals in _CAMERA_TAG_MAP.items():
                if _cam_key in cam:
                    cam_tags = _cam_vals
                    break
            if cam_tags:
                _search_and_add(cam_tags, f"camera: {cam}", panel_idx=_pidx, signal_type="camera")

            if pos >= max_total:
                break

            # Signal 2: Emotion / mood
            # Primary source: character_emotions from beat plan
            # Fallback: scan suggested_action text for emotion keywords directly
            # (many panels have no character_emotions data but the action text says
            #  "trembling", "desperate", "weeping" etc. — mine those too)
            raw_emo = bp.get("character_emotions") or {}
            emo_tags: List[str] = []
            emo_label_parts: List[str] = []
            if isinstance(raw_emo, dict):
                for _emo_str in raw_emo.values():
                    _emo_lower = str(_emo_str).lower()
                    emo_label_parts.append(str(_emo_str))
                    for _ek, _ev in _EMOTION_TAG_MAP.items():
                        if _ek in _emo_lower:
                            emo_tags.extend(_ev)
            emo_tags = list(dict.fromkeys(emo_tags))
            if not emo_tags:
                # Fallback: scan action text for emotion keywords
                _action_lower = str(bp.get("suggested_action") or "").lower()
                for _ek, _ev in _EMOTION_TAG_MAP.items():
                    if _ek in _action_lower:
                        emo_tags.extend(_ev)
                        emo_label_parts.append(_ek)
                emo_tags = list(dict.fromkeys(emo_tags))
            if emo_tags:
                _search_and_add(emo_tags, f"emotion: {', '.join(emo_label_parts[:2])}",
                                panel_idx=_pidx, signal_type="mood")

            if pos >= max_total:
                break

            # Signal 3: Action / scene type
            action_text = str(bp.get("suggested_action") or "").lower()
            scene_type  = str(bp.get("scene_type") or "").upper()
            act_tags: List[str] = list(_SCENE_TYPE_TAG_MAP.get(scene_type, []))
            for _kws, _atags in _ACTION_KEYWORD_TAG_MAP:
                if any(k in action_text for k in _kws):
                    act_tags.extend(_atags)
            act_tags = list(dict.fromkeys(act_tags))
            if act_tags:
                _search_and_add(act_tags, f"action: {action_text[:40]}", panel_idx=_pidx, signal_type="action")

    if not refs:
        return [], "", gaps, {}, {}

    # ── Build structured prompt prefix ────────────────────────────────────────
    # Each ref gets ONE narrow job. "Use Image N" lines are injected inline into
    # each PANEL section by director.py — the prefix here is the catalogue only.
    _div = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    parts: List[str] = [_div, "REFERENCE ROLES", _div, ""]

    all_role_lines = identity_lines + emotion_lines + (setting_lines or [])
    parts.extend(all_role_lines)

    parts.append(
        "\n"
        "IDENTITY RULE: Maintain each character's exact face, hair, eye color, skin tone, and outfit "
        "from their identity image in EVERY panel — do not redesign or age them.\n"
        "SPECIES LOCK: Each character's body type is permanently fixed. Never apply non-human anatomy "
        "(scales, serpentine form, fur, claws) to a human, and never apply human anatomy "
        "(human face, skin, limbs, clothing) to a creature. Keep species completely separate.\n"
        "MANNEQUIN RULE: All people shown inside non-identity reference images are invisible mannequins — "
        "never copy their face, hair, skin tone, clothing, or identity.\n"
        "LANGUAGE: All text, signs, labels, dialogue, captions must be in ENGLISH.\n"
        f"{_div}\n"
    )

    prefix = "\n".join(parts) + "\n\n"
    _panel_ref_data: Dict[str, Any] = {
        "panel_map": panel_ref_map,   # {panel_0idx: [ref_num, ...]}
        "loc_refs":  page_loc_refs,   # [ref_num, ...] — shared across all panels
    }
    return refs, prefix, gaps, appearance_overrides, _panel_ref_data


def log_generation_gaps(gaps: List[Dict[str, Any]], project_id: str = "") -> None:
    """
    Persist real-time generation gaps (missing emotion/setting refs) to the library,
    so they show up in the Gaps tab alongside structural gap analysis.
    """
    if not gaps:
        return
    _ensure_dirs()
    gap_file = os.path.join(_LIB_ROOT, "generation_gaps.json")
    try:
        with _LIB_LOCK:
            existing: List[Dict] = []
            if os.path.exists(gap_file):
                try:
                    with open(gap_file, "r", encoding="utf-8") as f:
                        existing = json.load(f)
                except Exception:
                    existing = []
            # Deduplicate by (type, needed)
            seen = {(g.get("type"), g.get("needed")) for g in existing}
            for g in gaps:
                key = (g.get("type"), g.get("needed"))
                if key not in seen:
                    existing.append({**g, "project_id": project_id})
                    seen.add(key)
            with open(gap_file, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


# ── Gap Detection ─────────────────────────────────────────────────────────────

# Mapping from story SCENE_TYPES and common beat keywords → library tag/attribute clusters
_SCENE_TAG_MAP: Dict[str, List[str]] = {
    "ACTION":     ["warrior", "fierce", "athletic", "muscular", "action"],
    "EMOTION":    ["drama", "sad", "warm", "gentle", "mysterious"],
    "DIALOGUE":   ["adult", "serious", "confident", "support"],
    "MEMORY":     ["slice_of_life", "warm", "cheerful"],
    "AWAKENING":  ["protagonist", "fierce", "confident"],
    "AFTERMATH":  ["sad", "tired", "drama"],
    # Extended visual keywords detected from beat text
    "fight":      ["warrior", "action", "fierce", "athletic"],
    "battle":     ["warrior", "action", "muscular"],
    "romance":    ["romance", "warm", "gentle", "romantic_lead"],
    "kiss":       ["romance", "warm"],
    "sleep":      ["slice_of_life", "tired"],
    "run":        ["action", "athletic"],
    "cry":        ["drama", "sad"],
    "school":     ["slice_of_life", "teen", "young_adult"],
    "train":      ["warrior", "action", "athletic"],
    "eat":        ["slice_of_life", "cheerful"],
    "walk":       ["slice_of_life"],
    "confront":   ["fierce", "drama", "serious"],
    "reveal":     ["drama", "mysterious"],
}


def detect_library_gaps(projects_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Scan all project beat_plans for scene types, compare against library coverage.

    Returns a list of gap dicts:
      {scene_key, label, beats_using_it, template_count, tags_needed, severity}
    sorted by severity (high first).
    """
    if projects_dir is None:
        projects_dir = os.path.join(_MODULE_DIR, "projects")

    # ── Count scene appearances across all projects ───────────────────────────
    scene_counts: Dict[str, int] = {}

    if os.path.isdir(projects_dir):
        for proj in os.listdir(projects_dir):
            pjson = os.path.join(projects_dir, proj, "project.json")
            if not os.path.isfile(pjson):
                continue
            try:
                with open(pjson, "r", encoding="utf-8") as f:
                    data = json.load(f)
                beat_plans = data.get("beat_plans") or {}
                beats = data.get("beats") or []
                for bp in beat_plans.values():
                    st = (bp.get("scene_type") or "").upper()
                    if st:
                        scene_counts[st] = scene_counts.get(st, 0) + 1
                # Also scan raw beat text for keyword signals
                for beat in beats:
                    low = beat.lower()
                    for kw in ("fight", "battle", "romance", "kiss", "sleep",
                               "run", "cry", "school", "train", "eat", "walk",
                               "confront", "reveal"):
                        if kw in low:
                            scene_counts[kw] = scene_counts.get(kw, 0) + 1
            except Exception:
                pass

    # ── Count library coverage per tag cluster ────────────────────────────────
    lib = load_library()
    templates = list(lib.get("templates", {}).values())

    def _coverage(tags_needed: List[str]) -> int:
        count = 0
        for t in templates:
            ttags = set(t.get("tags") or [])
            scene_suit = set(t.get("scene_suitability") or [])
            archetype = t.get("archetype", "")
            combined = ttags | scene_suit | {archetype}
            if any(tag in combined for tag in tags_needed):
                count += 1
        return count

    # ── Build gap report ──────────────────────────────────────────────────────
    gaps: List[Dict[str, Any]] = []
    seen: set = set()

    for scene_key, beat_count in sorted(scene_counts.items(), key=lambda x: -x[1]):
        if beat_count < 2:
            continue
        tags_needed = _SCENE_TAG_MAP.get(scene_key, _SCENE_TAG_MAP.get(scene_key.lower(), []))
        if not tags_needed:
            continue
        canon_key = tuple(sorted(tags_needed))
        if canon_key in seen:
            continue
        seen.add(canon_key)

        coverage = _coverage(tags_needed)
        ratio = coverage / max(beat_count, 1)

        if ratio >= 2.0:
            severity = "ok"
        elif ratio >= 1.0:
            severity = "low"
        elif ratio >= 0.3:
            severity = "medium"
        else:
            severity = "high"

        gaps.append({
            "scene_key":    scene_key,
            "label":        scene_key.replace("_", " ").title(),
            "beats_using_it": beat_count,
            "template_count": coverage,
            "tags_needed":  tags_needed,
            "severity":     severity,
        })

    severity_order = {"high": 0, "medium": 1, "low": 2, "ok": 3}
    gaps.sort(key=lambda g: (severity_order.get(g["severity"], 9), -g["beats_using_it"]))
    return gaps


# ── Casting helpers ───────────────────────────────────────────────────────────

# Common tags used for matching
COMMON_TAGS = [
    "young", "teen", "adult", "mature", "elderly",
    "slim", "athletic", "muscular", "curvy", "tall", "short",
    "elegant", "noble", "refined", "rough", "fierce", "gentle",
    "protagonist", "antagonist", "warrior", "scholar", "leader",
    "serious", "cheerful", "cold", "warm", "mysterious",
    "fair-skin", "tan-skin", "dark-skin",
    "long-hair", "short-hair", "bald",
    "light-eyes", "dark-eyes",
]


def filter_templates_for_cast(
    gender: str,
    required_tags: Optional[List[str]] = None,
    lib: Optional[Dict] = None,
) -> List[str]:
    """
    Return template IDs eligible for casting a character of given gender.
    Sorted by tag-match score (best first). Returns empty list if none found.
    """
    if lib is None:
        lib = load_library()
    cat_map = {"male": "male", "m": "male", "female": "female", "f": "female", "woman": "female", "man": "male"}
    cat = cat_map.get((gender or "").strip().lower(), "other")

    candidates = [
        t for t in lib.get("templates", {}).values()
        if t.get("category", "").lower() == cat
    ]

    if not required_tags:
        return [t["template_id"] for t in candidates]

    req_set = {tg.lower() for tg in required_tags}
    scored = []
    for t in candidates:
        ttags = {tg.lower() for tg in (t.get("tags") or [])}
        score = len(req_set & ttags)
        scored.append((score, t["template_id"]))
    scored.sort(key=lambda x: -x[0])
    return [tid for _, tid in scored]


def _template_cast_score(t: Dict, required_tags: List[str], is_lead: bool = False) -> float:
    """
    Score a template for casting quality.
    Higher = better candidate.
      +2.0  has "solo" tag (clearly shows only this character)
      +1.0  per matching required tag
      +0.5  has local face image on disk
      +0.5  has FAL face URL (already uploaded, usable immediately)
      +0.0–1.0  ai reusability.as_identity_ref (0–1 scale from analysis)
      +2.0  (lead only) archetype is protagonist or romantic lead
    """
    tags = {tg.lower() for tg in (t.get("tags") or [])}
    score = 0.0
    if "solo" in tags or "alone" in tags:
        score += 2.0
    req_set = {tg.lower() for tg in required_tags}
    score += len(req_set & tags)
    if t.get("local_face") and os.path.exists(t.get("local_face", "")):
        score += 0.5
    if t.get("fal_face_url"):
        score += 0.5
    reuse = (t.get("reusability") or {})
    score += float(reuse.get("as_identity_ref", 0) or 0)
    if is_lead:
        arch = ((t.get("ai_analysis") or {}).get("archetype") or "").lower()
        if arch in ("protagonist", "romantic lead"):
            score += 2.0
    return score


def auto_cast_characters(
    st_characters: Dict,
    st_project_id: str,
    existing_cast: Optional[Dict] = None,
) -> Dict[str, str]:
    """
    Auto-cast story characters to library templates.

    Rules:
    - Prefer templates tagged "solo"/"alone" (character clearly visible on their own).
    - Score by reusability.as_identity_ref from AI analysis.
    - First character in the story is assumed protagonist → defaults to male unless
      the story bible already set a gender.
    - Lead characters (first 2 by position, or role flagged as lead/protagonist/main)
      are restricted to attractive templates (protagonist / romantic lead archetype)
      and cannot be cast as monsters, creatures, crowd shots, or damaged-appearance
      templates.  Supporting characters have no such restriction.
    - Respects existing_cast: never silently replaces an already-cast character.
    - Returns {char_name: template_id} for newly cast characters.
    """
    # Tags / archetypes that disqualify a template for lead roles.
    # Only excludes genuinely non-human / multi-person templates — NOT appearance
    # qualities like bruised or old, since those may be exactly right for the story.
    _LEAD_EXCLUDE_ARCHETYPES = frozenset({"monster", "crowd"})
    _LEAD_EXCLUDE_TAGS = frozenset({
        "monster", "creature", "beast", "non-human",
        "crowd", "group", "zombie", "undead",
    })

    lib = load_library()
    existing = existing_cast or {}
    seed = abs(hash(st_project_id or "default")) % (2 ** 31)
    rng = random.Random(seed)

    new_cast: Dict[str, str] = {}
    used_templates = set(existing.values())
    char_list = list((st_characters or {}).items())

    for char_idx, (char_name, char_data) in enumerate(char_list):
        if char_name in existing:
            continue  # already locked

        fields = char_data.get("fields") or {}
        gender = (fields.get("gender") or "").strip().lower()

        # First character defaults to male (protagonist) unless explicitly set otherwise
        if char_idx == 0 and gender not in ("male", "female", "m", "f", "woman", "man"):
            gender = "male"

        # Detect lead characters: first two by position, or role text says so
        char_role = (
            fields.get("role") or fields.get("character_type") or
            char_data.get("character_type") or ""
        ).lower()
        is_lead = char_idx < 2 or any(
            w in char_role for w in ("lead", "protagonist", "main", "hero", "heroine")
        )

        # Extract trait tags from DNA prompt for better matching
        dna = (char_data.get("dna_prompt") or "").lower()
        char_type = char_role
        tags: List[str] = []
        if any(w in dna for w in ["young", "teen", "student", "junior"]):
            tags.append("young adult")
        if any(w in dna for w in ["mature", "middle-aged"]):
            tags.append("adult")
        if any(w in dna for w in ["elegant", "noble", "refined", "graceful"]):
            tags.append("elegant")
        if any(w in dna for w in ["warrior", "fighter", "soldier", "knight"]):
            tags.append("warrior")
        if any(w in dna for w in ["slim", "slender", "lean"]):
            tags.append("slim")
        if any(w in dna for w in ["muscular", "built", "strong", "broad"]):
            tags.append("muscular")
        if "villain" in char_type or any(w in dna for w in ["villain", "antagonist", "evil"]):
            tags.append("dark tone")
        # Leads get archetype-matching tags to pull attractive protagonist templates
        if is_lead:
            tags.extend(["protagonist", "romantic lead"])

        # Get gender-matched candidates then score them
        candidate_ids = filter_templates_for_cast(gender, tags or None, lib)
        all_templates = lib.get("templates", {})

        # For animal/beast characters the lead exclusion is inverted:
        # we WANT creature templates, not human protagonist ones.
        char_is_nonhuman = fields.get("character_type", "").lower() in ("animal", "beast", "creature")

        scored: List[tuple] = []
        for tid in candidate_ids:
            t = all_templates.get(tid)
            if not t:
                continue
            # Hard exclusion for lead roles: no monster/creature/damaged templates
            # — but skip this filter entirely for non-human characters, since
            # a snake protagonist SHOULD be matched to a creature template.
            if is_lead and not char_is_nonhuman:
                t_arch = ((t.get("ai_analysis") or {}).get("archetype") or "").lower()
                t_tags = {tg.lower() for tg in (t.get("tags") or [])}
                if t_arch in _LEAD_EXCLUDE_ARCHETYPES:
                    continue
                if t_tags & _LEAD_EXCLUDE_TAGS:
                    continue
            sc = _template_cast_score(t, tags, is_lead=is_lead)
            # Penalty for already-used templates (still allowed as fallback)
            if tid in used_templates:
                sc -= 10.0
            scored.append((sc, tid))

        if not scored:
            # Relaxed fallback: remove lead exclusions but keep the score bias
            for tid in candidate_ids:
                t = all_templates.get(tid)
                if not t:
                    continue
                sc = _template_cast_score(t, tags, is_lead=is_lead)
                if tid in used_templates:
                    sc -= 10.0
                scored.append((sc, tid))

        if not scored:
            continue

        scored.sort(key=lambda x: -x[0])
        # Pick from top-5 scored to allow some variety across projects
        top = scored[:min(5, len(scored))]
        top_fresh = [x for x in top if x[1] not in used_templates]
        pool = top_fresh if top_fresh else top
        chosen_score, chosen = rng.choice(pool)
        new_cast[char_name] = chosen
        used_templates.add(chosen)

    return new_cast


# (duplicate resolve_scene_refs removed — see the full version above)
