import os
import re
import io
import json
import uuid
import time
import zipfile
import hashlib
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional

import cloud_storage as _cloud
import reasoning_provider as _rp

import gradio as gr
from PIL import Image

try:
    import fal_client
    from fal_client import FalClientHTTPError as _FalHTTPError
except Exception:
    fal_client = None
    _FalHTTPError = None


def _rewrite_flagged_prompt(prompt: str, reason: str) -> Optional[str]:
    """Rewrite a rejected image prompt using the active reasoning provider."""
    if _rp.is_deepseek_mode():
        text, _status = _rp.call_text(
            "Rewrite the image-generation prompt while preserving its visual intent and details. Return only the final prompt.",
            {"reason": reason or "content filter", "prompt": prompt},
            max_tokens=4000,
            temperature=0,
        )
        return text.strip() if text else None

    api_key = (os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY") or "").strip()
    if not api_key:
        return None
    try:
        import requests as _rq
        r = _rq.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={
                "model": "claude-haiku-4-5",   # fast/cheap for this simple task
                "max_tokens": 4000,
                "temperature": 0,
                "system": (
                    "You rewrite image-generation prompts that were rejected by an AI safety filter. "
                    "Your job is to preserve EVERY creative and visual detail while rephrasing "
                    "anything that could trigger safety checks. "
                    "Rules:\n"
                    "- Keep character descriptions, scene details, art style, camera angles — everything.\n"
                    "- Replace direct violence/gore language with action synonyms (clash→duel, "
                    "  blood→crimson marks, injury→wounds, killing→defeating, dead→fallen).\n"
                    "- Replace romantic/sexual wording with tasteful equivalents.\n"
                    "- Do not remove named characters, clothing, or plot-relevant objects.\n"
                    "- Return ONLY the rewritten prompt text. No preamble, no explanation."
                ),
                "messages": [{"role": "user", "content":
                    f"This prompt was flagged (reason: {reason or 'content policy'}).\n\n"
                    f"ORIGINAL PROMPT:\n{prompt}\n\n"
                    "Rewrite it so it passes the safety filter without losing visual detail."}],
            },
            timeout=(10, 60),
        )
        r.raise_for_status()
        parts = [p.get("text", "") for p in r.json().get("content", []) if p.get("type") == "text"]
        result = "".join(parts).strip()
        return result if result else None
    except Exception as _e:
        print(f"[rewrite_flagged_prompt] error: {_e}", flush=True)
        return None

DEFAULT_STYLE = (
    "Korean manhwa webtoon art style, visible bold black ink line art — distinct ink strokes clearly readable as lines, "
    "thick contour edges with thin interior cross-hatching detail, "
    "hard-edged cel shading flat color fills separated by sharp ink line borders — no painterly blending, no airbrush, no soft gradients, "
    "vivid saturated color palette with strong warm-cool temperature contrast, "
    "large luminous eyes with gradient iris fill, fine lash detail, and hard white catchlight, "
    "THEATRICAL EXAGGERATED EXPRESSIONS — jaw fully dropped, mouth wide open, eyes stretched wide in shock, "
    "brow furrowed in rage, full grin showing teeth, sweat drops, manga-style reaction lines around face — "
    "every emotion must be readable from across a room, "
    "dynamic cinematic composition with strong foreground framing elements and depth, "
    "clean professional Korean webtoon digital illustration, 2D flat art, not photorealistic, not 3D render"
)
DEFAULT_NEGATIVE = (
    "photorealistic, realistic, 3d render, cgi, photograph, live action, "
    "low quality, blurry, pixelated, jpeg artifacts, extra limbs, bad anatomy, deformed hands, missing fingers, "
    "text, words, watermark, logo, UI elements, speech bubble, "
    "non-English text, foreign script, Japanese text, Chinese characters, Korean hangul text, Arabic script, "
    "flat colors, no shading, pastel washed-out palette, chalky, desaturated, muddy, "
    "soft painterly blending, airbrush shading, smooth gradients, oil painting texture, "
    "neutral expression, bland face, mild expression, subtle emotion, resting face, closed mouth when surprised, "
    "western comics, Disney style, chibi, overly cute, amateur sketch, ugly, mutated, "
    "panel overflow, figure breaking panel border, character crossing panel edge, panel bleed, "
    "limb extending outside frame, body part outside panel, figure bursting through border, "
    "overlapping panel borders, elements outside panel boundary, "
    "nude, naked, nudity, topless, bottomless, exposed skin, bare chest, bare breasts, nsfw, explicit, sexual"
)
REF_NEGATIVE = (
    f"{DEFAULT_NEGATIVE}, realistic skin pores, camera grain, DSLR photograph, lens bokeh, hyperreal texture"
)
REF_ANIME_LOCK = (
    "Korean manhwa webtoon illustrated style, bold clean line art, high-contrast cel shading, "
    "vibrant saturated colors, 2D drawn illustration, not photorealistic, not live action"
)
FAL_MODEL = "fal-ai/z-image/turbo"
FAL_OUTPUT_EXT = "png"
FORCED_IMAGE_SIZE = "landscape_16_9"  # FAL native preset — 1360×768, cleaner than custom dict for turbo model
CLAUDE_MODEL = "claude-sonnet-4-5"
CLAUDE_FAST_MODEL = "claude-haiku-4-5"  # faster/cheaper — used for extraction tasks
OPENAI_PROMPT_MODEL = os.getenv("OPENAI_PROMPT_MODEL", "gpt-4.1-mini")
PANELS_PER_PAGE: int = 10        # beats per page in Panel mode (2-col × 5-row grid)
SHORTS_PANELS_PER_PAGE: int = 6  # beats per page in Shorts mode (2-col × 3-row grid)

PLACE_WORD_BLOCKLIST = {
    "Academy", "School", "Classroom", "Hallway", "Room", "Bedroom", "Kitchen", "Bathroom",
    "Hospital", "Office", "Factory", "Warehouse", "Street", "Alley", "Forest", "Mountain",
    "River", "Lake", "Station", "Subway", "House", "Apartment", "Backyard", "Yard",
    "Rooftop", "Garage", "Porch", "Lab", "Parking", "Lot", "University", "Campus",
    "Dorm", "Gym", "Cafeteria", "Library", "Courtyard", "Gate", "Arena", "Training", "Grounds",
}

# Generic words that appear capitalised in manhwa/fantasy stories but are NOT character names.
# Used to filter both the regex extractor and Claude's returned character list.
GENERIC_NOUN_BLOCKLIST = {
    # Elements / nature
    "Stone", "Water", "Fire", "Wind", "Earth", "Shadow", "Light", "Dark", "Darkness",
    "Blood", "Ice", "Thunder", "Lightning", "Sky", "Sun", "Moon", "Star", "Stars",
    "Night", "Day", "Void", "Flame", "Smoke", "Dust", "Ash", "Mist", "Fog", "Wave",
    "Storm", "Snow", "Rain", "Sand", "Iron", "Steel", "Gold", "Silver", "Crystal",
    # Abstract / concept
    "Spirit", "Soul", "Mind", "Heart", "Body", "Form", "Voice", "Force", "Energy",
    "Power", "Mana", "Aura", "Qi", "Chi", "Chakra", "Life", "Death", "Fate", "Time",
    "Space", "World", "Heaven", "Hell", "Realm", "Domain", "Void", "Chaos", "Order",
    "Truth", "Law", "Karma", "Curse", "Blessing", "Will",
    # Titles / roles (when used as nouns, not names)
    "Master", "King", "Queen", "Lord", "Emperor", "Empress", "Prince", "Princess",
    "Knight", "Warrior", "Hunter", "Demon", "Dragon", "God", "Goddess", "Devil",
    "Monster", "Beast", "Human", "Mortal", "Immortal", "Ancient", "Elder", "Ancestor",
    "Hero", "Villain", "Assassin", "Mage", "Wizard", "Witch", "Sage", "Oracle",
    "Guardian", "Protector", "Servant", "Slave", "Puppet", "Ghost", "Phantom",
    # System / game terms common in manhwa
    "System", "Interface", "Level", "Rank", "Grade", "Class", "Stage", "Phase",
    "Tier", "Guild", "Clan", "Party", "Quest", "Mission", "Dungeon", "Gate",
    "Player", "User", "Admin", "Host",
    # Story structure words
    "Chapter", "Part", "Scene", "Arc", "Volume", "Episode",
    # Common descriptors used as nouns
    "True", "False", "Sacred", "Holy", "Fallen", "Lost", "Broken", "Chosen",
    "First", "Last", "Only", "Final", "New", "Old", "Young",
    # Common body/action words capitalised in dramatic prose
    "Eye", "Eyes", "Hand", "Hands", "Face", "Head", "Voice", "Words", "Step",
    # Number words that appear capitalised in dramatic prose
    "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
    "Eleven", "Twelve", "Hundred", "Thousand", "Million", "Billion", "Trillion",
    "Once", "Twice", "Thrice",
    # Pronouns / possessives / determiners
    "You", "Your", "His", "Her", "Its", "Our", "Their", "My", "Me", "Him", "They",
    "We", "He", "She", "It", "Who", "Whom", "Whose", "Which", "What", "That", "This",
    "These", "Those", "There", "Here",
    # Conjunctions / prepositions / articles
    "And", "But", "Or", "Nor", "Yet", "So", "For", "The", "An",
    "With", "From", "Into", "Over", "Under", "After", "Before", "During", "Between",
    "Through", "About", "Against", "Along", "Among", "Around", "Behind", "Below",
    "Beside", "Beyond", "Near", "Since", "Until", "Upon", "Within", "Without",
    # Auxiliary / common verbs used as sentence starters
    "Was", "Were", "Has", "Have", "Had", "Been", "Being", "Does", "Did", "Could",
    "Would", "Should", "Might", "Must", "Shall", "Will", "Can", "May", "Let",
    "Got", "Get", "Went", "Said", "Told", "Felt", "Knew", "Saw", "Took", "Came",
    # Common adverbs / discourse markers used capitalised in prose
    "When", "Where", "While", "Then", "Than", "Even", "Just", "Only", "Still",
    "Also", "Again", "Always", "Never", "Every", "Each", "Both", "Either",
    "How", "Why", "More", "Much", "Many", "Some", "Most", "Such", "Same",
    "Not", "No", "Yes", "Now", "Soon", "Already", "Often", "Far", "Away",
    # Generic creature / race / role nouns used as chapter labels in fantay prose
    "Vampire", "Vampires", "Werewolf", "Werewolves", "Demon", "Demons",
    "Angel", "Angels", "Dragon", "Dragons", "Ghost", "Ghosts", "Undead",
    "Witch", "Witches", "Warlock", "Warlocks", "Hunter", "Hunters",
    "Slayer", "Slayers", "Monster", "Monsters", "Creature", "Creatures",
    "Warrior", "Warriors", "Soldier", "Soldiers", "Knight", "Knights",
    "Guard", "Guards", "Assassin", "Assassins", "Rogue", "Rogues",
    # Titles / ranks (single-word) — only block alone; "Dark Lord" multi-word still passes
    "Lord", "Lady", "Count", "Baron", "Duke", "Duchess",
    "King", "Queen", "Prince", "Princess", "Emperor", "Empress",
    "Master", "Mistress", "Leader", "Commander", "Captain",
    "Elder", "Elders", "Ancient", "Ancients",
    # Generic place nouns that get capitalised in fantasy prose
    "Kingdom", "Kingdoms", "Empire", "Empires", "Realm", "Realms",
    "Spire", "Spires", "Tower", "Towers", "Keep", "Castle", "Fortress",
    "Temple", "Citadel", "Village", "Town", "City",
    # Generic group / collective nouns
    "Fledgling", "Fledglings", "Clan", "Clans", "Tribe", "Tribes",
    "Horde", "Hordes", "Faction", "Factions", "Order", "Orders",
    "Council", "Councils", "Guild", "Guilds",
    # Generic adjectives that start sentences and look like proper names
    "Good", "Evil", "Bad", "Dark", "Light", "True", "False",
    "Crimson", "Scarlet", "Golden", "Silver", "Obsidian", "Ivory",
    "Tactical", "Strategic", "Ancient", "Sacred", "Fallen",
    # Abstract nouns capitalised for dramatic effect
    "Resonance", "Essence", "Balance", "Chaos", "Order", "Fate",
    "Destiny", "Truth", "Honor", "Justice", "Vengeance", "Wrath",
    "Darkness", "Shadows", "Silence", "Void",
}

CHARACTER_TYPE_CHOICES = ["human", "humanoid", "animal", "beast", "spirit", "orb", "object"]

LOC_KEYS = [
    "academy", "school", "classroom", "hallway", "room", "kitchen", "bedroom", "bathroom",
    "house", "apartment", "rooftop", "warehouse", "street", "alley", "forest", "mountain",
    "river", "lake", "hospital", "office", "factory", "lab", "subway", "station",
    "parking lot", "backyard", "yard", "porch", "garage", "campus", "university",
    "cafeteria", "library", "courtyard", "gym", "dorm", "bunker", "underground bunker",
]

GENERIC_PROP_STOP = {
    "door","window","table","chair","floor","wall","grass","sky","clouds","page","book",
    "notes","room","hallway","class","classes","students","student","library","courtyard"
}

IMPORTANCE_CUES = {
    "system": 4, "artifact": 4, "cursed": 4, "curse": 4, "weapon": 3, "blade": 3,
    "key": 3, "seal": 3, "contract": 3, "blood": 2, "badge": 2, "ring": 2,
    "collar": 3, "knife": 3, "sword": 3, "phone": 2, "interface": 4, "core": 3,
    "storage": 3, "inventory": 3, "mana": 3, "aura": 3, "lightning": 3, "fire": 2,
    "shadow": 3, "ice": 2, "wind": 2,
}

PERSPECTIVES = [
    "Establishing Wide", "Medium Scene", "Close Detail", "Over-the-Shoulder", "Low Angle", "High Angle"
]

# Legacy perspective labels are kept for backward compatibility in older project JSON.
# New generation flow uses camera types only in Tab 2.
CAMERA_TYPES = [
    "dramatic medium character shot",
    "over-the-shoulder shot",
    "tight combat close-up",
    "low-angle attack shot",
    "wide chaos shot",
    "extreme impact close-up",
]

SCENE_TYPES = [
    "ACTION",
    "EMOTION",
    "DIALOGUE",
    "MEMORY",
    "AWAKENING",
    "AFTERMATH",
]

SCENE_CAMERA_POOLS: Dict[str, List[str]] = {
    "ACTION": [
        "low-angle attack shot",
        "tight combat close-up",
        "dramatic medium character shot",
        "wide chaos shot",
        "extreme impact close-up",
    ],
    "EMOTION": [
        "dramatic medium character shot",
        "over-the-shoulder shot",
        "tight combat close-up",
    ],
    "DIALOGUE": [
        "over-the-shoulder shot",
        "dramatic medium character shot",
        "tight combat close-up",
    ],
    "MEMORY": [
        "dramatic medium character shot",
        "wide chaos shot",
        "over-the-shoulder shot",
    ],
    "AWAKENING": [
        "low-angle attack shot",
        "dramatic medium character shot",
        "tight combat close-up",
    ],
    "AFTERMATH": [
        "wide chaos shot",
        "dramatic medium character shot",
        "tight combat close-up",
    ],
}

COLOR_POOL = ["electric blue", "violet", "emerald green", "crimson red", "icy cyan", "golden white"]
SHAPE_POOL = [
    "clean angular panels", "thin rectangular panes", "hexagonal holographic tiles",
    "curved translucent windows", "layered rune circles", "stacked glowing cards"
]
HAIR_MALE = [
    "short dark brown messy hair", "short black textured hair", "short ash-brown hair with loose fringe",
    "medium dark brown hair brushed back", "short charcoal hair with tapered sides"
]
HAIR_FEMALE = [
    "long dark brown hair", "long black straight hair", "shoulder-length chestnut hair",
    "long wavy brown hair", "dark auburn hair tied in a low ponytail"
]
EYES = ["brown eyes", "dark brown eyes", "hazel eyes", "amber-brown eyes", "charcoal-gray eyes"]
BUILDS_M = ["lean athletic build", "athletic lean build", "slim but fit build", "broad-shouldered athletic build"]
BUILDS_F = ["athletic slim build", "slim fit build", "lean athletic build", "graceful fit build"]
SKIN_TONES = ["light skin", "warm fair skin", "light warm skin", "pale fair skin", "light beige skin"]
ANCHORS_M = ["small scar on left eyebrow", "thin black wristband", "silver chain necklace", "distinct mole beneath left eye"]
ANCHORS_F = ["thin silver necklace", "small hairpin on right side", "bracelet on left wrist", "beauty mark near lower lip"]
OUTFITS_M = [
    "dark gray long-sleeve shirt, black jogger pants, indoor slippers",
    "faded black hoodie, dark jeans, black sneakers",
    "white school shirt, dark blazer, striped tie, black slacks, school loafers",
    "dark bomber jacket, charcoal t-shirt, dark jeans, worn boots",
]
OUTFITS_F = [
    "dark fitted long-sleeve top, slim jeans, ankle boots",
    "school blazer, white blouse, dark skirt, knee socks, loafers",
    "oversized knit sweater, pleated skirt, tights, ankle boots",
    "black jacket, dark jeans, lace-up boots",
]


@dataclass
class ProjectState:
    project_id: str = ""
    project_name: str = ""
    project_dir: str = ""
    story: str = ""
    beats: List[str] = field(default_factory=list)
    characters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    locations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    items: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    beat_plans: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    image_paths: List[str] = field(default_factory=list)
    images_by_beat: Dict[int, List[str]] = field(default_factory=dict)
    next_image_index: int = 1
    image_prompts: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    total_cost: float = 0.0
    total_images: int = 0
    # All paths from the manifest (including cloud-only ones not yet on disk)
    all_manifest_paths: List[str] = field(default_factory=list)
    # Beat index for each manifest path (key = image_path)
    manifest_beat_index: Dict[str, int] = field(default_factory=dict)
    # Per-character age phases: {char_name: [{label, beat_start, beat_end, appearance_prompt, ref_image_path}]}
    character_age_phases: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    # Story continuation parts: [{name, start_beat, end_beat}] (end_beat=None means "to end")
    story_parts: List[Dict[str, Any]] = field(default_factory=list)
    # Panel mode page scripts: {page_idx: compact multi-panel image prompt}
    page_scripts: Dict[int, str] = field(default_factory=dict)
    # ISO timestamp of last page script regeneration, e.g. "2026-07-14 03:45:22"
    page_scripts_generated_at: str = ""
    # Build mode: "Normal" (beat-per-image) or "Panel" (10-panel page-per-image)
    build_mode: str = "Panel"
    # Omni/style reference image URL (FAL storage). Applied when model supports reference_images.
    style_reference_url: str = ""
    # Optional world context (lore, power system, tone, etc.) supplied by the user at build time.
    world_context: str = ""
    # The raw story text as the user typed it — never overwritten by beat expansion.
    original_story: str = ""
    # Path to the generated cover image (Shorts mode only).
    cover_image_path: str = ""
    # NB2 Edit character cast: {story_char_name: template_id} — locked per-project assignment.
    character_cast: Dict[str, str] = field(default_factory=dict)
    # Seed used for deterministic auto-cast (stored so the cast is reproducible).
    casting_seed: str = ""
    # Library reference URLs already used in this story — prevents reuse across pages.
    used_reference_urls: List[str] = field(default_factory=list)
    # Real-time gap log: [{type, needed, description, page}] from generation calls.
    generation_gaps: List[Dict[str, Any]] = field(default_factory=list)
    # Usage counters for reference balancing:
    #   ref_url_usage:  {url → times sent as a reference} — used to deprioritise heavy-use URLs
    #   char_ref_usage: {char_name → pages they appeared in} — used to give rare chars priority
    ref_url_usage: Dict[str, int] = field(default_factory=dict)
    char_ref_usage: Dict[str, int] = field(default_factory=dict)
    # Manual storyboard assignments: {page_1idx: {"page_setting": template_id, "panels": {panel_n: {"chars": [name,...]}}}}
    manual_page_refs: Dict[int, Any] = field(default_factory=dict)
    # Session cache: {0-indexed page_num → [fal_url, ...]} for last-N panel crops
    # from the previous page.  Not persisted to disk — rebuilt on first use each session.
    prev_page_panel_urls: Dict[int, List[str]] = field(default_factory=dict)



def _sync_int(value, bump: int = 0) -> int:
    try:
        base = int(float(value or 0))
    except Exception:
        base = 0
    return base + bump

def safe_filename(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_\-]+", "_", (s or "").strip())
    return s[:80] if s else "untitled"


def ensure_dirs(base_dir: str) -> Dict[str, str]:
    os.makedirs(base_dir, exist_ok=True)
    paths = {
        "base": base_dir,
        "images": os.path.join(base_dir, "images"),
        "thumbnails": os.path.join(base_dir, "thumbnails"),
        "refs": os.path.join(base_dir, "refs"),
        "refs_chars": os.path.join(base_dir, "refs", "characters"),
        "refs_locs": os.path.join(base_dir, "refs", "locations"),
        "refs_items": os.path.join(base_dir, "refs", "items"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def save_json(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def zip_folder(folder_path: str, zip_path: str) -> str:
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(folder_path):
            for fn in files:
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, folder_path)
                zf.write(full, rel)
    return zip_path


def new_project(name: str = "") -> ProjectState:
    pid = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    base = os.path.join("projects", pid)
    ensure_dirs(base)
    display_name = name.strip() if name and name.strip() else pid
    return ProjectState(project_id=pid, project_name=display_name, project_dir=base)


def load_most_recent_project(fast: bool = False) -> "ProjectState | None":
    """Load the most recently modified project from disk (used on session start).

    Args:
        fast: When True, skip the cloud-sync step.  Use this for session
              auto-load where a 30-second wait is unacceptable.
    """
    projects_root = "projects"
    if not os.path.isdir(projects_root):
        return None
    if not fast:
        # Pull cloud-only project metadata so they show up too
        try:
            local_pids = set(os.listdir(projects_root))
            for cloud_pid in _cloud.list_cloud_project_ids():
                if cloud_pid not in local_pids:
                    _cloud.restore_project_meta(os.path.join(projects_root, cloud_pid))
        except Exception:
            pass
    best_dir = None
    best_mtime = 0.0
    for pid in os.listdir(projects_root):
        pdir = os.path.join(projects_root, pid)
        proj_json = os.path.join(pdir, "project.json")
        if os.path.isfile(proj_json):
            try:
                # Score by the most recent activity in the project: either the
                # project.json itself OR any image file in images/.  This ensures
                # a project where many images were generated long ago but no new
                # story was built still beats a brand-new empty project whose
                # project.json was just written today.
                mt = os.path.getmtime(proj_json)
                images_dir = os.path.join(pdir, "images")
                if os.path.isdir(images_dir):
                    for img in os.listdir(images_dir):
                        try:
                            imt = os.path.getmtime(os.path.join(images_dir, img))
                            if imt > mt:
                                mt = imt
                        except Exception:
                            pass
                if mt > best_mtime:
                    best_mtime = mt
                    best_dir = pdir
            except Exception:
                pass
    if not best_dir:
        return None
    try:
        return _load_project_from_disk(best_dir)
    except Exception:
        return None


def _list_all_projects() -> List[Dict[str, Any]]:
    projects_root = "projects"
    os.makedirs(projects_root, exist_ok=True)
    # Pull any cloud-only projects down (just project.json + manifest — not images yet)
    try:
        local_pids = set(os.listdir(projects_root))
        for cloud_pid in _cloud.list_cloud_project_ids():
            if cloud_pid not in local_pids:
                _cloud.restore_project_meta(os.path.join(projects_root, cloud_pid))
    except Exception:
        pass
    results = []
    for pid in os.listdir(projects_root):
        pdir = os.path.join(projects_root, pid)
        proj_json = os.path.join(pdir, "project.json")
        if not os.path.isfile(proj_json):
            continue
        try:
            with open(proj_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Prefer total_images from project.json (always up-to-date, survives container restart)
            img_count = int(data.get("total_images") or 0)
            if img_count == 0:
                # Fallback: count manifest entries
                manifest_path = os.path.join(pdir, "manifest.jsonl")
                if os.path.isfile(manifest_path):
                    with open(manifest_path, "r", encoding="utf-8") as _mf:
                        img_count = sum(1 for _l in _mf if _l.strip())
            mtime = os.path.getmtime(proj_json)
            modified = time.strftime("%Y-%m-%d %H:%M", time.localtime(mtime))
            results.append({
                "project_id": data.get("project_id", pid),
                "project_name": data.get("project_name") or data.get("project_id", pid),
                "beats": len(data.get("beats", [])),
                "images": img_count,
                "modified": modified,
                "project_dir": pdir,
                "_mtime": mtime,
            })
        except Exception:
            continue
    # Sort by actual last-modified time so recently-worked-on projects float to the top
    results.sort(key=lambda r: r["_mtime"], reverse=True)
    for r in results:
        r.pop("_mtime", None)
    return results


def _load_project_from_disk(project_dir: str) -> "ProjectState":
    proj_json = os.path.join(project_dir, "project.json")
    with open(proj_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    st = ProjectState(
        project_id=data.get("project_id", ""),
        project_name=data.get("project_name", "") or data.get("project_id", ""),
        project_dir=project_dir,
        story=data.get("story", ""),
        beats=data.get("beats", []),
        characters=data.get("characters", {}),
        locations=data.get("locations", {}),
        items=data.get("items", {}),
        beat_plans={int(k): v for k, v in (data.get("beat_plans") or {}).items()},
        next_image_index=data.get("next_image_index", 1),
        image_prompts={int(k): v for k, v in (data.get("image_prompts") or {}).items()},
        total_cost=float(data.get("total_cost", 0.0)),
        total_images=int(data.get("total_images", 0)),
    )
    manifest_path = os.path.join(project_dir, "manifest.jsonl")
    # Collect all image paths listed in the manifest (whether or not they exist locally yet)
    all_manifest_paths = []
    manifest_beat_index: Dict[str, int] = {}
    if os.path.isfile(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as mf:
            for line in mf:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    img_path = entry.get("image_path", "")
                    if not img_path:
                        continue
                    # Normalise to absolute path so Gradio 6.x can serve it
                    # (allowed_paths check uses absolute path comparison).
                    img_path = os.path.abspath(img_path)
                    beat_idx = int(entry.get("beat_index", 0) or 0)
                    all_manifest_paths.append(img_path)
                    manifest_beat_index[img_path] = beat_idx
                    if os.path.isfile(img_path) and img_path not in st.image_paths:
                        st.image_paths.append(img_path)
                        if beat_idx > 0:
                            st.images_by_beat.setdefault(beat_idx, [])
                            if img_path not in st.images_by_beat[beat_idx]:
                                st.images_by_beat[beat_idx].append(img_path)
                except Exception:
                    pass
    # Store ALL manifest paths so the poll timer can pick up cloud-restored images
    st.all_manifest_paths = all_manifest_paths
    st.manifest_beat_index = manifest_beat_index
    st.character_age_phases = data.get("character_age_phases", {})
    st.story_parts = data.get("story_parts", [])
    st.page_scripts = {int(k): v for k, v in (data.get("page_scripts") or {}).items()}
    st.page_scripts_generated_at = data.get("page_scripts_generated_at", "")
    st.build_mode = data.get("build_mode", "Panel")
    st.style_reference_url = data.get("style_reference_url", "")
    st.world_context = data.get("world_context", "")
    st.cover_image_path = data.get("cover_image_path", "")
    st.original_story = data.get("original_story", "")
    st.character_cast = data.get("character_cast", {})
    st.casting_seed = data.get("casting_seed", "")
    st.manual_page_refs = {int(k): v for k, v in (data.get("manual_page_refs") or {}).items()}
    # Restore story text: prefer the saved original_story (never touched by beat expansion),
    # then the saved story field, and only fall back to reconstructing from beats when
    # neither is present (very old projects with no saved story text at all).
    saved_story = (st.original_story or data.get("story") or "").strip()
    if saved_story:
        st.story = saved_story
    elif st.beats:
        st.story = "\n\n".join(st.beats)
    # Restore any missing images from cloud in the background
    if all_manifest_paths:
        _cloud.restore_missing_images_bg(project_dir, all_manifest_paths)
    return st


def _delete_project(project_dir: str) -> str:
    import shutil
    if not project_dir or not os.path.isdir(project_dir):
        return "❌ Project folder not found."
    try:
        shutil.rmtree(project_dir)
        return f"✅ Deleted project at {project_dir}"
    except Exception as e:
        return f"❌ Delete failed: {e}"


def clean_for_prompt(text: str) -> str:
    t = (text or "").strip()
    t = t.replace("•", " ")
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()


def strip_shot_labels(text: str) -> str:
    """Remove bracketed shot-type annotations written by Claude for framing guidance
    (e.g. [ECU], [MS], [DUTCH], [OTS], [LOW]) from prompts before they go to the
    image model — the model renders them as literal on-screen text otherwise.
    Only strips all-uppercase bracket groups (2–12 chars each); lowercase tags like
    [Child age 8] or phase markers are untouched."""
    # Matches things like [ECU], [MS], [DUTCH ANGLE], [OTS], [HIGH ANGLE] etc.
    return re.sub(r"\[\s*[A-Z]{2,12}(?:\s[A-Z]{2,12})?\s*\]", "", text)


def _stable_int(seed_text: str) -> int:
    h = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    return int(h[:12], 16)


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _call_claude_json(system: str, user_payload: Dict[str, Any], max_tokens: int = 4000, model: Optional[str] = None, _log=None) -> Optional[Dict[str, Any]]:
    if _rp.is_deepseek_mode():
        result, status = _rp.call_json(system, user_payload, max_tokens=max_tokens, temperature=0)
        if result is None and _log:
            _log(f"DeepSeek V4.1 Flash: {status}")
        return result

    api_key = (os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY") or "").strip()
    if not api_key:
        if _log:
            _log("Claude: no API key found")
        return None
    try:
        import requests
    except Exception:
        return None
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model or CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "temperature": 0,
        "system": system,
        "messages": [{"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}],
    }
    import time as _claude_time
    for _claude_attempt in range(3):
        try:
            r = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=(15, 120))
        except Exception as exc:
            if _log:
                _log(f"Claude: request failed — {type(exc).__name__}: {exc}")
            return None
        if r.status_code == 429:
            if _claude_attempt < 2:
                _wait = int(r.headers.get("Retry-After", 0)) or (20 * (_claude_attempt + 1))
                if _log:
                    _log(f"Claude: rate-limited (429) — waiting {_wait}s (attempt {_claude_attempt + 1}/3)")
                _claude_time.sleep(_wait)
                continue
            if _log:
                _log("Claude: rate-limited after 3 attempts — using fallback")
            return None
        break
    try:
        r.raise_for_status()
        data = r.json()
        stop_reason = data.get("stop_reason", "")
        text_parts = []
        for part in data.get("content", []):
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        raw_text = "\n".join(text_parts)
        if stop_reason == "max_tokens":
            if _log:
                _log(f"Claude: response truncated (max_tokens hit) — JSON may be incomplete, attempting parse anyway")
        result = _extract_json_object(raw_text)
        if result is None and _log:
            _log(f"Claude: JSON parse failed. stop_reason={stop_reason!r}. First 200 chars: {raw_text[:200]!r}")
        return result
    except Exception as exc:
        if _log:
            _log(f"Claude: request failed — {type(exc).__name__}: {exc}")
        return None


VISUAL_SPLIT_MARKERS = [r"\bwhile\b", r"\bas\b", r"\bwhereas\b", r"\bbut\b", r"\bhowever\b", r"\bthen\b", r"\bsuddenly\b", r"\bafter\b"]


def split_sentences(story: str) -> List[str]:
    story = (story or "").strip()
    if not story:
        return []
    story = story.replace("\r\n", "\n")
    blocks = [b.strip() for b in story.split("\n\n") if b.strip()]
    sents: List[str] = []
    for b in blocks:
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])", b.strip())
        sents.extend([p.strip() for p in parts if p.strip()])
    return sents


def split_visual_beats(sentence: str) -> List[str]:
    s = sentence.strip()
    if not s:
        return []
    low = s.lower()
    if low.startswith("while ") and "," in s:
        a, b = s.split(",", 1)
        return [a.strip() + ".", b.strip()]
    pattern = "(" + "|".join(VISUAL_SPLIT_MARKERS) + ")"
    tokens = re.split(pattern, s, flags=re.IGNORECASE)
    if len(tokens) <= 1:
        return [s]
    segs: List[str] = []
    current = tokens[0].strip()
    i = 1
    while i < len(tokens):
        marker = (tokens[i] or "").strip()
        chunk = (tokens[i + 1] if i + 1 < len(tokens) else "").strip()
        if current:
            segs.append(current)
        current = (marker + " " + chunk).strip()
        i += 2
    if current:
        segs.append(current)
    out = []
    for seg in segs:
        seg = seg.strip()
        if not seg:
            continue
        if seg[-1] not in ".!?":
            seg += "."
        out.append(seg)
    return out or [s]


def merge_micro_beats(beats: List[str], min_words: int = 7) -> List[str]:
    merged: List[str] = []
    for b in beats:
        if merged and len(b.split()) < min_words:
            merged[-1] = (merged[-1].rstrip() + " " + b.lstrip()).strip()
        else:
            merged.append(b.strip())
    return merged


def split_beats(story: str) -> List[str]:
    sents = split_sentences(story)
    beats: List[str] = []
    for sent in sents:
        beats.extend(split_visual_beats(sent))
    beats = merge_micro_beats([re.sub(r"\s+", " ", b).strip() for b in beats if b.strip()])
    return beats


def _validate_shorts_beats_with_claude(story: str, expanded_beats: List[str]) -> List[str]:
    """
    Second-pass story-alignment validator. Runs AFTER beat expansion, BEFORE page scripts.

    Maps every expanded beat proportionally back to the story sentence it covers,
    then asks Claude to review each (sentence → visual) pair and rewrite any beat
    whose visual description does not faithfully show what its sentence actually says.

    This is intentionally story-agnostic — no hard-coded rules about headaches,
    coin dimensions, or any specific situation. The prompt just asks one question:
    'Does this visual match the sentence? If not, fix it so it does.'
    """
    api_key = (os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY") or "").strip()
    if not _rp.has_text_provider("anthropic") or not expanded_beats:
        return expanded_beats

    import re as _re
    # Split story into sentences for mapping
    sents = [s.strip() for s in _re.split(r'(?<=[.!?…])\s+', story.strip()) if s.strip()]
    if not sents:
        return expanded_beats

    n_beats = len(expanded_beats)
    n_sents = len(sents)

    # Build proportional beat → sentence mapping (same formula as _orig_excerpt in director.py)
    def _sent_for(beat_idx: int) -> str:
        frac   = beat_idx / max(n_beats - 1, 1)
        sent_j = min(int(round(frac * (n_sents - 1))), n_sents - 1)
        return sents[sent_j]

    # Build the review payload: one line per beat showing [sentence] → [visual]
    pairs_text = "\n".join(
        f"Beat {i+1} | Sentence: {_sent_for(i)} | Visual: {b}"
        for i, b in enumerate(expanded_beats)
    )

    system = (
        "You are a story-visual alignment editor for a Korean manhwa webtoon. "
        "Your only job is to check that each beat's visual description faithfully shows "
        "what its tagged story sentence actually describes — and fix it if it doesn't.\n\n"
        "For each beat you receive:\n"
        "  • If the visual ALREADY faithfully shows the story sentence: output it UNCHANGED.\n"
        "  • If the visual has DRIFTED — it shows the wrong location, wrong character, wrong action, "
        "    a metaphor instead of the literal event, content from a neighbouring sentence, or "
        "    continues a previous scene when the story has moved on — REWRITE it so it shows "
        "    exactly what the sentence says.\n\n"
        "Rewrite rules:\n"
        "  • Keep the same camera prefix (ECU, CU, MS, LS, POV, etc.) unless it's wrong for the content.\n"
        "  • The corrected visual must show the LOCATION named in the sentence, the CHARACTER doing the "
        "    action in the sentence, and the SPECIFIC OBJECTS or EVENTS named in the sentence.\n"
        "  • Do not invent story events not present in the sentence.\n"
        "  • Keep each beat 1-2 sentences max.\n\n"
        "Output ONLY a raw JSON array of strings, same length as input, same order. "
        "No preamble, no explanation, no markdown fences."
    )

    user = (
        f"Story (ground truth):\n{story.strip()}\n\n"
        f"Beats to review ({n_beats} total):\n{pairs_text}\n\n"
        "Return a JSON array of exactly {n_beats} strings. "
        "Keep correct beats unchanged. Rewrite only drifted beats."
    ).replace("{n_beats}", str(n_beats))

    try:
        if _rp.is_deepseek_mode():
            raw_text, _status = _rp.call_text(system, user, max_tokens=6000, temperature=0)
            if not raw_text:
                return expanded_beats
        else:
            import requests as _rq
            r = _rq.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": CLAUDE_MODEL, "max_tokens": 6000, "temperature": 0, "system": system,
                      "messages": [{"role": "user", "content": user}]},
                timeout=(15, 150),
            )
            r.raise_for_status()
            parts = [p.get("text", "") for p in r.json().get("content", []) if p.get("type") == "text"]
            raw_text = "".join(parts).strip()
        raw_text = re.sub(r"^```[a-z]*\n?", "", raw_text, flags=re.IGNORECASE)
        raw_text = re.sub(r"\n?```$", "", raw_text)
        validated = json.loads(raw_text.strip())
        if isinstance(validated, list) and len(validated) == n_beats:
            return [str(v) for v in validated]
        # If length mismatch, return original — expansion is still better than nothing
        return expanded_beats
    except Exception:
        return expanded_beats


def _expand_shorts_beats_with_claude(story: str, raw_beats: List[str], target_beats: int = 50) -> List[str]:
    """Expand sparse story beats into visual micro-beats for Shorts mode.
    Each beat = one image = 1-3 seconds on screen. Targets ~50 beats for a ~1-minute short."""
    api_key = (os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY") or "").strip()
    if not _rp.has_text_provider("anthropic"):
        return raw_beats

    beats_text = "\n".join(f"{i+1}. {b}" for i, b in enumerate(raw_beats))

    system = (
        "You are a visual story director for a Korean manhwa webtoon short video. "
        "A Short is a 1-2 minute video where each image stays on screen for 1-3 seconds. "
        f"Expand the story into exactly {target_beats} visual micro-beats — one beat = one image.\n\n"

        "CULTURAL SETTING — NON-NEGOTIABLE:\n"
        "This story is set in a Korean or Chinese cultural world (default: modern Seoul or ancient/modern China), "
        "NOT in America, Europe, or a generic Western city. "
        "Streets look like Seoul alleys, Hong Kong back-lanes, or Chinese urban districts — "
        "hangul/hanja signage, pojangmacha stalls, traditional tile-roofed buildings mixed with neon, "
        "PC bang storefronts, convenience stores with Korean/Chinese branding. "
        "NO graffiti-covered New York walls. NO American English signage. NO baseball caps unless story states them. "
        "All human characters are Korean or Chinese — East Asian facial features, black or very dark hair (default), "
        "East Asian skin tone. ONLY override if the story explicitly describes a non-Asian character.\n\n"

        "RULES:\n"
        "1. Every beat describes ONE specific image, ONE composition, ONE frozen moment in time.\n"
        "2. Start every beat with a camera direction: ECU (extreme close-up), CU (close-up), "
        "MS (medium shot), LS (long shot), POV, low-angle, overhead, dutch-tilt, tracking.\n"
        "3. Vary camera distance across consecutive beats — never two identical framings in a row.\n"
        "4. Show emotion and context through objects, environment, body language — NO dialogue, "
        "NO narration text, NO captions in the beat description.\n"
        "5. STORY FIDELITY — SOURCE SENTENCE LOCK — CRITICAL:\n"
        "   Each beat expands ONLY the sentence(s) it is assigned to. "
        "   It must use ONLY visual elements that actually appear in, or are directly implied by, that sentence. "
        "   NEVER borrow a prop, object, or person from a later or earlier sentence to fill a beat.\n"
        "   SETTING-INTRODUCTION RULE: A short sentence that names a PLACE or GROUP ('This sect.', 'This city.', "
        "   'The village.') is introducing a setting, not an action. Its beats MUST show that setting — "
        "   wide establishing shots of the location, its architecture, its atmosphere. "
        "   NEVER fill a setting-introduction beat with objects or actions from later sentences (food, people, props). "
        "   Example: 'This sect.' → show the sect gates, courtyard, cold stone walls. NOT a bowl of food.\n"
        "   SHORT FRAGMENT GROUPING: When consecutive sentences are very short (fewer than 5 words each) AND "
        "   they are all about the SAME subject/situation (e.g. 'Three years. One bean. Daily.' all describe "
        "   Chen Ping's daily reality), treat them as ONE story moment — beats may draw from any sentence in the cluster. "
        "   Do NOT group a setting-introduction sentence ('This sect.') with an action sentence that follows — "
        "   they are different moments and must stay separate.\n"
        "6. PHYSICAL EVENT FIDELITY — CRITICAL:\n"
        "   If a sentence describes a concrete physical action — someone being beaten/struck/hit, "
        "   someone collapsing or falling, someone being left abandoned, someone dying — "
        "   the beat for that sentence MUST depict that action directly (bodies, people, the act itself). "
        "   Do NOT substitute an architectural shot, an environmental wide, a food close-up, or a metaphor. "
        "   Show the event. Specific examples:\n"
        "   • 'got beaten' → show the attacker, the weapon, the victim being struck. "
        "     A bowl of food is NOT a substitute for a beating scene.\n"
        "   • 'collapsed from exhaustion got left in the dirt' → show a person lying face-down or slumped "
        "     on the ground in an outdoor dirt or stone surface, alone, ignored, no one helping. "
        "     A person kneeling upright at a bowl is NOT collapsed. A person sitting at a table is NOT left in the dirt.\n"
        "   • 'died' or 'killed' → show the body or the killing moment, not an empty room.\n"
        "7. PROP REPETITION LIMIT — ABSOLUTE RULE: The same specific prop (bowl, bean, sword, phone, etc.) "
        "   must NOT appear as the SOLE SUBJECT in more than 2 consecutive beats. "
        "   This rule overrides everything — it overrides Beat Fidelity, it overrides short fragment grouping, "
        "   it overrides the 'show the item in close-up' rule. After 1–2 beats showing the prop, "
        "   the NEXT beat MUST shift to the CHARACTER: their face, their body, their reaction, their context.\n"
        "   FAILURE EXAMPLE — never do this:\n"
        "     Beat 7: ECU of dried bean in cracked bowl.\n"
        "     Beat 8: ECU of dried bean in cracked bowl, different angle.\n"
        "     Beat 9: ECU of dried bean in cracked bowl, focus on texture.\n"
        "     Beat 10: ECU of dried bean in cracked bowl, bean off-center.\n"
        "     Beat 11: ECU of dried bean in cracked bowl, bowl held at angle.\n"
        "     Beat 12: ECU of dried bean in cracked bowl, bean appears smaller.\n"
        "   CORRECT version of the same beats:\n"
        "     Beat 7: ECU of single dried bean sitting alone in a cracked ceramic bowl.\n"
        "     Beat 8: CU of Chen Ping's gaunt hollow face staring down, eyes dark with despair.\n"
        "     Beat 9: MS of Chen Ping seated alone in the empty courtyard, dwarfed by cold stone walls.\n"
        "     Beat 10: CU of his trembling hand reaching for the bean, knuckles scarred.\n"
        "     Beat 11: ECU of his sunken eyes, dead and hollow, staring at nothing.\n"
        "     Beat 12: ELS of Chen Ping — a tiny figure alone in the vast sect compound, stone silence around him.\n"
        "   RULE: maximum 2 beats on any single prop per page. Then switch to character or environment.\n"
        "8. CHARACTER PRESENCE MINIMUM — ABSOLUTE RULE: At least HALF the beats on any page must show "
        "   a story character (face, body, hands, silhouette — any part of a person counts). "
        "   A page of 6 beats must have at least 3 beats with a character visible. "
        "   A page of 10 beats must have at least 5 beats with a character visible. "
        "   Pure object or environment beats are allowed, but they must be balanced with character beats.\n"
        "9. HOOK QUALITY — beats 1–5 must be visually gripping, but the drama must come FROM "
        "   the actual story content of those sentences — not from borrowing imagery from later sentences. "
        "   Find the inherent drama in what the sentence actually says.\n"
        "9. CHRONOLOGICAL LOCK — CRITICAL: Beats must advance the story in STRICT forward order.\n"
        "   • Never show an action in beat N that belongs to a later sentence — that is borrowing from the future.\n"
        "   • Never show an action in beat N that was ALREADY assigned to an earlier beat — that is repeating the past.\n"
        "   • 'Then', 'immediately', 'suddenly', 'next', 'after' in a source sentence signal a hard scene cut — \n"
        "     the new event must appear NOW, not deferred to a later beat.\n"
        "   • If the previous beat showed character A in location X, and the next source sentence \n"
        "     is about character B doing action Y, CUT to character B doing action Y — do not linger on A.\n"
        "10. Each beat: 1-2 sentences maximum.\n"
        "11. Do NOT invent story events not implied by the source — only expand visually.\n"
        "12. BEAT FIDELITY — CRITICAL: Every concrete noun and visual element named in the source text MUST "
        "appear in at least one beat assigned to that sentence. Never silently drop any named visual element.\n"
        "   • If the source says 'trucks AND a trash can', BOTH must appear in their beats.\n"
        "   • If the source names a specific food item — 'one dried bean', 'a single scrap', 'half a bowl of rice' — "
        "     that EXACT item must be clearly visible (as a close-up or focal object) in at least one beat for "
        "     that sentence. Do NOT substitute a different food. Do NOT show a bowl without the named item in it. "
        "     Do NOT replace the food scene with a character scene or action scene.\n"
        "   • If the source says a character 'gets exactly one [item] per day', a beat must show that item "
        "     in extreme close-up so its scarcity and smallness are unmistakable.\n\n"
        "Output ONLY a raw JSON array of strings. No preamble, no explanation, no markdown code fences.\n"
        "Example: [\"ECU: crumpled eviction notice on cracked concrete floor, single overhead bulb casting "
        "a harsh cone of amber light.\", \"CU: young man's hand picking up the notice, knuckles scraped "
        "and calloused, ink-smudged fingers.\"]"
    )

    user = (
        f"Original story:\n{story.strip()}\n\n"
        f"Raw beats parsed ({len(raw_beats)} beats):\n{beats_text}\n\n"
        f"Expand into exactly {target_beats} visual micro-beats. Return ONLY a JSON array of strings."
    )

    try:
        if _rp.is_deepseek_mode():
            raw_text, _status = _rp.call_text(system, user, max_tokens=6000, temperature=0)
            if not raw_text:
                return raw_beats
        else:
            import requests as _rq
            r = _rq.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": CLAUDE_MODEL, "max_tokens": 6000, "temperature": 0, "system": system,
                      "messages": [{"role": "user", "content": user}]},
                timeout=(15, 120),
            )
            r.raise_for_status()
            parts = [p.get("text", "") for p in r.json().get("content", []) if p.get("type") == "text"]
            raw_text = "".join(parts).strip()
        # Strip markdown fences if model added them anyway
        raw_text = re.sub(r"^```[a-z]*\n?", "", raw_text, flags=re.IGNORECASE)
        raw_text = re.sub(r"\n?```$", "", raw_text)
        expanded = json.loads(raw_text.strip())
        if isinstance(expanded, list) and len(expanded) >= 10:
            return [str(b).strip() for b in expanded if str(b).strip()]
    except Exception:
        pass
    return raw_beats


def detect_gender_from_story(name: str, story: str) -> Optional[str]:
    s = (story or "")
    low = s.lower()
    idxs = [m.start() for m in re.finditer(re.escape(name), s)]
    for idx in idxs[:10]:
        start = max(0, idx - 140)
        end = min(len(s), idx + 140)
        window = low[start:end]
        if " she " in window or " her " in window:
            return "female"
        if " he " in window or " his " in window:
            return "male"
    return None


def detect_first_person_gender(story: str) -> Optional[str]:
    s = (story or "")
    low = s.lower()
    idxs = [m.start() for m in re.finditer(r"\bI\b", s)]
    for idx in idxs[:8]:
        start = max(0, idx - 120)
        end = min(len(s), idx + 120)
        w = low[start:end]
        if " she " in w or " her " in w:
            return "female"
        if " he " in w or " his " in w:
            return "male"
    return None


def extract_characters(story: str, locations: List[str]) -> List[str]:
    text = story or ""
    cands = re.findall(r"\b[A-Z][a-z]{2,}(?:'[a-z]+)?\b", text)
    stop = {"The","A","An","And","But","Then","Now","I","We","He","She","They","It"}
    freq: Dict[str,int] = {}
    for c in cands:
        if c in stop:
            continue
        base = c.split("'")[0]
        freq[base] = freq.get(base, 0) + 1
    # Require at least 3 occurrences to reduce noise from incidental capitalisation
    characters_raw = [k for k, v in sorted(freq.items(), key=lambda x: -x[1]) if v >= 3]
    loc_set = set(locations)
    loc_key_set = set(LOC_KEYS)
    chars: List[str] = []
    for c in characters_raw:
        if c in PLACE_WORD_BLOCKLIST or c in GENERIC_NOUN_BLOCKLIST:
            continue
        if c in loc_set or c.lower() in loc_key_set:
            continue
        chars.append(c)
    # Hard cap — if regex still returns too many, keep only the most frequent 20
    return chars[:20]


def extract_locations(story: str) -> List[str]:
    low = (story or "").lower()
    return sorted({kw.title() for kw in LOC_KEYS if kw in low})


def _default_sub_location_specs(main_location: str) -> Dict[str, str]:
    low = (main_location or "").lower()
    if any(k in low for k in ["bedroom", "room"]):
        return {
            "window view": "viewpoint near the bedroom window with outside light spill, curtain edges, and partial exterior depth cues",
            "doorway": "viewpoint facing the bedroom doorway with frame trim, hallway spill light, and threshold floor detail",
            "closet": "viewpoint near the closet doors with wardrobe texture, handles, and nearby wall anchors",
            "bedside": "viewpoint beside the bed with headboard, side table, and bedding folds as stable anchors",
        }
    if any(k in low for k in ["classroom"]):
        return {
            "front board": "viewpoint facing the board and teacher zone with front desks and clear perspective lines",
            "window row": "viewpoint along the classroom windows with desk row alignment and daylight direction cues",
            "doorway": "viewpoint near the classroom door with corridor-facing threshold and side wall details",
            "back row": "viewpoint from the rear desks toward the front with long depth lines",
        }
    if any(k in low for k in ["apartment", "house", "dorm"]):
        return {
            "entry doorway": "viewpoint from the unit doorway with shoe rack or entry wall anchors",
            "living area": "viewpoint centered on seating zone and table anchors",
            "kitchen corner": "viewpoint on counters, sink, and cabinet geometry",
            "hallway": "viewpoint down interior hallway with door spacing anchors",
        }
    if any(k in low for k in ["school", "academy", "campus", "university"]):
        return {
            "main hallway": "viewpoint along school corridor with locker/wall rhythm and ceiling lights",
            "stair landing": "viewpoint at stair transition with railings and floor split cues",
            "rooftop access door": "viewpoint around rooftop doorway and safety rail",
            "courtyard edge": "viewpoint at courtyard boundary with paths and fence anchors",
        }
    if any(k in low for k in ["warehouse", "factory", "lab"]):
        return {
            "entrance lane": "viewpoint from entry corridor with rack lines and floor markings",
            "work floor": "viewpoint at active floor zone with machinery or benches as anchors",
            "storage aisle": "viewpoint down shelf aisle with repeating geometry and vanishing lines",
        }
    if any(k in low for k in ["hospital"]):
        return {
            "corridor": "viewpoint in hospital corridor with room doors and overhead lighting rhythm",
            "patient room doorway": "viewpoint at room entry with bed placement and monitor wall",
            "stairwell": "viewpoint in stairwell landing with railings and concrete textures",
        }
    if any(k in low for k in ["street", "alley"]):
        return {
            "intersection": "viewpoint toward crossing lanes with curb lines and signage anchors",
            "side alley": "viewpoint into narrow alley depth with wall texture continuity",
            "storefront edge": "viewpoint near storefront facade with door/window anchors",
        }
    return {
        "main area": "primary viewpoint with stable architectural anchors",
        "entry": "viewpoint from entry threshold",
        "corner": "angled viewpoint from side corner to preserve depth continuity",
    }


def _default_sub_locations(main_location: str) -> List[str]:
    return list(_default_sub_location_specs(main_location).keys())


def extract_important_props(story: str, top_n: int = 12) -> List[str]:
    low = (story or "").lower()
    scores: Dict[str, float] = {}
    for m in re.finditer(r"\b(?:a|an|the|my|his|her|their)\s+([a-z][a-z\- ]{1,24})", low):
        phrase = m.group(1).strip()
        phrase = re.sub(r"\b(of|with|that|which|who)\b.*$", "", phrase).strip()
        phrase = phrase.split(",")[0].strip()
        if not phrase or len(phrase.split()) > 3:
            continue
        head = phrase.split()[-1]
        if head in GENERIC_PROP_STOP:
            continue
        scores[phrase] = scores.get(phrase, 0.0) + 1.0
        for cue, bonus in IMPORTANCE_CUES.items():
            if cue in phrase:
                scores[phrase] += bonus
    _all_blocked_lower = {w.lower() for w in PLACE_WORD_BLOCKLIST} | {w.lower() for w in GENERIC_NOUN_BLOCKLIST}
    titlecase = re.findall(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b", story or "")
    for phrase in titlecase:
        low_phrase = phrase.lower().strip()
        # Skip single-word phrases that are in any blocklist
        if len(low_phrase.split()) == 1 and low_phrase in _all_blocked_lower:
            continue
        if low_phrase in {x.lower() for x in PLACE_WORD_BLOCKLIST}:
            continue
        if any(loc in low_phrase for loc in LOC_KEYS):
            continue
        if len(low_phrase.split()) <= 3:
            scores[low_phrase] = scores.get(low_phrase, 0.0) + 1.5
    forced = []
    for w in ["knife", "ring", "system", "interface", "storage space", "inventory", "sword", "badge"]:
        if w in low:
            scores[w] = max(scores.get(w, 0.0), 4.5)
            forced.append(w)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    top = [t[0] for t in ranked[:top_n]]
    return [p.title() for p in top]


def extract_power_items(story: str) -> List[str]:
    low = (story or "").lower()
    found = []
    rules = [
        ("system", "System Interface"),
        ("inventory", "Inventory Interface"),
        ("storage space", "Infinite Storage Space Interface"),
        ("lightning", "Lightning Aura"),
        ("electricity", "Electric Aura"),
        ("shadow", "Shadow Aura"),
        ("fire", "Fire Aura"),
        ("ice", "Ice Aura"),
        ("wind", "Wind Pressure Aura"),
        ("mana circle", "Mana Circle"),
        ("summoned sword", "Summoned Sword Manifestation"),
    ]
    for key, label in rules:
        if key in low and label not in found:
            found.append(label)
    return found


def build_location_shots(location_name: str) -> Dict[str, str]:
    """
    Legacy helper retained so older saved projects can still load.
    New system uses scene camera types in Tab 2 and sub-location refs in Tab 1.
    """
    base = location_name.strip().lower()
    style = "illustrated anime manhwa environment shot, drawn 2d background, clean sharp lineart, cel shading, cinematic anime lighting, vibrant colors, not photorealistic, not live action"
    return {
        "Establishing Wide": f"{style}, establishing wide shot of the {base}",
        "Medium Scene": f"{style}, medium scene inside the {base}",
        "Close Detail": f"{style}, close detail inside the {base}",
        "Over-the-Shoulder": f"{style}, over-the-shoulder inside the {base}",
        "Low Angle": f"{style}, low angle view inside the {base}",
        "High Angle": f"{style}, high angle view inside the {base}",
    }


def _pick(seq: List[str], seed: int, offset: int = 0) -> str:
    return seq[(seed + offset) % len(seq)]


def _seeded_gender(project_id: str, name: str, story: str, force_gender: Optional[str] = None) -> str:
    if force_gender:
        return force_gender
    detected = detect_gender_from_story(name, story)
    if detected:
        return detected
    return "male"


def _infer_character_type(name: str, story: str, explicit_type: Optional[str] = None) -> str:
    if explicit_type:
        ct = clean_for_prompt(str(explicit_type)).lower()
        if ct in CHARACTER_TYPE_CHOICES:
            return ct
    low_name = (name or "").lower()
    low_story = (story or "").lower()

    keyword_groups = [
        ("orb", ["orb", "sphere", "core", "floating light", "energy ball"]),
        ("spirit", ["spirit", "ghost", "phantom", "specter", "wraith", "apparition", "soul"]),
        ("object", ["sword", "shield", "ring", "amulet", "book", "statue", "doll", "puppet", "artifact", "relic", "object"]),
        ("animal", ["dog", "cat", "wolf", "fox", "rabbit", "bird", "horse", "hawk", "rat", "bear", "tiger", "lion", "deer", "snake", "serpent", "viper", "cobra", "python", "lizard", "crocodile", "turtle", "fish", "spider", "insect", "beetle"]),
        ("beast", ["dragon", "monster", "beast", "demon dog", "creature", "chimera", "wyvern", "hydra", "gryphon"]),
        ("humanoid", ["elf", "dwarf", "orc", "goblin", "android", "robot", "humanoid"]),
    ]

    # Check name itself for embedded type hints (e.g. "Dragon King", "Snake Elder")
    if any(k in low_name for grp_keys in [g[1] for g in keyword_groups] for k in grp_keys):
        for ctype, keys in keyword_groups:
            if any(k in low_name for k in keys):
                return ctype

    # ── Transformation / reincarnation detection ────────────────────────────
    # Scan the FULL story for "reborn as X", "woke up as X", "became a X",
    # "is now a X", "body of a X" patterns. This handles isekai/reincarnation
    # where a human protagonist becomes an animal/beast — their human name
    # never contains an animal keyword, but the story text does.
    _TRANSFORM_PATTERNS = [
        r"(?:reborn|reincarnated|transmigrated)\s+as\s+a[n]?\s+(\w+)",
        r"(?:woke|awakened|found\s+(?:himself|herself|myself|itself))\s+(?:up\s+)?as\s+a[n]?\s+(\w+)",
        r"(?:became|become|turned\s+into)\s+a[n]?\s+(\w+)",
        r"(?:now|was\s+now|is\s+now)\s+a[n]?\s+(\w+)",
        r"(?:his|her|my|their)\s+(?:new\s+)?body\s+(?:was|is)\s+(?:that\s+of\s+)?a[n]?\s+(\w+)",
        r"(?:the\s+)?protagonist\s+(?:was|is)\s+a[n]?\s+(\w+)",
        r"system[:\s]+(?:you\s+(?:are|have\s+become))\s+a[n]?\s+(\w+)",
    ]
    for pattern in _TRANSFORM_PATTERNS:
        for m in re.finditer(pattern, low_story, re.IGNORECASE):
            species = m.group(1).lower().rstrip("s")   # singularise
            for ctype, keys in keyword_groups:
                if species in keys or any(k == species for k in keys):
                    return ctype

    # ── Local context around name mentions ─────────────────────────────────
    # Only search ±200 chars around each name occurrence (wider than before).
    name_contexts: List[str] = []
    if low_name:
        for match in re.finditer(re.escape(low_name), low_story):
            start = max(0, match.start() - 200)
            end = min(len(low_story), match.end() + 200)
            name_contexts.append(low_story[start:end])
    joined = " ".join(name_contexts) if name_contexts else ""
    for ctype, keys in keyword_groups:
        if joined and any(k in joined for k in keys):
            return ctype

    return "humanoid"


def _default_fields_for_character_type(character_type: str, gender: str, seed: int) -> Dict[str, Any]:
    ctype = clean_for_prompt(character_type).lower() or "human"
    g = clean_for_prompt(gender).lower() or "male"
    if ctype in {"human", "humanoid"}:
        if g == "female":
            hair = _pick(HAIR_FEMALE, seed)
            build = _pick(BUILDS_F, seed, 1)
            outfit = _pick(OUTFITS_F, seed, 2)
            anchor = _pick(ANCHORS_F, seed, 3)
            neg = "male, man, beard, mustache, masculine jaw, broad shoulders"
        else:
            hair = _pick(HAIR_MALE, seed)
            build = _pick(BUILDS_M, seed, 1)
            outfit = _pick(OUTFITS_M, seed, 2)
            anchor = _pick(ANCHORS_M, seed, 3)
            neg = "female, woman, breasts, feminine makeup, long eyelashes"
        return {
            "character_type": ctype,
            "gender": g if g in {"male", "female"} else "male",
            "hair": hair,
            "eyes": _pick(EYES, seed, 4),
            "build": build,
            "skin": _pick(SKIN_TONES, seed, 5),
            "outfit": outfit,
            "anchor": anchor,
            "negative_lock": neg,
            "alt_outfits": [],
        }
    if ctype in {"animal", "beast"}:
        surface = _pick([
            "short tawny fur",
            "dense charcoal fur",
            "striped brown-and-black fur",
            "snow-white fur",
            "sleek reddish fur",
            "dark scaled hide",
        ], seed)
        eyes = _pick(["golden eyes", "amber eyes", "pale blue eyes", "emerald eyes"], seed, 1)
        build = _pick(["lean agile body", "compact sturdy body", "large muscular body", "long serpentine body"], seed, 2)
        anchor = _pick(["distinct white chest marking", "torn left ear", "glowing patterned markings", "ringed tail tip", "scar across the snout"], seed, 3)
        return {
            "character_type": ctype,
            "gender": g if g in {"male", "female"} else "male",
            "hair": surface,
            "eyes": eyes,
            "build": build,
            "skin": surface,
            "outfit": "no clothing",
            "anchor": anchor,
            "negative_lock": "human body, human hands, human face, human clothing, human proportions",
            "alt_outfits": [],
        }
    if ctype == "spirit":
        if g == "female":
            hair = "long flowing ethereal hair shaped from elemental essence"
            build = "lean humanoid spirit body with feminine proportions"
            outfit = "flowing feminine spirit robes formed from the same energy"
            anchor = "bright core at the chest with trailing energy wisps"
            neg = "featureless blob form, tiny orb only, animal anatomy, masculine beard, bulky armor, chibi proportions, photorealism, nude, naked, nudity, exposed skin, bare skin, topless, nsfw"
        else:
            hair = "medium-length windswept ethereal hair shaped from elemental essence"
            build = "lean masculine humanoid spirit body with broad shoulders and flat chest"
            outfit = "layered male spirit robes formed from the same energy, masculine robe silhouette"
            anchor = "bright core at the chest with controlled trailing energy wisps"
            neg = "featureless blob form, tiny orb only, animal anatomy, feminine face, feminine body shape, dress, skirt, chibi proportions, photorealism, nude, naked, nudity, exposed skin, bare skin, topless, nsfw"
        return {
            "character_type": ctype,
            "gender": g if g in {"male", "female"} else "male",
            "hair": hair,
            "eyes": "bright glowing eyes",
            "build": build,
            "skin": "semi-transparent elemental body with stable humanoid face and limbs",
            "outfit": outfit,
            "anchor": anchor,
            "negative_lock": neg,
            "alt_outfits": [],
        }
    if ctype == "orb":
        if g == "female":
            hair = "long energy-shaped hair flowing backward from the head"
            build = "lean humanoid energy body with feminine proportions and an orb core at the chest"
            outfit = "fitted feminine spirit robes formed from flowing light"
            neg = "featureless floating ball only, blob form, animal anatomy, beard, bulky armor, chibi proportions, photorealism, nude, naked, nudity, exposed skin, bare skin, topless, nsfw"
        else:
            hair = "short-to-medium energy-shaped hair flowing backward from the head"
            build = "lean masculine humanoid energy body with broad shoulders, flat chest, and an orb core at the chest"
            outfit = "structured male spirit robes formed from flowing light, masculine silhouette"
            neg = "featureless floating ball only, blob form, animal anatomy, feminine face, feminine body shape, dress, skirt, chibi proportions, photorealism, nude, naked, nudity, exposed skin, bare skin, topless, nsfw"
        return {
            "character_type": ctype,
            "gender": g if g in {"male", "female"} else "male",
            "hair": hair,
            "eyes": "bright glowing eyes",
            "build": build,
            "skin": "glowing translucent energy skin with a stable humanoid face and limbs",
            "outfit": outfit,
            "anchor": "bright orb core embedded in the chest with a faint halo ring",
            "negative_lock": neg,
            "alt_outfits": [],
        }
    if ctype == "object":
        return {
            "character_type": ctype,
            "gender": g if g in {"male", "female"} else "male",
            "hair": "no hair",
            "eyes": "glowing slit-like eyes built into the object design",
            "build": "solid inanimate object form",
            "skin": _pick(["weathered wood surface", "polished metal surface", "smooth stone surface", "ceramic glazed surface"], seed),
            "outfit": "no clothing",
            "anchor": _pick(["distinct engraved emblem", "cracked corner detail", "glowing rune line", "gold band accent"], seed, 1),
            "negative_lock": "human body, limbs, human face, clothing, photorealism",
            "alt_outfits": [],
        }
    return _default_fields_for_character_type("humanoid", g, seed)




def _default_character_fields(
    project_id: str,
    name: str,
    story: str,
    force_gender: Optional[str] = None,
    force_character_type: Optional[str] = None,
) -> Dict[str, Any]:
    seed = _stable_int(f"{project_id}:{name}")
    character_type = _infer_character_type(name, story, explicit_type=force_character_type)
    default_gender = _seeded_gender(project_id, name, story, force_gender)
    if force_gender:
        default_gender = force_gender
    return _default_fields_for_character_type(character_type, default_gender, seed)

def _sanitize_unknown(v: str, fallback: str) -> str:
    s = clean_for_prompt(v)
    if not s or s.lower() in {"unknown", "not specified", "n/a", "none"}:
        return fallback
    return s



def character_prompts_from_fields(fields: Dict[str, Any]) -> Tuple[str, str, str]:
    character_type = _sanitize_unknown(str(fields.get("character_type", "human")), "human").lower()
    if character_type not in CHARACTER_TYPE_CHOICES:
        character_type = "human"
    gender = _sanitize_unknown(str(fields.get("gender", "male")), "male").lower()
    if gender == "none":
        gender = "male"
    male_guard = "male-presenting, masculine facial structure, flat chest, masculine silhouette, male clothing, no dress, no skirt, no feminine styling"
    female_guard = "female-presenting, feminine facial structure, feminine silhouette, feminine clothing, no beard, no masculine styling"
    hair = _sanitize_unknown(str(fields.get("hair", "")), "short dark brown messy hair")
    eyes = _sanitize_unknown(str(fields.get("eyes", "")), "brown eyes")
    build = _sanitize_unknown(str(fields.get("build", "")), "lean athletic build")
    skin = _sanitize_unknown(str(fields.get("skin", "")), "warm beige skin")
    outfit = clean_for_prompt(str(fields.get("outfit", "")))
    anchor = _sanitize_unknown(str(fields.get("anchor", "")), "thin black wristband")

    if character_type in {"human", "humanoid"}:
        parts = [character_type, gender, skin, hair, eyes, build]
        if outfit and outfit.lower() not in {"none", "no clothing"}:
            parts.append(outfit)
        parts.append(f"signature detail: {anchor}")
        dna = clean_for_prompt("single consistent character design, " + ", ".join([p for p in parts if clean_for_prompt(p)]))
        ref_prompt = clean_for_prompt(
            f"{DEFAULT_STYLE}\n"
            f"{REF_ANIME_LOCK}\n"
            f"full body character reference, front view, neutral standing pose, clean plain background, no powers active, no floating UI,\n"
            f"{dna}, high detail face, consistent hairstyle and clothing folds, clear shoes, readable accessories,\n"
            f"NO TEXT, NO SIGNS, NO UI, NO WORDS, no watermark, no logo"
        )
    elif character_type in {"animal", "beast"}:
        guard = "male animal" if gender != "female" else "female animal"
        parts = [character_type, guard, skin, hair, eyes, build, f"signature detail: {anchor}"]
        dna = clean_for_prompt("single consistent non-human character design, " + ", ".join([p for p in parts if clean_for_prompt(p)]))
        ref_prompt = clean_for_prompt(
            f"{DEFAULT_STYLE}\n"
            f"{REF_ANIME_LOCK}\n"
            f"full body creature reference, side-facing three-quarter stance, isolated on clean plain background, no humans, no harness, no rider,\n"
            f"{dna}, readable paws or claws, clear silhouette, stable anatomy, illustrated 2d creature design,\n"
            f"NO TEXT, NO SIGNS, NO UI, NO WORDS, no watermark, no logo"
        )
    elif character_type == "spirit":
        parts = ["humanoid spirit", gender, skin, hair, eyes, build]
        if outfit and outfit.lower() not in {"none", "no clothing"}:
            parts.append(outfit)
        parts.append(f"signature detail: {anchor}")
        dna = clean_for_prompt("single consistent humanoid spirit character design, " + ", ".join([p for p in parts if clean_for_prompt(p)]))
        ref_prompt = clean_for_prompt(
            f"{DEFAULT_STYLE}\n"
            f"{REF_ANIME_LOCK}\n"
            f"full body humanoid spirit reference, front view, neutral standing pose, clean plain background, no environment, no extra people,\n"
            f"{dna}, stable human silhouette, readable face, consistent hair shape, elemental body texture, illustrated 2d spirit design,\n"
            f"NO TEXT, NO SIGNS, NO UI, NO WORDS, no watermark, no logo"
        )
    elif character_type == "orb":
        parts = ["humanoid energy avatar", gender, skin, hair, eyes, build]
        if outfit and outfit.lower() not in {"none", "no clothing"}:
            parts.append(outfit)
        parts.append(f"signature detail: {anchor}")
        dna = clean_for_prompt("single consistent humanoid orb-spirit character design, " + ", ".join([p for p in parts if clean_for_prompt(p)]))
        ref_prompt = clean_for_prompt(
            f"{DEFAULT_STYLE}\n"
            f"{REF_ANIME_LOCK}\n"
            f"full body humanoid spirit reference, front view, neutral standing pose, clean plain background, no environment, no extra people,\n"
            f"{dna}, bright orb core integrated into the body design, stable human silhouette, readable face, illustrated 2d spirit design,\n"
            f"NO TEXT, NO SIGNS, NO UI, NO WORDS, no watermark, no logo"
        )
    else:
        parts = [skin, build]
        if outfit and outfit.lower() not in {"none", "no clothing"}:
            parts.append(outfit)
        parts.append(f"signature detail: {anchor}")
        dna = clean_for_prompt("single consistent sentient object design, " + ", ".join([p for p in parts if clean_for_prompt(p)]))
        ref_prompt = clean_for_prompt(
            f"{DEFAULT_STYLE}\n"
            f"{REF_ANIME_LOCK}\n"
            f"isolated object reference, centered object on clean plain background, no environment, no humans,\n"
            f"{dna}, readable materials and edges, stable silhouette, illustrated 2d object design,\n"
            f"NO TEXT, NO SIGNS, NO UI, NO WORDS, no watermark, no logo"
        )
    negative_lock = clean_for_prompt(str(fields.get("negative_lock", "")))
    return dna, ref_prompt, negative_lock

def build_character_profile(project_id: str, name: str, story: str, force_gender: Optional[str] = None, force_character_type: Optional[str] = None, override_fields: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    fields = _default_character_fields(project_id, name, story, force_gender=force_gender, force_character_type=force_character_type)
    override_fields = override_fields or {}
    for k in ["character_type", "gender", "hair", "eyes", "build", "skin", "outfit", "anchor", "negative_lock", "alt_outfits"]:
        if k in override_fields and override_fields[k] not in [None, ""]:
            fields[k] = override_fields[k]
    dna, ref_prompt, neg_lock = character_prompts_from_fields(fields)
    return {"fields": fields, "dna_prompt": dna, "ref_prompt": ref_prompt, "negative_lock": neg_lock}


def build_location_profile(
    name: str,
    bible_prompt: Optional[str] = None,
    shot_overrides: Optional[Dict[str, str]] = None,
    ref_prompt_base: Optional[str] = None,
    sub_locations: Optional[Any] = None,
) -> Dict[str, Any]:
    # shot_overrides is accepted for backward compatibility but no longer drives generation.
    _ = shot_overrides
    base = name.lower()
    bible = clean_for_prompt(bible_prompt or (
        f"anime manhwa environment canon for the {base}, illustrated 2d background design, repeatable layout anchors, "
        f"specific architecture, fixed furniture placement, stable lighting direction, clear wall and floor materials, consistent mood, no photorealism"
    ))
    ref_base = clean_for_prompt(ref_prompt_base or (
        f"{DEFAULT_STYLE}\n"
        f"{REF_ANIME_LOCK}\n"
        f"illustrated anime manhwa environment reference image of the {base}, drawn 2d background, no people, no powers, no text, readable layout,\n"
        f"specific architecture and materials, fixed furniture and prop placement, cinematic lighting, no live-action realism, no photorealism,\n"
        f"NO TEXT, NO SIGNS, NO UI, NO WORDS, no watermark"
    ))
    parsed_subs: Dict[str, Dict[str, str]] = {}
    if isinstance(sub_locations, dict):
        for sub_name, raw in sub_locations.items():
            sname = clean_for_prompt(str(sub_name))
            if not sname:
                continue
            if isinstance(raw, dict):
                sbible = clean_for_prompt(str(raw.get("bible_prompt") or "")) or f"sub-location inside {base}: {sname.lower()}, consistent layout anchors and materials"
                sref = clean_for_prompt(str(raw.get("ref_prompt") or ""))
            else:
                sbible = f"sub-location inside {base}: {sname.lower()}, consistent layout anchors and materials"
                sref = clean_for_prompt(str(raw))
            if not sref:
                sref = clean_for_prompt(
                    f"{ref_base}\nsub-location focus: {sname.lower()} inside the {base}, environment only, stable layout, consistent architectural anchors"
                )
            parsed_subs[sname] = {"bible_prompt": sbible, "ref_prompt": sref}
    elif isinstance(sub_locations, list):
        for raw in sub_locations:
            if isinstance(raw, dict):
                sname = clean_for_prompt(str(raw.get("name") or ""))
                sbible = clean_for_prompt(str(raw.get("bible_prompt") or ""))
                sref = clean_for_prompt(str(raw.get("ref_prompt") or ""))
            else:
                sname = clean_for_prompt(str(raw))
                sbible = ""
                sref = ""
            if not sname:
                continue
            if not sbible:
                sbible = f"sub-location inside {base}: {sname.lower()}, consistent layout anchors and materials"
            if not sref:
                sref = clean_for_prompt(
                    f"{ref_base}\nsub-location focus: {sname.lower()} inside the {base}, environment only, stable layout, consistent architectural anchors"
                )
            parsed_subs[sname] = {
                "bible_prompt": sbible,
                "ref_prompt": sref,
            }
    if not parsed_subs:
        for sname, sdesc in _default_sub_location_specs(name).items():
            parsed_subs[sname] = {
                "bible_prompt": f"sub-location inside {base}: {sname.lower()}, {sdesc}, consistent architecture/material continuity",
                "ref_prompt": clean_for_prompt(
                    f"{ref_base}\nsub-location focus: {sname.lower()} inside the {base}, {sdesc}, environment only, stable layout, consistent architectural anchors"
                ),
            }
    return {
        "bible_prompt": bible,
        "ref_prompt_base": ref_base,
        "sub_locations": parsed_subs,
        # legacy key kept to avoid breaking older logic paths
        "shots": {},
    }


def build_item_profile(name: str, kind: str = "prop", bible_prompt: Optional[str] = None, ref_prompt: Optional[str] = None) -> Dict[str, Any]:
    kind = (kind or "prop").strip().lower()
    seed = _stable_int(name)
    if kind == "power_system":
        color = _pick(COLOR_POOL, seed)
        shape = _pick(SHAPE_POOL, seed, 1)
        bible = clean_for_prompt(bible_prompt or (
            f"toggleable visual power effect: {name.lower()}, visible only when active, {color} glow, {shape}, crisp outer edges, layered inner light, stable silhouette, no permanent physical redesign, no random extra symbols, no full screen takeover"
        ))
        ref = clean_for_prompt(ref_prompt or (
            f"{DEFAULT_STYLE}\n"
            f"{REF_ANIME_LOCK}\n"
            f"single isolated power effect reference for {name.lower()}, centered on plain background, no full person, no extra objects,\n"
            f"specific look: {color} emission, {shape}, tight glow halo, readable energy core, fixed shape language,\n"
            f"{bible},\n"
            f"NO TEXT, NO SIGNS, NO UI, NO WORDS, no watermark, no logo"
        ))
    else:
        bible = clean_for_prompt(bible_prompt or (
            f"story-important prop design: {name.lower()}, clear silhouette, fixed proportions, consistent material, distinctive small details, readable scale, repeatable shape language"
        ))
        ref = clean_for_prompt(ref_prompt or (
            f"{DEFAULT_STYLE}\n"
            f"{REF_ANIME_LOCK}\n"
            f"single prop reference image of {name.lower()}, centered on plain background, no hands, no people,\n"
            f"{bible}, studio lighting, high detail,\n"
            f"NO TEXT, NO SIGNS, NO UI, NO WORDS, no watermark, no logo"
        ))
    return {"bible_prompt": bible, "ref_prompt": ref, "kind": kind}


# Per-model capability flags — controls which params are safe to send
_FAL_MODEL_CAPS: Dict[str, Dict] = {
    "fal-ai/z-image/turbo":        {"steps": 8,    "guidance": 7.5, "neg": True,  "img_size": True,  "aspect": None,   "simple": False, "ref_images": False, "edit_mode": False},
    "fal-ai/nano-banana-2":        {"steps": None,  "guidance": 0,   "neg": False, "img_size": False, "aspect": "9:16", "simple": True,  "ref_images": True,  "edit_mode": False},
    "fal-ai/nano-banana-pro":      {"steps": None,  "guidance": 0,   "neg": False, "img_size": False, "aspect": "9:16", "simple": True,  "ref_images": True,  "edit_mode": False},
    # NB2 Edit — character-consistent generation; uses image_urls (flat list), not reference_images dicts
    "fal-ai/nano-banana-2/edit":   {"steps": None,  "guidance": 0,   "neg": False, "img_size": False, "aspect": "9:16", "simple": True,  "ref_images": False, "edit_mode": True},
}
_FAL_DEFAULT_CAPS: Dict = {"steps": 28, "guidance": 3.5, "neg": True, "img_size": True, "aspect": None, "simple": False}


def upload_pil_to_fal(pil_img: Image.Image) -> str:
    """Upload a PIL image to FAL storage and return its URL for use as a reference."""
    if fal_client is None:
        raise RuntimeError("fal_client not installed.")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    url = fal_client.upload(buf.getvalue(), content_type="image/png")
    return url


def call_fal_generate(
    prompt: str,
    negative_prompt: str,
    enable_safety_checker: Optional[bool] = None,
    model: Optional[str] = None,
    num_inference_steps: Optional[int] = None,
    guidance_scale: Optional[float] = None,
    skip_esrgan: bool = False,
    aspect_ratio_override: Optional[str] = None,
    upscale_factor: int = 2,
    reference_images: Optional[List[Dict[str, str]]] = None,
) -> Image.Image:
    if fal_client is None:
        raise RuntimeError("fal_client not installed. Add 'fal-client' to requirements.txt.")
    if not os.getenv("FAL_KEY"):
        raise RuntimeError("Missing FAL_KEY env var. Set it in HF Space Secrets.")
    use_model = model or FAL_MODEL
    caps = _FAL_MODEL_CAPS.get(use_model, _FAL_DEFAULT_CAPS)
    steps = num_inference_steps if num_inference_steps is not None else caps["steps"]
    guidance = guidance_scale if guidance_scale is not None else caps["guidance"]

    if caps.get("simple"):
        # Minimal argument set for models like Nano Banana that don't accept
        # loras, acceleration, enable_prompt_expansion, steps, or guidance
        arguments: Dict[str, Any] = {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio_override or caps.get("aspect") or "9:16",
        }
        if caps.get("edit_mode") and reference_images:
            # NB2 Edit endpoint — image_urls + quality parameters that make
            # the model actually USE the references and produce premium output.
            # thinking_level "high" is the single biggest lever for reference compliance.
            arguments["image_urls"] = [r["url"] for r in reference_images if r.get("url")]
            arguments["thinking_level"] = "high"
            arguments["output_format"] = "png"
            arguments["num_images"] = 1
            arguments["limit_generations"] = False   # false = maximum quality
            arguments["system_prompt"] = (
                "You create premium Korean action-fantasy manhwa / webtoon illustrations. "
                "Your top priority is preserving locked character identity across every panel — "
                "exact face shape, hair, eye color, skin tone, and outfit must remain identical. "
                "Apply reference images exactly as directed: identity images lock who a character IS, "
                "composition images control only shot angle and framing (treat their people as invisible "
                "mannequins), mood images control only lighting and color palette (people invisible), "
                "setting images control only background architecture (people invisible). "
                "Render beautiful, refined facial anatomy with gradient irises, sharp catchlights, "
                "and individually rendered hair strands. Never copy reference people's appearance "
                "into a character slot that belongs to someone else."
            )
        elif reference_images and caps.get("ref_images"):
            # Regular NB2 / NB Pro use reference_images: [{url, tag}, ...]
            arguments["reference_images"] = reference_images
        if enable_safety_checker is not None:
            arguments["enable_safety_checker"] = bool(enable_safety_checker)
    else:
        arguments: Dict[str, Any] = {
            "prompt": prompt,
            "num_images": 1,
            "output_format": FAL_OUTPUT_EXT,
            "acceleration": "regular",
            "enable_prompt_expansion": False,
            "loras": [],
        }
        if caps["img_size"]:
            arguments["image_size"] = FORCED_IMAGE_SIZE
        if caps["aspect"]:
            arguments["aspect_ratio"] = caps["aspect"]
        if steps is not None:
            arguments["num_inference_steps"] = steps
        if guidance and guidance > 0:
            arguments["guidance_scale"] = guidance
        if caps["neg"] and (negative_prompt or "").strip():
            arguments["negative_prompt"] = negative_prompt
        if enable_safety_checker is not None:
            arguments["enable_safety_checker"] = bool(enable_safety_checker)

    import concurrent.futures as _cf
    # Edit-mode models (NB2 Edit) can legitimately take 300 s+ when FAL queues are
    # busy — give them 600 s so we never cut off a finished image.  Fast models keep
    # the 120 s cap.
    # IMPORTANT: never use `with ThreadPoolExecutor` around a timed future — the context-manager
    # __exit__ calls shutdown(wait=True) which blocks until the FAL thread finishes even after
    # TimeoutError is raised, turning a "90 s timeout" into a full-length hang.
    # Use shutdown(wait=False) via finally so the thread is abandoned on timeout.
    _fal_timeout = 600 if caps.get("edit_mode") else 120

    # ── FAL call with automatic content-policy retry (up to 2 rewrites) ──────
    _active_prompt = arguments.get("prompt", prompt)
    _last_err = None
    for _attempt in range(3):   # attempt 0 = original; 1,2 = rewrites
        if _attempt > 0:
            print(f"[fal] content-policy retry {_attempt}/2 — rewriting prompt…", flush=True)
            _rewritten = _rewrite_flagged_prompt(
                _active_prompt,
                str(_last_err)[:200] if _last_err else "content policy",
            )
            if not _rewritten:
                raise RuntimeError(
                    f"Content flagged and Claude could not rewrite the prompt. "
                    f"Original error: {_last_err}"
                )
            _active_prompt = _rewritten
            arguments = dict(arguments)   # shallow copy so we don't mutate the original
            arguments["prompt"] = _active_prompt
            print(f"[fal] rewritten prompt (first 200 chars): {_active_prompt[:200]}", flush=True)

        _ex = _cf.ThreadPoolExecutor(max_workers=1)
        try:
            _fut = _ex.submit(fal_client.run, use_model, arguments=arguments)
            try:
                result = _fut.result(timeout=_fal_timeout)
                _last_err = None
                break   # success
            except _cf.TimeoutError:
                raise RuntimeError(
                    f"FAL timed out after {_fal_timeout}s — queue may be busy, please retry."
                )
        except Exception as _exc:
            _is_422 = (
                (_FalHTTPError and isinstance(_exc, _FalHTTPError) and getattr(_exc, "status_code", 0) == 422)
                or "422" in str(_exc)
                or "content" in str(_exc).lower() and "flag" in str(_exc).lower()
                or "content could not be processed" in str(_exc).lower()
            )
            if _is_422 and _attempt < 2:
                _last_err = _exc
                continue   # retry with rewrite
            raise   # non-422 or out of retries → propagate
        finally:
            _ex.shutdown(wait=False, cancel_futures=True)
    else:
        raise RuntimeError(f"FAL content policy: prompt flagged after 2 rewrites. Last error: {_last_err}")

    imgs = result.get("images") or []
    if not imgs or not imgs[0].get("url"):
        raise RuntimeError(f"fal returned no image url: {result}")

    image_url = imgs[0]["url"]

    # ESRGAN upscale — skip_esrgan=True for reference/thumbnail images that don't need full res
    _uf_factor = max(1, int(upscale_factor or 2))
    if not skip_esrgan and _uf_factor > 1:
        try:
            _ux = _cf.ThreadPoolExecutor(max_workers=1)
            try:
                _uf = _ux.submit(
                    fal_client.subscribe, "fal-ai/esrgan",
                    arguments={"image_url": image_url, "upscale_factor": _uf_factor}
                )
                try:
                    up_res = _uf.result(timeout=60)
                    upscaled = (up_res or {}).get("image") or {}
                    if upscaled.get("url"):
                        image_url = upscaled["url"]
                except _cf.TimeoutError:
                    pass  # fall back to original resolution if ESRGAN stalls
            finally:
                _ux.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass  # never let upscale failure break generation

    import requests
    r = requests.get(image_url, timeout=30)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    return img


def make_char_table_rows(st: ProjectState) -> List[List[Any]]:
    rows = []
    for name, c in st.characters.items():
        f = c.get("fields", {})
        rows.append([name, f.get("character_type","human"), f.get("gender",""), f.get("hair",""), f.get("eyes",""), f.get("outfit",""), c.get("dna_prompt",""), c.get("negative_lock","")])
    return rows


def make_loc_table_rows(st: ProjectState) -> List[List[Any]]:
    rows = []
    for name, l in st.locations.items():
        subs = l.get("sub_locations") or {}
        sub_lines = "\n".join([f"{k}: {(v or {}).get('bible_prompt', '')}" for k, v in subs.items()])
        rows.append([name, l.get("bible_prompt", ""), sub_lines])
    return rows


def make_item_table_rows(st: ProjectState) -> List[List[Any]]:
    rows = []
    for name, it in st.items.items():
        rows.append([name, it.get("kind", "prop"), it["bible_prompt"], it["ref_prompt"]])
    return rows


def _save_project_json(st: ProjectState) -> None:
    if not st or not st.project_dir:
        return
    save_json(os.path.join(st.project_dir, "project.json"), {
        "project_id": st.project_id,
        "project_name": st.project_name,
        "story": st.story,
        "beats": st.beats,
        "characters": st.characters,
        "locations": st.locations,
        "items": st.items,
        "beat_plans": st.beat_plans,
        "next_image_index": st.next_image_index,
        "image_prompts": st.image_prompts,
        "forced_image_size": FORCED_IMAGE_SIZE,
        "total_cost": st.total_cost,
        "total_images": st.total_images,
        "character_age_phases": getattr(st, "character_age_phases", {}),
        "story_parts": getattr(st, "story_parts", []),
        "page_scripts": {str(k): v for k, v in getattr(st, "page_scripts", {}).items()},
        "page_scripts_generated_at": getattr(st, "page_scripts_generated_at", ""),
        "build_mode": getattr(st, "build_mode", "Panel"),
        "style_reference_url": getattr(st, "style_reference_url", ""),
        "world_context": getattr(st, "world_context", ""),
        "original_story": getattr(st, "original_story", ""),
        "cover_image_path": getattr(st, "cover_image_path", ""),
        "character_cast": getattr(st, "character_cast", {}),
        "casting_seed": getattr(st, "casting_seed", ""),
        "manual_page_refs": {str(k): v for k, v in (getattr(st, "manual_page_refs", None) or {}).items()},
    })
    _cloud.upload_project_meta_bg(st.project_dir)


def _sanitize_character_payload(st: ProjectState, story: str, payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    chars = payload.get("characters") or []
    out: Dict[str, Dict[str, Any]] = {}
    for ch in chars:
        name = clean_for_prompt(str(ch.get("name") or ""))
        if not name or name in PLACE_WORD_BLOCKLIST or name in GENERIC_NOUN_BLOCKLIST:
            continue
        # Reject single-word generic nouns even with different casing
        if name.lower() in {w.lower() for w in GENERIC_NOUN_BLOCKLIST}:
            continue
        raw_type = str(ch.get("character_type") or ch.get("type") or "").strip()
        fields = _default_character_fields(
            st.project_id,
            name,
            story,
            force_gender=(str(ch.get("gender") or "") or None),
            force_character_type=(raw_type or None),
        )
        fields["character_type"] = _sanitize_unknown(raw_type, fields.get("character_type", "human")).lower()
        if fields["character_type"] not in CHARACTER_TYPE_CHOICES:
            fields["character_type"] = "human"
        fallback_gender = fields.get("gender") or "male"
        fields["gender"] = _sanitize_unknown(str(ch.get("gender") or fields.get("gender") or "male"), fallback_gender).lower()
        fields["hair"] = _sanitize_unknown(str(ch.get("hair") or ""), fields["hair"])
        fields["eyes"] = _sanitize_unknown(str(ch.get("eyes") or ""), fields["eyes"])
        fields["build"] = _sanitize_unknown(str(ch.get("build") or ""), fields["build"])
        fields["skin"] = _sanitize_unknown(str(ch.get("skin") or ""), fields["skin"])
        fields["outfit"] = _sanitize_unknown(str(ch.get("default_outfit") or ch.get("outfit") or ""), fields["outfit"])
        fields["anchor"] = _sanitize_unknown(str(ch.get("signature_detail") or ""), fields["anchor"])
        # never allow powers inside signature detail
        if any(word in fields["anchor"].lower() for word in ["power", "aura", "lightning", "shadow", "fire", "ice", "system", "mana", "glow"]):
            fields["anchor"] = _default_character_fields(st.project_id, name, story, force_gender=fields["gender"])["anchor"]
        alt = ch.get("alt_outfits") or []
        if not isinstance(alt, list):
            alt = []
        fields["alt_outfits"] = [clean_for_prompt(str(x)) for x in alt if clean_for_prompt(str(x))]
        out[name] = build_character_profile(st.project_id, name, story, force_character_type=fields.get("character_type"), override_fields=fields)
        # Store named forms from Claude's story bible (for situational appearance lookup per beat)
        raw_forms = ch.get("forms") or []
        if isinstance(raw_forms, list):
            cleaned_forms = []
            valid_types = {"base", "outfit", "partial_transform", "full_transform"}
            for f in raw_forms:
                if not isinstance(f, dict):
                    continue
                fname = str(f.get("name") or "").strip()
                ftype = str(f.get("type") or "base").strip().lower()
                if ftype not in valid_types:
                    ftype = "outfit"
                fdesc = clean_for_prompt(str(f.get("description") or ""))
                fnote = str(f.get("context_note") or "")
                raw_locs = f.get("activates_at_locations") or []
                locs = [str(l).strip() for l in raw_locs if str(l).strip()] if isinstance(raw_locs, list) else []
                if fname and fdesc:
                    cleaned_forms.append({
                        "name": fname,
                        "type": ftype,
                        "description": fdesc,
                        "context_note": fnote,
                        "activates_at_locations": locs,
                    })
            if cleaned_forms:
                out[name]["forms"] = cleaned_forms
    return out


def _sanitize_location_payload(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    _blocked_lower = {w.lower() for w in GENERIC_NOUN_BLOCKLIST}
    out: Dict[str, Dict[str, Any]] = {}
    for loc in (payload.get("locations") or []):
        name = clean_for_prompt(str(loc.get("name") or ""))
        if not name:
            continue
        # Reject generic nouns / stop words as location names
        if name.lower() in _blocked_lower:
            continue
        raw_subs = loc.get("sub_locations")
        # Support legacy payloads that only provide "shots" by converting to default sub-locations.
        if raw_subs is None:
            raw_subs = loc.get("subLocations")
        out[name] = build_location_profile(
            name,
            bible_prompt=str(loc.get("bible_prompt") or ""),
            shot_overrides={},
            ref_prompt_base=str(loc.get("ref_prompt_base") or ""),
            sub_locations=raw_subs,
        )
    return out


def _sanitize_item_payload(payload: Dict[str, Any], character_names: Optional[set] = None) -> Dict[str, Dict[str, Any]]:
    _blocked_lower = {w.lower() for w in PLACE_WORD_BLOCKLIST} | {w.lower() for w in GENERIC_NOUN_BLOCKLIST}
    _char_lower = {n.lower() for n in (character_names or set())}
    out: Dict[str, Dict[str, Any]] = {}
    for it in (payload.get("items") or []):
        name = clean_for_prompt(str(it.get("name") or ""))
        if not name:
            continue
        # Reject single-word generic nouns / stop words as item names
        if name.lower() in _blocked_lower:
            continue
        # Reject anything that is already a character name (Claude often cross-lists them)
        if name.lower() in _char_lower:
            continue
        kind = clean_for_prompt(str(it.get("kind") or "prop")).lower()
        if kind not in {"prop", "power_system"}:
            kind = "prop"
        out[name] = build_item_profile(name, kind=kind, bible_prompt=str(it.get("bible_prompt") or ""), ref_prompt=str(it.get("ref_prompt") or ""))
    return out


def _library_cover(st: "ProjectState") -> Optional[str]:
    """Pick the best library template image as a Shorts cover, overlay the project title,
    and save to project_dir/cover/cover.jpg.  No FAL call — instant and free."""
    try:
        from character_library import load_library as _ll
        lib = _ll()
        templates = lib.get("templates") or {}
        if not templates:
            return None

        title = (st.project_name or "Untitled").strip()

        # ── Score every template that has a usable local image ──────────────
        # Prefer: cast characters first, then dramatic/action/cinematic solo shots.
        cast_ids = set((st.character_cast or {}).values())

        _COVER_POSITIVE = {"solo", "action", "dramatic", "intense", "cinematic",
                           "confident", "fierce", "powerful", "hero", "protagonist"}
        _COVER_NEGATIVE = {"group", "crowd", "background only", "setting", "environment",
                           "couple", "duo"}

        scored: list = []
        for tid, t in templates.items():
            img_path = t.get("local_face") or t.get("local_body") or ""
            if not img_path or not os.path.exists(img_path):
                continue
            tags = {tg.lower() for tg in (t.get("tags") or [])}
            score = 0.0
            if tid in cast_ids:
                score += 5.0           # cast characters are ideal
            score += len(tags & _COVER_POSITIVE)
            score -= len(tags & _COVER_NEGATIVE)
            reuse = (t.get("reusability") or {})
            score += float(reuse.get("as_identity_ref", 0) or 0)
            if "solo" in tags or "alone" in tags:
                score += 2.0
            scored.append((score, tid, img_path))

        if not scored:
            return None

        scored.sort(key=lambda x: -x[0])
        _, best_tid, best_img_path = scored[0]

        # ── Load the image ───────────────────────────────────────────────────
        from PIL import Image as _PImg, ImageDraw as _PDraw, ImageFont as _PFont
        img = _PImg.open(best_img_path).convert("RGB")

        # Resize to 9:16 portrait crop if needed
        w, h = img.size
        target_ratio = 9 / 16
        cur_ratio = w / h
        if cur_ratio > target_ratio:
            # too wide — crop sides
            new_w = int(h * target_ratio)
            offset = (w - new_w) // 2
            img = img.crop((offset, 0, offset + new_w, h))
        elif cur_ratio < target_ratio * 0.7:
            # very tall — pad sides
            new_w = int(h * target_ratio)
            canvas = _PImg.new("RGB", (new_w, h), (0, 0, 0))
            canvas.paste(img, ((new_w - w) // 2, 0))
            img = canvas
        w, h = img.size

        # ── Title overlay (same logic as _generate_shorts_cover) ────────────
        _FONT_CANDIDATES = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
        ]
        def _load_font(size):
            for _fp in _FONT_CANDIDATES:
                if os.path.isfile(_fp):
                    try:
                        return _PFont.truetype(_fp, size=size)
                    except Exception:
                        pass
            return _PFont.load_default()

        # Gradient scrim — bottom 40%
        scrim_h = int(h * 0.40)
        overlay = _PImg.new("RGBA", (w, h), (0, 0, 0, 0))
        ov_d = _PDraw.Draw(overlay)
        for row in range(scrim_h):
            alpha = int(230 * (row / scrim_h) ** 1.4)
            ov_d.line([(0, h - scrim_h + row), (w, h - scrim_h + row)],
                      fill=(0, 0, 0, alpha))
        img = img.convert("RGBA")
        img = _PImg.alpha_composite(img, overlay)
        img = img.convert("RGB")
        draw = _PDraw.Draw(img)

        def _text_size(d, text, font):
            try:
                bb = d.textbbox((0, 0), text, font=font)
                return bb[2] - bb[0], bb[3] - bb[1]
            except AttributeError:
                return d.textsize(text, font=font)

        def _draw_shadow_text(d, pos, text, font, fill=(255, 255, 255)):
            x, y = pos
            for ox, oy in [(-3, 3), (3, 3), (0, 5), (-5, 5), (5, 5)]:
                d.text((x + ox, y + oy), text, font=font, fill=(0, 0, 0))
            for ox, oy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                d.text((x + ox, y + oy), text, font=font, fill=(30, 30, 30))
            d.text((x, y), text, font=font, fill=fill)

        title_upper = title.upper()
        font_size = max(60, w // 7)
        font_title = _load_font(font_size)
        tw_px, _ = _text_size(draw, title_upper, font_title)
        while tw_px > w * 0.90 and font_size > 28:
            font_size -= 4
            font_title = _load_font(font_size)
            tw_px, _ = _text_size(draw, title_upper, font_title)

        title_lines = [title_upper]
        tw_check, _ = _text_size(draw, title_upper, font_title)
        if tw_check > w * 0.90:
            words = title_upper.split()
            mid = max(1, len(words) // 2)
            title_lines = [" ".join(words[:mid]), " ".join(words[mid:])]

        _, line_h = _text_size(draw, "Ag", font_title)
        scrim_top = h - scrim_h
        title_y = scrim_top + int(scrim_h * 0.30)
        for i, line in enumerate(title_lines):
            lw, _ = _text_size(draw, line, font_title)
            lx = (w - lw) // 2
            ly = title_y + i * (line_h + 8)
            _draw_shadow_text(draw, (lx, ly), line, font_title, fill=(255, 255, 255))

        # ── Save ─────────────────────────────────────────────────────────────
        cover_dir = os.path.join(st.project_dir, "cover")
        os.makedirs(cover_dir, exist_ok=True)
        cover_path = os.path.join(cover_dir, "cover.jpg")
        img.save(cover_path, format="JPEG", quality=95)
        return cover_path
    except Exception as _e:
        print(f"[_library_cover] failed: {_e}", flush=True)
        return None


def _generate_shorts_cover(st: "ProjectState", world_context: str = "") -> Optional[str]:
    """Generate a cinematic movie-poster cover (Marvel/Hollywood style) for a Shorts project.
    Overlays the story title with bold PIL text and saves to project_dir/cover/cover.jpg.
    Returns the saved path, or None on failure."""
    if fal_client is None or not os.getenv("FAL_KEY"):
        return None
    try:
        title = (st.project_name or "Untitled").strip()

        # Pull a few story details for a richer prompt
        char_desc = ""
        if st.characters:
            first_char = next(iter(st.characters.values()), {})
            char_desc = (
                f"{first_char.get('hair', '')} {first_char.get('build', '')} "
                f"{first_char.get('skin', '')} young man"
            ).strip()
        char_desc = char_desc or "lone hero"

        tone_hint = ""
        if st.beats:
            # Grab first and last beat for mood context
            tone_hint = f" Story tone: {st.beats[0][:80]}. Final moment: {st.beats[-1][:80]}."

        loc_hint = ""
        if st.locations:
            first_loc = next(iter(st.locations.values()), {})
            loc_hint = f" Setting: {str(first_loc.get('bible', ''))[:100]}." if first_loc.get('bible') else ""

        world_hint = f" {world_context.strip()[:120]}" if (world_context or "").strip() else ""

        cover_prompt = (
            "Epic cinematic movie poster, Hollywood blockbuster quality, photorealistic CGI render, "
            "dramatic studio lighting, deep shadows and vivid highlights, "
            "rich color grading (teal-orange or warm golden-hour), "
            "highly detailed, 8K resolution, professional poster photography. "
            "9:16 vertical format, full bleed edge-to-edge. "
            f"{char_desc} standing in a powerful hero pose, dominant foreground subject.{loc_hint}{tone_hint}{world_hint} "
            "Atmosphere: epic, intense, cinematic scale — the kind of image that makes you want to watch immediately. "
            "NO text, NO watermarks, NO logos, NO subtitles, NO foreign language characters. "
            "Pure image only — text will be added separately."
        )
        cover_neg = (
            "anime, manga, webtoon, cartoon, illustration, 2D art, flat colors, cel shading, "
            "ink lines, comic panels, speech bubbles, caption boxes, text, watermark, "
            "multiple panels, split screen, low quality, blurry, ugly, distorted, "
            "Korean text, Chinese text, Japanese text, hangul, kanji, foreign language"
        )

        img = call_fal_generate(cover_prompt, cover_neg, skip_esrgan=False)

        # ── Poster title overlay ──────────────────────────────────────────────
        try:
            from PIL import ImageDraw as _PDraw, ImageFont as _PFont, Image as _PImg
            import textwrap as _tw

            w, h = img.size

            # Find the best available bold font — try several common system paths
            _FONT_CANDIDATES = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
                "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
                "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
            ]
            def _load_font(size):
                for _fp in _FONT_CANDIDATES:
                    if os.path.isfile(_fp):
                        try:
                            return _PFont.truetype(_fp, size=size)
                        except Exception:
                            pass
                return _PFont.load_default()

            # ── Deep gradient scrim covering bottom 35% of image ──────────────
            scrim_h = int(h * 0.40)
            overlay = _PImg.new("RGBA", (w, h), (0, 0, 0, 0))
            ov_d = _PDraw.Draw(overlay)
            for row in range(scrim_h):
                # Ramp from transparent at top of scrim to near-black at bottom
                alpha = int(230 * (row / scrim_h) ** 1.4)
                ov_d.line([(0, h - scrim_h + row), (w, h - scrim_h + row)],
                          fill=(0, 0, 0, alpha))
            img = img.convert("RGBA")
            img = _PImg.alpha_composite(img, overlay)
            img = img.convert("RGB")
            draw = _PDraw.Draw(img)

            # ── Helper: measure text size ──────────────────────────────────────
            def _text_size(d, text, font):
                try:
                    bb = d.textbbox((0, 0), text, font=font)
                    return bb[2] - bb[0], bb[3] - bb[1]
                except AttributeError:
                    return d.textsize(text, font=font)

            # ── Helper: draw text with multi-layer shadow + stroke ─────────────
            def _draw_shadow_text(d, pos, text, font, fill=(255, 255, 255)):
                x, y = pos
                # Outer shadow (dark, offset)
                for ox, oy in [(-3, 3), (3, 3), (0, 5), (-5, 5), (5, 5)]:
                    d.text((x + ox, y + oy), text, font=font, fill=(0, 0, 0))
                # Inner glow / stroke
                for ox, oy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    d.text((x + ox, y + oy), text, font=font, fill=(30, 30, 30))
                d.text((x, y), text, font=font, fill=fill)

            # ── Title — wrap if too wide, auto-size font ───────────────────────
            title_upper = title.upper()
            font_size = max(60, w // 7)
            font_title = _load_font(font_size)
            tw_px, _ = _text_size(draw, title_upper, font_title)
            # Shrink until it fits with 5% padding per side
            while tw_px > w * 0.90 and font_size > 28:
                font_size -= 4
                font_title = _load_font(font_size)
                tw_px, _ = _text_size(draw, title_upper, font_title)

            # If still too wide, wrap into 2 lines
            title_lines = [title_upper]
            tw_check, th_check = _text_size(draw, title_upper, font_title)
            if tw_check > w * 0.90:
                words = title_upper.split()
                mid = max(1, len(words) // 2)
                title_lines = [" ".join(words[:mid]), " ".join(words[mid:])]

            _, line_h = _text_size(draw, "Ag", font_title)
            block_h = line_h * len(title_lines) + 8 * (len(title_lines) - 1)

            # Position: bottom 28% of image, vertically centered in scrim
            scrim_top = h - scrim_h
            title_y = scrim_top + int(scrim_h * 0.30)

            for i, line in enumerate(title_lines):
                lw, lh = _text_size(draw, line, font_title)
                lx = (w - lw) // 2
                ly = title_y + i * (line_h + 8)
                _draw_shadow_text(draw, (lx, ly), line, font_title, fill=(255, 255, 255))

            # ── Tagline — small italic-style subtitle ──────────────────────────
            if st.beats and len(st.beats) >= 3:
                # Use 2nd beat as tagline seed, keep it short
                tagline_raw = st.beats[1].strip()
                # strip leading camera tags like "ECU:", "MS:", etc.
                tagline_raw = re.sub(r"^[A-Z]{2,5}:\s*", "", tagline_raw)
                tagline_raw = re.sub(r"\[[A-Z]{2,5}\]\s*", "", tagline_raw)
                tagline = tagline_raw[:55].rstrip(" .,") + ("…" if len(tagline_raw) > 55 else "")
                font_tag = _load_font(max(22, w // 22))
                tw2, th2 = _text_size(draw, tagline, font_tag)
                tag_y = title_y + block_h + max(18, int(h * 0.025))
                tag_x = (w - tw2) // 2
                _draw_shadow_text(draw, (tag_x, tag_y), tagline, font_tag,
                                  fill=(220, 200, 160))  # warm gold tint

        except Exception:
            pass  # Image without overlay still ships

        cover_dir = os.path.join(st.project_dir, "cover")
        os.makedirs(cover_dir, exist_ok=True)
        cover_path = os.path.join(cover_dir, "cover.jpg")
        img.save(cover_path, format="JPEG", quality=95)
        return cover_path
    except Exception as _e:
        return None


def _call_claude_story_bible(st: ProjectState, story: str, beats: List[str], build_mode: str = "Panel", world_context: str = "") -> Optional[Dict[str, Any]]:
    system = (
        "You are a story-to-visual-bible planner for a manhwa director tool. "
        "Return ONLY valid JSON. No markdown. No commentary. "
        "Create specific, repeatable visual canon. Be decisive. Never use 'unknown'. "
        "Character signature_detail must be a permanent physical/accessory detail only. "
        "Do NOT put powers, auras, interfaces, glowing eyes, or temporary abilities inside signature_detail. "
        "Visible powers, interfaces, summoned effects, elemental effects, and system overlays must be separate items with kind='power_system'."
    )
    user = {
        "task": "Build the initial visual bible from the story.",
        "story": story,
        **({"world_context": world_context.strip()} if (world_context or "").strip() else {}),
        "beats_preview": beats[:20],
        "json_schema": {
            "characters": [{
                "name": "string",
                "character_type": "human/humanoid/animal/beast/spirit/orb/object",
                "gender": "male/female",
                "hair": "specific color+style",
                "eyes": "specific eye color",
                "build": "specific body build",
                "skin": "Korean/East Asian light skin (default) — ONLY use a darker tone if the story explicitly states it",
                "default_outfit": "specific repeatable everyday outfit description — what they wear MOST of the time in normal situations",
                "signature_detail": "permanent physical or accessory detail only",
                "alt_outfits": ["only if the story explicitly mentions a clothing change"],
                "forms": [
                    {"name": "base", "type": "base", "description": "specific everyday default outfit", "context_note": "worn when nothing special is happening", "activates_at_locations": []},
                    {"name": "example_form_name", "type": "outfit|partial_transform|full_transform", "description": "specific visual description", "context_note": "when and why this form appears", "activates_at_locations": ["LocationName1", "LocationName2"]}
                ],
            }],
            "locations": [{
                "name": "string",
                "bible_prompt": "specific location canon prompt",
                "ref_prompt_base": "specific location reference prompt",
                "sub_locations": [{
                    "name": "string",
                    "bible_prompt": "specific sub-location canon prompt",
                    "ref_prompt": "single consistent environment-only reference prompt"
                }]
            }],
            "items": [{
                "name": "string",
                "kind": "prop or power_system",
                "bible_prompt": "specific item or power appearance",
                "ref_prompt": "specific isolated reference prompt"
            }]
        },
        "rules": [
            "Be specific and visually repeatable.",
            "CRITICAL — Only list ACTUAL NAMED CHARACTERS: people, creatures, or entities that have a proper name and appear as individuals who act, speak, or are directly described in the story. Do NOT list generic nouns, concepts, elements, or environmental words as characters even if capitalised (e.g. Stone, Spirit, Shadow, Fire, Wind, Blood, System, World, Ancient, Master are NOT characters unless they are the character's actual proper name).",
            "Maximum 15 characters total. If the story has fewer named individuals, list fewer. Never pad the list with background characters, concepts, or unnamed roles.",
            "CRITICAL — Describe every character as they appear at the BEGINNING of the story, in their earliest/base form. Do NOT include scales, mutations, corruption marks, power awakening side effects, physical transformations, or any body change that happens mid-story. Those changes will be captured separately as transformation phases. If a character gains scales, a scar, a brand, or any other visible mark mid-story, their base description must NOT include it.",
            "SKIN TONE — This is a Korean manhwa. All human characters are Korean/East Asian by default. Unless the story text explicitly and directly states that a character has brown, dark, Black, South Asian, or non-East-Asian skin, you MUST use 'light skin' or 'warm fair skin'. Never infer a darker skin tone from occupation, lifestyle, setting, or any other indirect cue. Only override to a non-East-Asian tone when the story literally says so.",
            "Classify every character with character_type. Use 'human' for normal people with no supernatural traits. Use 'animal' for snakes, wolves, spiders, insects, fish, or any real-world creature species. Use 'beast' for fictional monsters, dragons, chimeras. Use 'spirit' only for ghosts/deities with no physical body. Use 'humanoid' for elves/orcs/androids. CRITICAL — REINCARNATION/TRANSFORMATION RULE: If the story says a character was reborn as, woke up as, became, or is now a specific creature (e.g. 'reborn as a snake', 'I became a slime', 'woke up as a wolf'), their character_type MUST be the creature type — NEVER 'human' — even if they are the protagonist or retain human memories. The protagonist of a 'reborn as snake' story is character_type='animal', NOT 'human'.",
            "Default gender to male unless the story clearly states female, even for spirits, animals, beasts, or object-like beings.",
            "For spirit or orb protagonists, prefer a stable humanoid male manifestation with masculine face, masculine clothing, and a readable consistent silhouette unless the story clearly says female or fully non-humanoid.",
            "OUTFIT INFERENCE — If the story does NOT explicitly describe a character's clothing: (1) Look at the dominant setting/environment they appear in most often. (2) Infer the most realistic outfit for that context: a student in a school with uniforms → school uniform; a worker at an office → business casual; a person in a fantasy village → simple tunic/peasant clothing; a person mostly at home → casual home clothes. (3) Once chosen, this becomes the 'base' form and must NOT randomly change — it stays consistent every page unless a form override applies. (4) Only deviate from base when the beat provides explicit evidence of a different outfit.",
            "Only add alt_outfits when the story explicitly says the clothes changed.",
            "NON-LINEAR STORY STRUCTURE — Many stories open with a 'hook' showing a future/present scene, then cut back with phrases like 'X days earlier', 'X days ago', 'X months before', 'flashback', 'that same morning'. CRITICAL RULE: The opening hook scene is NOT part of the flashback. The character's BASE form is what they wear during the HOOK / PRESENT-DAY scenes. The outfit they wear inside the flashback is a SEPARATE named form. Example: story opens at an arena (hook), then cuts to '32 days earlier' where the character streams in a maid costume — the maid costume is a 'streaming_costume' form that only activates during flashback stream scenes; the base form is what he wears at the arena (normal clothes). NEVER let a flashback outfit bleed into the hook scenes.",
            "FORMS — Every character MUST have a 'forms' list. Rules: (1) The FIRST form MUST be named 'base' (type='base') — this is the character's TRUE default look for the MAJORITY of normal scenes, inferred from dominant setting if not stated explicitly. (2) CRITICAL: A performance costume, streaming outfit, stage costume, uniform worn for a stunt, or any clothing worn specifically FOR a single event/broadcast/performance/act is ABSOLUTELY NEVER the base form, even if it appears on the very first page of the story. The base form is always what the character wears in ordinary daily life OUTSIDE of that event — casual home clothes, everyday clothes, or their dominant setting outfit. Example: a streamer who wears a maid costume only during streams has a base form of casual home clothes (t-shirt, sweats), and a separate 'streaming_costume' form. (3) If the story begins mid-event (at an arena, a stage, a performance), the outfit worn AT that event is NOT the base — look for what the character normally wears outside that event. (4) Type rules: 'partial_transform' layers ON TOP of clothing (wings from back, claws extended — clothes still visible); 'full_transform' REPLACES everything (full armor, full beast body — no clothing visible); 'outfit' is a different set of clothes for specific situations. (5) For location-specific outfits (e.g., school uniform only when AT school, formal wear only at parties), set 'activates_at_locations' to the relevant location names — the beat planner will auto-select this form when the character is at those locations. (6) If a character never changes appearance, include only the base form. (7) Do NOT put powers, auras, or elemental effects in forms.",
            "Locations should have fixed architecture/material/furniture/layout details.",
            "Provide useful sub_locations (hallway, kitchen, classroom, rooftop, etc.) and keep each one visually stable.",
            "Power systems should describe exact glow color, shape language, and activation look.",
            "Keep item list lean and useful. Items are physical objects or power effects — weapons, tools, artifacts, magical interfaces. Do NOT put character names in the items list. If something is already in the characters list, it must NOT also appear in items.",
            "CRITICAL — item ref_prompt must show ONLY the isolated item, object, or power effect on a plain background. NO people, NO faces, NO characters, NO portraits. The ref_prompt for a 'book' should show a book; for a 'sword' show a sword; for a power system show the isolated glow/effect. Never describe a person holding or using the item.",
        ] + (
            [
                "SHORTS MODE — This is a 1–2 minute short story. The source text may be very brief (even just a few sentences). "
                "You MUST invent rich, specific cinematic visual detail even where the story does not spell it out. "
                "Every character must have a visually unique, immediately recognisable silhouette: specific hair colour and cut, a distinctive outfit with exact colours and textures, and a strong signature accessory or feature. "
                "Every location must have a strong atmospheric colour palette, specific materials and lighting direction (e.g. 'rain-slicked cobblestones under amber street lamps', not just 'alley'). "
                "If the story only describes mood or action without specifying appearance, infer the most cinematic, high-contrast visual that fits — then lock it down as canon. "
                "Make it stunning. Behave as if you are the art director for a premium animated short, not just extracting facts from a script. "
                "SHORTS CAST COMPLETENESS — Include ALL characters in the characters list who appear in more than one sentence OR who participate in any action, speak, or are directly interacted with — "
                "even if they are referred to only by role or descriptor (e.g. 'the guard', 'the butcher', 'the old woman') and have no proper name. "
                "In a short story every person who does something matters. Do not omit a character just because they have little detail — invent a distinctive visual for them.",

                "CULTURAL SETTING (SHORTS) — MANDATORY: Unless the story explicitly states otherwise, all Shorts stories "
                "are set in a Korean or Chinese cultural world. Modern settings default to Seoul-style streets: "
                "narrow alleys with hangul neon signs, pojangmacha food stalls, PC bangs, convenience stores with Korean/Chinese "
                "branding, traditional tile-roofed buildings alongside glass towers. Fantasy/historical settings default to "
                "East Asian architecture — wooden pagodas, stone courtyards, silk-draped halls. "
                "NEVER default to New York, generic American city, European streets, or Western locations unless the story explicitly says so.",

                "CHARACTER APPEARANCE (SHORTS) — MANDATORY: All human characters default to Korean or Chinese appearance. "
                "Hair MUST be black or very dark brown by default — only use lighter hair if the story explicitly states it. "
                "Facial features must read as East Asian: sharp defined eyes with manhwa styling (single or double-lid), "
                "high cheekbones, smooth East Asian skin tone (light warm or fair). "
                "Outfits: contemporary Korean streetwear (slim-cut joggers, oversized hoodies, school uniforms, track jackets) "
                "or Chinese-influenced clothing depending on setting — NOT American hoodies with English text, baggy Western jeans, "
                "or generic Western casual wear unless the story explicitly calls for it.",
            ]
            if build_mode == "Shorts" else []
        ),
    }
    return _call_claude_json(system, user, max_tokens=8096, model=CLAUDE_MODEL)


def _call_claude_names_only(st: "ProjectState", story: str, _log=None) -> Optional[Dict[str, Any]]:
    """Lightweight Claude Haiku call — just extract character names and locations.
    Used as fallback when the full Sonnet bible call fails or is truncated."""
    system = (
        "You are a character and location extractor. Return ONLY valid JSON. No markdown. No commentary."
    )
    user = {
        "task": "Extract every actual named character and every named location from this story.",
        "story": story[:8000],  # truncate input if very long
        "json_schema": {
            "characters": [{"name": "proper name only", "gender": "male/female", "character_type": "human/humanoid/animal/beast/spirit/orb/object"}],
            "locations": [{"name": "place name"}],
            "items": [],
        },
        "rules": [
            "Characters must be INDIVIDUALS with their own proper name (e.g. Malachar, Elena, Kazimir).",
            "Do NOT include generic creature types (vampire, demon, hunter, elder, fledgling), titles (lord, king, count), "
            "adjectives (crimson, dark, good), group nouns (clan, horde, council), or any common English word.",
            "A word is a character name ONLY if it is used in the story as that specific individual's personal identifier.",
            "Locations are named places where scenes happen (room, alley, school, city name, etc.).",
            "Return an empty list if genuinely none found — never pad with made-up entries.",
        ],
    }
    return _call_claude_json(system, user, max_tokens=2000, model=CLAUDE_FAST_MODEL, _log=_log)


def _fallback_story_bible(st: "ProjectState", story: str, _log=None) -> None:
    """Last-resort fallback: try Claude Haiku for names, then build profiles from those names.
    Only falls back to the regex extractor if Haiku also fails."""
    haiku_result = _call_claude_names_only(st, story, _log=_log)
    if haiku_result and (haiku_result.get("characters") or haiku_result.get("locations")):
        if _log:
            _log("STEP 3b: Haiku name extraction succeeded — building profiles from names")
        chars_raw = haiku_result.get("characters") or []
        locs_raw = haiku_result.get("locations") or []
        _blocked = {w.lower() for w in GENERIC_NOUN_BLOCKLIST} | {w.lower() for w in PLACE_WORD_BLOCKLIST}
        char_names = [
            clean_for_prompt(str(c.get("name") or ""))
            for c in chars_raw
            if clean_for_prompt(str(c.get("name") or "")).lower() not in _blocked
        ]
        char_names = [n for n in char_names if n]
        loc_names = [
            clean_for_prompt(str(l.get("name") or ""))
            for l in locs_raw
            if clean_for_prompt(str(l.get("name") or ""))
        ]
        if char_names:
            st.characters = {}
            for c in chars_raw:
                name = clean_for_prompt(str(c.get("name") or ""))
                if not name or name.lower() in _blocked:
                    continue
                gender = str(c.get("gender") or "male").lower()
                ctype = str(c.get("character_type") or "human").lower()
                if ctype not in CHARACTER_TYPE_CHOICES:
                    ctype = "human"
                st.characters[name] = build_character_profile(st.project_id, name, story, force_gender=gender, force_character_type=ctype)
        st.locations = {n: build_location_profile(n) for n in loc_names}
        power_items = extract_power_items(story)
        st.items = {}
        for n in power_items:
            st.items[n] = build_item_profile(n, kind="power_system")
        return

    # True last resort — regex only (should rarely be reached)
    if _log:
        _log("STEP 3b: Haiku also failed — using regex extractor (least accurate)")
    loc_names = extract_locations(story)
    char_names = extract_characters(story, loc_names)
    if not char_names:
        inferred = detect_first_person_gender(story)
        force_gender = inferred if inferred else "male"
        st.characters = {"Protagonist": build_character_profile(st.project_id, "Protagonist", story, force_gender=force_gender or "male", force_character_type="humanoid")}
    else:
        st.characters = {n: build_character_profile(st.project_id, n, story) for n in char_names}
    st.locations = {n: build_location_profile(n) for n in loc_names}
    prop_names = extract_important_props(story, top_n=10)
    for p in extract_power_items(story):
        if p not in prop_names:
            prop_names.append(p)
    items = {}
    for n in prop_names:
        kind = "power_system" if any(k in n.lower() for k in ["aura", "interface", "mana", "storage", "inventory", "summoned"]) else "prop"
        items[n] = build_item_profile(n, kind=kind)
    st.items = items


def _call_claude_batch_beat_plans(st: ProjectState, start_index: int, beats_batch: List[str], prior_location: str, prior_active_forms: Optional[Dict[str, str]] = None) -> Dict[int, Dict[str, Any]]:
    # Identify protagonist — first character in the ordered dict
    char_names = list((st.characters or {}).keys())
    protagonist = char_names[0] if char_names else ""

    system = (
        "You are a beat planner for a manhwa director tool. Return ONLY valid JSON. No markdown. No commentary. "
        f"The protagonist is '{protagonist}'. "
        "Maintain location continuity: keep the current place until the story clearly moves somewhere else. "
        "The action line should describe what the character DOES or FEELS — keep it concise and visual. Do NOT include the location name in the action; location is handled separately. "
        "For active_forms: assign a form name from each character's forms list, or 'base' for their default state. "
        "LOCATION RULE: if a form has activates_at_locations matching the current beat's location, prefer that form over 'base' automatically — even without explicit mention in the beat text. "
        "EXPLICIT FORM RULE: only activate non-location, non-base forms (transforms, special costumes) when the beat EXPLICITLY shows that form in use. "
        "ACQUISITION RULE: Only assign a full_transform or partial_transform form when the beat confirms the character has ALREADY acquired that ability. If the character is currently IN THE PROCESS of earning the ability (fighting, struggling, being offered something), they do NOT yet have it — keep 'base' or their current non-transform form. "
        "DEACTIVATION RULE: if a transform/special form was active and the current beat shows the character resting, at home, eating, sleeping, or in any clearly casual scene — set them back to 'base' (or the location-appropriate form if applicable). "
        "CONTINUITY RULE: If no explicit change is shown in this beat, CARRY FORWARD the active_forms from the immediately prior beat. Only switch a character's form when the beat clearly shows a change happening. "
        "FLASHBACK RULE: If the beat contains a time-skip phrase ('X days earlier', 'X days ago', 'X months before', 'flashback', 'back then', 'that morning', 'years before') it signals a scene change to a DIFFERENT time period. The outfit for flashback scenes comes from what the story describes for that time period — it does NOT carry forward from the preceding present-day scenes. Likewise, when the story returns to the present after a flashback, revert to the character's base form unless the present-day context explicitly states otherwise. "
        "Use attacker->target phrasing for action scenes to avoid subject confusion. "
        "CRITICAL — suggested_characters: include the protagonist in MOST beats. "
        "The protagonist is present whenever the beat uses first-person narration ('I', 'me', 'my', 'myself'), third-person pronouns ('he', 'she', 'they', 'him', 'her', 'his'), "
        "or any action/emotion that implies a subject. "
        "Only use an empty suggested_characters list for beats that are PURELY environment or world description with ZERO character reference — no pronouns, no named person, no implied subject. "
        "When in doubt, include the protagonist. "
        "Use the character descriptions and location descriptions provided to make accurate decisions — do not guess."
    )

    # Attach 3 context beats immediately before this batch so Claude understands flow
    context_window = 3
    context_start = max(1, start_index - context_window)
    context_beats = (st.beats or [])[context_start - 1: start_index - 1]

    user = {
        "task": "Plan multiple beats at once.",
        "prior_location_before_batch": prior_location or "",
        "prior_active_forms_before_batch": prior_active_forms or {},
        "context_beats_before_batch": [
            {"beat_index": context_start + i, "beat_text": b}
            for i, b in enumerate(context_beats)
        ],
        "beats": [{"beat_index": start_index + i, "beat_text": b} for i, b in enumerate(beats_batch)],
        "available_characters": [
            {
                "name": name,
                "role": "protagonist" if name == protagonist else "supporting",
                "type": ((cdata.get("fields") or {}).get("character_type") or "human"),
                "gender": ((cdata.get("fields") or {}).get("gender") or ""),
                "description": (cdata.get("dna_prompt") or "")[:150],
                "forms": [
                    {
                        "name": f["name"],
                        "type": f["type"],
                        "context_note": f.get("context_note", ""),
                        "activates_at_locations": f.get("activates_at_locations") or [],
                    }
                    for f in (cdata.get("forms") or [])
                ],
            }
            for name, cdata in (st.characters or {}).items()
        ],
        "available_locations": [
            {
                "name": name,
                "description": (ldata or {}).get("bible_prompt", "")[:150],
                "sub_locations": list(((ldata or {}).get("sub_locations") or {}).keys()),
            }
            for name, ldata in (st.locations or {}).items()
        ] + [{"name": "None", "sub_locations": []}],
        "available_items": [{"name": k, "kind": (v or {}).get("kind", "prop")} for k, v in (st.items or {}).items()],
        "allowed_scene_types": SCENE_TYPES,
        "allowed_camera_types": CAMERA_TYPES,
        "json_schema": {
            "plans": [{
                "beat_index": "integer",
                "suggested_location": "string or None",
                "suggested_sub_location": "string",
                "scene_type": "one of allowed_scene_types",
                "camera_type": "one of allowed_camera_types",
                "suggested_characters": ["names"],
                "suggested_items": ["names"],
                "suggested_action": "one concise visual sentence",
                "attacker_name": "string or empty",
                "target_name": "string or empty",
                "character_emotions": {"Character": "emotion + expression"},
                "active_forms": {"Character": "form_name from that character's forms list, or 'base' for default"}
            }]
        }
    }
    data = _call_claude_json(system, user, max_tokens=3500, model=CLAUDE_FAST_MODEL)
    out: Dict[int, Dict[str, Any]] = {}
    if not data:
        return out
    plans = data.get("plans") or []
    if not isinstance(plans, list):
        return out

    # Build lookup sets from the project — used to snap Claude's output back to
    # the exact keys it was given, in case of casing/spacing/abbreviation drift.
    valid_chars: set = set((st.characters or {}).keys())
    valid_locs: set = set((st.locations or {}).keys())

    def _snap_name(name: str, valid: set) -> str:
        """Return the exact key from `valid` that best matches `name`, or ''."""
        if not name:
            return ""
        if name in valid:
            return name
        nl = name.lower()
        # 1. exact case-insensitive
        m = next((v for v in valid if v.lower() == nl), None)
        if m:
            return m
        # 2. substring — name is contained in key or key is contained in name (min 3 chars)
        if len(nl) >= 3:
            m = next((v for v in valid if nl in v.lower() or v.lower() in nl), None)
        return m or ""

    for row in plans:
        try:
            idx = int(row.get("beat_index"))
        except Exception:
            continue

        # Snap character names to exact project keys
        raw_chars = row.get("suggested_characters") or []
        snapped_chars = [r for r in (_snap_name(c, valid_chars) for c in raw_chars) if r]
        # Deduplicate while preserving order
        seen: set = set()
        row["suggested_characters"] = [c for c in snapped_chars if not (c in seen or seen.add(c))]  # type: ignore[func-returns-value]

        # Snap location to exact project key
        raw_loc = str(row.get("suggested_location") or "None")
        if raw_loc and raw_loc != "None":
            snapped_loc = _snap_name(raw_loc, valid_locs)
            row["suggested_location"] = snapped_loc if snapped_loc else "None"
        else:
            row["suggested_location"] = "None"

        # Snap attacker/target names
        for field in ("attacker_name", "target_name"):
            raw = str(row.get(field) or "")
            row[field] = _snap_name(raw, valid_chars) if raw else ""

        out[idx] = row
    return out


def _classify_scene_type(beat_text: str) -> str:
    low = (beat_text or "").lower()
    if any(k in low for k in ["hit", "punch", "kick", "slash", "stab", "shoot", "attack", "fight", "crash", "smash"]):
        return "ACTION"
    if any(k in low for k in ["system", "level up", "notification", "quest", "awaken", "awakens", "status window"]):
        return "AWAKENING"
    if any(k in low for k in ["blood", "corpse", "body", "silent", "aftermath", "ruins", "smoke"]):
        return "AFTERMATH"
    if any(k in low for k in ["remember", "memory", "flashback", "used to", "once"]):
        return "MEMORY"
    if any(k in low for k in ["says", "asks", "replies", "whispers", "talks", "dialogue", "conversation"]):
        return "DIALOGUE"
    return "EMOTION"


def _choose_camera_type(scene_type: str, beat_index: int, seed_text: str = "") -> str:
    pool = SCENE_CAMERA_POOLS.get(scene_type) or CAMERA_TYPES
    seed = _stable_int(f"{scene_type}:{beat_index}:{seed_text}") % max(len(pool), 1)
    return pool[seed]


def _infer_sub_location(st: ProjectState, location_name: str, beat_text: str) -> str:
    if not location_name or location_name == "None":
        return ""
    subs = (st.locations.get(location_name) or {}).get("sub_locations") or {}
    if not subs:
        return ""
    low = (beat_text or "").lower()
    for sub_name in subs.keys():
        if sub_name.lower() in low:
            return sub_name
    return list(subs.keys())[0]


def _infer_attacker_target(chars: List[str], beat_text: str) -> Tuple[str, str]:
    if len(chars) >= 2:
        return chars[0], chars[1]
    return "", ""


def _heuristic_beat_plan(st: ProjectState, beat_text: str, beat_index: int, prior_location: str = "", prior_active_forms: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    low = (beat_text or "").lower()
    suggested_characters = [name for name in (st.characters or {}).keys() if name.lower() in low]
    if not suggested_characters:
        # Only fall back to the single main character if the beat text implies a person is
        # physically present — personal pronouns suggest a character even if not named.
        # Pure location/environment beats (no pronouns, no action subject) stay empty.
        # Prepend a space so " i " catches first-person "I" at sentence start too
        _low_padded = " " + low + " "
        has_pronoun = any(p in _low_padded for p in [
            " i ", " i'", " he ", " she ", " they ", " his ", " her ", " their ", " him ",
            "himself", "herself", "myself",
            "the boy", "the girl", "the man", "the woman", "the child", "the infant",
            "the protagonist", "the young", "the old",
        ])
        if has_pronoun and (len(st.characters or {}) == 1 or has_pronoun):
            suggested_characters = [list(st.characters.keys())[0]]
    suggested_location = ""
    for loc_name in (st.locations or {}).keys():
        if loc_name.lower() in low:
            suggested_location = loc_name
            break
    if not suggested_location:
        suggested_location = prior_location or (list(st.locations.keys())[0] if st.locations else "None")
    suggested_sub_location = _infer_sub_location(st, suggested_location, beat_text)
    scene_type = _classify_scene_type(beat_text)
    camera_type = _choose_camera_type(scene_type, beat_index, beat_text)
    suggested_items = [name for name in (st.items or {}).keys() if name.lower() in low]
    emotions = {n: "focused, controlled expression" for n in suggested_characters}
    attacker, target = _infer_attacker_target(suggested_characters, beat_text)
    action = clean_for_prompt(beat_text)
    if scene_type == "ACTION" and attacker and target:
        action = f"{attacker} striking {target} directly in front of {attacker}"
    # Determine active form for each suggested character.
    # Priority order: explicit keyword match > location match > carry-forward from prior beat > base.
    # Deactivation: calm/home/rest/sleep/eat keywords → force "base" (or location form), ignoring carry-forward.
    _deactivation_keywords = ["rest", "resting", "sleep", "sleeping", "home", "house", "apartment",
                               "relaxing", "eating", "meal", "breakfast", "lunch", "dinner", "casual", "calm down"]
    _is_deactivating = any(k in low for k in _deactivation_keywords)
    _prior = prior_active_forms or {}
    active_forms = {}
    for c in suggested_characters:
        forms = (st.characters.get(c) or {}).get("forms") or []
        valid_form_names = {"base"} | {f["name"] for f in forms if f.get("name")}
        # Sanitize carry-forward — only trust it if the form name is still valid
        prior_form = _prior.get(c, "base")
        if prior_form not in valid_form_names:
            prior_form = "base"
        selected = prior_form  # default: carry forward unless something overrides
        if forms:
            # 1. Location-based auto-selection
            location_form = None
            for f in forms:
                if (f.get("name") or "base") == "base":
                    continue
                locs_lower = [(l or "").lower() for l in (f.get("activates_at_locations") or [])]
                if suggested_location and any(suggested_location.lower() == ll or suggested_location.lower() in ll for ll in locs_lower):
                    location_form = f["name"]
                    break
            # 2. Explicit keyword match
            if not _is_deactivating:
                explicit_form = None
                for f in forms:
                    if (f.get("name") or "base") == "base":
                        continue
                    fname_low = (f.get("name") or "").lower().replace("_", " ").replace("-", " ")
                    if fname_low and fname_low in low:
                        explicit_form = f["name"]
                        break
                if explicit_form:
                    selected = explicit_form
                elif location_form:
                    selected = location_form
                # else: keep carry-forward (prior_form) — no change in this beat
            else:
                # Deactivating — location form wins if applicable, else hard-reset to base
                selected = location_form if location_form else "base"
        active_forms[c] = selected
    return {
        "suggested_location": suggested_location or "None",
        "suggested_sub_location": suggested_sub_location,
        "scene_type": scene_type,
        "camera_type": camera_type,
        "suggested_perspective": camera_type,  # legacy compatibility
        "suggested_characters": suggested_characters,
        "suggested_items": suggested_items,
        "suggested_action": action,
        "attacker_name": attacker,
        "target_name": target,
        "character_emotions": emotions,
        "active_forms": active_forms,
        "outfit_overrides": {},  # legacy key kept for backward compat with old UI reads
        "plan_source": "heuristic",
    }


def _sanitize_plan(st: ProjectState, plan: Dict[str, Any], beat_text: str) -> Dict[str, Any]:
    valid_chars = set((st.characters or {}).keys())
    valid_locs = set(["None"] + list((st.locations or {}).keys()))
    valid_items = set((st.items or {}).keys())
    loc = clean_for_prompt(str(plan.get("suggested_location") or "None")) or "None"
    if loc not in valid_locs:
        loc = "None"
    sub_loc = clean_for_prompt(str(plan.get("suggested_sub_location") or ""))
    if loc != "None":
        valid_subs = set(((st.locations.get(loc) or {}).get("sub_locations") or {}).keys())
        if sub_loc not in valid_subs:
            sub_loc = (list(valid_subs)[0] if valid_subs else "")
    else:
        sub_loc = ""
    scene_type = clean_for_prompt(str(plan.get("scene_type") or _classify_scene_type(beat_text))).upper()
    if scene_type not in SCENE_TYPES:
        scene_type = _classify_scene_type(beat_text)
    camera_type = clean_for_prompt(str(plan.get("camera_type") or plan.get("suggested_perspective") or ""))
    if camera_type not in CAMERA_TYPES:
        camera_type = _choose_camera_type(scene_type, _stable_int(beat_text) % 1000, beat_text)
    chars = [c for c in (plan.get("suggested_characters") or []) if c in valid_chars]
    items = [i for i in (plan.get("suggested_items") or []) if i in valid_items]
    attacker = clean_for_prompt(str(plan.get("attacker_name") or ""))
    target = clean_for_prompt(str(plan.get("target_name") or ""))
    if attacker and attacker not in chars:
        attacker = ""
    if target and target not in chars:
        target = ""
    action = clean_for_prompt(str(plan.get("suggested_action") or beat_text))
    if scene_type == "ACTION" and attacker and target:
        action = f"{attacker} striking {target} directly in front of {attacker}"
    if loc != "None" and loc.lower() not in action.lower():
        if sub_loc:
            action = f"In the {sub_loc} of the {loc}, {action}"
        else:
            action = f"In the {loc}, {action}"
    if loc == "None":
        for ln in st.locations.keys():
            action = re.sub(rf"\b{re.escape(ln)}\b", "", action, flags=re.IGNORECASE)
        action = clean_for_prompt(re.sub(r"\s+,", ",", action))
    raw_emotions = plan.get("character_emotions") or {}
    if not isinstance(raw_emotions, dict):
        raw_emotions = {}
    emotions = {c: clean_for_prompt(str(raw_emotions.get(c) or "focused, controlled expression")) for c in chars}
    raw_outfits = plan.get("outfit_overrides") or {}
    if not isinstance(raw_outfits, dict):
        raw_outfits = {}
    outfits = {}
    low = (beat_text or "").lower()
    explicit_change = any(k in low for k in ["change clothes", "changed clothes", "changed into", "put on", "wearing a different", "uniform"])
    if explicit_change:
        for c in chars:
            v = clean_for_prompt(str(raw_outfits.get(c) or ""))
            if v:
                outfits[c] = v
    # Carry through active_forms from Claude's response (or heuristic), validated against project forms
    raw_active_forms = plan.get("active_forms") or {}
    if not isinstance(raw_active_forms, dict):
        raw_active_forms = {}
    valid_form_names: Dict[str, set] = {}
    for cn in (st.characters or {}).keys():
        forms_list = ((st.characters or {}).get(cn) or {}).get("forms") or []
        valid_form_names[cn] = {"base"} | {f["name"] for f in forms_list if f.get("name")}
    active_forms_out: Dict[str, str] = {}
    for cn in chars:
        raw_fn = clean_for_prompt(str(raw_active_forms.get(cn) or "base")) or "base"
        # Snap to valid form name; fall back to "base" if unrecognised
        if raw_fn in (valid_form_names.get(cn) or {"base"}):
            active_forms_out[cn] = raw_fn
        else:
            active_forms_out[cn] = "base"
    return {
        "suggested_location": loc,
        "suggested_sub_location": sub_loc,
        "scene_type": scene_type,
        "camera_type": camera_type,
        "suggested_perspective": camera_type,
        "suggested_characters": chars,
        "suggested_items": items,
        "suggested_action": action,
        "attacker_name": attacker,
        "target_name": target,
        "character_emotions": emotions,
        "outfit_overrides": outfits,
        "active_forms": active_forms_out,
        "plan_source": clean_for_prompt(str(plan.get("plan_source") or "claude")) or "claude",
    }


def _precompute_beat_plans(st: ProjectState, batch_size: int, progress=None, log=None, use_sonnet: bool = True) -> None:
    st.beat_plans = {}
    prior_location = ""
    prior_active_forms: Dict[str, str] = {}
    total = max(len(st.beats), 1)
    for start in range(0, len(st.beats), batch_size):
        batch = st.beats[start:start + batch_size]
        batch_start_idx = start + 1
        batch_end_idx = start + len(batch)
        if progress:
            progress(0.35 + 0.50 * (batch_end_idx / total), desc=f"Planning beats {batch_start_idx:03d}-{batch_end_idx:03d}...")
        if log:
            log(f"STEP 6: Planning beats {batch_start_idx:03d}-{batch_end_idx:03d}")
        rows = _call_claude_batch_beat_plans(st, batch_start_idx, batch, prior_location, prior_active_forms=prior_active_forms) if use_sonnet else {}
        for rel_i, beat_text in enumerate(batch):
            idx = batch_start_idx + rel_i
            raw = rows.get(idx) if rows else None
            plan = _sanitize_plan(st, raw or _heuristic_beat_plan(st, beat_text, idx, prior_location=prior_location, prior_active_forms=prior_active_forms), beat_text)
            st.beat_plans[idx] = plan
            if plan.get("suggested_location") and plan.get("suggested_location") != "None":
                prior_location = plan["suggested_location"]
            # Carry forward any active forms set in this beat for the next beat
            if plan.get("active_forms"):
                prior_active_forms = {**prior_active_forms, **plan["active_forms"]}




def _compose_sonnet_prompt_fallback(st: ProjectState, beat_text: str, plan: Dict[str, Any]) -> str:
    chars = list(plan.get("suggested_characters") or [])
    loc = clean_for_prompt(str(plan.get("suggested_location") or ""))
    sub_loc = clean_for_prompt(str(plan.get("suggested_sub_location") or ""))
    camera = clean_for_prompt(str(plan.get("camera_type") or "dramatic medium character shot"))
    scene_type = clean_for_prompt(str(plan.get("scene_type") or "EMOTION"))
    action = clean_for_prompt(str(plan.get("suggested_action") or beat_text))
    emotions = plan.get("character_emotions") or {}
    lines = [DEFAULT_STYLE]
    if loc and loc != "None" and loc in (st.locations or {}):
        loc_obj = st.locations.get(loc) or {}
        sub_obj = ((loc_obj.get("sub_locations") or {}).get(sub_loc) or {}) if sub_loc and sub_loc != "None" else {}
        env = clean_for_prompt(str(sub_obj.get("bible_prompt") or loc_obj.get("bible_prompt") or loc))
        if env:
            lines.append(env)
    lines.append(camera)
    for name in chars:
        c = (st.characters or {}).get(name) or {}
        dna = clean_for_prompt(str(c.get("dna_prompt") or ""))
        if dna:
            lines.append(dna)
        emo = clean_for_prompt(str(emotions.get(name) or ""))
        if emo:
            lines.append(f"{name} expression: {emo}")
    for item_name in (plan.get("suggested_items") or []):
        item = (st.items or {}).get(item_name) or {}
        bible = clean_for_prompt(str(item.get("bible_prompt") or ""))
        if bible:
            lines.append(bible)
    if action:
        lines.append(action)
    lines.append(
        f"scene type: {scene_type.lower()}, "
        "bold ink line art, deep shadow pools contrasting vivid highlights, "
        "dramatic high-contrast cel shading, vivid saturated color, "
        "cinematic asymmetric composition, strong depth, foreground framing element, "
        "highly expressive face and eyes, 2D Korean manhwa webtoon illustration"
    )
    return clean_for_prompt("\n".join([x for x in lines if clean_for_prompt(x)]))


def _call_claude_batch_image_prompts(st: ProjectState, start_index: int, beats_batch: List[str]) -> Dict[int, Dict[str, Any]]:
    api_key = (os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY") or "").strip()
    if not api_key:
        return {}
    plans_payload = []
    for rel_i, beat_text in enumerate(beats_batch):
        idx = start_index + rel_i
        plan = (st.beat_plans or {}).get(idx) or _heuristic_beat_plan(st, beat_text, idx)
        plans_payload.append({
            "beat_index": idx,
            "beat_text": beat_text,
            "scene_type": plan.get("scene_type", "EMOTION"),
            "camera_type": plan.get("camera_type", "dramatic medium character shot"),
            "location": plan.get("suggested_location", "None"),
            "sub_location": plan.get("suggested_sub_location", "None"),
            "characters": list(plan.get("suggested_characters") or []),
            "items": list(plan.get("suggested_items") or []),
            "action": plan.get("suggested_action") or beat_text,
            "emotions": plan.get("character_emotions") or {},
        })
    system = (
        "You are the main image prompt director for a Korean manhwa / webtoon storyboard tool. "
        "Return ONLY valid JSON. No markdown. No commentary. "
        "For each beat, write one final production-ready manhwa panel prompt that is visually concrete and cinematic. "
        "\n\n"
        "=== MANHWA VISUAL DNA — MANDATORY IN EVERY PROMPT ===\n"
        "Open every prompt with: 'Korean manhwa webtoon art style, bold expressive line art with dynamic weight variation, "
        "dramatic high-contrast cel shading with deep shadow pools, vivid saturated color palette, 2D illustrated not photorealistic'\n"
        "Then build the scene description:\n"
        "LIGHTING (always name one dominant source): amber candlelight / cold blue window light / hard side rim light / "
        "neon glow on wet pavement / explosion flash backlight / volumetric shaft of light / harsh overhead fluorescent / "
        "golden hour backlight / moonlight with deep shadow pools\n"
        "COLOR: vivid and saturated — intentional warm-cool contrast — never desaturated, muddy, or washed out\n"
        "EYES: describe them specifically — 'luminous irises with sharp catchlights, expressive lashes, emotion visible in gaze'\n"
        "SHADING: 'deep shadow areas contrasting with bright vivid highlights' — never flat or pastel\n"
        "COMPOSITION: cinematic asymmetric — foreground framing elements, strong depth, dynamic perspective angle\n"
        "\n"
        "Never use: 'masterpiece', 'best quality', '4k', 'ultra HD' — Flux ignores these. Describe what you SEE.\n"
        "Make camera type materially change the framing (low angle attack / extreme close-up / wide establishing / OTS).\n"
        "Default unclear beings to humanoid. Default unspecified gender to male. "
        "Keep male characters visually male unless the story explicitly requires otherwise."
    )
    user = {
        "style": DEFAULT_STYLE,
        "beats": plans_payload,
        "json_schema": {
            "prompts": [{
                "beat_index": "integer",
                "prompt": "string",
                "negative": "string",
                "summary": "short string"
            }]
        },
        "rules": [
            "One prompt per beat.",
            "Use anime/manhwa 2d illustrated language, not photorealistic language.",
            "The prompt must reflect the exact moment to be shown in the image.",
            "Keep the framing specific to the chosen camera_type.",
            "Do not add text overlays unless the beat explicitly requires them.",
            "Negative prompt should stay concise and focused on common image defects and photorealism drift."
        ]
    }
    data = _call_claude_json(system, user, max_tokens=2500, model=CLAUDE_FAST_MODEL)
    out: Dict[int, Dict[str, Any]] = {}
    if not data:
        return out
    rows = data.get("prompts") or []
    if not isinstance(rows, list):
        return out
    for row in rows:
        try:
            idx = int(row.get("beat_index"))
        except Exception:
            continue
        out[idx] = {
            "prompt": clean_for_prompt(str(row.get("prompt") or "")),
            "negative": clean_for_prompt(str(row.get("negative") or DEFAULT_NEGATIVE)) or DEFAULT_NEGATIVE,
            "summary": clean_for_prompt(str(row.get("summary") or "")),
            "source": "deepseek_v4_1_flash" if _rp.is_deepseek_mode() else "sonnet_batch",
        }
    return out


def _precompute_image_prompts(st: ProjectState, batch_size: int, progress=None, log=None, use_sonnet: bool = True) -> None:
    st.image_prompts = {}
    total = max(len(st.beats), 1)
    for start in range(0, len(st.beats), batch_size):
        batch = st.beats[start:start + batch_size]
        batch_start_idx = start + 1
        batch_end_idx = start + len(batch)
        if progress:
            progress(0.86 + 0.12 * (batch_end_idx / total), desc=f"Generating prompts {batch_start_idx:03d}-{batch_end_idx:03d}...")
        if log:
            log(f"STEP 7: Generating Sonnet prompts {batch_start_idx:03d}-{batch_end_idx:03d}")
        rows = _call_claude_batch_image_prompts(st, batch_start_idx, batch) if use_sonnet else {}
        for rel_i, beat_text in enumerate(batch):
            idx = batch_start_idx + rel_i
            plan = (st.beat_plans or {}).get(idx) or _heuristic_beat_plan(st, beat_text, idx)
            prompt_row = rows.get(idx) or {}
            prompt_text = clean_for_prompt(str(prompt_row.get("prompt") or ""))
            if not prompt_text:
                prompt_text = _compose_sonnet_prompt_fallback(st, beat_text, plan)
                source = "local_fallback"
            else:
                source = prompt_row.get("source") or "sonnet_batch"
            st.image_prompts[idx] = {
                "prompt": prompt_text,
                "negative": clean_for_prompt(str(prompt_row.get("negative") or DEFAULT_NEGATIVE)) or DEFAULT_NEGATIVE,
                "summary": clean_for_prompt(str(prompt_row.get("summary") or plan.get("suggested_action") or beat_text)),
                "source": source,
                "beat_text": beat_text,
                "plan": plan,
            }

def _entity_choices(st: ProjectState, entity_type: str) -> List[str]:
    if not st:
        return []
    if entity_type == "Location":
        return list((st.locations or {}).keys())
    if entity_type == "Item":
        return list((st.items or {}).keys())
    return list((st.characters or {}).keys())


_AGE_TIMELINE_SYSTEM = """
You are a visual story analyst for a manhwa webtoon image generator.

Given a story, beats, and characters, produce an appearance timeline for each character. This covers TWO types of visual change:

TYPE 1 — AGE CHANGES: character appears younger in flashbacks/memories, or ages over the story.
TYPE 2 — PHYSICAL TRANSFORMATIONS: character gains or loses visible physical traits mid-story.
  Examples: scales erupting on skin, corruption marks spreading, power awakening brands, mutation after absorbing energy, a scar appearing, an eye color change after power-up, a creature evolving to a new form.
  IMPORTANT: If a character's body changes visibly at a specific beat, that IS a new phase — even if their age does not change.
  CRITICAL TIMING RULE: A transformation/acquisition phase only begins at the beat AFTER the triggering event fully completes.
  - If a character gains armor/ability/power BY defeating an enemy, beat_start must be set to AFTER the defeat is confirmed — NOT during the fight.
  - If a character is offered something and hasn't accepted yet, they do NOT have it yet.
  - When in doubt, set beat_start one beat later rather than earlier. It is worse to show a power too early than too late.

For each character, identify every distinct visual phase:
- Name phases concisely (e.g. "Before awakening", "After scale eruption", "Corrupted form", "Final evolution")
- The FIRST phase must describe the character in their BASE/EARLIEST form — NO transformation marks unless they have them from beat 1
- beat_end: -1 means "from beat_start through the end of the story"
- Only include characters with actual visual changes. A character with no changes gets one phase: beat_start=1, beat_end=-1

For appearance_prompt: write COMPLETE visual description for this phase including body form, any transformation marks, distinguishing features. DO NOT include hair color or eye color (those are fixed by dna_prompt).

CRITICAL RULES:
- NEVER write "bald head", "no hair", or any hair color/style that contradicts the character's dna_prompt. Even newborn characters with defined hair should say "fine sparse [character's actual hair color] hair barely visible" — not bald.
- NEVER use the word "infant" alone — generation models interpret it as a small standing child. Use physical constraint language instead.
- ALWAYS include the developmental motor stage so posture is correct.

Age vocabulary and physical constraint examples:
  0-3 months: "newborn baby, tiny helpless body, cannot hold head up, must be shown lying flat on back or cradled in adult arms, cannot sit or stand, scrunched newborn face, red-pink delicate skin, tiny clenched fists"
  3-6 months: "three-month-old baby, small and fragile, being held cradled or lying flat, cannot sit or stand independently, round baby face beginning to soften"
  6-12 months: "six-month-old baby, chubby round baby body, may sit propped with support but cannot walk, large baby head proportions"
  toddler 1-2 yrs: "toddler age one, unsteady on feet just learning to walk, very chubby short limbs, oversized baby head, wide waddling stance"
  toddler 2-3 yrs: "toddler age 2-3, small round chubby face, short pudgy limbs, confident toddler walk"
  child 7-8 yrs: "young child age 7-8, lean small build, round boyish face, childlike proportions, front teeth gap"
  teen 15-16 yrs: "teenager age 15-16, lanky adolescent build, angular face beginning to sharpen, growing taller"

Return ONLY valid JSON. No explanation. Schema:
{
  "character_age_phases": {
    "CharacterName": [
      {"label": "Phase name", "beat_start": 1, "beat_end": -1, "appearance_prompt": "..."}
    ]
  }
}
""".strip()


def _extract_age_timeline(st: "ProjectState", story: str, beats: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """Call Claude to auto-extract an age/appearance timeline for each character."""
    if not beats or not (st.characters or {}):
        return {}
    # Sample beats across the FULL story so late-story transformations are detected.
    # Strategy: first 30 + evenly spaced middle + last 20, capped at 150 total.
    n_beats = len(beats)
    if n_beats <= 150:
        sampled_beats = list(enumerate(beats))
    else:
        head = list(enumerate(beats[:30]))
        tail = list(enumerate(beats[-20:], start=n_beats - 20))
        step = max(1, (n_beats - 50) // 100)
        middle = [(i, beats[i]) for i in range(30, n_beats - 20, step)]
        combined = {i: b for i, b in head + middle + tail}
        sampled_beats = sorted(combined.items())
    user_payload = {
        "story_excerpt": story[:3000],
        "total_beats": n_beats,
        "beats": {str(i + 1): b for i, b in sampled_beats},
        "characters": [
            {"name": name, "dna_prompt": (cdata or {}).get("dna_prompt", "")}
            for name, cdata in (st.characters or {}).items()
        ],
    }
    result = _call_claude_json(_AGE_TIMELINE_SYSTEM, user_payload, max_tokens=2000, model=CLAUDE_FAST_MODEL)
    if not result:
        return {}
    phases = result.get("character_age_phases") or {}
    # Validate and sanitize
    out: Dict[str, List[Dict[str, Any]]] = {}
    for char_name, phase_list in phases.items():
        if not isinstance(phase_list, list):
            continue
        clean_phases = []
        for p in phase_list:
            try:
                clean_phases.append({
                    "label": clean_for_prompt(str(p.get("label", "Unknown"))),
                    "beat_start": max(1, int(p.get("beat_start", 1))),
                    "beat_end": int(p.get("beat_end", -1)),
                    "appearance_prompt": clean_for_prompt(str(p.get("appearance_prompt", ""))),
                    "ref_image_path": str(p.get("ref_image_path", "")),
                })
            except Exception:
                pass
        if clean_phases:
            out[char_name] = clean_phases
    return out


def get_active_form_for_beat(st: "ProjectState", char_name: str, beat_index: int) -> Optional[Dict[str, Any]]:
    """Return the active form dict for a character at a given beat, or None if the base form is active.

    The beat plan stores ``active_forms: {char_name: form_name}`` for each planned beat.
    A value of "base" (or missing) means no special form — use the character's dna_prompt as-is.
    Any other value is a form name looked up in ``character["forms"]``.
    """
    plan = (getattr(st, "beat_plans", {}) or {}).get(beat_index)
    if not plan:
        return None
    active_name = (plan.get("active_forms") or {}).get(char_name)
    if not active_name or active_name == "base":
        return None
    char = (st.characters or {}).get(char_name) or {}
    for f in (char.get("forms") or []):
        if f.get("name") == active_name:
            return f
    return None


def age_phase_for_beat(st: "ProjectState", char_name: str, beat_index: int) -> Optional[Dict[str, Any]]:
    """Return the age phase for char_name at beat_index.
    If beat falls in a gap between phases, returns the most recent phase that started before
    this beat — prevents falling back to dna_prompt (which may describe a future evolved form)."""
    phases = (getattr(st, "character_age_phases", None) or {}).get(char_name, [])
    if not phases:
        return None
    n_beats = len(st.beats or [])
    # Direct match first
    for phase in phases:
        lo = int(phase.get("beat_start", 1))
        hi = int(phase.get("beat_end", -1))
        if hi == -1:
            hi = max(n_beats, beat_index)
        if lo <= beat_index <= hi:
            return phase
    # No direct match — find the most recent phase that started before this beat.
    # This means "stay in the last known form" rather than jump to dna_prompt.
    before = [p for p in phases if int(p.get("beat_start", 1)) <= beat_index]
    if before:
        return max(before, key=lambda p: int(p.get("beat_start", 1)))
    # Beat is before all phases — return the earliest phase (base form)
    return min(phases, key=lambda p: int(p.get("beat_start", 1)))


def _group_beats_into_pages(beats: List[str], n: int = PANELS_PER_PAGE) -> List[List[str]]:
    """Split beats list into chunks of n for panel page grouping."""
    return [beats[i:i + n] for i in range(0, len(beats), n)]


def regenerate_page_scripts_cb(st: "ProjectState", progress=gr.Progress(track_tqdm=False)):
    """Regenerate all panel page scripts for a project (Step 9 only — no full rebuild).
    Runs up to 5 pages in parallel for speed (~30-45s for 70 pages).
    Each page still receives the previous page's BEATS as context for PANEL 1 continuity
    (prev_beats is always available upfront; prev_script is skipped to enable parallelism).
    Yields (st, status) so the Status box updates live as batches complete."""
    if not st or not st.project_dir or not st.beats:
        yield st, "❌ Load a project with beats first."
        return
    _mode_ppp = SHORTS_PANELS_PER_PAGE if getattr(st, 'build_mode', 'Panel') == 'Shorts' else PANELS_PER_PAGE
    pages = _group_beats_into_pages(st.beats, _mode_ppp)
    n_pages = len(pages)
    st.page_scripts = {}
    st.page_scripts_generated_at = ""
    progress(0, desc=f"Generating {n_pages} page scripts…")
    yield st, f"⏳ Writing page scripts: 0 / {n_pages}…"

    import concurrent.futures as _cf
    done_count = 0
    with _cf.ThreadPoolExecutor(max_workers=10) as ex:
        futs = {
            ex.submit(
                _call_claude_panel_page_prompt,
                pi, pg, st,
                pages[pi - 1] if pi > 0 else None,  # prev_beats for PANEL 1 continuity
                None,                                 # prev_script skipped (parallel mode)
                getattr(st, 'build_mode', 'Panel'),
                getattr(st, 'world_context', ''),
                _mode_ppp,
            ): pi
            for pi, pg in enumerate(pages)
        }
        for f in _cf.as_completed(futs):
            pi = futs[f]
            try:
                st.page_scripts[pi] = f.result() or ""
            except Exception:
                st.page_scripts[pi] = ""
            done_count += 1
            progress(done_count / n_pages, desc=f"Page scripts {done_count}/{n_pages}…")
            yield st, f"⏳ Writing page scripts: {done_count} / {n_pages}…"

    from datetime import datetime as _dt
    st.page_scripts_generated_at = _dt.now().strftime("%Y-%m-%d %H:%M:%S")
    _save_project_json(st)
    yield st, f"✅ {len(st.page_scripts)} page scripts ready — generated at {st.page_scripts_generated_at}"


def _beat_plan_ref_summary(st: "ProjectState", page_idx: int, n_panels: int) -> List[str]:
    """Return a one-line ref-signal summary per panel so Claude knows what refs will handle.
    Returns a list of n_panels strings (empty string if no beat plan for that panel)."""
    summaries: List[str] = []
    for pi in range(n_panels):
        beat_idx = page_idx * n_panels + pi + 1
        bp = (getattr(st, "beat_plans", None) or {}).get(beat_idx) or {}
        if not bp:
            summaries.append("")
            continue
        parts: List[str] = []
        cam = (bp.get("camera_type") or bp.get("suggested_perspective") or "").strip()
        if cam:
            parts.append(f"camera: {cam}")
        stype = (bp.get("scene_type") or "").strip()
        if stype:
            parts.append(f"scene type: {stype}")
        emo = bp.get("character_emotions") or {}
        if isinstance(emo, dict) and emo:
            emo_str = ", ".join(f"{k}: {v}" for k, v in list(emo.items())[:3])
            parts.append(f"emotions: {emo_str}")
        loc = (bp.get("suggested_location") or "").strip()
        if loc and loc.lower() not in ("", "none"):
            parts.append(f"location: {loc}")
        summaries.append(" | ".join(parts) if parts else "")
    return summaries


def _call_claude_panel_page_prompt(page_idx: int, page_beats: List[str], st: "ProjectState",
                                   prev_beats: Optional[List[str]] = None,
                                   prev_script: Optional[str] = None,
                                   build_mode: str = "Panel",
                                   world_context: str = "",
                                   n_panels: int = PANELS_PER_PAGE) -> str:
    """Generate a compact image-model prompt for one manhwa page (10 panels).
    prev_beats: raw story beats of the previous page, for narrative context.
    prev_script: the GENERATED visual script of the previous page — used to make PANEL 1
                 a direct visual continuation of the previous page's last panel."""
    api_key = (os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY") or "").strip()
    if not _rp.has_text_provider("anthropic"):
        header = "Korean manhwa webtoon art style, bold black ink outlines, hard-edged cel shading, vivid saturated palette, large luminous eyes. 9:16 vertical page, 10 panels, white gutters, left-right top-bottom reading, clean 2D Korean webtoon illustration not photorealistic. ALL figures and objects fully contained within their panel boundaries — no character or limb crosses a panel border."
        return header + "\n\n" + "\n".join(f"PANEL {i+1} [MS]: {b}" for i, b in enumerate(page_beats[:10]))

    loc_bibles = "\n".join(
        f"- {n}: {d.get('bible_prompt', '')}" for n, d in (st.locations or {}).items()
    )[:800]
    beats_text = "\n".join(f"{i+1}. {b}" for i, b in enumerate(page_beats))
    prev_context = ""
    if prev_script:
        # Extract the last 2 panels from the previous page's generated visual script
        _lines = [l.strip() for l in prev_script.splitlines() if l.strip().startswith("PANEL")]
        _last_panels = "\n".join(_lines[-2:]) if _lines else ""
        if _last_panels:
            prev_context = (
                "\n\nPREVIOUS PAGE — LAST 2 PANELS (visual script already generated — "
                "PANEL 1 of this page must be a direct visual continuation, same location, "
                "same characters mid-action, picking up exactly where this left off):\n"
                + _last_panels
            )
    if not prev_context and prev_beats:
        prev_context = (
            "\n\nPREVIOUS PAGE ENDING (last story beats — use for visual/narrative continuity; "
            "PANEL 1 must pick up exactly where the previous page left off):\n"
            + "\n".join(f"- {b}" for b in prev_beats[-3:])
        )

    # Build comprehensive per-character appearance section for this page's position in the story.
    # Lists ALL characters with NO truncation — uses age phase if defined, else dna_prompt.
    # This is the single source of truth for character descriptions; no separate char_dna needed.
    first_beat_of_page = page_idx * n_panels + 1
    char_entries = []
    for n, d in (st.characters or {}).items():
        phase = age_phase_for_beat(st, n, first_beat_of_page)
        if phase and phase.get("appearance_prompt"):
            appearance = phase["appearance_prompt"].strip()
            label = phase.get("label", "")
            phase_tag = f" [{label}]" if label else ""
            # Use phase appearance ONLY — do NOT include dna_prompt here.
            # dna_prompt often describes a later/evolved form and will override the phase.
            char_entries.append(
                f"• {n}{phase_tag}:\n"
                f"  Current appearance: {appearance}"
            )
        else:
            dna = (d.get("dna_prompt", "") or "").strip()
            if dna:
                char_entries.append(f"• {n}:\n  Appearance: {dna}")
    char_section = (
        "ALL STORY CHARACTERS — reference these whenever a character appears in a panel:\n"
        + "\n".join(char_entries)
        + "\n\nIMPORTANT: Each panel description MUST name the character and include their key physical traits "
        "(gender, age/size, distinguishing features) so the image model renders them correctly. "
        "Multiple characters may appear on the same page — describe each one accurately in their panels. "
        "Do NOT invent or change any character's gender, age, or body proportions."
    ) if char_entries else "No named characters."

    is_shorts = build_mode == "Shorts"
    is_hook_page = is_shorts and page_idx == 0  # first page of a short = the hook

    shorts_system_addon = (
        "\n\nSHORTS MODE — SHOW DON'T TELL (this is non-negotiable):\n"
        "This page is part of a 1–2 minute visual short. Context must ALWAYS be shown through images, "
        "never stated as narration text. If a beat implies a situation — debt, grief, fear, loneliness, "
        "power, wealth — you must translate it into a specific visible object or composition. "
        "Examples:\n"
        "  • Debt / financial ruin → stacked final-notice envelopes, a phone screen showing '47 missed calls', "
        "an empty kitchen drawer with a few coins, an eviction notice slid under a door, worn-down sneakers.\n"
        "  • Grief / loss → an empty chair at a dinner table set for two, a photo face-down on a shelf, "
        "a dusty coat still hanging by the door.\n"
        "  • Danger / surveillance → a shadow at the end of a hallway, a car that has been parked on the "
        "same street for three days, a curtain that moves after the character passes.\n"
        "  • Power / ability → environment reacts — cracks radiate from a footstep, candle flames lean in "
        "without wind, small objects orbit the character at waist height.\n"
        "NEVER put exposition in a Narration box. Let the image carry 100% of the meaning.\n"
        "Every panel must have a strong composition: dominant focal point, deliberate camera angle, "
        "and a clear mood established by lighting and colour temperature."
    ) if is_shorts else ""

    hook_system_addon = (
        "\n\nHOOK PAGE RULE — panels 1–5 are the opening hook of the entire short:\n"
        "These five panels are the ONLY chance to make a viewer stop scrolling and commit to watching. "
        "They must be cinematic showstoppers. Rules:\n"
        "  1. PANEL 1 — extreme close-up or macro detail that instantly creates a question in the viewer's "
        "mind. Never a wide establishing shot. Think: a crumpled bill on cracked pavement, a single drop "
        "of blood on white tile, a pair of eyes through a door-crack.\n"
        "  2. PANEL 2 — pull back slightly; reveal one layer of context without resolving the mystery.\n"
        "  3. PANEL 3 — introduce the character or key subject, but frame them in a way that immediately "
        "communicates their emotional state through body language and environment alone.\n"
        "  4. PANEL 4 — the most visually striking composition on the page: dramatic angle, "
        "high-contrast lighting, or a visual juxtaposition that feels charged and tense.\n"
        "  3. PANEL 3 — the most visually striking composition on the page: dramatic angle, "
        "high-contrast lighting, or a visual juxtaposition that feels charged and tense. "
        "Raises the story question through a visual event (a door opening, an object changing, something arriving).\n"
        f"Panels 4–{n_panels} continue the story naturally. The hook is panels 1–3."
    ) if is_hook_page else ""

    system = (
        "You are a visual story director writing panel descriptions for a Korean manhwa AI image generator. "
        "Output ONLY the final prompt text — no preamble, no explanation, no meta-commentary. "
        f"Always produce exactly {n_panels} panels numbered PANEL 1 through PANEL {n_panels}. "
        "PANEL 1 must visually continue from the previous page's cliffhanger or final moment if one exists.\n\n"

        "━━━ REFERENCE IMAGES WILL BE ATTACHED ━━━\n"
        "When this page is generated, visual reference images will be automatically attached to guide "
        "camera/composition, mood/lighting, and setting/architecture. "
        "YOUR JOB IS TO DESCRIBE THE STORY, NOT THE VISUAL STYLE.\n\n"
        "WHAT REFERENCES HANDLE (do NOT write these in panel text):\n"
        "  • Camera framing, depth, shot geometry, compositional arrangement\n"
        "  • Lighting quality — whether it is harsh, soft, warm, cold, overhead, directional\n"
        "  • Color palette and tonal mood — cool/warm contrast, saturation\n"
        "  • Setting architecture and background atmosphere\n"
        "  • Shadow quality, ambient light color, environmental depth\n\n"
        "WHAT YOUR PANEL TEXT MUST DESCRIBE:\n"
        "  • WHO is in the panel and their exact physical appearance (hair, eyes, outfit, age/build)\n"
        "  • WHAT they are doing — the specific physical action or pose\n"
        "  • WHAT THEY ARE FEELING — exact theatrical manhwa expression (see below)\n"
        "  • THE KEY PROP or object that carries story meaning (the bean, the stone, the contract)\n"
        "  • THE SETTING in one grounding phrase — 'underground mine chamber with wooden beam supports'\n\n"
        "BANNED PHRASES (references handle these — never write them in panel text):\n"
        "  • Any lighting description: 'harsh cold overhead light', 'warm orange glow', 'cool shadow', "
        "'dramatic side lighting', 'soft diffuse light', 'backlit', 'rim light', 'shadow across face'\n"
        "  • Any color/palette description: 'cyan tones', 'warm palette', 'desaturated', 'high contrast'\n"
        "  • Any texture/macro instruction: 'extreme macro detail', 'fine texture visible', 'deep cracks in skin'\n"
        "  • Camera prose: 'shot from below', 'overhead perspective', 'camera pulls back'\n"
        "Use the bracketed [SHOT TYPE] tag only — never write camera direction as prose.\n\n"

        "PANEL BOUNDARY RULE — NON-NEGOTIABLE:\n"
        "Every panel is a sealed rectangle with white gutters. No character, limb, or object may cross a panel border. "
        "Each panel must be 100% self-contained. If a full-body shot won't fit, use a closer shot type.\n"
        "GUTTERS: uniform white, never black. Consistent 6–8px white gutter on all sides.\n\n"

        "SHOT TYPE RULE:\n"
        "Every panel MUST open with one bracketed shot type:\n"
        "  [ECU] extreme close-up — one detail fills the frame (eye, hand, single object)\n"
        "  [CU] close-up — head and shoulders\n"
        "  [MCU] medium close-up — chest up\n"
        "  [MS] medium shot — waist up\n"
        "  [MLS] medium long shot — knees up\n"
        "  [LS] long shot — full body\n"
        "  [ELS] extreme long shot — small figure in large environment\n"
        "  [OTS] over-the-shoulder\n"
        "  [POV] through a character's eyes\n"
        "  [LOW] low angle — camera below, looking up\n"
        "  [HIGH] high angle — camera above, looking down\n"
        "  [DUTCH] dutch tilt — diagonal frame for tension\n\n"

        "CHARACTER PRESENCE MINIMUM — ABSOLUTE RULE:\n"
        "At least HALF the panels on every page must show a story character (face, body, hands, or silhouette — "
        "any part of a person counts). For a 6-panel page: minimum 3 panels with a character. "
        "For a 10-panel page: minimum 5 panels with a character.\n"
        "When a beat describes only an object or environment (e.g. 'a dried bean in a bowl'), "
        "you MUST still place the character in frame — either reacting to it, holding it, observing it, "
        "or shown in the background. The object can be the FOCUS but the character must be PRESENT.\n"
        "A page with zero characters is always WRONG, no matter what the beats say.\n\n"

        "EXPRESSION RULE — MANDATORY FOR EVERY PANEL WITH A CHARACTER:\n"
        "Korean manhwa expressions are THEATRICAL and EXAGGERATED. Always name the exact expression.\n"
        "  Shock → 'jaw dropped, mouth wide open, eyes stretched huge with white all around iris'\n"
        "  Fear → 'pupils shrunk to dots, cold sweat bead on temple, lips trembling'\n"
        "  Rage → 'eyebrows V-shaped downward, teeth bared in snarl, forehead vein'\n"
        "  Determination → 'eyes half-lidded and sharp, jaw set, slight smirk'\n"
        "  Awe → 'eyes wide sparkling with triple catchlight, mouth open in O'\n"
        "  Pain → 'eyes screwed shut, mouth open in cry, sweat lines radiating'\n"
        "  Despair → 'eyes downcast and hollow, corners of mouth pulled down, shoulders curved inward'\n"
        "  Smugness → 'lazy half-lidded eyes, one eyebrow raised, corner of mouth up'\n"
        "Reaction lines, sweat drops, and speed lines are ENCOURAGED — core manhwa language.\n\n"

        "CHARACTER APPEARANCE RULE — MANDATORY:\n"
        "Every panel with a character MUST include their appearance inline: hair color/style, eye color, "
        "age/build, distinctive features, outfit. Never just write a name. "
        "Never swap genders. Never change ages. Never omit hair/eye color.\n\n"
        "GOOD panel example:\n"
        "  PANEL 3 [MCU]: Chen Ping — elderly man, hollow sunken cheeks, grey stubble, dark exhausted eyes, "
        "ragged brown robe — stares at the single bean in his trembling palm, eyes downcast and hollow, "
        "corners of mouth pulled down in silent despair. Underground mine resting chamber, wooden beam supports.\n\n"
        "BAD panel example (do not write like this):\n"
        "  PANEL 3 [ECU]: A bean rests in a calloused palm. Harsh cold overhead light creates sharp shadows. "
        "Extreme macro detail on bean texture and skin surface.\n\n"

        "BEAT FIDELITY RULE — NON-NEGOTIABLE:\n"
        "Every object, person, location, or action named in a beat MUST appear in that panel. "
        "You may emphasise one element (foreground vs background) but may NEVER omit any named element.\n\n"

        "CULTURAL SETTING RULE:\n"
        "Default setting is Korean or Chinese. Modern scenes: Seoul-style streets, hangul/hanja signs, "
        "pojangmacha stalls, narrow alleys. NOT generic Western cities. "
        "Characters are Korean/Chinese — black or dark hair by default, East Asian features.\n\n"

        "LANGUAGE RULE: All text inside images — signs, labels, captions, speech bubbles — must be in ENGLISH.\n"
        "Format: PANEL N [SHOT TYPE]: [character appearance + expression + action + key prop + setting phrase]. Narration: \"text\" (if any)."
        + shorts_system_addon
        + hook_system_addon
    )
    # Build per-panel ref signal summary — tells Claude what refs will cover per panel
    _ref_signals = _beat_plan_ref_summary(st, page_idx, n_panels)
    _ref_signal_lines = []
    for _pi, _sig in enumerate(_ref_signals):
        if _sig:
            _ref_signal_lines.append(f"  Panel {_pi + 1}: {_sig}")
    _ref_signal_block = (
        "\nREFERENCE SIGNALS PER PANEL (these aspects will be provided by reference images — "
        "do NOT describe them in your panel text):\n"
        + "\n".join(_ref_signal_lines)
        + "\n"
    ) if _ref_signal_lines else ""

    n_beats = len(page_beats)
    shorts_user_addon = (
        "\n\nSHORTS REMINDER: Zero narration text for context. Every beat must become a specific visible detail — "
        "a prop, a texture, a lighting condition, a body-language micro-expression, an environmental clue. "
        "If the beat says 'he is broke', show the empty wallet. If it says 'she is afraid', show her knuckles "
        "white on a door handle. Make every panel a painting someone would want to look at for five seconds."
    ) if is_shorts else ""
    hook_user_addon = (
        "\n\nHOOK REMINDER: Panels 1–5 are the entire reason a viewer keeps watching. "
        "Panel 1 must be a tight, mysterious detail shot. No wide shots until panel 3 minimum. "
        "Use extreme angles, strong shadows, or macro textures. Make it irresistible."
    ) if is_hook_page else ""
    world_ctx_block = (
        f"\nWORLD CONTEXT (use to inform locations, tone, and visual language):\n{world_context.strip()}\n"
        if (world_context or "").strip() else ""
    )
    user = (
        f"Page {page_idx + 1} of a Korean manhwa.{prev_context}\n\n"
        f"{char_section}\n\n"
        f"LOCATIONS:\n{loc_bibles or 'Various environments.'}\n\n"
        f"{world_ctx_block}"
        f"{_ref_signal_block}"
        f"STORY BEATS FOR THIS PAGE ({n_beats} beats — expand into {n_panels} panels):\n{beats_text}\n\n"
        + (
        "Begin the prompt with this exact header line (copy it verbatim):\n"
        "\"Korean manhwa webtoon art style, bold black ink outlines with thick-thin weight variation, "
        "hard-edged cel shading flat color fills sharp shadow cutoffs, vivid saturated palette warm-cool contrast, "
        f"large luminous eyes gradient iris sharp catchlight, 9:16 vertical page, {n_panels} panels in a 2-column {n_panels // 2}-row grid, "
        "white gutters, left-right top-bottom reading, clean 2D Korean webtoon illustration not photorealistic, "
        "ALL figures and objects fully contained within their panel boundaries — no character or limb crosses a panel border.\"\n\n"
        f"Then write PANEL 1 through PANEL {n_panels}. Each panel must look like a professional Korean webtoon panel — "
        "bold ink lines, flat cel-shaded colors, dramatic lighting, large expressive eyes. "
        "CONTAINMENT RULE: every figure must fit fully inside its panel. If a full-body shot won't fit, use a closer shot type. "
        "REMINDER: embed each character's full appearance (hair color/style, eye color, age, build, outfit) "
        "directly inside every panel description where they appear — never rely on the reader knowing who they are."
        if is_shorts else
        "Begin the prompt with this exact header line (copy it verbatim):\n"
        "\"Korean manhwa webtoon art style, bold black ink outlines, hard-edged cel shading, vivid saturated palette, "
        "large expressive eyes, 9:16 vertical page, 10 panels, white gutters, left-right top-bottom reading, "
        "clean 2D Korean webtoon illustration not photorealistic, "
        "ALL figures and objects fully contained within their panel boundaries — no character or limb crosses a panel border.\"\n\n"
        "Then write PANEL 1 through PANEL 10. "
        "CONTAINMENT RULE: every figure must fit fully inside its panel. If a full-body shot won't fit, use a closer shot type. "
        "REMINDER: embed each character's full appearance (hair color/style, eye color, age, build, outfit) "
        "directly inside every panel description where they appear — never rely on the reader knowing who they are."
        )
        + shorts_user_addon
        + hook_user_addon
    )
    try:
        if _rp.is_deepseek_mode():
            text, _status = _rp.call_text(system, user, max_tokens=2500, temperature=0)
            if text:
                return text.strip()
            raise RuntimeError(_status)
        import requests as _rq
        r = _rq.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": CLAUDE_MODEL, "max_tokens": 2500, "temperature": 0, "system": system,
                  "messages": [{"role": "user", "content": user}]},
            timeout=(15, 90),
        )
        r.raise_for_status()
        parts = [p.get("text", "") for p in r.json().get("content", []) if p.get("type") == "text"]
        return "".join(parts).strip()
    except Exception:
        header = "Korean manhwa webtoon art style, bold black ink outlines, hard-edged cel shading, vivid saturated palette, large luminous eyes. 9:16 vertical page, 10 panels, white gutters, left-right top-bottom reading, clean 2D Korean webtoon illustration not photorealistic. ALL figures and objects fully contained within their panel boundaries — no character or limb crosses a panel border."
        return header + "\n\n" + "\n".join(f"PANEL {i+1} [MS]: {b}" for i, b in enumerate(page_beats[:10]))


def build_everything_cb(story, project_name, gen_char, gen_loc, gen_item, loc_persps, precompute_beats, batch_size, build_mode="Panel", world_context="", current_sync=0, progress=gr.Progress(track_tqdm=False)):
    use_sonnet = True
    logs = []
    def log(m): logs.append(m)

    # These are updated in place so the closure always reads the latest value
    _beats_table: list = []
    _char_rows:   list = []
    _loc_rows:    list = []
    _item_rows:   list = []

    def _mid(msg):
        """Intermediate yield — keeps browser alive and shows live progress."""
        return (st, msg, _beats_table, _char_rows, _loc_rows, _item_rows,
                gr.update(), gr.update(), gr.update(), "\n".join(logs),
                gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                _sync_int(current_sync))

    if not (story or "").strip():
        empty = ProjectState()
        yield (empty, "❌ Paste a story first.", [], [], [], [], [], [], [], "❌ Story is empty.", gr.Dropdown(choices=[], value=None), gr.Dropdown(choices=["Character","Location","Item"], value="Character"), gr.Dropdown(choices=[], value=None), gr.Dropdown(choices=[], value=None), gr.Dropdown(choices=[], value=None), current_sync)
        return

    st = new_project(name=project_name or "")
    st.story = story
    st.original_story = story  # preserve the user's raw text; never overwritten by beat expansion
    st.build_mode = build_mode
    st.world_context = (world_context or "").strip()
    dirs = ensure_dirs(st.project_dir)
    log(f"STEP 1: New project {st.project_id}")

    progress(0.10, desc="Splitting beats...")
    st.beats = split_beats(story)
    log(f"STEP 2: Raw beats = {len(st.beats)}")

    if build_mode == "Shorts":
        yield _mid(f"🔄 {len(st.beats)} raw beats — expanding into cinematic micro-beats for Shorts…")
        progress(0.13, desc="Expanding Shorts beats…")
        # Scale beat count to story length.
        # Rule: 1 image every ~2 seconds on screen → 30 beats = 1 min, 60 = 2 min.
        # Old hard minimum of 50 wasn't the problem itself — the problem was even a
        # 5-sentence story got forced to 50 beats.  Now we scale by word count but
        # keep a floor of 30 (= 1 minute minimum) and a cap of 90 (= 3 min max).
        _story_words = len((story or "").split())
        if _story_words < 100:
            target = 30        # shortest possible short — 1 min exactly
        elif _story_words < 250:
            target = 35
        elif _story_words < 500:
            target = 45
        elif _story_words < 800:
            target = 60
        else:
            target = 75
        target = min(target, 90)   # hard cap: 90 beats = 3-min short
        st.beats = _expand_shorts_beats_with_claude(story, st.beats, target_beats=target)
        log(f"STEP 2b: Shorts beats expanded → {len(st.beats)} (target={target}, words={_story_words})")
        yield _mid(f"🔄 Validating {len(st.beats)} beats against story sentences…")
        progress(0.17, desc="Aligning beats to story…")
        st.beats = _validate_shorts_beats_with_claude(story, st.beats)
        log(f"STEP 2c: Beat validation pass complete → {len(st.beats)} beats")

    _beats_table = [[str(i + 1), b] for i, b in enumerate(st.beats)]
    n_beats = len(st.beats)

    yield _mid(f"🔄 Found {n_beats} beats — building story bible…")

    progress(0.22, desc="Building story bible...")
    if use_sonnet:
        bible = _call_claude_story_bible(st, story, st.beats, build_mode=build_mode, world_context=world_context)
    else:
        bible = None
    if bible:
        st.characters = _sanitize_character_payload(st, story, bible)
        st.locations  = _sanitize_location_payload(bible)
        st.items      = _sanitize_item_payload(bible, character_names=set(st.characters.keys()))
        log(f"STEP 3: Claude story bible OK -> chars={len(st.characters)} locs={len(st.locations)} items={len(st.items)}")
    else:
        log("STEP 3: Sonnet bible call returned None — trying Haiku name extractor as fallback…")
        _fallback_story_bible(st, story, _log=log)
        log(f"STEP 3: Fallback complete -> chars={len(st.characters)} locs={len(st.locations)} items={len(st.items)}")

    if not st.characters:
        inferred    = detect_first_person_gender(story)
        force_gender = inferred if inferred else "male"
        st.characters = {"Protagonist": build_character_profile(st.project_id, "Protagonist", story, force_gender=force_gender or "male", force_character_type="humanoid")}
        log(f"STEP 4: No characters found -> created Protagonist ({force_gender})")

    # ── Auto-cast characters from library ─────────────────────────────────
    try:
        from character_library import auto_cast_characters as _auto_cast, load_library as _load_lib
        _new_cast = _auto_cast(st.characters, st.project_id, st.character_cast or {})
        if _new_cast:
            st.character_cast = {**(st.character_cast or {}), **_new_cast}
            log(f"STEP 4c: Auto-cast {len(_new_cast)} characters from library → {list(_new_cast.items())}")
        else:
            log("STEP 4c: Auto-cast — library empty or all characters already cast")
    except Exception as _ce:
        log(f"STEP 4c: Auto-cast failed (non-fatal): {_ce}")

    # ── Age timeline extraction ────────────────────────────────────────────
    yield _mid("🔄 Extracting character age timeline…")
    try:
        age_phases = _extract_age_timeline(st, story, st.beats)
        if age_phases:
            st.character_age_phases = age_phases
            log(f"STEP 4b: Age timeline extracted for {list(age_phases.keys())}")
        else:
            log("STEP 4b: Age timeline — no phases found (single-age story or extraction failed)")
    except Exception as _ae:
        log(f"STEP 4b: Age timeline extraction failed: {_ae}")

    _char_rows = make_char_table_rows(st)
    _loc_rows  = make_loc_table_rows(st)
    _item_rows = make_item_table_rows(st)

    if precompute_beats and st.beats and build_mode != "Panel":
        bsz   = int(batch_size or 20)
        total = max(n_beats, 1)

        import concurrent.futures as _cf_plan
        import threading as _plan_th

        BP_WORKERS    = 4   # beat plans: runs first on a clean API slate
        IP_WORKERS    = 4   # image prompts: 4 workers halves wall-clock time vs 2
        HARD_TIMEOUT_SECS = 55  # must be > requests read timeout (45s) so threads complete cleanly

        def _call_with_hard_timeout(fn, *args):
            """Run fn(*args) in a daemon thread; return None if it takes > HARD_TIMEOUT_SECS."""
            result = [None]
            def _run():
                try:
                    result[0] = fn(*args)
                except Exception:
                    pass
            t = _plan_th.Thread(target=_run, daemon=True)
            t.start()
            t.join(timeout=HARD_TIMEOUT_SECS)
            return result[0]  # None if still running (timed out)

        yield _mid(f"🔄 {len(st.characters)} chars, {len(st.locations)} locs — planning {n_beats} beats ({BP_WORKERS} beat workers / {IP_WORKERS} prompt workers)…")

        # ── Beat plans — parallel ──────────────────────────────────────────────
        _bp_batches = []
        for _s in range(0, n_beats, bsz):
            _b = st.beats[_s:_s + bsz]
            _bp_batches.append((_s + 1, _s + len(_b), _b))

        st.beat_plans = {}
        _bp_done = 0
        _bp_timeouts = 0
        log(f"STEP 6: Planning {len(_bp_batches)} beat-plan batches in parallel (workers={BP_WORKERS})")
        _bp_ex = _cf_plan.ThreadPoolExecutor(max_workers=BP_WORKERS)
        try:
            _bp_futures = {
                _bp_ex.submit(_call_with_hard_timeout, _call_claude_batch_beat_plans, st, si, batch, ""): (si, ei, batch)
                for si, ei, batch in _bp_batches
            }
            _bp_pending = set(_bp_futures.keys())
            while _bp_pending:
                _bp_done_set, _bp_pending = _cf_plan.wait(_bp_pending, timeout=8)
                for _fut in _bp_done_set:
                    si, ei, batch = _bp_futures[_fut]
                    try:
                        rows = _fut.result() or {}
                    except Exception as _fe:
                        log(f"STEP 6: future error beat {si}: {_fe}")
                        rows = {}
                        _bp_timeouts += 1
                    if not rows:
                        _bp_timeouts += 1
                    for rel_i, beat_text in enumerate(batch):
                        idx  = si + rel_i
                        raw  = rows.get(idx)
                        plan = _sanitize_plan(st, raw or _heuristic_beat_plan(st, beat_text, idx), beat_text)
                        st.beat_plans[idx] = plan
                    _bp_done += len(batch)
                _warn = f" ⚠️ {_bp_timeouts} batches used fallback" if _bp_timeouts else ""
                progress(0.35 + 0.35 * (_bp_done / total), desc=f"Beat plans {_bp_done}/{n_beats}…")
                yield _mid(f"🔄 Beat plans: {_bp_done}/{n_beats}…{_warn}")
        finally:
            _bp_ex.shutdown(wait=False, cancel_futures=True)

        # ── Image prompts — parallel ───────────────────────────────────────────
        # Use smaller batches (5) so each call stays well under the 45s timeout.
        # max_tokens=2500 means we can fit ~5 detailed prompts comfortably.
        _ip_bsz = 5
        _ip_batches = []
        for _s in range(0, n_beats, _ip_bsz):
            _b = st.beats[_s:_s + _ip_bsz]
            _ip_batches.append((_s + 1, _s + len(_b), _b))

        st.image_prompts = {}
        _ip_done = 0
        _ip_timeouts = 0
        log(f"STEP 7: Generating {len(_ip_batches)} image-prompt batches in parallel (workers={IP_WORKERS}, bsz={_ip_bsz})")
        _ip_ex = _cf_plan.ThreadPoolExecutor(max_workers=IP_WORKERS)
        try:
            _ip_futures = {
                _ip_ex.submit(_call_with_hard_timeout, _call_claude_batch_image_prompts, st, si, batch): (si, ei, batch)
                for si, ei, batch in _ip_batches
            }
            _ip_pending = set(_ip_futures.keys())
            while _ip_pending:
                _ip_done_set, _ip_pending = _cf_plan.wait(_ip_pending, timeout=8)
                for _fut in _ip_done_set:
                    si, ei, batch = _ip_futures[_fut]
                    try:
                        rows = _fut.result() or {}
                    except Exception as _fe:
                        log(f"STEP 7: future error beat {si}: {_fe}")
                        rows = {}
                        _ip_timeouts += 1
                    if not rows:
                        _ip_timeouts += 1
                    for rel_i, beat_text in enumerate(batch):
                        idx        = si + rel_i
                        plan       = (st.beat_plans or {}).get(idx) or _heuristic_beat_plan(st, beat_text, idx)
                        prompt_row = rows.get(idx) or {}
                        prompt_text = clean_for_prompt(str(prompt_row.get("prompt") or ""))
                        if not prompt_text:
                            prompt_text = _compose_sonnet_prompt_fallback(st, beat_text, plan)
                            source      = "local_fallback"
                        else:
                            source = prompt_row.get("source") or "sonnet_batch"
                        st.image_prompts[idx] = {
                            "prompt":    prompt_text,
                            "negative":  clean_for_prompt(str(prompt_row.get("negative") or DEFAULT_NEGATIVE)) or DEFAULT_NEGATIVE,
                            "summary":   clean_for_prompt(str(prompt_row.get("summary") or plan.get("suggested_action") or beat_text)),
                            "source":    source,
                            "beat_text": beat_text,
                            "plan":      plan,
                        }
                    _ip_done += len(batch)
                _warn = f" ⚠️ {_ip_timeouts} batches used fallback" if _ip_timeouts else ""
                progress(0.72 + 0.14 * (_ip_done / total), desc=f"Prompts {_ip_done}/{n_beats}…")
                yield _mid(f"🔄 Prompts: {_ip_done}/{n_beats}…{_warn}")
        finally:
            _ip_ex.shutdown(wait=False, cancel_futures=True)

        log(f"STEP 8: Precomputed prompts = {len(st.image_prompts)}")
    else:
        st.beat_plans    = {}
        st.image_prompts = {}
        log("STEP 6: Beat plans and prompts will be generated on demand.")

    # STEP 9: Panel page scripts
    # Panel mode: sequential (each page references the previous page's script for continuity).
    # Shorts mode: parallel (individual beats are self-contained; continuity less critical and
    #              sequential was a major wall-clock bottleneck for short stories).
    if build_mode in ("Panel", "Shorts") and st.beats:
        _ppp_val = SHORTS_PANELS_PER_PAGE if build_mode == "Shorts" else PANELS_PER_PAGE
        _pages = _group_beats_into_pages(st.beats, _ppp_val)
        _n_pages = len(_pages)

        # Always start fresh — stale scripts from a previous build (different beats,
        # different rules) must never survive into the new image generation.
        st.page_scripts = {}

        if build_mode == "Shorts":
            # ── Shorts: parallel page scripts ────────────────────────────────
            import concurrent.futures as _cf_pg
            _PS_WORKERS = 4
            log(f"STEP 9: Generating {_n_pages} Shorts page scripts (parallel, workers={_PS_WORKERS})…")
            yield _mid(f"🔄 Shorts: generating {_n_pages} page scripts (parallel)…")

            def _gen_page_script(_pi):
                try:
                    return _pi, _call_claude_panel_page_prompt(
                        _pi, _pages[_pi], st,
                        prev_beats=None, prev_script=None,
                        build_mode=build_mode, world_context=world_context,
                        n_panels=_ppp_val,
                    ) or ""
                except Exception as _pe:
                    log(f"STEP 9: Shorts page {_pi} failed: {_pe}")
                    return _pi, ""

            _pg_done = 0
            with _cf_pg.ThreadPoolExecutor(max_workers=_PS_WORKERS) as _pg_ex:
                _pg_futs = {_pg_ex.submit(_gen_page_script, pi): pi for pi in range(_n_pages)}
                _pg_pending = set(_pg_futs.keys())
                while _pg_pending:
                    _pg_done_set, _pg_pending = _cf_pg.wait(_pg_pending, timeout=8)
                    for _pf in _pg_done_set:
                        _pi, _script = _pf.result()
                        st.page_scripts[_pi] = _script
                        _pg_done += 1
                    progress(0.86 + 0.02 * (_pg_done / max(_n_pages, 1)),
                             desc=f"Page scripts {_pg_done}/{_n_pages}…")
                    yield _mid(f"🔄 Page scripts {_pg_done}/{_n_pages}…")
        else:
            # ── Panel mode: sequential for cross-page continuity ──────────────
            log(f"STEP 9: Generating {_n_pages} Panel page scripts (sequential for continuity)…")
            yield _mid(f"🔄 Panel mode: generating {_n_pages} page scripts…")
            for _pi, _pg in enumerate(_pages):
                _prev_beats = _pages[_pi - 1] if _pi > 0 else None
                _prev_script = st.page_scripts.get(_pi - 1) if _pi > 0 else None
                try:
                    st.page_scripts[_pi] = _call_claude_panel_page_prompt(
                        _pi, _pg, st, prev_beats=_prev_beats, prev_script=_prev_script,
                        build_mode=build_mode, world_context=world_context,
                        n_panels=_ppp_val,
                    ) or ""
                except Exception as _pe:
                    log(f"STEP 9: Page {_pi} failed: {_pe}")
                    st.page_scripts[_pi] = ""
                if (_pi + 1) % 5 == 0 or _pi == _n_pages - 1:
                    yield _mid(f"🔄 Page scripts {_pi + 1}/{_n_pages}…")

        log(f"STEP 9: Panel page scripts = {len(st.page_scripts)}")
        yield _mid(f"✅ {_n_pages} panel page scripts ready")

    st.next_image_index = 1
    _beats_table = [[str(i + 1), b] for i, b in enumerate(st.beats)]
    _char_rows   = make_char_table_rows(st)
    _loc_rows    = make_loc_table_rows(st)
    _item_rows   = make_item_table_rows(st)

    char_gallery, loc_gallery, item_gallery = [], [], []
    can_generate = (fal_client is not None) and bool(os.getenv("FAL_KEY"))

    # ── Character gallery: pull face images directly from library cast ────────
    # Never generate character refs via FAL — the library IS the reference.
    if st.character_cast:
        try:
            from character_library import load_library as _clib_load
            _clib = _clib_load()
            for cname in st.characters:
                tid = st.character_cast.get(cname)
                if not tid:
                    continue
                t = _clib.get("templates", {}).get(tid)
                if not t:
                    continue
                face_path = t.get("local_face", "")
                body_path = t.get("local_body", "")
                img_path = face_path if (face_path and os.path.exists(face_path)) else body_path
                if img_path and os.path.exists(img_path):
                    char_gallery.append(img_path)
            if char_gallery:
                log(f"STEP 9: Character gallery from library — {len(char_gallery)} images")
        except Exception as _cge:
            log(f"STEP 9: Library char gallery failed (non-fatal): {_cge}")

    if can_generate:
        ref_tasks = []
        if gen_char:
            pass  # Characters come from the library — no FAL generation needed
        if gen_loc:
            for lname, ldata in st.locations.items():
                ref_tasks.append(("loc", lname, _location_single_ref_prompt(lname, ldata)))
        if gen_item:
            for iname, idata in st.items.items():
                ref_tasks.append(("item", iname, idata["ref_prompt"]))

        if ref_tasks:
            import concurrent.futures as _cf_ref
            n_ref = len(ref_tasks)
            yield _mid(f"🔄 Generating {n_ref} reference images…")
            progress(0.88, desc=f"Generating {n_ref} reference images in parallel (no upscale)…")
            with _cf_ref.ThreadPoolExecutor(max_workers=min(n_ref, 6)) as ref_ex:
                future_to_ref = {
                    ref_ex.submit(call_fal_generate, task[2], REF_NEGATIVE, skip_esrgan=True): task
                    for task in ref_tasks
                }
                for future in _cf_ref.as_completed(future_to_ref):
                    kind, name, _ = future_to_ref[future]
                    try:
                        img = future.result()
                        if kind == "char":
                            _clear_old_ref_files(dirs["refs_chars"], safe_filename(name))
                            out = _stamp_ref_path(dirs["refs_chars"], safe_filename(name))
                            img.save(out)
                            char_gallery.append(out)
                        elif kind == "loc":
                            _clear_old_ref_files(dirs["refs_locs"], f"loc_{safe_filename(name)}")
                            out = _stamp_ref_path(dirs["refs_locs"], f"loc_{safe_filename(name)}")
                            img.save(out)
                            loc_gallery.append(out)
                        elif kind == "item":
                            _clear_old_ref_files(dirs["refs_items"], f"item_{safe_filename(name)}")
                            out = _stamp_ref_path(dirs["refs_items"], f"item_{safe_filename(name)}")
                            img.save(out)
                            item_gallery.append(out)
                    except Exception as e:
                        log(f"  ❌ {kind} ref failed {name}: {e}")
    else:
        log("⚠️ Skipping ref generation (missing fal_client or FAL_KEY).")

    # Stamp the initial part entry so ZIP download can filter by part
    if not getattr(st, "story_parts", None):
        st.story_parts = [{"name": "Original", "start_beat": 1, "end_beat": None}]

    # STEP 10: Shorts cover image — pick best library image, no FAL generation
    if build_mode == "Shorts":
        yield _mid("🎨 Building cover from library…")
        try:
            cover_path = _library_cover(st)
            if cover_path:
                st.cover_image_path = cover_path
                log(f"STEP 10: Cover from library → {cover_path}")
            else:
                log("STEP 10: No suitable library image found for cover (non-fatal).")
        except Exception as _ce:
            log(f"STEP 10: Library cover failed (non-fatal): {_ce}")

    _save_project_json(st)
    progress(1.0, desc="Done")
    status = f"✅ Built {st.project_id} | Beats={n_beats} | Chars={len(st.characters)} | Locs={len(st.locations)} | Items={len(st.items)} | Beat plans={len(st.beat_plans)} | Prompts={len(st.image_prompts)}"
    default_entity_type = "Character"
    entity_choices  = _entity_choices(st, default_entity_type)
    entity_pick_val = gr.Dropdown(choices=entity_choices, value=(entity_choices[0] if entity_choices else None))
    yield (st, status, _beats_table, _char_rows, _loc_rows, _item_rows,
           char_gallery, loc_gallery, item_gallery, "\n".join(logs),
           gr.Dropdown(choices=list(st.characters.keys()), value=(list(st.characters.keys())[0] if st.characters else None)),
           gr.Dropdown(choices=["Character", "Location", "Item"], value=default_entity_type),
           entity_pick_val,
           gr.Dropdown(choices=list(st.locations.keys()), value=(list(st.locations.keys())[0] if st.locations else None)),
           gr.Dropdown(choices=list(st.items.keys()), value=(list(st.items.keys())[0] if st.items else None)),
           _sync_int(current_sync, 1))


def continue_story_cb(st, continuation_text, continuation_name, precompute_beats, current_sync=0, progress=gr.Progress(track_tqdm=False)):
    """Append new beats to an existing project, continuing the story from where it left off."""

    def _emit(status_msg, table=None):
        tbl = table if table is not None else ([[str(i+1), b] for i, b in enumerate(st.beats)] if st else [])
        return (st, status_msg, tbl, _sync_int(current_sync))

    if not st or not st.project_dir or not st.beats:
        yield _emit("❌ Load or build a project first — no beats found.")
        return

    txt = (continuation_text or "").strip()
    if not txt:
        yield _emit("❌ Paste the new story text before continuing.")
        return

    old_beat_count = len(st.beats)

    # Initialise or close previous part
    if not getattr(st, "story_parts", None):
        st.story_parts = [{"name": "Original", "start_beat": 1, "end_beat": old_beat_count}]
    elif st.story_parts[-1].get("end_beat") is None:
        st.story_parts[-1]["end_beat"] = old_beat_count

    part_name = (continuation_name or "").strip() or f"Part {len(st.story_parts) + 1}"

    yield _emit(f"🔄 Splitting new story text into beats…")
    progress(0.05, desc="Splitting beats…")

    new_beats = split_beats(txt)
    if not new_beats:
        yield _emit("❌ Could not split text into beats — check the story text.")
        return

    # Record new part boundary
    st.story_parts.append({"name": part_name, "start_beat": old_beat_count + 1, "end_beat": None})

    # Append to story and beats
    st.beats.extend(new_beats)
    st.story = (st.story or "").rstrip() + "\n\n" + txt

    beats_table = [[str(i + 1), b] for i, b in enumerate(st.beats)]
    yield (st, f"🔄 Added {len(new_beats)} beats ({old_beat_count + 1}–{len(st.beats)}) — building full story context…", beats_table, _sync_int(current_sync))

    if precompute_beats:
        import threading as _cont_th

        HARD_TIMEOUT = 55
        bsz = 20

        def _call_timeout(fn, *args):
            result = [None]
            def _run():
                try:
                    result[0] = fn(*args)
                except Exception:
                    pass
            t = _cont_th.Thread(target=_run, daemon=True)
            t.start()
            t.join(timeout=HARD_TIMEOUT)
            return result[0]

        # ── Re-run story bible on the FULL combined story ────────────────────
        # This gives the beat planner total context — all characters and locations
        # across both episodes. We only ADD new entities; existing ones are kept
        # as-is so Episode 1 beat plans and images are never affected.
        yield (st, f"🔄 Rebuilding story bible with full {len(st.beats)}-beat context…", beats_table, _sync_int(current_sync))
        progress(0.05, desc="Rebuilding story bible…")
        try:
            _full_bible = _call_claude_story_bible(st, st.story, st.beats, build_mode=getattr(st, 'build_mode', 'Panel'))
            if _full_bible:
                _exist_chars_lower = {n.lower() for n in (st.characters or {}).keys()}
                _exist_locs_lower  = {n.lower() for n in (st.locations  or {}).keys()}

                # Filter to only entities NOT already known so we don't waste
                # LLM calls re-building profiles for Episode 1 characters/locations.
                _bible_new = dict(_full_bible)
                _bible_new["characters"] = [
                    c for c in (_full_bible.get("characters") or [])
                    if (c.get("name") or "").lower() not in _exist_chars_lower
                ]
                _bible_new["locations"] = [
                    l for l in (_full_bible.get("locations") or [])
                    if (l.get("name") or "").lower() not in _exist_locs_lower
                ]
                _bible_new["items"] = _full_bible.get("items") or []

                _new_chars = _sanitize_character_payload(st, st.story, _bible_new) if _bible_new["characters"] else {}
                _new_locs  = _sanitize_location_payload(_bible_new)                 if _bible_new["locations"]  else {}
                _new_items = _sanitize_item_payload(_bible_new, character_names=set((st.characters or {}).keys()) | set(_new_chars.keys())) if _bible_new["items"] else {}

                # Merge — update() only adds keys not already present when using |=
                # but dict.update() will overwrite, so filter before merging.
                _added_chars = {k: v for k, v in _new_chars.items() if k not in (st.characters or {})}
                _added_locs  = {k: v for k, v in _new_locs.items()  if k not in (st.locations  or {})}
                _added_items = {k: v for k, v in _new_items.items()  if k not in (st.items      or {})}

                if not st.characters: st.characters = {}
                if not st.locations:  st.locations  = {}
                if not st.items:      st.items      = {}
                st.characters.update(_added_chars)
                st.locations.update(_added_locs)
                st.items.update(_added_items)

                _msg = f"🔄 Story bible refreshed — +{len(_added_chars)} new chars, +{len(_added_locs)} new locs — planning {len(new_beats)} new beats…"
                yield (st, _msg, beats_table, _sync_int(current_sync))
            else:
                yield (st, f"⚠️ Story bible refresh returned nothing — continuing with existing context…", beats_table, _sync_int(current_sync))
        except Exception as _bible_err:
            yield (st, f"⚠️ Story bible refresh failed ({_bible_err}) — continuing with existing context…", beats_table, _sync_int(current_sync))

        new_slice = st.beats[old_beat_count:]
        batches = []
        for s in range(0, len(new_slice), bsz):
            chunk = new_slice[s:s + bsz]
            si = old_beat_count + s + 1
            ei = si + len(chunk) - 1
            batches.append((si, ei, chunk))

        # Find the last assigned location from the original story to give Claude continuity
        last_known_loc = ""
        for _bi in range(old_beat_count, 0, -1):
            _plan = (st.beat_plans or {}).get(_bi)
            if _plan:
                _loc = (_plan.get("suggested_location") or "").strip()
                if _loc and _loc.lower() != "none":
                    last_known_loc = _loc
                    break

        # Run batches SEQUENTIALLY so each batch's location decision feeds the next,
        # exactly as Part 1 planning works — parallel batches lose location continuity.
        current_loc = last_known_loc
        for done_count, (si, ei, chunk) in enumerate(batches):
            try:
                plans = _call_timeout(_call_claude_batch_beat_plans, st, si, chunk, current_loc)
                if plans:
                    for bidx, plan in plans.items():
                        st.beat_plans[bidx] = plan
                    # Chain: find last assigned location in this batch to pass to next
                    for _bi in range(ei, si - 1, -1):
                        _p = plans.get(_bi)
                        if _p:
                            _loc = (_p.get("suggested_location") or "").strip()
                            if _loc and _loc.lower() != "none":
                                current_loc = _loc
                                break
            except Exception:
                pass
            progress(0.1 + 0.85 * (done_count + 1) / len(batches), desc=f"Planned beats {si}–{ei}")
            yield (st, f"🔄 Planned beats {si}–{ei}…", beats_table, _sync_int(current_sync))

    _save_project_json(st)
    progress(1.0, desc="Done")
    status = f"✅ '{part_name}' added: {len(new_beats)} new beats (#{old_beat_count + 1}–#{len(st.beats)}). Total: {len(st.beats)} beats."
    yield (st, status, beats_table, _sync_int(current_sync, 1))


def replan_part_cb(st, part_name, current_sync=0, progress=gr.Progress(track_tqdm=False)):
    """Clear and re-run beat planning for a single part without deleting images or beats."""

    def _emit(msg, table=None):
        tbl = table if table is not None else [[str(i+1), b] for i, b in enumerate(st.beats or [])]
        return (st, msg, tbl, _sync_int(current_sync))

    if not st or not st.beats:
        yield _emit("❌ No project loaded.")
        return
    parts = getattr(st, "story_parts", []) or []
    target = next((p for p in parts if p.get("name") == part_name), None)
    if not target:
        yield _emit(f"❌ Part '{part_name}' not found.")
        return

    lo = int(target.get("start_beat") or 1)
    hi = target.get("end_beat")
    hi = int(hi) if hi is not None else len(st.beats)
    beats_table = [[str(i+1), b] for i, b in enumerate(st.beats)]
    n = hi - lo + 1

    yield _emit(f"🔄 Clearing {n} beat plans for '{part_name}'…", beats_table)

    # Clear stale beat plans for this part
    for idx in range(lo, hi + 1):
        if st.beat_plans and idx in st.beat_plans:
            del st.beat_plans[idx]

    import threading as _rp_th

    HARD_TIMEOUT = 55
    def _call_timeout(fn, *args):
        result = [None]
        def _run():
            try:
                result[0] = fn(*args)
            except Exception:
                pass
        t = _rp_th.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout=HARD_TIMEOUT)
        return result[0]

    # Re-run story bible on the full combined story so the planner has complete context
    yield _emit(f"🔄 Rebuilding story bible with full {len(st.beats)}-beat context…", beats_table)
    progress(0.05, desc="Rebuilding story bible…")
    try:
        _full_bible = _call_claude_story_bible(st, st.story, st.beats, build_mode=getattr(st, 'build_mode', 'Panel'))
        if _full_bible:
            _exist_chars_lower = {n.lower() for n in (st.characters or {}).keys()}
            _exist_locs_lower  = {n.lower() for n in (st.locations  or {}).keys()}
            _bible_new = dict(_full_bible)
            _bible_new["characters"] = [
                c for c in (_full_bible.get("characters") or [])
                if (c.get("name") or "").lower() not in _exist_chars_lower
            ]
            _bible_new["locations"] = [
                l for l in (_full_bible.get("locations") or [])
                if (l.get("name") or "").lower() not in _exist_locs_lower
            ]
            _bible_new["items"] = _full_bible.get("items") or []
            _new_chars = _sanitize_character_payload(st, st.story, _bible_new) if _bible_new["characters"] else {}
            _new_locs  = _sanitize_location_payload(_bible_new)                 if _bible_new["locations"]  else {}
            _new_items = _sanitize_item_payload(_bible_new, character_names=set((st.characters or {}).keys()) | set(_new_chars.keys())) if _bible_new["items"] else {}
            _added_chars = {k: v for k, v in _new_chars.items() if k not in (st.characters or {})}
            _added_locs  = {k: v for k, v in _new_locs.items()  if k not in (st.locations  or {})}
            _added_items = {k: v for k, v in _new_items.items()  if k not in (st.items      or {})}
            if not st.characters: st.characters = {}
            if not st.locations:  st.locations  = {}
            if not st.items:      st.items      = {}
            st.characters.update(_added_chars)
            st.locations.update(_added_locs)
            st.items.update(_added_items)
            yield _emit(f"🔄 Bible updated (+{len(_added_chars)} chars, +{len(_added_locs)} locs) — replanning {n} beats…", beats_table)
    except Exception as _be:
        yield _emit(f"⚠️ Story bible refresh failed ({_be}) — continuing with existing context…", beats_table)

    # Sequential batch planning for this part's beat range
    part_beats = st.beats[lo - 1: hi]
    bsz = 20
    batches = []
    for s in range(0, len(part_beats), bsz):
        chunk = part_beats[s:s + bsz]
        si = lo + s
        ei = si + len(chunk) - 1
        batches.append((si, ei, chunk))

    # Seed from the beat just before this part for location continuity
    prior_loc = ""
    for _bi in range(lo - 1, 0, -1):
        _p = (st.beat_plans or {}).get(_bi)
        if _p:
            _l = (_p.get("suggested_location") or "").strip()
            if _l and _l.lower() != "none":
                prior_loc = _l
                break

    current_loc = prior_loc
    for done_count, (si, ei, chunk) in enumerate(batches):
        try:
            plans = _call_timeout(_call_claude_batch_beat_plans, st, si, chunk, current_loc)
            if plans:
                for bidx, plan in plans.items():
                    st.beat_plans[bidx] = plan
                for _bi in range(ei, si - 1, -1):
                    _p = plans.get(_bi)
                    if _p:
                        _l = (_p.get("suggested_location") or "").strip()
                        if _l and _l.lower() != "none":
                            current_loc = _l
                            break
        except Exception:
            pass
        progress(0.1 + 0.85 * (done_count + 1) / len(batches), desc=f"Planned beats {si}–{ei}")
        yield _emit(f"🔄 Replanned beats {si}–{ei}…", beats_table)

    _save_project_json(st)
    progress(1.0, desc="Done")
    yield _emit(f"✅ '{part_name}' replanned: {n} beats ({lo}–{hi}) now use clean beat plans.", beats_table)


def delete_part_cb(st, part_name, current_sync=0):
    """Remove a story part and re-index all beats, plans, prompts and manifest entries."""
    if not st or not st.beats:
        return st, "❌ No project loaded.", gr.Dropdown(choices=[], value=None), [], _sync_int(current_sync)

    parts = getattr(st, "story_parts", []) or []
    target = next((p for p in parts if p.get("name") == part_name), None)
    if not target:
        choices = [p["name"] for p in parts]
        return st, f"❌ Part '{part_name}' not found.", gr.Dropdown(choices=choices, value=None), [[str(i+1), b] for i, b in enumerate(st.beats)], _sync_int(current_sync)

    lo = int(target.get("start_beat") or 1)
    hi = target.get("end_beat")
    hi = int(hi) if hi is not None else len(st.beats)
    offset = hi - lo + 1  # beats removed

    # Collect paths that belong to deleted beats BEFORE re-indexing
    deleted_beat_paths = {
        path for path, bidx in (st.manifest_beat_index or {}).items()
        if lo <= int(bidx or 0) <= hi
    }

    # ── Remove beats and rebuild story text ───────────────────────────────────
    st.beats = st.beats[:lo - 1] + st.beats[hi:]
    st.story = "\n\n".join(st.beats)

    # ── Re-index beat_plans ───────────────────────────────────────────────────
    new_plans: dict = {}
    for idx, plan in (st.beat_plans or {}).items():
        i = int(idx)
        if lo <= i <= hi:
            continue
        new_plans[i - offset if i > hi else i] = plan
    st.beat_plans = new_plans

    # ── Re-index image_prompts ────────────────────────────────────────────────
    new_prompts: dict = {}
    for idx, prompt in (st.image_prompts or {}).items():
        i = int(idx)
        if lo <= i <= hi:
            continue
        new_prompts[i - offset if i > hi else i] = prompt
    st.image_prompts = new_prompts

    # ── Re-index manifest_beat_index (path → beat) ────────────────────────────
    new_mbi: dict = {}
    for path, bidx in (st.manifest_beat_index or {}).items():
        i = int(bidx or 0)
        if lo <= i <= hi:
            continue
        new_mbi[path] = (i - offset if i > hi else i)
    st.manifest_beat_index = new_mbi

    # ── Prune all_manifest_paths ──────────────────────────────────────────────
    if st.all_manifest_paths:
        st.all_manifest_paths = [p for p in st.all_manifest_paths if p not in deleted_beat_paths]

    # ── Re-index story_parts ──────────────────────────────────────────────────
    new_parts = []
    for p in parts:
        if p.get("name") == part_name:
            continue
        plo = int(p.get("start_beat") or 1)
        phi = p.get("end_beat")
        phi_int = int(phi) if phi is not None else None
        if phi_int is not None and phi_int < lo:
            new_parts.append(p)           # entirely before deleted range
        elif plo > hi:
            new_parts.append({            # entirely after — shift both bounds
                "name": p["name"],
                "start_beat": plo - offset,
                "end_beat": (phi_int - offset) if phi_int is not None else None,
            })
    # Last part is always open-ended
    if new_parts and new_parts[-1].get("end_beat") is not None:
        new_parts[-1]["end_beat"] = None
    st.story_parts = new_parts

    # ── Rewrite manifest.jsonl ────────────────────────────────────────────────
    if st.project_dir:
        mf_path = os.path.join(st.project_dir, "manifest.jsonl")
        if os.path.isfile(mf_path):
            try:
                kept: list = []
                with open(mf_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                            bidx = int(entry.get("beat_index") or 0)
                            if lo <= bidx <= hi:
                                continue
                            if bidx > hi:
                                entry["beat_index"] = bidx - offset
                            kept.append(json.dumps(entry, ensure_ascii=False))
                        except Exception:
                            kept.append(line)
                with open(mf_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(kept))
                    if kept:
                        f.write("\n")
            except Exception:
                pass

    _save_project_json(st)

    beats_table = [[str(i + 1), b] for i, b in enumerate(st.beats)]
    new_choices = [p["name"] for p in st.story_parts]
    status = (
        f"✅ Deleted '{part_name}' (beats {lo}–{hi}, {offset} beats removed). "
        f"Project now has {len(st.beats)} beats."
    )
    return st, status, gr.Dropdown(choices=new_choices, value=None), beats_table, _sync_int(current_sync, 1)


def import_zip_cb(zip_file_obj, current_sync=0):
    """Import a project from a ZIP file exported from this tool. Returns loaded ProjectState."""
    import shutil
    import tempfile

    if not zip_file_obj:
        return None, "❌ No file uploaded.", _sync_int(current_sync)

    # Gradio passes either a path string or an object with .name
    zip_path = zip_file_obj if isinstance(zip_file_obj, str) else getattr(zip_file_obj, "name", None)
    if not zip_path or not os.path.isfile(zip_path):
        return None, "❌ Could not read uploaded file.", _sync_int(current_sync)

    tmp_dir = tempfile.mkdtemp(prefix="manhwa_import_")
    try:
        import zipfile as _zf
        with _zf.ZipFile(zip_path, "r") as zf:
            zf.extractall(tmp_dir)

        # Find project.json — may be at root or one level deep
        proj_json_path = None
        src_root = tmp_dir
        for candidate_root in [tmp_dir] + [os.path.join(tmp_dir, d) for d in os.listdir(tmp_dir)
                                             if os.path.isdir(os.path.join(tmp_dir, d))]:
            candidate = os.path.join(candidate_root, "project.json")
            if os.path.isfile(candidate):
                proj_json_path = candidate
                src_root = candidate_root
                break

        if not proj_json_path:
            return None, "❌ No project.json found in ZIP — is this a valid manhwa project export?", _sync_int(current_sync)

        with open(proj_json_path, "r", encoding="utf-8") as f:
            proj_data = json.load(f)

        # Create fresh project dir
        pid = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        dest_dir = os.path.join("projects", pid)
        os.makedirs(dest_dir, exist_ok=True)

        # Copy everything except images_ordered (we'll handle that separately)
        for item in os.listdir(src_root):
            src_item = os.path.join(src_root, item)
            dst_item = os.path.join(dest_dir, item)
            if item == "images_ordered":
                continue  # handled below
            if item in ("images", "thumbnails"):
                continue  # don't import old absolute-path images dir
            if os.path.isdir(src_item):
                shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
            else:
                shutil.copy2(src_item, dst_item)

        # Move images_ordered → images/ with correct names
        imgs_src = os.path.join(src_root, "images_ordered")
        imgs_dst = os.path.join(dest_dir, "images")
        os.makedirs(imgs_dst, exist_ok=True)
        if os.path.isdir(imgs_src):
            for fn in os.listdir(imgs_src):
                shutil.copy2(os.path.join(imgs_src, fn), os.path.join(imgs_dst, fn))

        # Rewrite project.json with new project_id and dir
        proj_data["project_id"] = pid
        with open(os.path.join(dest_dir, "project.json"), "w", encoding="utf-8") as f:
            json.dump(proj_data, f, indent=2, ensure_ascii=False)

        # Rebuild manifest.jsonl so image paths point to the new dest_dir
        mf_src = os.path.join(dest_dir, "manifest.jsonl")
        new_mf_lines = []
        if os.path.isfile(mf_src):
            with open(mf_src, "r", encoding="utf-8") as mf:
                for line in mf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        old_path = entry.get("image_path", "")
                        if old_path:
                            fn = os.path.basename(old_path)
                            entry["image_path"] = os.path.join(dest_dir, "images", fn)
                        new_mf_lines.append(json.dumps(entry, ensure_ascii=False))
                    except Exception:
                        new_mf_lines.append(line)
            with open(mf_src, "w", encoding="utf-8") as mf:
                mf.write("\n".join(new_mf_lines) + "\n")
        else:
            # Reconstruct manifest from beat_image_map.json if present
            bim_path = os.path.join(dest_dir, "beat_image_map.json")
            if os.path.isfile(bim_path):
                with open(bim_path, "r", encoding="utf-8") as f:
                    bim = json.load(f)
                with open(mf_src, "w", encoding="utf-8") as mf:
                    for row in bim:
                        fn = os.path.basename(str(row.get("image", "")))
                        img_path = os.path.join(dest_dir, "images", fn)
                        entry = {"beat_index": row.get("beat", 0), "image_path": img_path}
                        mf.write(json.dumps(entry, ensure_ascii=False) + "\n")

        loaded = _load_project_from_disk(dest_dir)
        msg = (
            f"✅ Imported '{loaded.project_name}' — "
            f"{len(loaded.beats)} beats, {len(loaded.image_paths)} images available locally."
        )
        return loaded, msg, _sync_int(current_sync, 1)

    except Exception as e:
        import traceback
        return None, f"❌ Import failed: {e}\n{traceback.format_exc()[:400]}", _sync_int(current_sync)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def entity_editor_load_cb(st: ProjectState, entity_type: str, entity_name: str, *args):
    empty = ("human", "", "", "", "", "", "", "", "", "", "", "", "", "", "", "prop")
    if not st or not entity_name:
        return empty
    if entity_type == "Location" and entity_name in st.locations:
        l = st.locations[entity_name]
        sub_locations = l.get("sub_locations") or {}
        sub_lines = "\n".join([
            f"{sub}: {(data or {}).get('bible_prompt', '')}" for sub, data in sub_locations.items()
        ])
        return ("human", "", "", "", "", "", "", l.get("bible_prompt", ""), l.get("ref_prompt_base", ""), sub_lines, "", "", "", "", "", "prop")
    if entity_type == "Item" and entity_name in st.items:
        it = st.items[entity_name]
        return ("human", "", "", "", "", "", "", it.get("bible_prompt", ""), it.get("ref_prompt", ""), "", "", "", "", "", "", it.get("kind", "prop"))
    if entity_name in st.characters:
        f = st.characters[entity_name]["fields"]
        alt_text = "\n".join(f.get("alt_outfits") or [])
        return (f.get("character_type","human"), f.get("gender","male"), f.get("hair",""), f.get("eyes",""), f.get("build",""), f.get("skin",""), f.get("outfit",""), f.get("anchor",""), alt_text, "", "", "", "", "", "", "prop")
    return empty


def entity_pick_refresh_cb(st: ProjectState, entity_type: str):
    choices = _entity_choices(st, entity_type)
    return gr.Dropdown(choices=choices, value=(choices[0] if choices else None))


def _stamp_ref_path(folder: str, prefix: str) -> str:
    # Use a versioned filename so Gradio/browser caches do not keep showing the old image
    # after Apply Changes or Regenerate Ref. Old versions are removed first by _clear_old_ref_files.
    stamp = int(time.time() * 1000)
    return os.path.join(folder, f"{prefix}__{stamp}.{FAL_OUTPUT_EXT}")


def _parse_sub_locations_text(raw_text: str, main_location: str) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    def _auto_detail(sub_name: str) -> str:
        low = sub_name.lower()
        if "window" in low:
            return "viewpoint anchored near window frame with clear daylight direction and exterior depth hint"
        if "door" in low or "doorway" in low:
            return "viewpoint anchored at doorway threshold with frame lines and adjacent wall continuity"
        if "closet" in low:
            return "viewpoint anchored near closet doors with panel texture and handle placement continuity"
        if "bed" in low:
            return "viewpoint anchored near bed area with headboard and bedding fold continuity"
        return "viewpoint anchored to this zone with stable nearby architecture and prop placement"
    for line in (raw_text or "").splitlines():
        text = clean_for_prompt(line)
        if not text:
            continue
        if ":" in text:
            name, bp = text.split(":", 1)
            sub_name = clean_for_prompt(name)
            sub_bible = clean_for_prompt(bp)
        else:
            sub_name = text
            sub_bible = ""
        if not sub_name:
            continue
        if not sub_bible:
            sub_bible = f"sub-location inside {main_location.lower()}: {sub_name.lower()}, {_auto_detail(sub_name)}"
        out[sub_name] = {
            "bible_prompt": sub_bible,
            "ref_prompt": clean_for_prompt(
                f"{DEFAULT_STYLE}\n{REF_ANIME_LOCK}\nillustrated anime manhwa environment reference for {sub_name.lower()} in the {main_location.lower()}, "
                f"background only, no people, stable layout anchors\nNO TEXT, NO SIGNS, NO UI, NO WORDS"
            ),
        }
    return out


def _rewrite_location_prompts_with_openai(
    location_name: str,
    bible_prompt: str,
    ref_prompt_base: str,
    sub_locations: Dict[str, Dict[str, str]],
) -> Tuple[str, str, Dict[str, Dict[str, str]], str]:
    clean_bible = clean_for_prompt(bible_prompt)
    clean_ref_base = clean_for_prompt(ref_prompt_base)
    clean_subs: Dict[str, Dict[str, str]] = {}
    for sub_name, sub_data in (sub_locations or {}).items():
        sname = clean_for_prompt(str(sub_name))
        if not sname:
            continue
        sdata = sub_data or {}
        clean_subs[sname] = {
            "bible_prompt": clean_for_prompt(str(sdata.get("bible_prompt") or "")),
            "ref_prompt": clean_for_prompt(str(sdata.get("ref_prompt") or "")),
        }
    if not _rp.has_text_provider("openai"):
        return clean_bible, clean_ref_base, clean_subs, "local (missing active reasoning API key)"
    system = (
        "You refine prompt text for a manhwa reference-image tool. Return ONLY valid JSON. "
        "Keep anime/manhwa 2d style and avoid photorealism. "
        "Preserve location identity and sub-location intent exactly."
    )
    payload = {
        "location_name": location_name,
        "inputs": {
            "bible_prompt": clean_bible,
            "ref_prompt_base": clean_ref_base,
            "sub_locations": clean_subs,
        },
        "output_schema": {
            "bible_prompt": "string",
            "ref_prompt_base": "string",
            "sub_locations": {
                "Sub Name": {"bible_prompt": "string", "ref_prompt": "string"}
            },
        },
        "rules": [
            "Keep each sub-location distinct (doorway vs window view vs closet, etc).",
            "Keep environment-only framing for reference generation.",
            "Keep wording concrete and visually repeatable.",
            "Do not introduce photorealistic language.",
        ],
    }
    raw = _call_openai_text(system, payload, model=OPENAI_PROMPT_MODEL, max_output_tokens=1600)
    data = _extract_json_object(raw or "")
    if not isinstance(data, dict):
        return clean_bible, clean_ref_base, clean_subs, f"local fallback ({OPENAI_PROMPT_MODEL} failed)"
    out_bible = clean_for_prompt(str(data.get("bible_prompt") or clean_bible)) or clean_bible
    out_ref_base = clean_for_prompt(str(data.get("ref_prompt_base") or clean_ref_base)) or clean_ref_base
    out_subs: Dict[str, Dict[str, str]] = {}
    raw_subs = data.get("sub_locations") or {}
    if isinstance(raw_subs, dict):
        for sname, sdata in raw_subs.items():
            sn = clean_for_prompt(str(sname))
            if not sn:
                continue
            d = sdata if isinstance(sdata, dict) else {}
            sb = clean_for_prompt(str(d.get("bible_prompt") or "")) or clean_subs.get(sn, {}).get("bible_prompt", "")
            sr = clean_for_prompt(str(d.get("ref_prompt") or "")) or clean_subs.get(sn, {}).get("ref_prompt", "")
            out_subs[sn] = {"bible_prompt": sb, "ref_prompt": sr}
    if not out_subs:
        out_subs = clean_subs
    return out_bible, out_ref_base, out_subs, _rp.active_model_label(f"OpenAI {OPENAI_PROMPT_MODEL}")


def _rewrite_item_prompts_with_openai(item_name: str, kind: str, bible_prompt: str, ref_prompt: str) -> Tuple[str, str, str]:
    clean_bible = clean_for_prompt(bible_prompt)
    clean_ref = clean_for_prompt(ref_prompt)
    if not _rp.has_text_provider("openai"):
        return clean_bible, clean_ref, "local (missing active reasoning API key)"
    system = (
        "You refine prompt text for anime/manhwa reference-image generation. "
        "Return ONLY valid JSON. Keep 2d illustrated style and avoid photorealism."
    )
    payload = {
        "item_name": item_name,
        "kind": kind,
        "inputs": {"bible_prompt": clean_bible, "ref_prompt": clean_ref},
        "output_schema": {"bible_prompt": "string", "ref_prompt": "string"},
        "rules": [
            "Preserve item identity.",
            "Keep reference prompt isolated and environment-free.",
            "Avoid photorealistic language.",
        ],
    }
    raw = _call_openai_text(system, payload, model=OPENAI_PROMPT_MODEL, max_output_tokens=900)
    data = _extract_json_object(raw or "")
    if not isinstance(data, dict):
        return clean_bible, clean_ref, f"local fallback ({OPENAI_PROMPT_MODEL} failed)"
    out_bible = clean_for_prompt(str(data.get("bible_prompt") or clean_bible)) or clean_bible
    out_ref = clean_for_prompt(str(data.get("ref_prompt") or clean_ref)) or clean_ref
    return out_bible, out_ref, _rp.active_model_label(f"OpenAI {OPENAI_PROMPT_MODEL}")


def _location_single_ref_prompt(location_name: str, location_data: Dict[str, Any]) -> str:
    ldata = location_data or {}
    base = clean_for_prompt(str(ldata.get("ref_prompt_base") or ldata.get("bible_prompt") or ""))
    subs = list(((ldata.get("sub_locations") or {}).keys()))
    subs_text = ""
    if subs:
        preview = ", ".join([clean_for_prompt(s).lower() for s in subs[:6] if clean_for_prompt(s)])
        if preview:
            subs_text = f"\nlocation includes: {preview}"
    return clean_for_prompt(
        f"{base}{subs_text}\nillustrated anime manhwa background only, no people, no powers, no text, stable layout anchors, one master reference image for the full location\nNO TEXT, NO SIGNS, NO UI, NO WORDS"
    )


def _clear_old_ref_files(folder: str, startswith: str) -> None:
    if not os.path.isdir(folder):
        return
    for fn in os.listdir(folder):
        if fn.startswith(startswith) and fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            try:
                os.remove(os.path.join(folder, fn))
            except Exception:
                pass


def entity_editor_apply_cb(st: ProjectState, entity_type: str, entity_name: str, character_type: str, gender: str, hair: str, eyes: str, build: str, skin: str, outfit: str, anchor_or_bible: str, alt_or_ref: str, est: str, med: str, close: str, ots: str, low: str, high: str, item_kind: str, auto_regen_ref: bool, current_sync=0):
    if not st or not entity_name:
        return st, make_char_table_rows(st) if st else [], make_loc_table_rows(st) if st else [], make_item_table_rows(st) if st else [], "❌ Pick an entity first.", [], [], [], _sync_int(current_sync)
    dirs = ensure_dirs(st.project_dir) if st and st.project_dir else None
    char_gallery, loc_gallery, item_gallery = [], [], []
    if entity_type == "Location" and entity_name in st.locations:
        parsed_subs = _parse_sub_locations_text(est, entity_name)
        rewritten_bible, rewritten_ref_base, rewritten_subs, rewrite_source = _rewrite_location_prompts_with_openai(
            entity_name,
            anchor_or_bible,
            alt_or_ref,
            parsed_subs,
        )
        st.locations[entity_name] = build_location_profile(
            entity_name,
            bible_prompt=rewritten_bible,
            shot_overrides={},
            ref_prompt_base=rewritten_ref_base,
            sub_locations=rewritten_subs if rewritten_subs else None,
        )
        if dirs and fal_client is not None and os.getenv("FAL_KEY"):
            _clear_old_ref_files(dirs["refs_locs"], f"loc_{safe_filename(entity_name)}")
            try:
                prompt = _location_single_ref_prompt(entity_name, st.locations[entity_name])
                img = call_fal_generate(prompt, REF_NEGATIVE, skip_esrgan=True)
                out = _stamp_ref_path(dirs["refs_locs"], f"loc_{safe_filename(entity_name)}")
                img.save(out)
            except Exception:
                pass
        status = f"✅ Updated location {entity_name} | Prompt rewrite: {rewrite_source}"
    elif entity_type == "Item" and entity_name in st.items:
        resolved_kind = item_kind or st.items[entity_name].get("kind", "prop")
        rewritten_bible, rewritten_ref, rewrite_source = _rewrite_item_prompts_with_openai(
            entity_name,
            resolved_kind,
            anchor_or_bible,
            alt_or_ref,
        )
        st.items[entity_name] = build_item_profile(
            entity_name,
            kind=resolved_kind,
            bible_prompt=rewritten_bible,
            ref_prompt=rewritten_ref,
        )
        if dirs and fal_client is not None and os.getenv("FAL_KEY"):
            try:
                _clear_old_ref_files(dirs["refs_items"], f"item_{safe_filename(entity_name)}")
                img = call_fal_generate(st.items[entity_name]["ref_prompt"], REF_NEGATIVE, skip_esrgan=True)
                out = _stamp_ref_path(dirs["refs_items"], f"item_{safe_filename(entity_name)}")
                img.save(out)
            except Exception:
                pass
        status = f"✅ Updated item {entity_name} | Prompt rewrite: {rewrite_source}"
    elif entity_name in st.characters:
        profile = st.characters[entity_name]
        fields = dict(profile.get("fields") or {})
        ctype = _sanitize_unknown(character_type, fields.get("character_type", "human")).lower()
        if ctype not in CHARACTER_TYPE_CHOICES:
            ctype = "human"
        fields["character_type"] = ctype
        default_gender = fields.get("gender", "male") or "male"
        fields["gender"] = _sanitize_unknown(gender, default_gender).lower()
        fields["hair"] = _sanitize_unknown(hair, fields.get("hair", "short dark brown messy hair"))
        fields["eyes"] = _sanitize_unknown(eyes, fields.get("eyes", "brown eyes"))
        fields["build"] = _sanitize_unknown(build, fields.get("build", "lean athletic build"))
        fields["skin"] = _sanitize_unknown(skin, fields.get("skin", "warm beige skin"))
        fields["outfit"] = _sanitize_unknown(outfit, fields.get("outfit", "dark gray long-sleeve shirt, dark pants, slippers"))
        anchor_val = _sanitize_unknown(anchor_or_bible, fields.get("anchor", "thin black wristband"))
        if any(word in anchor_val.lower() for word in ["power", "aura", "lightning", "shadow", "system", "mana", "glow"]):
            anchor_val = fields.get("anchor", "thin black wristband")
        fields["anchor"] = anchor_val
        fields["alt_outfits"] = [clean_for_prompt(x) for x in (alt_or_ref or "").splitlines() if clean_for_prompt(x)]
        st.characters[entity_name] = build_character_profile(st.project_id, entity_name, st.story, force_character_type=fields.get("character_type"), override_fields=fields)
        if dirs and fal_client is not None and os.getenv("FAL_KEY"):
            try:
                _clear_old_ref_files(dirs["refs_chars"], safe_filename(entity_name))
                img = call_fal_generate(st.characters[entity_name]["ref_prompt"], REF_NEGATIVE, skip_esrgan=True)
                out = _stamp_ref_path(dirs["refs_chars"], safe_filename(entity_name))
                img.save(out)
            except Exception:
                pass
        status = f"✅ Updated character {entity_name}"
    else:
        status = "❌ Entity not found."
    if dirs:
        for folder, sink in [(dirs["refs_chars"], char_gallery), (dirs["refs_locs"], loc_gallery), (dirs["refs_items"], item_gallery)]:
            if os.path.exists(folder):
                all_fns = sorted(fn for fn in os.listdir(folder) if fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
                for fn in all_fns[-12:]:
                    sink.append(os.path.join(folder, fn))
    _save_project_json(st)
    return st, make_char_table_rows(st), make_loc_table_rows(st), make_item_table_rows(st), status, char_gallery, loc_gallery, item_gallery, _sync_int(current_sync, 1)


def entity_ref_regen_cb(st: ProjectState, entity_type: str, entity_name: str, current_sync=0):
    if not st or not st.project_dir:
        return "❌ Build first.", [], [], [], _sync_int(current_sync)
    if fal_client is None or not os.getenv("FAL_KEY"):
        return "❌ Missing fal_client or FAL_KEY.", [], [], [], _sync_int(current_sync)
    dirs = ensure_dirs(st.project_dir)
    try:
        if entity_type == "Location" and entity_name in st.locations:
            loc_gallery = []
            _clear_old_ref_files(dirs["refs_locs"], f"loc_{safe_filename(entity_name)}")
            prompt = _location_single_ref_prompt(entity_name, st.locations[entity_name])
            img = call_fal_generate(prompt, REF_NEGATIVE)
            out = _stamp_ref_path(dirs["refs_locs"], f"loc_{safe_filename(entity_name)}")
            img.save(out)
            all_locs = sorted(fn for fn in os.listdir(dirs["refs_locs"]) if fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
            loc_gallery = [os.path.join(dirs["refs_locs"], fn) for fn in all_locs[-12:]]
            return f"✅ Regenerated ref for {entity_name}", [], loc_gallery, [], _sync_int(current_sync, 1)
        if entity_type == "Item" and entity_name in st.items:
            _clear_old_ref_files(dirs["refs_items"], f"item_{safe_filename(entity_name)}")
            img = call_fal_generate(st.items[entity_name]["ref_prompt"], REF_NEGATIVE)
            out = _stamp_ref_path(dirs["refs_items"], f"item_{safe_filename(entity_name)}")
            img.save(out)
            all_items = sorted(fn for fn in os.listdir(dirs["refs_items"]) if fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
            item_gallery = [os.path.join(dirs["refs_items"], fn) for fn in all_items[-12:]]
            return f"✅ Regenerated ref for {entity_name}", [], [], item_gallery, _sync_int(current_sync, 1)
        if entity_name in st.characters:
            _clear_old_ref_files(dirs["refs_chars"], safe_filename(entity_name))
            img = call_fal_generate(st.characters[entity_name]["ref_prompt"], REF_NEGATIVE)
            out = _stamp_ref_path(dirs["refs_chars"], safe_filename(entity_name))
            img.save(out)
            all_chars = sorted(fn for fn in os.listdir(dirs["refs_chars"]) if fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
            char_gallery = [os.path.join(dirs["refs_chars"], fn) for fn in all_chars[-12:]]
            return f"✅ Regenerated ref for {entity_name}", char_gallery, [], [], _sync_int(current_sync, 1)
    except Exception as e:
        return f"❌ Ref regen failed: {e}", [], [], [], _sync_int(current_sync)
    return "❌ Pick an entity.", [], [], [], _sync_int(current_sync)


def entity_editor_visibility_cb(entity_type: str):
    is_char = entity_type == "Character"
    is_loc = entity_type == "Location"
    is_item = entity_type == "Item"
    return (
        gr.update(visible=is_char),
        gr.update(visible=is_char or is_loc or is_item),
        gr.update(visible=is_loc),
        gr.update(visible=is_loc),
        gr.update(visible=is_item),
    )



def char_panel_load_cb(st: ProjectState, entity_name: str):
    vals = entity_editor_load_cb(st, "Character", entity_name)
    return vals[0], vals[1], vals[2], vals[3], vals[4], vals[5], vals[6], vals[7], vals[8]


def loc_panel_load_cb(st: ProjectState, entity_name: str):
    vals = entity_editor_load_cb(st, "Location", entity_name)
    return vals[7], vals[8], vals[9]


def item_panel_load_cb(st: ProjectState, entity_name: str):
    vals = entity_editor_load_cb(st, "Item", entity_name)
    kind = vals[15] if vals[15] in ["prop", "power_system"] else "prop"
    return kind, vals[7], vals[8]


def char_panel_apply_cb(st: ProjectState, entity_name: str, character_type: str, gender: str, hair: str, eyes: str, build: str, skin: str, outfit: str, anchor: str, alt: str, auto_regen_ref: bool, current_sync=0):
    return entity_editor_apply_cb(st, "Character", entity_name, character_type, gender, hair, eyes, build, skin, outfit, anchor, alt, "", "", "", "", "", "", "prop", auto_regen_ref, current_sync)


def loc_panel_apply_cb(st: ProjectState, entity_name: str, bible: str, ref_base: str, est: str, auto_regen_ref: bool, current_sync=0):
    # `est` now carries multiline sub-location definitions ("name: bible prompt").
    return entity_editor_apply_cb(st, "Location", entity_name, "human", "", "", "", "", "", "", bible, ref_base, est, "", "", "", "", "", "prop", auto_regen_ref, current_sync)


def item_panel_apply_cb(st: ProjectState, entity_name: str, item_kind: str, bible: str, ref_prompt: str, auto_regen_ref: bool, current_sync=0):
    kind = item_kind if item_kind in ["prop", "power_system"] else "prop"
    return entity_editor_apply_cb(st, "Item", entity_name, "human", "", "", "", "", "", "", bible, ref_prompt, "", "", "", "", "", "", kind, auto_regen_ref, current_sync)

import threading as _bgt

# ── Background build state (module-level so timer can read it) ───────────────
_BUILD_STATE: dict = {
    "running": False,
    "done":    False,
    "gen_id":  0,
    "last_result": None,  # latest tuple from _mid()
    "error":   None,
}

def _launch_build_bg(story, pname, precompute, build_mode, world_context, sync):
    """Start build_everything_cb in a background thread. Returns immediately."""
    gen_id = _BUILD_STATE["gen_id"] + 1
    _BUILD_STATE.update({"running": True, "done": False, "last_result": None,
                          "error": None, "gen_id": gen_id})

    def _run():
        try:
            for result in build_everything_cb(story, pname, True, True, True,
                                              [], precompute, 10, build_mode, world_context, sync):
                if _BUILD_STATE.get("gen_id") != gen_id:
                    break   # newer build started — stop this one
                _BUILD_STATE["last_result"] = result
        except Exception as _e:
            import traceback as _tb
            _BUILD_STATE["error"] = f"❌ {_e}\n{_tb.format_exc()}"
        finally:
            if _BUILD_STATE.get("gen_id") == gen_id:
                _BUILD_STATE.update({"running": False, "done": True})

    _bgt.Thread(target=_run, daemon=True).start()


def build_tab1(state: gr.State, sync_token=None):
    with gr.Tab("Tab 1 — Build & Edit"):
        build_mode_radio = gr.Radio(
            choices=["Panel", "Normal", "Shorts"],
            value="Shorts",
            label="",
            elem_id="build-mode-toggle",
            info="PANEL — 10 beats → 1 vertical 9:16 page image (10 panels/image, ~$0.08/page with NB2).  |  SHORTS — same as Panel but built for 1–2 min short stories: richer world-building from sparse text, no caption boxes, ZIP includes full pages + every panel cropped individually.  |  NORMAL — 1 beat → 1 landscape 16:9 image.",
        )
        project_name_in = gr.Textbox(label="Project Name", placeholder="Give this project a name (e.g. 'Episode 3 — The Tower')", lines=1)
        story_in = gr.Textbox(label="Story", lines=14, placeholder="Paste your story here...")
        with gr.Accordion("🌍 World Context (optional)", open=False):
            gr.Markdown(
                "Anything here gets passed to the AI to help it understand your world — power systems, lore, "
                "the magic rules, how the society works, what era it is, tone, genre, any extra character notes. "
                "It won't appear in panels directly, but Claude reads it when building the visual bible and panel scripts."
            )
            world_context_in = gr.Textbox(
                label="",
                lines=6,
                placeholder="e.g. 'Gate-type hunter world (Solo Leveling style). Hunters are ranked E→S. The MC awakened as a hidden S-rank. Power system: shadow soldiers summoned from corpses. Setting: modern-day Seoul. Tone: dark shonen with power fantasy beats. The guild system is corrupt — the MC distrusts institutions.'"
            )
        precompute_beats = gr.Checkbox(label="Precompute beat suggestions during Build", value=True)

        _sync_in = sync_token if sync_token is not None else gr.Number(value=0, visible=False)

        with gr.Row():
            build_btn = gr.Button("BUILD EVERYTHING", variant="primary", scale=4)
            new_story_btn = gr.Button("🆕 New Story", variant="secondary", scale=1)
        build_status = gr.Textbox(label="Status", interactive=False)

        with gr.Accordion("📋 Beats", open=False):
            beats_table = gr.Dataframe(
                headers=["#", "Beat"], datatype=["str", "str"],
                interactive=False, elem_classes=["compact-table"],
            )
        with gr.Accordion("👥 Characters", open=False):
            char_table = gr.Dataframe(
                headers=["Name","Type","Gender","Hair/Fur/Surface","Eyes","Outfit/Exterior","DNA Prompt","Negative Lock"],
                datatype=["str","str","str","str","str","str","str","str"],
                interactive=False, elem_classes=["compact-table"],
            )
        with gr.Accordion("📅 Age Timeline", open=False):
            gr.Markdown("Auto-extracted at build time. Each phase defines how a character *looks* at that beat range (including flashbacks). The prompt builder uses this to inject the correct age appearance automatically.")
            with gr.Row():
                age_char_dd = gr.Dropdown(label="Character", choices=[], value=None, scale=2)
                age_extract_btn = gr.Button("🔄 Re-extract Timeline", scale=1)
            age_phases_df = gr.Dataframe(
                headers=["#", "Label", "Beats", "Appearance Prompt"],
                datatype=["str", "str", "str", "str"],
                interactive=False, elem_classes=["compact-table"],
            )
            age_phase_idx = gr.State(None)  # index of selected phase (0-based) or None for new
            with gr.Row():
                age_label_box = gr.Textbox(label="Phase label", placeholder="e.g. Newborn, Child age 8, Teen", scale=2)
                age_beat_from = gr.Number(label="Beat from", value=1, minimum=1, precision=0, scale=1)
                age_beat_to   = gr.Number(label="Beat to (-1=end)", value=-1, precision=0, scale=1)
            age_appearance_box = gr.Textbox(
                label="Appearance tokens",
                placeholder="e.g. newborn infant, tiny and fragile, bald head, eyes barely open, wrapped in white cloth",
                lines=2,
            )
            with gr.Row():
                age_ref_img = gr.Image(label="Reference image (upload or generate)", type="pil", scale=2, height=200)
                with gr.Column(scale=1):
                    age_gen_ref_btn   = gr.Button("🎨 Generate Ref Image", variant="secondary")
                    age_save_btn      = gr.Button("💾 Save Phase", variant="primary")
                    age_delete_btn    = gr.Button("🗑 Delete Phase", variant="stop")
                    age_new_btn       = gr.Button("➕ New Phase", variant="secondary")
            age_timeline_status = gr.Textbox(label="Status", interactive=False, lines=1)

        with gr.Accordion("📍 Locations", open=False):
            loc_table = gr.Dataframe(
                headers=["Name","Bible","Sub-locations"], datatype=["str","str","str"],
                interactive=False, elem_classes=["compact-table"],
            )
        with gr.Accordion("🎒 Items", open=False):
            item_table = gr.Dataframe(
                headers=["Item","Kind","Bible","Ref Prompt"], datatype=["str","str","str","str"],
                interactive=False, elem_classes=["compact-table"],
            )

        with gr.Accordion("🖼 Reference Images", open=False):
            with gr.Row():
                char_ref_gallery = gr.Gallery(label="Character Refs", columns=2, height=400)
                loc_ref_gallery = gr.Gallery(label="Location Refs", columns=2, height=400)
                item_ref_gallery = gr.Gallery(label="Item Refs", columns=2, height=400)

        with gr.Accordion("📝 Build Log", open=False):
            build_logs = gr.Textbox(label="Build Log", lines=10, interactive=False)

        with gr.Accordion("📖 Continue Story", open=False):
            gr.Markdown(
                "Add a new chapter or arc to your project. All existing characters, locations, and items "
                "are already known — the new beats will be planned with full context."
            )
            with gr.Row():
                continuation_name_in = gr.Textbox(
                    label="Part name",
                    placeholder='e.g. "Part 2", "Chapter 3", "The Tower Arc"',
                    lines=1,
                    scale=2,
                )
            continuation_text_in = gr.Textbox(
                label="New story text",
                lines=12,
                placeholder="Paste the new chapter or arc text here. It will be split into beats and appended to your existing project.",
            )
            continue_btn = gr.Button("➕ Continue Story", variant="primary")
            continue_status = gr.Textbox(label="Status", interactive=False, lines=1)

            gr.Markdown("---")
            gr.Markdown("### 🔄 Re-plan a Part\n*Clears the beat plans for a part and re-runs the planner with full story context. Use this to fix bad prompts from an old run without deleting images or beats.*")
            with gr.Row():
                replan_part_dd = gr.Dropdown(label="Part to re-plan", choices=[], value=None, scale=3)
                replan_part_btn = gr.Button("🔄 Re-plan Part", variant="primary", scale=1)
            replan_part_status = gr.Textbox(label="Status", interactive=False, lines=1)

            gr.Markdown("---")
            gr.Markdown("### 🗑 Delete a Part\n*Removes a part's beats and all planning data from the project so you can re-add it cleanly. Images already generated are NOT deleted — they stay in cloud storage.*")
            with gr.Row():
                delete_part_dd = gr.Dropdown(label="Part to delete", choices=[], value=None, scale=3)
                delete_part_btn = gr.Button("🗑 Delete Part", variant="stop", scale=1)
            delete_part_status = gr.Textbox(label="Status", interactive=False, lines=1)

        with gr.Column(visible=False):
            gr.Markdown("### Entity Editor (legacy hidden)")
            with gr.Row():
                char_pick = gr.Dropdown(label="Main character quick-pick", choices=[], value=None)
                entity_type = gr.Dropdown(label="Edit type", choices=["Character","Location","Item"], value="Character")
                entity_pick = gr.Dropdown(label="Pick entity", choices=[], value=None)
                apply_btn = gr.Button("Apply Changes", variant="primary")
            with gr.Row(visible=True) as char_fields_row:
                ed_character_type = gr.Dropdown(label="Character type", choices=CHARACTER_TYPE_CHOICES, value="human")
                ed_gender = gr.Dropdown(label="Gender", choices=["male","female"], value="male")
                ed_hair = gr.Textbox(label="Hair / Fur / Surface")
                ed_eyes = gr.Textbox(label="Eyes")
                ed_build = gr.Textbox(label="Build")
                ed_skin = gr.Textbox(label="Skin")
                ed_outfit = gr.Textbox(label="Outfit / Exterior / Shell")
            with gr.Row(visible=True) as shared_fields_row:
                ed_anchor = gr.Textbox(label="Signature detail / Bible prompt")
                ed_alt = gr.Textbox(label="Alt outfits (characters) OR Ref prompt (locations/items)", lines=3)
                ed_kind = gr.Dropdown(label="Item kind", choices=["prop","power_system"], value="prop", visible=False)
            with gr.Row(visible=False) as loc_fields_row_1:
                ed_est = gr.Textbox(label="Establishing Wide shot")
                ed_med = gr.Textbox(label="Medium Scene shot")
                ed_close = gr.Textbox(label="Close Detail shot")
            with gr.Row(visible=False) as loc_fields_row_2:
                ed_ots = gr.Textbox(label="Over-the-Shoulder shot")
                ed_low = gr.Textbox(label="Low Angle shot")
                ed_high = gr.Textbox(label="High Angle shot")

        with gr.Accordion("✏️ Separate Editors", open=False):
            auto_regen_ref = gr.Checkbox(label="Auto-regenerate ref after applying changes", value=True)
            edit_status = gr.Textbox(label="Edit Status", interactive=False)
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Character")
                    char_pick = gr.Dropdown(label="Character", choices=[], value=None)
                    char_type = gr.Dropdown(label="Character type", choices=CHARACTER_TYPE_CHOICES, value="human")
                    char_gender = gr.Dropdown(label="Gender", choices=["male","female"], value="male")
                    char_hair = gr.Textbox(label="Hair / Fur / Surface")
                    char_eyes = gr.Textbox(label="Eyes")
                    char_build = gr.Textbox(label="Build")
                    char_skin = gr.Textbox(label="Skin")
                    char_outfit = gr.Textbox(label="Outfit / Exterior / Shell")
                    char_anchor = gr.Textbox(label="Signature detail / Core marker")
                    char_alt = gr.Textbox(label="Alt outfits", lines=3)
                    char_apply_btn = gr.Button("Apply Changes", variant="primary")
                with gr.Column():
                    gr.Markdown("#### Location")
                    loc_pick = gr.Dropdown(label="Location", choices=[], value=None)
                    loc_bible = gr.Textbox(label="Location bible prompt", lines=3)
                    loc_ref_base = gr.Textbox(label="Location reference base prompt", lines=3)
                    loc_est = gr.Textbox(label="Sub-locations (one per line: name: bible prompt)", lines=6)
                    loc_apply_btn = gr.Button("Apply Changes", variant="primary")
                with gr.Column():
                    gr.Markdown("#### Item / Power")
                    item_pick = gr.Dropdown(label="Item / Power", choices=[], value=None)
                    item_kind_pick = gr.Dropdown(label="Item kind", choices=["prop","power_system"], value="prop")
                    item_bible = gr.Textbox(label="Item / Power bible prompt", lines=3)
                    item_ref = gr.Textbox(label="Item / Power reference prompt", lines=4)
                    item_apply_btn = gr.Button("Apply Changes", variant="primary")

        build_outputs = [state, build_status, beats_table, char_table, loc_table, item_table, char_ref_gallery, loc_ref_gallery, item_ref_gallery, build_logs, char_pick, entity_type, entity_pick, loc_pick, item_pick]
        if sync_token is not None:
            build_outputs.append(sync_token)

        n_out = len(build_outputs)

        # Timer fires every 2 s and pushes latest background-build state to UI
        build_timer = gr.Timer(value=2, active=False)

        def _start_build(story, pname, precompute, build_mode_val, world_ctx, sync):
            """Non-generator: launches build in background and returns immediately."""
            _launch_build_bg(story, pname, precompute, build_mode_val or "Shorts", world_ctx or "", sync)
            updates = [gr.update()] * n_out
            updates[1] = "🚀 Build started — updates every 2 s…"
            return tuple(updates) + (gr.update(active=True),)

        def _poll_build():
            """Called by gr.Timer every 2 s; reads module-level _BUILD_STATE."""
            err = _BUILD_STATE.get("error")
            result = _BUILD_STATE.get("last_result")
            is_done = _BUILD_STATE.get("done") and not _BUILD_STATE.get("running")
            timer_upd = gr.update(active=False) if is_done else gr.update()

            if err and result is None:
                updates = [gr.update()] * n_out
                updates[1] = err
                return tuple(updates) + (gr.update(active=False),)

            if result is None:
                # Still starting — no result yet
                return tuple([gr.update()] * n_out) + (timer_upd,)

            # result is a tuple from _mid() — same length as build_outputs
            result_list = list(result)
            # Pad or trim to match n_out exactly
            if len(result_list) < n_out:
                result_list += [gr.update()] * (n_out - len(result_list))
            else:
                result_list = result_list[:n_out]
            return tuple(result_list) + (timer_upd,)

        build_btn.click(
            _start_build,
            inputs=[story_in, project_name_in, precompute_beats, build_mode_radio, world_context_in, _sync_in],
            outputs=build_outputs + [build_timer],
        )

        def _new_story_cb():
            """Clear all Tab 1 fields so the user can paste a completely new story."""
            empty_beats  = [["", ""] for _ in range(0)]
            empty_chars  = [["", "", "", "", "", "", "", ""] for _ in range(0)]
            empty_locs   = [["", "", ""] for _ in range(0)]
            empty_items  = [["", "", "", ""] for _ in range(0)]
            return (
                None,          # state → blank project
                "",            # project_name_in
                "",            # story_in
                "",            # build_status
                empty_beats,   # beats_table
                empty_chars,   # char_table
                empty_locs,    # loc_table
                empty_items,   # item_table
                [],            # char_ref_gallery
                [],            # loc_ref_gallery
                [],            # item_ref_gallery
                "✅ Ready for a new story — paste it above and click BUILD EVERYTHING.",
            )

        _new_story_outputs = [state, project_name_in, story_in, build_status,
                               beats_table, char_table, loc_table, item_table,
                               char_ref_gallery, loc_ref_gallery, item_ref_gallery,
                               build_logs]
        new_story_btn.click(_new_story_cb, outputs=_new_story_outputs)

        _continue_outputs = [state, continue_status, beats_table]
        if sync_token is not None:
            _continue_outputs.append(sync_token)
        continue_btn.click(
            continue_story_cb,
            inputs=[state, continuation_text_in, continuation_name_in, precompute_beats, _sync_in],
            outputs=_continue_outputs,
        )

        # Refresh all part dropdowns after continuation or project load
        def _parts_choices(st):
            parts = getattr(st, "story_parts", []) or []
            choices = [p["name"] for p in parts]
            return gr.Dropdown(choices=choices, value=None), gr.Dropdown(choices=choices, value=None)

        continue_btn.click(_parts_choices, inputs=[state], outputs=[replan_part_dd, delete_part_dd])

        _replan_outputs = [state, replan_part_status, beats_table]
        if sync_token is not None:
            _replan_outputs.append(sync_token)
        replan_part_btn.click(
            replan_part_cb,
            inputs=[state, replan_part_dd, _sync_in],
            outputs=_replan_outputs,
        )

        _delete_outputs = [state, delete_part_status, delete_part_dd, beats_table]
        if sync_token is not None:
            _delete_outputs.append(sync_token)
        delete_part_btn.click(
            delete_part_cb,
            inputs=[state, delete_part_dd, _sync_in],
            outputs=_delete_outputs,
        )
        # Populate dropdowns whenever state changes (project load via sync token)
        if sync_token is not None:
            sync_token.change(_parts_choices, inputs=[state], outputs=[replan_part_dd, delete_part_dd])

        build_timer.tick(
            _poll_build,
            inputs=[],
            outputs=build_outputs + [build_timer],
        )

        def _char_pick_to_entity(name):
            return gr.update(value="Character"), gr.update(value=name)

        entity_type.change(entity_pick_refresh_cb, inputs=[state, entity_type], outputs=[entity_pick])
        entity_type.change(entity_editor_visibility_cb, inputs=[entity_type], outputs=[char_fields_row, shared_fields_row, loc_fields_row_1, loc_fields_row_2, ed_kind])
        char_pick.change(_char_pick_to_entity, inputs=[char_pick], outputs=[entity_type, entity_pick])
        entity_pick.change(entity_editor_load_cb, inputs=[state, entity_type, entity_pick], outputs=[ed_character_type, ed_gender, ed_hair, ed_eyes, ed_build, ed_skin, ed_outfit, ed_anchor, ed_alt, ed_est, ed_med, ed_close, ed_ots, ed_low, ed_high, ed_kind])

        apply_outputs = [state, char_table, loc_table, item_table, edit_status, char_ref_gallery, loc_ref_gallery, item_ref_gallery]
        if sync_token is not None:
            apply_outputs.append(sync_token)
        apply_btn.click(
            entity_editor_apply_cb,
            inputs=[state, entity_type, entity_pick, ed_character_type, ed_gender, ed_hair, ed_eyes, ed_build, ed_skin, ed_outfit, ed_anchor, ed_alt, ed_est, ed_med, ed_close, ed_ots, ed_low, ed_high, ed_kind, auto_regen_ref, _sync_in],
            outputs=apply_outputs,
        )

        char_pick.change(char_panel_load_cb, inputs=[state, char_pick], outputs=[char_type, char_gender, char_hair, char_eyes, char_build, char_skin, char_outfit, char_anchor, char_alt])
        loc_pick.change(loc_panel_load_cb, inputs=[state, loc_pick], outputs=[loc_bible, loc_ref_base, loc_est])
        item_pick.change(item_panel_load_cb, inputs=[state, item_pick], outputs=[item_kind_pick, item_bible, item_ref])

        char_apply_btn.click(
            char_panel_apply_cb,
            inputs=[state, char_pick, char_type, char_gender, char_hair, char_eyes, char_build, char_skin, char_outfit, char_anchor, char_alt, auto_regen_ref, _sync_in],
            outputs=apply_outputs,
        )
        loc_apply_btn.click(
            loc_panel_apply_cb,
            inputs=[state, loc_pick, loc_bible, loc_ref_base, loc_est, auto_regen_ref, _sync_in],
            outputs=apply_outputs,
        )
        item_apply_btn.click(
            item_panel_apply_cb,
            inputs=[state, item_pick, item_kind_pick, item_bible, item_ref, auto_regen_ref, _sync_in],
            outputs=apply_outputs,
        )

        # ── Populate Tab 1 when a project is loaded from the Projects tab ──────
        if sync_token is not None:
            def _tab1_populate(st: ProjectState):
                if not st or not st.beats:
                    return (
                        gr.update(), gr.update(), "No project loaded.", [], [], [], [],
                        [], [], [],
                        gr.Dropdown(choices=[], value=None),
                        gr.Dropdown(choices=[], value=None),
                        gr.Dropdown(choices=[], value=None),
                        gr.Dropdown(choices=[], value=None),
                        gr.update(), gr.update(),
                    )
                beats_tbl = [[str(i + 1), b] for i, b in enumerate(st.beats)]
                char_rows = make_char_table_rows(st)
                loc_rows = make_loc_table_rows(st)
                item_rows = make_item_table_rows(st)
                # Character gallery: pull from library cast (instant, no generation needed)
                char_gal, loc_gal, item_gal = [], [], []
                if st.character_cast:
                    try:
                        from character_library import load_library as _clib_load2
                        _clib2 = _clib_load2()
                        for cname in st.characters:
                            tid = st.character_cast.get(cname)
                            if not tid:
                                continue
                            t2 = _clib2.get("templates", {}).get(tid)
                            if not t2:
                                continue
                            face_p = t2.get("local_face", "")
                            body_p = t2.get("local_body", "")
                            img_p = face_p if (face_p and os.path.exists(face_p)) else body_p
                            if img_p and os.path.exists(img_p):
                                char_gal.append(img_p)
                    except Exception:
                        pass
                # Location / item refs still come from disk (they're generated images)
                if st.project_dir:
                    for folder, gallery in [
                        (os.path.join(st.project_dir, "refs", "locations"), loc_gal),
                        (os.path.join(st.project_dir, "refs", "items"), item_gal),
                    ]:
                        if os.path.isdir(folder):
                            gallery.extend([
                                os.path.join(folder, fn)
                                for fn in sorted(os.listdir(folder))
                                if fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
                            ])
                char_names = list(st.characters.keys())
                loc_names = list(st.locations.keys())
                item_names = list(st.items.keys())
                # Age timeline dropdown: chars with phases first, then remaining chars
                age_chars = list((st.character_age_phases or {}).keys())
                for c in char_names:
                    if c not in age_chars:
                        age_chars.append(c)
                status = (
                    f"✅ Loaded: {st.project_name} | "
                    f"{len(st.beats)} beats | {len(char_names)} chars | "
                    f"{len(loc_names)} locs | {len(item_names)} items"
                )
                return (
                    st.project_name,
                    getattr(st, "original_story", None) or st.story,
                    status,
                    beats_tbl, char_rows, loc_rows, item_rows,
                    char_gal, loc_gal, item_gal,
                    gr.Dropdown(choices=char_names, value=char_names[0] if char_names else None),
                    gr.Dropdown(choices=loc_names, value=loc_names[0] if loc_names else None),
                    gr.Dropdown(choices=item_names, value=item_names[0] if item_names else None),
                    gr.Dropdown(choices=age_chars, value=age_chars[0] if age_chars else None),
                    getattr(st, "world_context", ""),
                    getattr(st, "build_mode", "Shorts"),
                )

            _tab1_load_outputs = [
                project_name_in, story_in, build_status,
                beats_table, char_table, loc_table, item_table,
                char_ref_gallery, loc_ref_gallery, item_ref_gallery,
                char_pick, loc_pick, item_pick,
                age_char_dd,
                world_context_in, build_mode_radio,
            ]
            sync_token.change(_tab1_populate, inputs=[state], outputs=_tab1_load_outputs)

        # ── Age timeline callbacks ────────────────────────────────────────────

        def _age_phases_table_rows(phases_list):
            rows = []
            for i, p in enumerate(phases_list or []):
                hi = p.get("beat_end", -1)
                beat_range = f"{p.get('beat_start', 1)} → {'end' if hi == -1 else hi}"
                rows.append([str(i + 1), p.get("label", ""), beat_range, (p.get("appearance_prompt", ""))[:90]])
            return rows

        def _age_dd_choices(st):
            if not st:
                return []
            age_chars = list((getattr(st, "character_age_phases", None) or {}).keys())
            for c in list((st.characters or {}).keys()):
                if c not in age_chars:
                    age_chars.append(c)
            return age_chars

        def age_char_refresh_cb(st):
            choices = _age_dd_choices(st)
            return gr.Dropdown(choices=choices, value=choices[0] if choices else None)

        def age_phases_load_cb(st, char_name):
            phases = (getattr(st, "character_age_phases", None) or {}).get(char_name, []) if st and char_name else []
            return _age_phases_table_rows(phases), None, "", 1, -1, "", None, ""

        def age_phase_select_cb(st, char_name, evt: gr.SelectData):
            if not st or not char_name:
                return None, "", 1, -1, "", None, ""
            phases = (getattr(st, "character_age_phases", None) or {}).get(char_name, [])
            idx = evt.index[0] if (evt and hasattr(evt, "index")) else None
            if idx is None or int(idx) >= len(phases):
                return None, "", 1, -1, "", None, ""
            p = phases[int(idx)]
            ref_path = p.get("ref_image_path", "")
            ref_img = None
            if ref_path and os.path.isfile(ref_path):
                try:
                    ref_img = Image.open(ref_path)
                except Exception:
                    pass
            return int(idx), p.get("label", ""), p.get("beat_start", 1), p.get("beat_end", -1), p.get("appearance_prompt", ""), ref_img, f"Editing phase #{int(idx)+1}"

        def age_phase_save_cb(st, char_name, idx, label, beat_from, beat_to, appearance, ref_img_pil):
            if not st or not char_name:
                return st, _age_phases_table_rows([]), "No character selected.", None
            if not (label or "").strip():
                phases = (getattr(st, "character_age_phases", None) or {}).get(char_name, [])
                return st, _age_phases_table_rows(phases), "⚠ Label is required.", None
            if not st.character_age_phases:
                st.character_age_phases = {}
            phases = list(st.character_age_phases.get(char_name, []))
            ref_path = ""
            if idx is not None and int(idx) < len(phases):
                ref_path = phases[int(idx)].get("ref_image_path", "")
            if ref_img_pil is not None:
                stem = f"{char_name.replace(' ', '_')}_{label.strip().replace(' ', '_')}"
                out_dir = os.path.join(st.project_dir, "refs", "age_phases")
                os.makedirs(out_dir, exist_ok=True)
                ref_path = os.path.join(out_dir, f"{stem}.png")
                try:
                    ref_img_pil.save(ref_path, "PNG")
                except Exception:
                    pass
            phase = {
                "label": label.strip(),
                "beat_start": max(1, int(beat_from or 1)),
                "beat_end": int(beat_to if beat_to is not None else -1),
                "appearance_prompt": (appearance or "").strip(),
                "ref_image_path": ref_path,
            }
            if idx is None:
                phases.append(phase)
                msg = f"✅ Added phase '{label.strip()}' for {char_name}."
            else:
                phases[int(idx)] = phase
                msg = f"✅ Updated phase '{label.strip()}' for {char_name}."
            st.character_age_phases[char_name] = phases
            _save_project_json(st)
            return st, _age_phases_table_rows(phases), msg, None

        def age_phase_delete_cb(st, char_name, idx):
            if not st or not char_name or idx is None:
                phases = (getattr(st, "character_age_phases", None) or {}).get(char_name, []) if st and char_name else []
                return st, _age_phases_table_rows(phases), "Select a phase first."
            if not st.character_age_phases:
                st.character_age_phases = {}
            phases = list(st.character_age_phases.get(char_name, []))
            if int(idx) >= len(phases):
                return st, _age_phases_table_rows(phases), "Invalid selection."
            removed = phases.pop(int(idx))
            st.character_age_phases[char_name] = phases
            _save_project_json(st)
            return st, _age_phases_table_rows(phases), f"🗑 Deleted '{removed.get('label', '')}' for {char_name}."

        def age_new_phase_cb():
            return None, "", 1, -1, "", None, "Ready to add a new phase."

        def age_gen_ref_cb(st, char_name, idx, label, appearance):
            if not st or not char_name or not (appearance or "").strip():
                return None, "⚠ Provide a character and appearance tokens first."
            c = (st.characters or {}).get(char_name) or {}
            dna = c.get("dna_prompt", "")
            full_prompt = f"{appearance.strip()}, {dna}" if dna else appearance.strip()
            try:
                pil_img = call_fal_generate(full_prompt, negative_prompt="", skip_esrgan=True)
                if pil_img and st.project_dir:
                    stem = f"{char_name.replace(' ', '_')}_{(label or 'phase').replace(' ', '_')}_ref"
                    out_dir = os.path.join(st.project_dir, "refs", "age_phases")
                    os.makedirs(out_dir, exist_ok=True)
                    ref_path = os.path.join(out_dir, f"{stem}.png")
                    pil_img.save(ref_path, "PNG")
                    if idx is not None:
                        if not st.character_age_phases:
                            st.character_age_phases = {}
                        phases = list(st.character_age_phases.get(char_name, []))
                        if int(idx) < len(phases):
                            phases[int(idx)]["ref_image_path"] = ref_path
                            st.character_age_phases[char_name] = phases
                            _save_project_json(st)
                return pil_img, f"✅ Generated ref image for {char_name} / {label or 'phase'}."
            except Exception as e:
                return None, f"❌ Generation failed: {e}"

        def age_extract_manual_cb(st):
            if not st or not (st.story or "").strip() or not st.beats:
                return st, "No story/beats loaded.", gr.Dropdown(choices=[], value=None)
            try:
                phases = _extract_age_timeline(st, st.story, st.beats)
                if phases:
                    st.character_age_phases = phases
                    _save_project_json(st)
                choices = _age_dd_choices(st)
                found = list(phases.keys()) if phases else []
                msg = f"✅ Extracted timeline for: {', '.join(found)}" if found else "No age phases found (single-age story)."
                return st, msg, gr.Dropdown(choices=choices, value=choices[0] if choices else None)
            except Exception as e:
                return st, f"❌ Error: {e}", gr.Dropdown(choices=[], value=None)

        # Wire age timeline events
        state.change(age_char_refresh_cb, inputs=[state], outputs=[age_char_dd])
        age_char_dd.change(
            age_phases_load_cb, inputs=[state, age_char_dd],
            outputs=[age_phases_df, age_phase_idx, age_label_box, age_beat_from, age_beat_to, age_appearance_box, age_ref_img, age_timeline_status],
        )
        age_phases_df.select(
            age_phase_select_cb, inputs=[state, age_char_dd],
            outputs=[age_phase_idx, age_label_box, age_beat_from, age_beat_to, age_appearance_box, age_ref_img, age_timeline_status],
        )
        age_save_btn.click(
            age_phase_save_cb,
            inputs=[state, age_char_dd, age_phase_idx, age_label_box, age_beat_from, age_beat_to, age_appearance_box, age_ref_img],
            outputs=[state, age_phases_df, age_timeline_status, age_ref_img],
        )
        age_delete_btn.click(
            age_phase_delete_cb, inputs=[state, age_char_dd, age_phase_idx],
            outputs=[state, age_phases_df, age_timeline_status],
        )
        age_new_btn.click(
            age_new_phase_cb, inputs=[],
            outputs=[age_phase_idx, age_label_box, age_beat_from, age_beat_to, age_appearance_box, age_ref_img, age_timeline_status],
        )
        age_gen_ref_btn.click(
            age_gen_ref_cb,
            inputs=[state, age_char_dd, age_phase_idx, age_label_box, age_appearance_box],
            outputs=[age_ref_img, age_timeline_status],
        )
        age_extract_btn.click(
            age_extract_manual_cb, inputs=[state],
            outputs=[state, age_timeline_status, age_char_dd],
        )

        return build_mode_radio


def _call_openai_text_with_status(system: str, user_payload: Dict[str, Any], model: Optional[str] = None, max_output_tokens: int = 900) -> Tuple[Optional[str], str]:
    if _rp.is_deepseek_mode():
        return _rp.call_text(system, user_payload, max_tokens=max_output_tokens, temperature=0)

    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    chosen_model = model or OPENAI_PROMPT_MODEL
    if not api_key:
        return None, f"missing OPENAI_API_KEY (model: {chosen_model})"
    try:
        import requests
    except Exception as e:
        return None, f"requests import failed: {type(e).__name__}: {e}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": chosen_model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system}]},
            {"role": "user", "content": [{"type": "input_text", "text": json.dumps(user_payload, ensure_ascii=False)}]},
        ],
        "max_output_tokens": max_output_tokens,
    }
    import time as _time_mod
    for _attempt in range(3):
        try:
            r = requests.post("https://api.openai.com/v1/responses", headers=headers, json=payload, timeout=180)
        except Exception as e:
            return None, f"OpenAI request failed ({chosen_model}): {type(e).__name__}: {e}"
        status_code = r.status_code
        try:
            data = r.json()
        except Exception:
            data = {}
        if status_code == 429:
            # Rate-limited — wait and retry (up to 2 retries with backoff)
            retry_after = int(r.headers.get("Retry-After", 0)) or (5 * (2 ** _attempt))
            if _attempt < 2:
                _time_mod.sleep(min(retry_after, 30))
                continue
            return None, f"OpenAI rate-limited after 3 attempts ({chosen_model})"
        if status_code >= 400:
            err = data.get("error") if isinstance(data, dict) else None
            if isinstance(err, dict):
                message = err.get("message") or err.get("type") or str(err)
            else:
                message = r.text[:300].strip() or f"HTTP {status_code}"
            return None, f"OpenAI HTTP {status_code} ({chosen_model}): {message}"
        text_out = data.get("output_text") if isinstance(data, dict) else None
        if text_out:
            return str(text_out).strip(), f"ok ({chosen_model})"
        chunks: List[str] = []
        if isinstance(data, dict):
            for item in data.get("output", []):
                for content in item.get("content", []):
                    txt = content.get("text")
                    if txt:
                        chunks.append(txt)
        joined = "\n".join(chunks).strip()
        if joined:
            return joined, f"ok ({chosen_model})"
        return None, f"OpenAI returned no text ({chosen_model})"


def _call_openai_text(system: str, user_payload: Dict[str, Any], model: Optional[str] = None, max_output_tokens: int = 900) -> Optional[str]:
    text, _status = _call_openai_text_with_status(system, user_payload, model=model, max_output_tokens=max_output_tokens)
    return text


PROMPT_COMPOSER_SYSTEM = """
You write production-ready text-to-image prompts for a Korean manhwa / webtoon panel generator.
Return plain prompt text only. No JSON. No explanation. No markdown headers.

=== MANHWA VISUAL DNA — APPLY TO EVERY PROMPT ===
This is Korean manhwa / webtoon, which has a distinct visual language:
- LINE ART: visible bold black ink strokes clearly readable as distinct lines — thick contour edges, thin interior detail lines, NOT smooth painterly blending
- SHADING: hard-edged cel shading with flat color fills separated by sharp ink line borders — DEEP shadow pools — never airbrush, never soft gradient
- LIGHTING: always one dominant directional source — name it specifically (amber candlelight, cold blue window light, neon glow, hard rim light, volumetric shaft, explosion flash backlight)
- COLOR: vivid and saturated — intentional warm-cool contrast — never desaturated, muddy, or washed out
- EYES: large luminous irises with gradient fill, sharp catchlight reflections, fine detailed lashes — eyes must read emotion clearly
- EXPRESSIONS: THEATRICAL and EXAGGERATED — this is the most important element. Real manhwa expressions are never subtle. Use specific descriptors:
    Shock → "jaw dropped, mouth wide open, eyes stretched huge white showing all around iris"
    Fear → "pupils shrunk to dots, cold sweat bead, trembling lip"
    Joy/excitement → "wide crescent-eye grin, mouth open showing all teeth, eyes curved shut"
    Rage → "eyebrows angled V-shape, teeth bared in snarl, forehead vein"
    Awe → "eyes wide sparkling triple catchlight, mouth open O shape"
    Determination → "eyes half-lidded sharp, jaw set, slight smirk corner of mouth"
  Reaction lines, speed lines, and sweat drops around the face are core manhwa language — USE THEM.
  NEVER write: "surprised expression", "emotional look", "intense face" — always name the exact visual.
- COMPOSITION: cinematic and asymmetric — foreground framing elements, strong depth, not centered and static
- Every prompt must ground characters in a physical space with specific atmospheric detail

=== WHAT NOT TO WRITE ===
Never use: "masterpiece", "best quality", "4k", "ultra HD", "best resolution" — Flux models ignore these. Describe what you SEE instead.
Prefer one clear renderable moment over multiple simultaneous events.

=== STYLE LINE — REQUIRED IN EVERY PROMPT ===
Always open with: "Korean manhwa webtoon art style, visible bold black ink line art distinct ink strokes, hard-edged cel shading flat color fills sharp shadow cutoffs, vivid saturated palette warm-cool contrast, large luminous eyes gradient iris sharp catchlight, theatrical exaggerated expressions, 2D illustrated not photorealistic not 3D render"
Then add 2–3 scene-specific mood descriptors that name a specific light source and atmosphere (e.g. "cold blue fluorescent overhead light, rain-streaked window reflections, suffocating tension").
""".strip()


def _make_entity_summary(st: ProjectState, selected_characters: List[str], selected_location: str, selected_items: List[str]) -> Dict[str, Any]:
    chars = []
    for name in selected_characters:
        c = (st.characters or {}).get(name) or {}
        chars.append({
            "name": name,
            "dna_prompt": c.get("dna_prompt", ""),
            "fields": c.get("fields", {}),
        })
    loc = None
    if selected_location and selected_location != "None" and selected_location in (st.locations or {}):
        l = st.locations[selected_location]
        loc = {
            "name": selected_location,
            "bible_prompt": l.get("bible_prompt", ""),
            "shots": l.get("shots", {}),
        }
    items = []
    for name in selected_items:
        it = (st.items or {}).get(name) or {}
        items.append({
            "name": name,
            "kind": it.get("kind", "prop"),
            "bible_prompt": it.get("bible_prompt", ""),
        })
    return {"characters": chars, "location": loc, "items": items}


def _compose_prompt_fallback(st: ProjectState, scene_brief: str, selected_characters: List[str], selected_location: str, selected_items: List[str], perspective: str, shot_notes: str, emotion_notes: str) -> str:
    lines = [DEFAULT_STYLE]
    if selected_location and selected_location != "None" and selected_location in (st.locations or {}):
        loc = st.locations[selected_location]
        lines.append(clean_for_prompt(loc.get("bible_prompt", "")))
    for name in selected_characters:
        c = (st.characters or {}).get(name) or {}
        dna = clean_for_prompt(c.get("dna_prompt", ""))
        if dna:
            lines.append(f"single consistent character design: {dna}")
    item_bits = []
    for name in selected_items:
        it = (st.items or {}).get(name) or {}
        bit = clean_for_prompt(it.get("bible_prompt", ""))
        if bit:
            item_bits.append(bit)
    if item_bits:
        lines.append("story elements: " + "; ".join(item_bits))
    action_block = clean_for_prompt(scene_brief)
    if action_block:
        lines.append(f"dynamic scene action: {action_block}")
    frame_bits = []
    if perspective:
        frame_bits.append(perspective)
    if shot_notes:
        frame_bits.append(clean_for_prompt(shot_notes))
    if frame_bits:
        lines.append("framing: " + ", ".join(frame_bits))
    mood_bits = []
    if emotion_notes:
        mood_bits.append(clean_for_prompt(emotion_notes))
    if mood_bits:
        lines.append("mood: " + ", ".join(mood_bits))
    lines.append("cinematic clarity, visible interaction, readable action space, strong depth, anime manhwa illustration")
    return clean_for_prompt("\n".join([x for x in lines if clean_for_prompt(x)]))


def _get_plan_for_beat(st: ProjectState, beat_number: int) -> Dict[str, Any]:
    if not st or beat_number < 1 or beat_number > len(st.beats or []):
        return {}
    existing = (st.beat_plans or {}).get(int(beat_number))
    if existing:
        return existing
    prior_location = ""
    for idx in range(1, beat_number):
        plan = (st.beat_plans or {}).get(idx)
        if plan and plan.get("suggested_location") and plan.get("suggested_location") != "None":
            prior_location = plan["suggested_location"]
    return _heuristic_beat_plan(st, st.beats[beat_number - 1], beat_number, prior_location=prior_location)


def tab2_refresh_choices_cb(st: ProjectState):
    beat_choices = [str(i + 1) for i in range(len(st.beats or []))]
    char_choices = list((st.characters or {}).keys())
    loc_choices = ["None"] + list((st.locations or {}).keys())
    item_choices = list((st.items or {}).keys())
    return (
        gr.Dropdown(choices=beat_choices, value=(beat_choices[0] if beat_choices else None)),
        gr.CheckboxGroup(choices=char_choices, value=[]),
        gr.Dropdown(choices=loc_choices, value=(loc_choices[0] if loc_choices else "None")),
        gr.CheckboxGroup(choices=item_choices, value=[]),
    )


def tab2_load_beat_cb(st: ProjectState, beat_value: str):
    if not st or not beat_value:
        return "", [], "None", [], "Medium Scene", "", "", "", ""
    try:
        beat_number = int(str(beat_value))
    except Exception:
        return "", [], "None", [], "Medium Scene", "", "", "", ""
    if beat_number < 1 or beat_number > len(st.beats or []):
        return "", [], "None", [], "Medium Scene", "", "", "", ""
    beat_text = st.beats[beat_number - 1]
    plan = _get_plan_for_beat(st, beat_number)
    chars = [c for c in (plan.get("suggested_characters") or []) if c in (st.characters or {})]
    loc = plan.get("suggested_location") or "None"
    if loc not in (["None"] + list((st.locations or {}).keys())):
        loc = "None"
    items = [i for i in (plan.get("suggested_items") or []) if i in (st.items or {})]
    persp = plan.get("suggested_perspective") or "Medium Scene"
    action = clean_for_prompt(plan.get("suggested_action") or beat_text)
    emotions = plan.get("character_emotions") or {}
    emotion_text = "; ".join([f"{k}: {v}" for k, v in emotions.items() if v])
    plan_source = str(plan.get("plan_source") or "")
    plan_summary = f"Beat {beat_number}: {beat_text}\nPlan source: {plan_source or 'none'}\nSuggested action: {action}\nCharacters: {', '.join(chars) if chars else 'None'}\nLocation: {loc}\nItems: {', '.join(items) if items else 'None'}\nPerspective: {persp}"
    return beat_text, chars, loc, items, persp, action, emotion_text, "", plan_summary


def compose_visual_prompt_cb(st: ProjectState, beat_value: str, scene_brief: str, selected_characters: List[str], selected_location: str, selected_items: List[str], perspective: str, shot_notes: str, emotion_notes: str):
    if not st:
        return "", "❌ Build the project first.", ""
    beat_number = 0
    if beat_value:
        try:
            beat_number = int(str(beat_value))
        except Exception:
            beat_number = 0
    beat_text = st.beats[beat_number - 1] if beat_number and 1 <= beat_number <= len(st.beats or []) else ""
    scene_text = clean_for_prompt(scene_brief or beat_text)
    if not scene_text:
        return "", "❌ Add a scene brief or select a beat.", ""
    entity_summary = _make_entity_summary(st, selected_characters or [], selected_location or "None", selected_items or [])
    payload = {
        "style_preset": DEFAULT_STYLE,
        "beat_number": beat_number,
        "beat_text": beat_text,
        "scene_brief": scene_text,
        "selected_perspective": perspective,
        "shot_notes": clean_for_prompt(shot_notes),
        "emotion_notes": clean_for_prompt(emotion_notes),
        "entities": entity_summary,
        "requirements": [
            "Write a visually rich anime/manhwa prompt.",
            "Be concrete about the environment, character appearance, action, framing, lighting, and mood.",
            "Keep the prompt focused on one renderable moment.",
            "Do not add NO TEXT, NO UI, no silhouettes, or similar bans unless explicitly requested.",
            "Keep the final prompt editable and production-ready.",
        ],
    }
    composed = _call_openai_text(PROMPT_COMPOSER_SYSTEM, payload, model=OPENAI_PROMPT_MODEL, max_output_tokens=900)
    source = _rp.active_model_label(f"OpenAI {OPENAI_PROMPT_MODEL}")
    if not composed:
        composed = _compose_prompt_fallback(st, scene_text, selected_characters or [], selected_location or "None", selected_items or [], perspective, shot_notes, emotion_notes)
        source = "local fallback composer"
    status = f"✅ Composed prompt with {source}. Edit it freely, then press Generate."
    return composed, status, source


def generate_scene_image_cb(st: ProjectState, beat_value: str, final_prompt: str, negative_prompt: str, enable_safety_checker: bool):
    if not st or not st.project_dir:
        return None, [], "❌ Build the project first."
    prompt = clean_for_prompt(final_prompt)
    if not prompt:
        return None, [], "❌ Compose or paste a prompt first."
    negative = clean_for_prompt(negative_prompt)
    try:
        img = call_fal_generate(prompt, negative, enable_safety_checker=enable_safety_checker)
        ensure_dirs(st.project_dir)
        beat_num = 0
        try:
            beat_num = int(str(beat_value or "0"))
        except Exception:
            beat_num = 0
        idx = int(st.next_image_index or 1)
        beat_tag = f"beat_{beat_num:03d}" if beat_num > 0 else "beat_manual"
        out_path = os.path.join(st.project_dir, "images", f"image_{idx:04d}_{beat_tag}.{FAL_OUTPUT_EXT}")
        img.save(out_path)
        st.next_image_index = idx + 1
        st.image_paths.append(out_path)
        if beat_num > 0:
            st.images_by_beat.setdefault(beat_num, []).append(out_path)
        _save_project_json(st)
        gallery = list(st.image_paths)
        status = f"✅ Generated image_{idx:04d} using the current editable composed prompt."
        return img, gallery, status
    except Exception as e:
        return None, list(st.image_paths or []), f"❌ Generate failed: {e}"


def build_tab2(state: gr.State, sync_token=None):
    with gr.Tab("Tab 2 — Compose & Generate"):
        gr.Markdown("Compose a specific prompt with GPT-4.1 mini, edit it manually, then generate from the edited text.")
        with gr.Row():
            beat_pick = gr.Dropdown(label="Beat", choices=[], value=None)
            refresh_btn = gr.Button("Refresh From Project")
        beat_text = gr.Textbox(label="Selected beat text", lines=3, interactive=False)
        plan_summary = gr.Textbox(label="Beat plan summary", lines=6, interactive=False)
        scene_brief = gr.Textbox(label="Scene brief / action line", lines=4, placeholder="Describe the exact visible moment you want.")
        with gr.Row():
            selected_characters = gr.CheckboxGroup(label="Characters", choices=[])
            selected_location = gr.Dropdown(label="Location", choices=["None"], value="None")
            selected_items = gr.CheckboxGroup(label="Items / powers", choices=[])
        with gr.Row():
            perspective = gr.Dropdown(label="Perspective", choices=PERSPECTIVES, value="Medium Scene")
            emotion_notes = gr.Textbox(label="Emotion / expression notes")
        shot_notes = gr.Textbox(label="Framing / composition notes", lines=2, placeholder="Example: medium wide shot, readable combat spacing, strong depth")
        with gr.Row():
            compose_btn = gr.Button("Compose Prompt", variant="primary")
            composer_source = gr.Textbox(label="Composer", interactive=False, scale=1)
        composed_prompt = gr.Textbox(label="Editable composed prompt", lines=18, placeholder="The generated prompt will appear here, and you can edit anything before generating.")
        negative_prompt = gr.Textbox(label="Negative prompt", lines=3, value=DEFAULT_NEGATIVE)
        enable_safety_checker = gr.Checkbox(label="Enable fal safety checker", value=False)
        compose_status = gr.Textbox(label="Compose status", interactive=False)
        with gr.Row():
            generate_btn = gr.Button("Generate From Current Prompt", variant="primary")
            gen_status = gr.Textbox(label="Generate status", interactive=False)
        output_image = gr.Image(label="Generated image", type="pil")
        image_gallery = gr.Gallery(label="Project images", columns=3, height=420)

        refresh_btn.click(tab2_refresh_choices_cb, inputs=[state], outputs=[beat_pick, selected_characters, selected_location, selected_items])
        if sync_token is not None:
            sync_token.change(tab2_refresh_choices_cb, inputs=[state], outputs=[beat_pick, selected_characters, selected_location, selected_items])
        beat_pick.change(
            tab2_load_beat_cb,
            inputs=[state, beat_pick],
            outputs=[beat_text, selected_characters, selected_location, selected_items, perspective, scene_brief, emotion_notes, composed_prompt, plan_summary],
        )
        compose_btn.click(
            compose_visual_prompt_cb,
            inputs=[state, beat_pick, scene_brief, selected_characters, selected_location, selected_items, perspective, shot_notes, emotion_notes],
            outputs=[composed_prompt, compose_status, composer_source],
        )
        generate_btn.click(
            generate_scene_image_cb,
            inputs=[state, beat_pick, composed_prompt, negative_prompt, enable_safety_checker],
            outputs=[output_image, image_gallery, gen_status],
        )
        gr.Markdown("Generate always uses the exact text currently inside the editable composed prompt box.")


def build_demo():
    state = gr.State(new_project())
    sync_token = gr.State(0)
    with gr.Blocks(title="Manhwa Director Tool") as demo:
        gr.Markdown("# Manhwa Director Tool")
        build_tab1(state, sync_token=sync_token)
        build_tab2(state, sync_token=sync_token)
    return demo

if __name__ == "__main__":
    demo = build_demo()
    demo.launch()


def build_projects_tab(state: gr.State, sync_token=None):
    with gr.Tab("📁 Projects"):
        gr.Markdown("## Your Projects\nAll projects are saved to disk and persist between sessions. Select a project row, then **Resume** to load it or **Delete** to remove it.")

        with gr.Row():
            refresh_btn = gr.Button("🔄 Refresh List", variant="secondary", scale=0)
            projects_status = gr.Textbox(label="Status", interactive=False, scale=1, show_label=False)

        projects_table = gr.Dataframe(
            headers=["Name", "Beats", "Images", "Last Modified", "Folder"],
            datatype=["str", "number", "number", "str", "str"],
            interactive=False,
            label="Saved Projects",
            wrap=True,
            elem_classes=["compact-table"],
        )

        selected_dir = gr.State(None)

        with gr.Row():
            selected_label = gr.Textbox(label="Selected Project", interactive=False, scale=3)
            resume_btn = gr.Button("▶ Resume Project", variant="primary", scale=1)
            delete_btn = gr.Button("🗑 Delete", variant="stop", scale=0)

        _sync_in = sync_token if sync_token is not None else gr.Number(value=0, visible=False)

        def _load_table():
            projects = _list_all_projects()
            if not projects:
                return [], "No projects found. Create one in Tab 1 — Build."
            rows = [[p["project_name"], p["beats"], p["images"], p["modified"], p["project_dir"]] for p in projects]
            return rows, f"Found {len(projects)} project(s)."

        def _on_select(evt: gr.SelectData, table_data):
            try:
                row_idx = evt.index[0]
                rows = table_data.values.tolist() if hasattr(table_data, "values") else table_data
                if row_idx < len(rows):
                    row = rows[row_idx]
                    name = row[0] if row else ""
                    folder = row[4] if len(row) > 4 else ""
                    return folder, f"{name}  ({folder})"
            except Exception:
                pass
            return None, ""

        def _resume_project(proj_dir, sync):
            if not proj_dir or not os.path.isdir(proj_dir):
                return None, "❌ No project selected. Click a row first.", _sync_int(sync)
            try:
                loaded = _load_project_from_disk(proj_dir)
                msg = (
                    f"✅ Loaded project '{loaded.project_name}' — "
                    f"{len(loaded.beats)} beats, {len(loaded.image_paths)} images locally "
                    f"({len(getattr(loaded, 'all_manifest_paths', None) or [])} in manifest). "
                    f"Tab 2 will refresh automatically."
                )
                return loaded, msg, _sync_int(sync, 1)
            except Exception as e:
                return None, f"❌ Failed to load project: {e}", _sync_int(sync)

        def _delete_project_cb(proj_dir, sync):
            if not proj_dir or not os.path.isdir(proj_dir):
                return "❌ No project selected. Click a row first.", [], _sync_int(sync)
            result = _delete_project(proj_dir)
            table_rows, status = _load_table()
            return result + " " + status, table_rows, _sync_int(sync, 1)

        demo_state = state
        resume_outputs = [demo_state, projects_status]
        delete_outputs = [projects_status, projects_table]
        if sync_token is not None:
            resume_outputs.append(sync_token)
            delete_outputs.append(sync_token)

        refresh_btn.click(_load_table, outputs=[projects_table, projects_status])
        projects_table.select(_on_select, inputs=[projects_table], outputs=[selected_dir, selected_label])
        resume_btn.click(_resume_project, inputs=[selected_dir, _sync_in], outputs=resume_outputs)
        delete_btn.click(_delete_project_cb, inputs=[selected_dir, _sync_in], outputs=delete_outputs)

        gr.Markdown("---")
        with gr.Accordion("📥 Import Project from ZIP", open=False):
            gr.Markdown(
                "Upload a `.zip` file exported from this tool. "
                "A new project will be created from the ZIP contents and loaded into Tab 2 automatically."
            )
            import_zip_file = gr.File(label="Upload ZIP", file_types=[".zip"], type="filepath")
            import_btn = gr.Button("📥 Import ZIP", variant="primary")
            import_status = gr.Textbox(label="Import Status", interactive=False, lines=1)

        import_outputs = [demo_state, import_status]
        if sync_token is not None:
            import_outputs.append(sync_token)
        import_btn.click(
            import_zip_cb,
            inputs=[import_zip_file, _sync_in],
            outputs=import_outputs,
        )

        gr.Markdown("---")
        with gr.Accordion("✂️ Cut & Zip", open=True):
            gr.Markdown(
                "Upload page images in any order — they'll be sorted by filename, "
                "cut into individual panels, and packaged into a ZIP with **pages/** and **panels/** folders ready to download."
            )
            with gr.Row():
                cz_files = gr.File(
                    label="Upload page images",
                    file_types=[".png", ".jpg", ".jpeg", ".webp"],
                    file_count="multiple",
                    type="filepath",
                    scale=3,
                )
                with gr.Column(scale=1):
                    cz_panels_dd = gr.Dropdown(
                        label="Panels per page",
                        choices=[2, 4, 6, 8, 9, 10, 12],
                        value=6,
                    )
                    cz_cols_dd = gr.Dropdown(
                        label="Columns",
                        choices=[1, 2, 3],
                        value=2,
                    )
                    cz_btn = gr.Button("✂️ Cut & Zip", variant="primary")
            with gr.Row():
                cz_out  = gr.File(label="Download ZIP", interactive=False, scale=2)
                cz_status = gr.Textbox(label="Status", interactive=False, lines=2, scale=3)

        def _do_cut_zip(file_objs, panels_per_page, n_cols):
            """Sort uploaded images naturally, crop into panels, return a ZIP."""
            try:
                if not file_objs:
                    return None, "⬆ Upload at least one image first."

                # Normalise to list of path strings
                paths = []
                for f in (file_objs if isinstance(file_objs, list) else [file_objs]):
                    p = f if isinstance(f, str) else getattr(f, "name", None)
                    if p and os.path.isfile(p):
                        paths.append(p)

                if not paths:
                    return None, "❌ No readable files found."

                # Natural-sort by basename so page001 < page002 < page010
                def _nat_key(p):
                    parts = re.split(r"(\d+)", os.path.basename(p).lower())
                    return [int(x) if x.isdigit() else x for x in parts]

                paths.sort(key=_nat_key)

                panels_per_page = int(panels_per_page or 6)
                n_cols          = int(n_cols or 2)
                n_rows          = (panels_per_page + n_cols - 1) // n_cols

                zp = f"/tmp/cut_zip_{int(time.time())}.zip"
                panel_total = 0

                with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as zf:
                    for pg_num, src in enumerate(paths, start=1):
                        ext      = os.path.splitext(src)[1].lstrip(".") or "png"
                        pg_name  = f"page_{pg_num:03d}.{ext}"
                        zf.write(src, f"pages/{pg_name}")

                        # Crop into panels
                        try:
                            from PIL import Image as _PI
                            im = _PI.open(src).convert("RGB")
                            w, h = im.size
                            pw = w // n_cols
                            ph = h // n_rows
                            for r in range(n_rows):
                                for c in range(n_cols):
                                    panel_num = r * n_cols + c + 1
                                    if panel_num > panels_per_page:
                                        break
                                    crop = im.crop((c * pw, r * ph, (c + 1) * pw, (r + 1) * ph))
                                    buf  = io.BytesIO()
                                    crop.save(buf, format="PNG")
                                    zf.writestr(f"panels/panel_{pg_num:03d}_{panel_num:02d}.png", buf.getvalue())
                                    panel_total += 1
                        except Exception:
                            pass  # skip panel crop for unreadable images

                return zp, f"✅ {len(paths)} pages, {panel_total} panel crops — ready to download."
            except Exception as e:
                return None, f"❌ Error: {e}"

        cz_btn.click(_do_cut_zip, inputs=[cz_files, cz_panels_dd, cz_cols_dd], outputs=[cz_out, cz_status])

    return _load_table, [projects_table, projects_status]
