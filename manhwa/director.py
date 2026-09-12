import os
import re
import json
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import List, Tuple, Optional, Any, Dict

# ── Per-generation step log ────────────────────────────────────────────────────
# Cleared at the start of each generate_cb run so the Tab 2 log box always shows
# only the most recent generation's steps in order.
_last_gen_steps: List[str] = []

import gradio as gr

# ── Patch Gradio components to tolerate stale browser-cached values ──────────────
# After a workflow restart the browser may have old selected values while the
# server-side components have choices=[].  The default Gradio preprocess raises
# an Error in that case.  These patches return a safe default instead.

def _lenient_cbg_preprocess(self, payload):
    """CheckboxGroup: filter out values that aren't in current choices."""
    if not payload:
        return []
    choice_values = [v for _, v in self.choices] if self.choices else []
    valid = [v for v in payload if not choice_values or v in choice_values]
    if self.type == "index":
        return [choice_values.index(v) for v in valid if v in choice_values]
    return valid

def _lenient_dd_preprocess(self, payload):
    """Dropdown: return None for stale values instead of raising."""
    if payload is None:
        return None
    choice_values = [v for _, v in self.choices] if self.choices else []
    if not self.allow_custom_value and choice_values:
        if isinstance(payload, list):
            payload = [v for v in payload if v in choice_values]
        elif payload not in choice_values:
            return None  # stale cached value — pass None gracefully
    if self.type == "value":
        return payload
    elif self.type == "index":
        if isinstance(payload, list):
            return [choice_values.index(c) if c in choice_values else None for c in payload]
        return choice_values.index(payload) if payload in choice_values else None
    return payload

import gradio.components.checkboxgroup as _cbg_mod
import gradio.components.dropdown as _dd_mod
_cbg_mod.CheckboxGroup.preprocess = _lenient_cbg_preprocess
_dd_mod.Dropdown.preprocess = _lenient_dd_preprocess
# ─────────────────────────────────────────────────────────────────────────────────

from build import (
    ProjectState,
    DEFAULT_STYLE,
    DEFAULT_NEGATIVE,
    FAL_OUTPUT_EXT,
    ensure_dirs,
    clean_for_prompt,
    strip_shot_labels,
    call_fal_generate,
    upload_pil_to_fal,
    zip_folder,
    _save_project_json,
    _call_openai_text_with_status,
    OPENAI_PROMPT_MODEL,
    age_phase_for_beat,
    get_active_form_for_beat,
    PANELS_PER_PAGE,
    SHORTS_PANELS_PER_PAGE,
    regenerate_page_scripts_cb,
)

GALLERY_PREVIEW_LIMIT = 40        # max images shown in "All beats" gallery
THUMB_SIZE = (270, 480)           # 9:16 thumbnail for portrait manhwa pages

def _ppp(st) -> int:
    """Panels per page — 6 for Shorts mode, 10 for Panel mode."""
    return SHORTS_PANELS_PER_PAGE if getattr(st, 'build_mode', 'Panel') == 'Shorts' else PANELS_PER_PAGE

import cloud_storage as _cloud

import threading
_BATCH_PAUSE_EVENT = threading.Event()
_BATCH_STOP_EVENT = threading.Event()
_BATCH_SAVE_LOCK = threading.Lock()  # serialises file writes during parallel generation

# ── Background generation job state ─────────────────────────────────────────
# Stores the latest result from the running batch so the UI timer can poll it
# even after the browser WebSocket drops (wifi loss, tab close, etc.).
_BG_JOB: dict = {
    "running": False,
    "cancel": False,
    "status": "idle",
    "latest_state": None,
    "latest_img": None,
    # next_start_consumed: True once the final post-batch beat advance has been
    # delivered to the UI. Reset to False each time a new job starts so the
    # auto-advance fires exactly once after completion, then stops so the user
    # can freely edit the field without the timer overwriting it every 4 s.
    "next_start_consumed": True,
}
_BG_LOCK = threading.Lock()

_NTFY_TOPIC: str = ""  # set from UI — ntfy.sh topic for push notifications


def _send_ntfy_notification(msg: str) -> None:
    """Send a push notification via ntfy.sh (fire-and-forget)."""
    topic = _NTFY_TOPIC.strip()
    if not topic:
        return
    try:
        import requests as _req
        _req.post(
            f"https://ntfy.sh/{topic}",
            data=msg.encode("utf-8"),
            headers={
                "Title": "Manhwa Generator",
                "Priority": "high",
                "Tags": "art,framed_picture",
            },
            timeout=8,
        )
    except Exception:
        pass


def _get_thumb_path(img_path: str) -> str:
    """Map a full-res image path → its thumbnail JPEG path."""
    project_dir = os.path.dirname(os.path.dirname(img_path))
    base = os.path.splitext(os.path.basename(img_path))[0]
    return os.path.join(project_dir, "thumbnails", base + ".jpg")


def _thumb_to_original(thumb_path: str) -> str:
    """Reverse _get_thumb_path: thumbnail path → best-guess original image path.
    Gallery shows thumbnails; manifest stores originals — this bridges the gap."""
    parts = thumb_path.replace("\\", "/").split("/")
    if "thumbnails" not in parts:
        return thumb_path
    base = os.path.splitext(os.path.basename(thumb_path))[0]
    project_dir = os.path.dirname(os.path.dirname(thumb_path))
    images_dir = os.path.join(project_dir, "images")
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        candidate = os.path.join(images_dir, base + ext)
        if os.path.isfile(candidate):
            return candidate
    return os.path.join(images_dir, base + ".png")


def _make_thumb(img_path: str) -> str:
    """Generate a JPEG thumbnail. Returns thumb path on success, original path on failure."""
    thumb = _get_thumb_path(img_path)
    if os.path.isfile(thumb):
        return thumb
    try:
        from PIL import Image as _PIL_Image
        os.makedirs(os.path.dirname(thumb), exist_ok=True)
        with _PIL_Image.open(img_path) as im:
            im = im.convert("RGB")
            im.thumbnail(THUMB_SIZE, _PIL_Image.LANCZOS)
            im.save(thumb, "JPEG", quality=82, optimize=True)
        _cloud.upload_file_bg(thumb)
        return thumb
    except Exception:
        return img_path


def _ensure_thumbs_bg(image_paths: List[str]) -> None:
    """Background thread: generate thumbnails for any images missing them."""
    for p in image_paths:
        try:
            if not os.path.isfile(_get_thumb_path(p)):
                _make_thumb(p)
        except Exception:
            pass


def _call_openai_text(system: str, user_payload: Dict[str, Any], model: Optional[str] = None, max_output_tokens: int = 900) -> Optional[str]:
    text, _status = _call_openai_text_with_status(system, user_payload, model=model, max_output_tokens=max_output_tokens)
    return text

CAMERA_TYPES: List[str] = [
    "extreme close-up — eyes fill frame, lashes visible",
    "tight face close-up — chin to brow, raw emotion",
    "over-the-shoulder — camera behind ear, looking past shoulder at scene",
    "profile shot — strict 90-degree side view, facing left or right",
    "dramatic medium — torso to head, three-quarter angle, slight low",
    "dutch-angle medium — camera tilted 20 degrees, diagonal tension",
    "low-angle shot — camera below waist, looking up at towering figure",
    "bird's-eye high-angle — camera far above, figure small below",
    "full-body silhouette — dark figure against bright backlit backdrop",
    "wide establishing shot — character tiny, vast environment dominates",
    "dynamic action blur — speed lines radiate, smear on moving element",
    "two-shot framing — two figures facing each other, tension between them",
]

# Rich prompt expansion for each camera type — used in prompt composition
WILSON_BRACKET_MAP: Dict[str, str] = {
    "extreme close-up":    "(ECU eye-level)",
    "tight face close-up": "(70mm Close-Up eye-level)",
    "over-the-shoulder":   "(50mm Over-the-Shoulder eye-level)",
    "profile shot":        "(50mm Medium Shot side-angle)",
    "dramatic medium":     "(50mm Medium Close-Up low angle)",
    "dutch-angle medium":  "(50mm Medium Shot Dutch Angle)",
    "low-angle shot":      "(24mm Low Angle looking up)",
    "bird's-eye high-angle": "(24mm Bird's Eye View overhead)",
    "full-body silhouette": "(24mm Wide Shot backlit)",
    "wide establishing shot": "(14mm Extreme Wide Shot eye-level)",
    "dynamic action blur": "(24mm Medium Shot low angle dynamic)",
    "two-shot framing":    "(50mm Medium Shot eye-level, two figures in frame)",
}

WILSON_ERA_MAP: Dict[str, str] = {
    "ACTION":    "2020s battle shonen anime-style",
    "EMOTION":   "2020s modern shonen anime-style",
    "DIALOGUE":  "2020s modern shonen anime-style",
    "MEMORY":    "2010s nostalgic shonen anime-style",
    "AWAKENING": "2020s battle shonen anime-style",
    "AFTERMATH": "2020s dark seinen anime-style",
}

WILSON_STUDIO_MAP: Dict[str, str] = {
    "ACTION":    "MAPPA style",
    "EMOTION":   "Kyoto Animation style",
    "DIALOGUE":  "Kyoto Animation style",
    "MEMORY":    "Kyoto Animation style",
    "AWAKENING": "Ufotable style",
    "AFTERMATH": "Studio Wit style",
}

WILSON_LIGHTING_MAP: Dict[str, str] = {
    "ACTION":    "dramatic rim lighting casting stark shadows and vivid highlights",
    "EMOTION":   "warm soft ambient lighting with gentle diffused shadows",
    "DIALOGUE":  "cinematic three-point lighting with soft rim highlight",
    "MEMORY":    "warm nostalgic amber lighting with soft diffusion",
    "AWAKENING": "intense burst lighting with brilliant god-rays and harsh rim",
    "AFTERMATH": "cold desaturated lighting with long harsh shadows",
}

CAMERA_SHOT_MAP: Dict[str, str] = {
    "extreme close-up — eyes fill frame, lashes visible":
        "EXTREME CLOSE-UP: both eyes fill entire frame edge-to-edge, individual eyelashes sharp, iris and pupil in crisp detail, no body visible, soft bokeh background",
    "tight face close-up — chin to brow, raw emotion":
        "TIGHT FACE CLOSE-UP: face from chin to brow fills the frame completely, jawline and cheekbones visible, raw unfiltered expression, no background",
    "over-the-shoulder — camera behind ear, looking past shoulder at scene":
        "OVER-THE-SHOULDER SHOT: camera positioned behind and above one character's ear, shoulder occupies lower corner of frame, camera looks past them toward the subject or scene ahead",
    "profile shot — strict 90-degree side view, facing left or right":
        "STRICT PROFILE SHOT: exact 90-degree side view, character faces directly left (or right), nose chin and ear in clean silhouette, flat side-profile composition, background recedes",
    "dramatic medium — torso to head, three-quarter angle, slight low":
        "DRAMATIC MEDIUM SHOT: framed from mid-torso to top of head, camera slightly below eye level at three-quarter angle, face positioned at upper third of frame",
    "dutch-angle medium — camera tilted 20 degrees, diagonal tension":
        "DUTCH ANGLE MEDIUM SHOT: camera rotated 20-25 degrees, horizon line slashes diagonally across frame, creating visual unease and tension, waist-to-head framing",
    "low-angle shot — camera below waist, looking up at towering figure":
        "LOW-ANGLE SHOT: camera placed below waist level, looking up at the figure from below, figure towers against sky or ceiling, conveying power or menace, feet visible at bottom",
    "bird's-eye high-angle — camera far above, figure small below":
        "BIRD'S-EYE HIGH-ANGLE SHOT: camera directly overhead or far above looking down, figure occupies small portion of frame, architecture or ground fills most of frame, lonely and exposed feeling",
    "full-body silhouette — dark figure against bright backlit backdrop":
        "FULL-BODY SILHOUETTE SHOT: character rendered as complete dark silhouette, intense backlight from window sun or fire creates rim-light edge, environment behind is bright and richly detailed",
    "wide establishing shot — character tiny, vast environment dominates":
        "WIDE ESTABLISHING SHOT: character occupies under 15 percent of frame height, vast environment completely dominates, conveying scale and isolation, character nearly lost in space",
    "dynamic action blur — speed lines radiate, smear on moving element":
        "DYNAMIC ACTION BLUR SHOT: radial speed lines emanate from focal point, motion smear on the fastest-moving element, camera appears to track rapid movement, background streaks",
    "two-shot framing — two figures facing each other, tension between them":
        "TWO-SHOT FRAMING: two characters both fully in frame facing each other, negative space between them charged with tension, neither dominates, relationship is the subject",
}

SCENE_TYPES: List[str] = ["ACTION", "EMOTION", "DIALOGUE", "MEMORY", "AWAKENING", "AFTERMATH"]

SCENE_CAMERA_POOLS: Dict[str, List[str]] = {
    "ACTION": [
        "low-angle shot — camera below waist, looking up at towering figure",
        "dynamic action blur — speed lines radiate, smear on moving element",
        "tight face close-up — chin to brow, raw emotion",
        "dutch-angle medium — camera tilted 20 degrees, diagonal tension",
        "wide establishing shot — character tiny, vast environment dominates",
        "extreme close-up — eyes fill frame, lashes visible",
    ],
    "EMOTION": [
        "extreme close-up — eyes fill frame, lashes visible",
        "tight face close-up — chin to brow, raw emotion",
        "profile shot — strict 90-degree side view, facing left or right",
        "full-body silhouette — dark figure against bright backlit backdrop",
        "dramatic medium — torso to head, three-quarter angle, slight low",
        "bird's-eye high-angle — camera far above, figure small below",
    ],
    "DIALOGUE": [
        "over-the-shoulder — camera behind ear, looking past shoulder at scene",
        "two-shot framing — two figures facing each other, tension between them",
        "tight face close-up — chin to brow, raw emotion",
        "profile shot — strict 90-degree side view, facing left or right",
        "extreme close-up — eyes fill frame, lashes visible",
        "dutch-angle medium — camera tilted 20 degrees, diagonal tension",
    ],
    "MEMORY": [
        "wide establishing shot — character tiny, vast environment dominates",
        "full-body silhouette — dark figure against bright backlit backdrop",
        "dramatic medium — torso to head, three-quarter angle, slight low",
        "profile shot — strict 90-degree side view, facing left or right",
        "bird's-eye high-angle — camera far above, figure small below",
    ],
    "AWAKENING": [
        "low-angle shot — camera below waist, looking up at towering figure",
        "extreme close-up — eyes fill frame, lashes visible",
        "dutch-angle medium — camera tilted 20 degrees, diagonal tension",
        "bird's-eye high-angle — camera far above, figure small below",
        "dramatic medium — torso to head, three-quarter angle, slight low",
        "full-body silhouette — dark figure against bright backlit backdrop",
    ],
    "AFTERMATH": [
        "wide establishing shot — character tiny, vast environment dominates",
        "bird's-eye high-angle — camera far above, figure small below",
        "full-body silhouette — dark figure against bright backlit backdrop",
        "profile shot — strict 90-degree side view, facing left or right",
        "dramatic medium — torso to head, three-quarter angle, slight low",
        "tight face close-up — chin to brow, raw emotion",
    ],
}

PERSPECTIVES: List[Tuple[str, str]] = [(c, "camera type") for c in CAMERA_TYPES]

TEXT_TYPES: List[str] = [
    "None",
    "🌶️ Auto (seasonal)",
    "Speech Bubble",
    "Thought Bubble",
    "Narration Box",
    "System Message",
    "SFX (Sound Effect)",
]

_SESSION_COST: dict = {"total": 0.0, "images": 0, "openai_calls": 0, "build_calls": 0}

FAL_COST_MAP: dict = {
    "fal-ai/z-image/turbo":        0.005,
    "fal-ai/nano-banana-2":        0.080,
    "fal-ai/nano-banana-2/edit":   0.080,
    "fal-ai/nano-banana-pro":      0.150,
}
# Resolution multipliers per FAL pricing page
FAL_RESOLUTION_MULTIPLIER: dict = {
    "0.5K": 0.75,
    "512":  0.75,
    "1K":   1.00,
    "2K":   1.50,
    "4K":   2.00,
}
THINKING_COST = 0.002          # flat fee when thinking_level="high" is used
OPENAI_PROMPT_COST = 0.0001   # ~gpt-4.1-mini per call estimate


def _calc_fal_cost(fal_model: str, resolution: str = "1K", use_thinking: bool = False) -> float:
    """Return the true per-image FAL cost including resolution multiplier and thinking fee."""
    base = FAL_COST_MAP.get(fal_model, 0.005)
    mult = FAL_RESOLUTION_MULTIPLIER.get(resolution, 1.0)
    thinking_fee = THINKING_COST if use_thinking else 0.0
    return round(base * mult + thinking_fee, 6)


def _add_session_cost(fal_model: str, used_openai: bool,
                      resolution: str = "1K", use_thinking: bool = False) -> None:
    global _SESSION_COST
    fal_cost = _calc_fal_cost(fal_model, resolution=resolution, use_thinking=use_thinking)
    _SESSION_COST["total"] = round(_SESSION_COST["total"] + fal_cost + (OPENAI_PROMPT_COST if used_openai else 0.0), 6)
    _SESSION_COST["images"] += 1
    if used_openai:
        _SESSION_COST["openai_calls"] += 1


def _format_session_cost() -> str:
    sc = _SESSION_COST
    return f"💰 Project total: ${sc['total']:.4f}  |  🖼 {sc['images']} imgs  |  🤖 {sc['openai_calls']} AI calls"


def _auto_season_text(beat_text: str, beat_index: int, build_mode: str = "Panel") -> Tuple[str, str]:
    """Decide whether this beat gets in-panel text, and what kind/content.
    High-priority cues always get text. Others use a ~50% gate.
    Extracts text directly from the beat — no extra API call needed.
    In Shorts mode, caption/narration boxes are suppressed — only speech
    bubbles, thought bubbles, and SFX are allowed."""
    import hashlib, re
    h = int(hashlib.md5(f"season_{beat_index}".encode()).hexdigest(), 16)
    roll = (h % 100) / 100.0
    txt = beat_text.lower()
    is_shorts = build_mode == "Shorts"

    sfx_cues = ["crash", "bang", "slam", "explosion", "shatter", "crack", "boom", "roar", "thud", "clang", "snap", "rattle", "swoosh", "slash"]
    thought_cues = ["thought", "wondered", "realized", "felt", "remembered", "thinking", "mind raced", "in his head", "in her head", "memories", "memory", "woke up", "waking", "inside my skull", "inside his skull", "inside her skull"]
    inner_voice_cues = ["my name", "i told myself", "i knew", "i couldn't", "i didn't", "i was", "i had", "i felt", "i tried", "i woke"]
    location_cues = ["entered", "arrived", "walked into", "stepped into", "stood at", "overlooked", "reached", "approached"]
    dual_reality_cues = ["two", "both", "simultaneously", "at once", "same time", "fighting inside", "two sets", "two worlds", "two versions", "couldn't tell"]

    def _short(text: str, n: int = 55) -> str:
        text = text.strip()
        return (text[:n] + "…") if len(text) > n else text

    # Beat 1 always gets a narration intro (suppressed in Shorts)
    if beat_index == 1:
        if is_shorts:
            return "None", ""
        return "Narration Box", _short(beat_text)

    # Dialogue — extract quoted text first (always show, even in Shorts)
    quoted = re.findall(r'["\u201c\u201d](.*?)["\u201c\u201d]', beat_text)
    if quoted:
        return "Speech Bubble", _short(quoted[0], 40)

    # Dual reality / dual memory beats — narration (suppressed in Shorts)
    if any(c in txt for c in dual_reality_cues):
        if is_shorts:
            return "None", ""
        return "Narration Box", _short(beat_text)

    # Inner voice narration — narration box (suppressed in Shorts)
    if any(c in txt for c in inner_voice_cues):
        if is_shorts:
            return "None", ""
        return "Narration Box", _short(beat_text)

    # Inner thought cues — thought bubble (allowed in Shorts)
    if any(c in txt for c in thought_cues):
        words = beat_text.split()
        snippet = " ".join(words[:9])
        if len(words) > 9:
            snippet += "…"
        return "Thought Bubble", snippet

    # Sound effect (always show, even in Shorts)
    for cue in sfx_cues:
        if cue in txt:
            return "SFX (Sound Effect)", cue.upper() + "!!"

    # Location arrival — narration card (suppressed in Shorts)
    if any(c in txt for c in location_cues):
        if is_shorts:
            return "None", ""
        return "Narration Box", _short(beat_text)

    # Dialogue verbs without quotes — speech bubble (allowed in Shorts)
    dialogue_verbs = ["said", "asked", "replied", "shouted", "whispered", "yelled", "told", "exclaimed", "muttered"]
    if any(c in txt for c in dialogue_verbs):
        return "Speech Bubble", ""

    # Generic — 50% gate; in Shorts skip narration fallback entirely
    if is_shorts or roll > 0.50:
        return "None", ""

    return "Narration Box", _short(beat_text)

CLAUDE_MODEL = "claude-sonnet-4-5"
CLAUDE_FAST_MODEL = "claude-3-5-haiku-20241022"
EMOTION_WORDS = [
    "shocked", "angry", "afraid", "terrified", "determined", "confident", "sad", "grieving",
    "tense", "nervous", "furious", "cold", "smug", "relieved", "exhausted", "hopeful",
    "suspicious", "calm", "desperate", "focused",
]

TAB2_PROMPT_ENHANCER_SYSTEM = """
You improve anime/manhwa image prompts for a text-to-image model.
Return only the final prompt text.
Preserve the exact named characters, selected location, selected items, and scene intent.
Make the image more specific, cinematic, and visually renderable.
Convert vague action into visible body motion and clear interaction.
Keep background crowds only in the distance when natural to the location, and avoid dominant unnamed foreground characters.
Do not add NO TEXT, NO WORDS, NO UI, no silhouettes, or other blanket bans unless they are already clearly requested in the prompt payload.
"""


def enhance_tab2_prompt_with_openai(base_prompt: str, context: Dict[str, Any]) -> Tuple[str, str]:
    payload = {
        "base_prompt": clean_for_prompt(base_prompt),
        "context": context,
        "requirements": [
            "Keep the exact subject and action intent.",
            "Improve cinematic clarity and specificity.",
            "Make action visible and readable.",
            "Allow only distant background crowd when natural to the location.",
            "Avoid dominant unnamed foreground characters.",
        ],
    }
    improved = _call_openai_text(TAB2_PROMPT_ENHANCER_SYSTEM, payload, model=OPENAI_PROMPT_MODEL, max_output_tokens=900)
    if improved:
        return clean_for_prompt(improved), f"OpenAI {OPENAI_PROMPT_MODEL}"
    return clean_for_prompt(base_prompt), "local prompt assembly"


TAB2_COMPREHENSIVE_PROMPT_SYSTEM = """
You write the final production prompt for a Korean manhwa / webtoon text-to-image model.
Return only the final prompt text. No JSON. No explanation. No markdown headers or bullets.
Write a compact multi-line prompt block, 8–12 lines total. Each line is prompt language, not prose narrative.

=== MANHWA VISUAL DNA — MANDATORY IN EVERY PROMPT ===
This is Korean manhwa / webtoon — not Japanese anime, not western comics.

STYLE LINE (Line 1 always): "Korean manhwa webtoon art style, bold expressive line art with dynamic weight variation, dramatic high-contrast cel shading with deep shadow pools, vivid saturated color palette, 2D illustrated not photorealistic"
Then add 2–3 scene-specific mood tokens (e.g. "cold blue night atmosphere, rain-slicked reflections, melancholy tension").

LINE ART: bold outer edges, fine inner detail lines — weight shifts with importance and emotion.
SHADING: DEEP shadow areas with sharp edges — never flat, never pastel. Rich blacks in shadow regions.
LIGHTING — pick ONE dominant source and name it specifically:
  - Interior: warm amber candlelight / cold blue window light / harsh overhead fluorescent / dim single lamp / neon sign glow
  - Exterior: golden hour backlight / hard midday top shadow / overcast diffuse / moonlight / rain-diffused street glow
  - Dramatic: explosion flash backlight / volumetric light shaft / power energy bloom / rim-lit silhouette against dark void
COLOR: always intentional — warm light + cool shadow OR cool light + warm shadow. Never neutral grey-beige.
EYES: always specific — "large luminous [color] irises, fine lash detail, sharp catchlight reflections, [emotion] reading in gaze"
COMPOSITION: cinematic and asymmetric — foreground framing element + strong depth. Never static center-frame.

=== WHAT NOT TO WRITE ===
Never use: "masterpiece", "best quality", "4k", "ultra HD", "8k", "highly detailed" as tokens — describe what you SEE instead.
Never use: "exposed", "bare", "naked", "nude", "topless", "undressed", or any word implying a character lacks clothing. Spirit and energy characters are always clothed in their elemental robes/light-formed garments — describe those garments explicitly, never imply skin is visible through them.

=== EMPTY CHARACTERS LIST — ENVIRONMENT / CROWD BEATS ===
If the payload "characters" array is empty, NO named main character appears in this scene.
This does NOT mean a barren or desolate scene — it means the camera focuses on the environment with naturally present anonymous people.
→ Show the environment populated with anonymous background figures appropriate to the setting:
   - Mall / market / shopping street → dense crowd of shoppers, bags, storefronts, mannequins, signage
   - Restaurant / café / bar → tables filled with anonymous diners, waitstaff weaving through, clinking glasses
   - City street / sidewalk → streams of pedestrians, parked bikes, storefronts, umbrellas
   - School hallway / office floor → students or workers moving, lockers, desks, noise and motion
   - Stadium / arena / concert → spectators, rows of seats, lights, flags, energy
   - Public plaza / park → varied people scattered naturally across the space
→ Anonymous figures should occupy near, middle, and far depth layers with varied heights, clothing, and poses.
→ Camera frames the environment as the hero; people fill it naturally. Never produce an empty ghost-town version of a public space just because no named character is listed.
→ Do NOT describe or imply any named character's DNA in this scene.

=== SCENE INTERPRETATION — YOU ARE A VISUAL DIRECTOR ===
You do NOT just pose a character in a location. You decide WHAT THE PANEL SHOWS to make the reader feel the story.
Read the beat_text carefully and ask: what is the most visually arresting way to SHOW this moment?

DUAL REALITY / DUAL MEMORY beats (character experiencing two worlds simultaneously):
→ Show BOTH worlds in the same frame. Use ghostly transparent overlay of the other world bleeding through the current scene, OR split the panel diagonally with each reality on one side, OR have one world appear as reflections/visions in glass/water/eyes.
→ Example for "modern city memories + fantasy palace memories fighting": character's face in the center, left half behind them shows blurry neon city traffic with car headlights, right half shows stone castle corridors with torchlight — both worlds simultaneously visible.

FLASHBACK / CHILDHOOD / PAST MEMORY beats:
→ Show a YOUNGER or CHILD version of the main character — smaller frame, same hair and eye color, softer younger face, same DNA but visibly a child or teenager.
→ OR show the specific remembered scene as a framed vignette panel floating inside the main image.
→ Example for "sword lessons as a child": young child version of character in courtyard, adult instructor looming above, rendered in slightly desaturated nostalgic palette.

ENVIRONMENT SHIFT beats (character thinking of or transported to a different location):
→ If the beat mentions coffee shops, cars, city streets, traffic, modern offices, schools, drive-throughs — SHOW THAT ENVIRONMENT. Do not stay in the fantasy palace.
→ Show the character physically present in the described environment OR show it as a dominant visual overlay.

INNER CONFLICT / MENTAL BATTLE beats (two forces fighting inside the mind):
→ Show the conflict VISUALLY: two versions of the character facing each other (light vs shadow), OR a storm of fragmented image shards swirling around the character, OR shadow-self emerging from behind, OR two hands reaching from opposite sides of the panel.

POV / OBSERVATION beats (character noticing something specific — ceiling, object, sky):
→ Show what the character SEES, not the character looking. Render their point of view.

PHYSICAL SENSATION beats (body feels heavy, disconnected, throbbing pain):
→ Show visual representation: blurred double-vision edges, wavy distortion radiating from the character, the environment tilting/warping around them.

=== FOR ALL EMOTIONAL / MEMORY / NARRATION BEATS — MANDATORY ===
1. Ground the character in a physical action (lying on silk sheets, standing at window, pacing, staring at hands).
2. The subject of the memory/thought MUST appear visually: ghostly figure, vision in glass, face forming in ceiling patterns, photograph held in hand.
3. COLOR TELLS THE EMOTION: warm amber/gold = nostalgia; cold desaturated blue = dread; orange-red = shame/anger; harsh white = anxiety; silver moonlight = grief; purple-grey = melancholy.
4. Environment MUST have specific atmospheric detail: silk sheets catching light, golden carved ceiling beams, blurred city windows, rain on glass.

=== CHARACTER APPEARANCE — NON-NEGOTIABLE ===
Copy each character's dna_prompt EXACTLY and COMPLETELY. Hair color/style, eye color, skin tone, build, outfit, signature anchor — verbatim. Do not paraphrase or drop any detail.
If a character has an "age_appearance" field, PREPEND those tokens before the dna_prompt to communicate developmental stage and body proportions. STRICT priority rules:
- Hair color and eye color from dna_prompt ALWAYS win. If age_appearance says anything about hair that conflicts, ignore it — use dna_prompt hair verbatim. A newborn with silver-white hair in dna_prompt has fine sparse silver-white hair, never bald.
- age_appearance only controls: body size, face maturity/proportions, developmental motor stage, skin texture.
- Never mix adult and child physical proportions in the same character.

=== INFANT / BABY DEVELOPMENTAL POSTURE — MANDATORY ===
NEVER use the word "infant" alone — generation models treat it as a small standing child. Always specify the physical developmental stage and what that means for posture:
- 0-3 months: character MUST be shown lying flat on back, or cradled/held in an adult's arms. They CANNOT hold their head up, sit, stand, or walk. Frame them horizontally or held against a chest.
- 3-6 months: still must be held or lying — cannot sit independently or stand.
- 6-12 months: may sit propped with support; still cannot walk or stand unsupported.
- 12-18 months: toddler pulling up on furniture, unsteady first steps only.
If the beat shows a very young baby in a scene with adults, the baby occupies a small horizontal portion of the frame — lying in a crib, swaddled in cloth, or cradled in arms. Never show them upright.

=== EXPRESSION — NEVER GENERIC ===
BANNED: "determined expression", "focused gaze", "intense look", "resolute face", "steely eyes".
USE: wide-eyed shock, jaw clenched in rage, trembling lower lip, hollow vacant stare, pupils dilated, brows furrowed in disbelief, knuckles white.

=== ADJACENT BEAT CONTRAST ===
When previous_beat_text is given — composition, framing, pose, and expression MUST visibly differ from it.
Close-up → go wider. Wide → go tighter. Stillness → show movement. Action → show quiet reaction.

=== CAMERA TYPE — MATERIALLY CHANGE FRAMING ===
Camera type must change the shot composition fundamentally. Apply the full camera description from the payload.

=== ON-IMAGE TEXT OVERLAY ===
Check the payload "text_type" field carefully:
- If text_type == "None" or is empty: write "NO TEXT, NO WORDS, NO CAPTIONS, NO UI, NO WATERMARK" as a standalone line.
- If text_type is NOT "None" (e.g. Narration Box, Speech Bubble, Thought Bubble, SFX): DO NOT write "NO TEXT". Instead, find the text instruction in the base_prompt (e.g. 'narration box with text: "..."') and include it VERBATIM in your output. The text overlay MUST appear in your final prompt.

=== GENDER + CHARACTER TYPE ===
Default unspecified to male. Male characters: masculine face, male clothing. Female: feminine presentation.

=== OUTPUT FORMAT ===
The payload contains a "prompt_format" field. Match it exactly.

FORMAT "token":
Dense comma-separated tokens. No sentences. 10–14 lines.
Order: manhwa style → text overlay OR NO TEXT → character dna → camera/shot → location/environment → action → emotion → lighting.

FORMAT "prose":
2–4 complete cinematic sentences. No comma lists.
Director describing a shot: environment/shot, character appearance, emotion/action, lighting/mood.

FORMAT "hybrid":
Line 1: manhwa style tokens (max 12 tokens).
Lines 2–7: cinematic prose/semi-prose. Expand the scene, don't repeat Line 1.

FORMAT "core":
11 lines, one element per line, exact order. No headers. No bullets.
Line 1:  STYLE — manhwa style anchor + scene mood tokens
Line 2:  TEXT — either the text overlay instruction verbatim OR "NO TEXT, NO WORDS, NO CAPTIONS, NO UI"
Line 3:  SCENE — what the panel SHOWS (dual world, flashback, environment, mental battle, or physical moment)
Line 4:  CHARACTER — if characters list is non-empty: verbatim dna_prompt: hair, eyes, skin, build, outfit, signature detail. If characters list is EMPTY: write "anonymous crowd, varied clothing, multiple depth layers, no named character"
Line 5:  ACTION — what is physically happening right now, specific and visual
Line 6:  EMOTION — facial expression + body language, vivid and tied to this beat
Line 7:  LOCATION — specific atmosphere from bible_prompt or described environment
Line 8:  SHOT TYPE — derived from camera_type, applied materially to framing
Line 9:  LIGHTING — named source with shadow + highlight behavior described
Line 10: COMPOSITION — foreground framing / depth / rule of thirds / silhouette / layered depth
Line 11: MANHWA RENDER QUALITY — "bold ink line art, deep shadow pools, vivid cel shading, dynamic perspective, highly expressive face"

FORMAT "original":
Follows the official z-image/turbo recommended 4-layer structure. Natural flowing language. No labels, no line numbers, no headers. ~80–120 words total.
Front-load the subject in the FIRST sentence — this is the model's highest-attention zone.

Layer 1 — SUBJECT + ACTION (write this first, ~40 tokens):
  If characters are present: start with the character's verbatim dna_prompt (hair, eye color, skin, build, outfit, signature detail), then immediately describe the specific physical action happening right now. Example: "A tall broad-shouldered man with silver-streaked black hair, piercing grey eyes, olive skin, wearing a deep-navy hanbok, lunging forward with blade raised."
  If characters list is EMPTY: start with the environment and crowd as the subject. Example: "A sprawling indoor shopping mall packed with afternoon crowds — shoppers streaming past gleaming storefronts, colorful window displays, and overhead banners, anonymous figures in near, middle, and far depth layers."

Layer 2 — TEXT (embed naturally, do not skip):
  If text_type is NOT "None": weave the text overlay instruction into the prose naturally (e.g. "A speech bubble reads '...' in the upper corner.").
  If text_type IS "None": append this exact phrase at the end of the prompt — "no text, no captions, no speech bubbles, no watermark, no UI elements."

Layer 3 — STYLE + SHOT + LOCATION (~30 tokens):
  manhwa webtoon illustration, bold ink line art, vivid cel shading, highly expressive face. Then apply the camera_type as a natural shot description (e.g. "extreme close-up" / "wide establishing shot" / "low-angle hero shot"). Then the location atmosphere from bible_prompt in 1–2 phrases.

Layer 4 — LIGHTING + CONSTRAINTS (~20 tokens):
  Name the light source and describe its shadow and highlight behavior (e.g. "harsh overhead fluorescent casting deep under-eye shadows" / "warm amber candlelight with soft rim highlight"). End with: "clean composition, no watermark." (Never add "no extra figures" when characters list is empty — the crowd IS the scene.)

Output as 2–4 natural sentences. Do NOT use bullet points, line numbers, or section labels.

FORMAT "wilson":
K.D.Wilson cinematography format. Single flowing line — no labels, no bullets, no line numbers.
This is a bracket-led cinematic prompt. Output MUST follow this exact structure in order:

  (Focal Length + Shot Type + Camera Angle) of a (Era + Anime Style) (character description), (action/pose), (background/environment), (lighting), (quality tags), (anime studio style reference), (color/mood), NO TEXT, NO WATERMARK

Rules:
1. SHOT BRACKET — translate the camera_type to Wilson notation using this map:
   extreme close-up              → (ECU eye-level)
   tight face close-up           → (70mm Close-Up eye-level)
   over-the-shoulder             → (50mm Over-the-Shoulder eye-level)
   profile shot                  → (50mm Medium Shot side-angle)
   dramatic medium               → (50mm Medium Close-Up low angle)
   dutch-angle medium            → (50mm Medium Shot Dutch Angle)
   low-angle shot                → (24mm Low Angle looking up)
   bird's-eye high-angle         → (24mm Bird's Eye View overhead)
   full-body silhouette          → (24mm Wide Shot backlit)
   wide establishing shot        → (14mm Extreme Wide Shot eye-level)
   dynamic action blur           → (24mm Medium Shot low angle dynamic)
   two-shot framing              → (50mm Medium Shot eye-level, two figures in frame)

2. ERA + ANIME STYLE — choose based on scene_type:
   ACTION   → "2020s battle shonen anime-style"
   EMOTION  → "2020s modern shonen anime-style"
   DIALOGUE → "2020s modern shonen anime-style"
   MEMORY   → "2010s nostalgic shonen anime-style"
   AWAKENING → "2020s battle shonen anime-style"
   AFTERMATH → "2020s dark seinen anime-style"

3. CHARACTER — if characters list is non-empty: extract from dna_prompt, describe hair, face, and clothing in natural language, outfit verbatim. If characters list is EMPTY: describe the environment as the subject and use "crowds of anonymous people, varied outfits, bustling [setting]" in place of a named character.

4. ACTION — if characters are present: what the character is physically doing right now, present tense, specific and visual. If characters list is EMPTY: describe the environmental activity and crowd movement instead.

5. BACKGROUND — describe environment in 1 short clause. If named characters are present and other figures exist, say "other [figures] blurred in background". If characters list is EMPTY, the crowd IS the foreground — no named character anywhere.

6. LIGHTING — one specific lighting description (e.g. "cinematic classroom lighting", "dramatic rim lighting", "warm amber candlelight").

7. QUALITY TAGS — always include at least 4 from: refined lineart, soft shadows, sharp focus, ultra detailed, clean anime rendering, dynamic lighting, moody atmosphere, vibrant colors, smooth shading, crisp textures.

8. STYLE REFERENCE — pick ONE studio that fits the scene mood:
   Polished/emotional → "Kyoto Animation style"
   Brutal/action → "MAPPA style"
   Epic/fantasy → "Ufotable style"
   Dramatic/dark → "Studio Wit style"

9. TEXT — If text_type is NOT "None": replace "NO TEXT, NO WATERMARK" with the text overlay instruction verbatim.
   If text_type IS "None": end with exactly "NO TEXT, NO WATERMARK".

Example output structure (do not copy content, only structure):
(50mm Medium Close-Up eye-level) of a 2025 modern shonen anime-style [character desc] wearing [outfit], [action], [background], [lighting], refined lineart, soft shadows, sharp focus, ultra detailed, [Studio] style, clean anime rendering, [color/mood], NO TEXT, NO WATERMARK
""".strip()


def build_comprehensive_prompt_cb(
    st: ProjectState,
    beat_label: str,
    scene_type_name: str,
    location_name: str,
    sub_location_name: str,
    camera_name: str,
    chars_sel: List[str],
    items_sel: List[str],
    action_line_val: str,
    emotion_notes_val: str,
    include_signature_val: bool,
    style_val: str,
    composed_prompt_val: str,
):
    if not st or not st.beats:
        return "", "❌ Build first in Tab 1."
    idx = _beat_index_from_label(beat_label or "", len(st.beats))
    beat_text = st.beats[idx - 1] if 1 <= idx <= len(st.beats) else ""
    stype = clean_for_prompt(str(scene_type_name or _classify_scene_type(beat_text))).upper()
    if stype not in SCENE_TYPES:
        stype = _classify_scene_type(beat_text)
    camera = camera_name if camera_name in CAMERA_TYPES else (_camera_choices(stype)[0] if _camera_choices(stype) else CAMERA_TYPES[0])
    chars_use = chars_sel or []
    items_use = items_sel or []
    attacker_name = chars_use[0] if stype == "ACTION" and len(chars_use) > 1 else ""
    target_name = chars_use[1] if stype == "ACTION" and len(chars_use) > 1 else ""
    action_text = clean_for_prompt(action_line_val or beat_text)
    if stype == "ACTION" and attacker_name and target_name:
        action_text = f"{attacker_name} striking {target_name} directly in front of {attacker_name}"
    if sub_location_name and sub_location_name != "None" and location_name and location_name != "None":
        if sub_location_name.lower() not in action_text.lower():
            action_text = f"In the {sub_location_name} of the {location_name}, {action_text}"
    action_text = _rewrite_action_for_location(st, action_text, location_name)
    base_seed = clean_for_prompt(composed_prompt_val or compose_tab2_preview_prompt(
        st,
        stype,
        location_name,
        sub_location_name,
        camera,
        chars_use,
        items_use,
        action_text,
        emotion_notes_val or "",
        bool(include_signature_val),
        style_val or DEFAULT_STYLE,
        attacker_name,
        target_name,
    ))
    payload = {
        "beat_text": beat_text,
        "scene_type": stype,
        "location": location_name,
        "sub_location": sub_location_name,
        "camera_type": camera,
        "characters": chars_use,
        "items": items_use,
        "action_text": action_text,
        "emotion_notes": clean_for_prompt(emotion_notes_val or ""),
        "composed_prompt": base_seed,
    }
    enhanced = _call_openai_text(TAB2_COMPREHENSIVE_PROMPT_SYSTEM, payload, model=OPENAI_PROMPT_MODEL, max_output_tokens=1200)
    final_prompt = clean_for_prompt(enhanced or base_seed)
    source = f"✅ Final prompt generated with OpenAI {OPENAI_PROMPT_MODEL}."
    if not enhanced:
        source = "⚠️ OpenAI unavailable; using composed prompt as final prompt."
    return final_prompt, source



def _beat_index_from_label(label: str, beats_len: int) -> int:
    m = re.match(r"^\s*(\d+)", label or "")
    idx = int(m.group(1)) if m else 1
    return max(1, min(idx, beats_len))


def _location_choices(st: ProjectState) -> List[str]:
    base = list((st.locations or {}).keys()) if st else []
    return ["None"] + base


def _sub_location_choices(st: ProjectState, location_name: Optional[str]) -> List[str]:
    if not st or not location_name or location_name == "None":
        return ["None"]
    loc = (st.locations or {}).get(location_name) or {}
    subs = list(((loc.get("sub_locations") or {}).keys()))
    return ["None"] + subs if subs else ["None"]


def _classify_scene_type(beat_text: str) -> str:
    low = (beat_text or "").lower()
    if any(k in low for k in ["hit", "punch", "kick", "slash", "stab", "shoot", "attack", "fight", "smash", "crash"]):
        return "ACTION"
    if any(k in low for k in ["system", "awakens", "awaken", "level up", "quest", "notification"]):
        return "AWAKENING"
    if any(k in low for k in ["blood", "corpse", "aftermath", "ruins", "silent", "smoke"]):
        return "AFTERMATH"
    if any(k in low for k in ["memory", "flashback", "remember", "used to", "once"]):
        return "MEMORY"
    if any(k in low for k in ["says", "asks", "replies", "whispers", "talk"]):
        return "DIALOGUE"
    return "EMOTION"


def _camera_choices(scene_type: Optional[str]) -> List[str]:
    stype = (scene_type or "").upper()
    if stype in SCENE_CAMERA_POOLS:
        return SCENE_CAMERA_POOLS[stype][:]
    return CAMERA_TYPES[:]


def _normalize_gallery_list(g: Any) -> List[str]:
    if not g:
        return []
    out: List[str] = []
    for item in g:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, (list, tuple)) and item and isinstance(item[0], str):
            out.append(item[0])
        elif isinstance(item, dict):
            p = item.get("path") or item.get("name") or item.get("data")
            if isinstance(p, str):
                out.append(p)
    return out


def _limit_gallery(paths: List[str], limit: int = GALLERY_PREVIEW_LIMIT) -> List[str]:
    if len(paths) <= limit:
        return paths
    return paths[-limit:]


LOCATION_CHANGE_WORDS = ["change clothes", "changed clothes", "changes clothes", "changed into", "change into", "puts on", "put on", "wears", "wearing", "dressed in", "changed outfit", "uniform"]

def _strip_location_prefixes(text: str, st: ProjectState) -> str:
    s = (text or "").strip()
    if not s or not st:
        return s
    locs = sorted([re.escape(x) for x in (st.locations or {}).keys()], key=len, reverse=True)
    if not locs:
        return s
    pattern = r'^(?:In the|In|Inside|At)\s+(?:' + '|'.join(locs) + r')\s*,?\s*'
    return re.sub(pattern, '', s, flags=re.IGNORECASE).strip(' ,:-')

def _remove_all_location_mentions(text: str, st: ProjectState) -> str:
    out = (text or "").strip()
    if not st:
        return out
    for ln in sorted((st.locations or {}).keys(), key=len, reverse=True):
        out = re.sub(rf'\b(?:the\s+)?{re.escape(ln)}\b', '', out, flags=re.IGNORECASE)
    out = re.sub(r'\s{2,}', ' ', out)
    out = re.sub(r'\s+,', ',', out)
    out = re.sub(r'^[,:;\-\s]+', '', out)
    return out.strip()

def _identity_only_appearance(text: str) -> str:
    """Strip camera/pose/setting/action context from a template appearance description.
    Reference photo captions often include the photo's camera angle, backdrop, and the
    model's pose/expression — none of which should bleed into a panel description.
    Keep only true identity features: hair, eye colour, skin, face, build, outfit name/colour."""
    import re as _re
    # Remove clauses/sentences containing reference-photo context patterns
    _strip_pats = [
        # Shot/camera angle
        r"[^.—,]*(?:bust shot|full[- ]?body shot|three[- ]?quarter|front view|back view|side view"
        r"|eye[- ]?level|bird'?s[- ]?eye|low[- ]?angle|high[- ]?angle|camera angle"
        r"|portrait shot|profile shot|close[- ]?up shot)[^.—,]*[.,]?",
        # Physical setting / backdrop — fixed to allow adjectives between "standing" and "in/on/against"
        r"[^.—,]*(?:set in a|standing (?:\w+\s+){0,3}(?:in|on|against)|sitting (?:\w+\s+){0,2}(?:in|on)"
        r"|lying on|kneeling on|positioned (?:in|on|against)"
        r"|on a (?:modern|gym|wooden|marble|stone|dark polished|tatami|wet)"
        r"|in a (?:sunlit|dim|dark|bright|modern|palace|corridor|hallway|studio|gym|forest|street)"
        r"|against a (?:pure white|black|grey|gray|dark|light|white|neutral))[^.—,]*[.,]?",
        # Explicit pose/position from reference
        r"[^.—,]*(?:plank position|push[- ]?up position|squat position|horizontal plank"
        r"|in a (?:seated|standing|crouching|prone|supine|lying|kneeling|running|jumping) (?:position|pose)"
        r"|(?:legs? |arms? )?(?:crossed|extended|raised|bent|flexed|spread)[^.—,]{0,30}position"
        r"|frontal pose|confident (?:frontal|standing|full[- ]?body) pose)[^.—,]*[.,]?",
        # Bare skin / shirtless — reference photo skin exposure, never a story costume
        r"[^.—,]*(?:shirtless|topless|bare[- ]?chested|bare[- ]?torso|bare[- ]?chest|bare[- ]?upper body"
        r"|no shirt|without (?:a )?shirt|exposed chest|exposed torso)[^.—,]*[.,]?",
        # Gym/fitness body description — reference photo body prose, not character identity
        r"[^.—,]*(?:physique is (?:highly |very |extremely )?(?:muscular|athletic|lean|toned|ripped|defined|sculpted)"
        r"|detailed anatomical definition|anatomical (?:detail|definition|rendering)"
        r"|musculature|muscle definition|defined (?:abs|muscles|chest|abdomen)"
        r"|six[- ]?pack|washboard abs|bulging (?:biceps|muscles))[^.—,]*[.,]?",
        # Opening physique label — "A muscular adult male", "A fit young man", etc. at start of template
        r"(?:^|(?<=\. ))(?:A |An )(?:muscular|fit|athletic|lean|slim|well-built|toned|burly|wiry)"
        r" (?:adult |young |tall |short )?(?:male|man|woman|female|guy|person)[,.]?",
        # Reference-photo lifestyle pose — "leans casually against", "sits relaxed at", etc.
        r"[^.—,]*(?:leans? casually (?:against|on|into)|sits? (?:casually|relaxed|comfortably) (?:on|at|in)"
        r"|reclines? (?:casually|relaxed)|lounges? (?:against|on))[^.—,]*[.,]?",
        # Gaze/expression directed at camera — reference-specific, not panel-specific
        r"[^.—,]*(?:gazes? (?:directly|intently|softly|warmly|coolly) at the (?:viewer|camera)"
        r"|looks? (?:directly|intently) (?:at|into) (?:the )?(?:camera|viewer|lens)"
        r"|direct(?:ly)? (?:facing|towards?) (?:the )?(?:camera|viewer)"
        r"|directed (?:past|at|toward) the (?:viewer|camera)|intense and slightly distant"
        r"|displays? (?:a )?(?:warm|cool|confident|knowing|soft) (?:smile|smirk|grin)"
        r"|(?:warm|cool|confident|knowing) (?:smile|smirk|grin) directed)[^.—,]*[.,]?",
        # Modern/anachronistic props and reference-photo set dressing
        r"[^.—,]*(?:smartphone|mobile phone|tablet|laptop|gym (?:floor|equipment|mat)"
        r"|fitness content|dumbbell|barbell|weight[- ]?rack|treadmill"
        r"|coffee mug|tea mug|energy drink|protein shake|water bottle"
        r"|tank top|sleeveless (?:shirt|top|vest|tee)"
        r"|combat boots|neon (?:sign|lettering|text|light|glow)|bold neon"
        r"|industrial metal (?:framework|structure|scaffolding)"
        r"|minimalist background (?:feature|with|of))[^.—,]*[.,]?",
        # Reference-photo action poses — "pushing against", "pulling on", "lunging at", etc.
        r"[^.—,]*(?:pushing (?:forcefully |hard )?(?:against|through|up|down)"
        r"|pulling (?:hard |forcefully )?(?:on|at|against)"
        r"|lunging (?:at|toward|forward)"
        r"|in a dramatic (?:action|fighting|battle|combat|power) pose"
        r"|braced (?:against|for)|pressed (?:against|flat))[^.—,]*[.,]?",
        # Reference-photo backdrop / set phrases not already caught
        r"[^.—,]*(?:the minimalist background features|minimalist (?:white|black|gray|grey|studio) background"
        r"|massive red (?:curved|flat|circular)|red (?:obstacle|surface|wall|fist))[^.—,]*[.,]?",
        # Studio/reference photo lighting — not panel mood lighting
        r"[^.—,]*(?:natural sunlight streams?|sunlight from the (?:upper|lower|left|right)"
        r"|warm golden highlight|rim light|studio light|backlit|soft box"
        r"|lighting is (?:soft|harsh|bright|dim|dramatic|professional|natural|warm|cool|golden|amber)"
        r"(?: and (?:professional|natural|studio|warm|soft|flattering))?"
        r"|creates? subtle shadows that emphasize|subtle shadows that emphasize"
        r"|softly illuminating his (?:face|body|figure)|gently illuminating"
        r"|emphasize(?:s)? his (?:features|muscles|physique|build|face))[^.—,]*[.,]?",
        # Mood description from reference shoot — never story character state
        r"[^.—,]*The mood is (?:relaxed|casual|confident|warm|approachable|intimate|friendly|playful"
        r"|powerful|intense|soft|cool|cheerful)[^.—,]*[.,]?",
        # Template metadata / art-style tags — never character appearance
        r"[^.—,]*(?:art style is|premium (?:Korean|manhwa|webtoon)|semi[- ]?realistic anatomical rendering"
        r"|manhwa (?:style|art|rendering)|webtoon (?:style|art|rendering))[^.—,]*[.,]?",
        # Sweat/shine — gym photo artifacts
        r"[^.—,]*(?:glistening with sweat|muscles defined and|gleaming (?:with|under)"
        r"|damp skin|sweat drop|sweat bead|sheen of sweat)[^.—,]*[.,]?",
        # Specific backdrop details from character reference photos
        r"[^.—,]*(?:green foliage|palace corridor|corridor with|sunlit (?:palace|corridor|hallway)"
        r"|dark polished floor|polished (?:marble|wood|stone) floor)[^.—,]*[.,]?",
    ]
    result = text
    for pat in _strip_pats:
        result = _re.sub(pat, " ", result, flags=_re.IGNORECASE)
    # Collapse whitespace and tidy punctuation
    result = _re.sub(r"[ \t]{2,}", " ", result)
    result = _re.sub(r"[\s,]{1,}([,.])", r"\1", result)
    result = _re.sub(r"\s{2,}", " ", result).strip().strip(",.").strip()
    return result


def _substitute_cast_in_script(script: str, cast_overrides: dict) -> str:
    """
    Replace inline character appearance blocks in panel text with cast template
    descriptions, stripping reference-photo context (camera angle, setting, pose from
    the reference shoot) so only identity features bleed into the panel description.
    Panel lines follow the pattern:
        CharName — [appearance description] — [expression / pose / action]
    Lines that don't match the double-dash pattern are left untouched.
    """
    if not cast_overrides or not script:
        return script
    EM_DASHES = "—–"
    lines = script.split("\n")
    out = []
    for line in lines:
        for char_name, appearance in cast_overrides.items():
            lo = line.lower()
            nm_idx = lo.find(char_name.lower())
            if nm_idx == -1:
                continue
            after_name = nm_idx + len(char_name)
            dash_positions = [i for i, c in enumerate(line) if c in EM_DASHES]
            dashes_after = [d for d in dash_positions if d >= after_name]
            if len(dashes_after) < 2:
                continue  # need two dashes to identify an appearance block
            d1, d2 = dashes_after[0], dashes_after[1]
            between = line[d1 + 1:d2].strip()
            if len(between) < 20:
                continue  # too short to be a real appearance block
            # Strip reference-photo context before injecting — keep identity only
            clean_app = _identity_only_appearance(appearance[:500]).replace("\n", " ").strip()
            line = line[:d1 + 1] + " " + clean_app + " " + line[d2:]
            break  # one substitution per line is enough
        out.append(line)
    return "\n".join(out)


def _rewrite_action_for_location(st: ProjectState, action: str, location_name: str) -> str:
    if not st:
        return clean_for_prompt(action or "")
    base = _remove_all_location_mentions(_strip_location_prefixes(action, st), st)
    if not location_name or location_name == "None":
        return clean_for_prompt(base)
    if not base:
        return f"In the {location_name}."
    return _ensure_location_in_action(clean_for_prompt(base), location_name)


def _location_overview_html(st) -> str:
    """Build an HTML table grouping consecutive beats by assigned location."""
    if not st or not getattr(st, "beats", None):
        return "<p style='color:#6868a0;font-size:13px;padding:6px'>Load a project to see the location map.</p>"
    total = len(st.beats)
    plans = st.beat_plans or {}

    # Group consecutive beats that share the same location
    groups: list = []
    for idx in range(1, total + 1):
        plan = plans.get(idx) or {}
        loc  = (plan.get("suggested_location") or "").strip() or "None"
        sub  = (plan.get("suggested_sub_location") or "").strip()
        label = loc if not sub or sub == "None" else f"{loc} › {sub}"
        if groups and groups[-1]["label"] == label:
            groups[-1]["end"] = idx
            groups[-1]["count"] += 1
        else:
            groups.append({"start": idx, "end": idx, "count": 1, "loc": loc, "label": label})

    rows_html = ""
    for g in groups:
        beat_range = f"{g['start']:03d}" if g["start"] == g["end"] else f"{g['start']:03d}–{g['end']:03d}"
        is_none = g["loc"] in ("None", "", "none")
        loc_color = "#f87171" if is_none else "#e2e8f0"
        bg = "background:#2d1a1a;" if is_none else ""
        rows_html += (
            f"<tr style='{bg}'>"
            f"<td style='padding:4px 8px;color:#94a3b8;font-size:12px;white-space:nowrap'>{beat_range}</td>"
            f"<td style='padding:4px 8px;color:#64748b;font-size:12px;text-align:center'>{g['count']}</td>"
            f"<td style='padding:4px 8px;color:{loc_color};font-size:12px'>{g['label'] if not is_none else '⚠️ None — not set'}</td>"
            f"</tr>"
        )

    return (
        "<div style='max-height:260px;overflow-y:auto;border:1px solid #27273e;border-radius:6px;margin-top:6px'>"
        "<table style='width:100%;border-collapse:collapse'>"
        "<thead><tr style='background:#1a1a2e;position:sticky;top:0'>"
        "<th style='padding:5px 8px;color:#6868a0;font-size:11px;text-align:left'>Beats</th>"
        "<th style='padding:5px 8px;color:#6868a0;font-size:11px;text-align:center'>#</th>"
        "<th style='padding:5px 8px;color:#6868a0;font-size:11px;text-align:left'>Location</th>"
        "</tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        "</table></div>"
    )

def _beat_has_explicit_outfit_change(beat_text: str) -> bool:
    low = (beat_text or '').lower()
    return any(k in low for k in LOCATION_CHANGE_WORDS)

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


def _call_claude_beat_plan(st: ProjectState, beat_text: str, beat_index: int) -> Optional[Dict[str, Any]]:
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    if not api_key:
        return None
    try:
        import requests
    except Exception:
        return None

    characters = []
    for name, c in (st.characters or {}).items():
        fields = (c or {}).get("fields", {})
        characters.append({
            "name": name,
            "character_type": fields.get("character_type", "human"),
            "gender": fields.get("gender", ""),
            "dna": (c or {}).get("dna_prompt", "")[:280],
            "forms": [
                {
                    "name": f["name"],
                    "type": f["type"],
                    "context_note": f.get("context_note", ""),
                    "activates_at_locations": f.get("activates_at_locations") or [],
                }
                for f in ((c or {}).get("forms") or [])
            ],
        })
    locations = []
    for name, l in (st.locations or {}).items():
        locations.append({
            "name": name,
            "bible": (l or {}).get("bible_prompt", "")[:220],
            "sub_locations": list(((l or {}).get("sub_locations") or {}).keys()),
        })
    items = []
    for name, it in (st.items or {}).items():
        items.append({
            "name": name,
            "bible": (it or {}).get("bible_prompt", "")[:180],
        })

    system = (
        "You are a CINEMATIC beat planner for a manhwa visual director. "
        "Return ONLY valid JSON. No markdown. No commentary. "
        "You are deciding what the IMAGE will SHOW — not just where the character stands. "
        "Think like a film director choosing a shot: every beat is a visual storytelling decision. "
        "Choose from provided characters, locations, items, scene types, and camera types. "
        "CRITICAL RULE: 'character thinks' or 'character reflects' are BANNED as suggested_action. "
        "Show HOW they think. Show WHAT they see. Show the memory itself as a visual scene."
    )
    user = {
        "task": "Create a CINEMATIC visual beat plan. Decide what the IMAGE will literally show.",
        "beat_index": beat_index,
        "beat_text": beat_text,
        "available_characters": characters,
        "available_locations": locations,
        "available_items": items,
        "allowed_scene_types": SCENE_TYPES,
        "allowed_camera_types": CAMERA_TYPES,
        "json_schema": {
            "suggested_location": "string or empty string",
            "suggested_sub_location": "string or empty string",
            "scene_type": "one allowed scene type or empty string",
            "camera_type": "one allowed camera type or empty string",
            "suggested_characters": ["character names only"],
            "suggested_items": ["item names only"],
            "suggested_action": "one vivid cinematic sentence describing exactly what the panel IMAGE SHOWS — not what the character feels",
            "visual_approach": "one of: single_character | dual_reality | flashback_child | environment_shift | mental_battle | pov_shot | physical_sensation",
            "attacker_name": "character name or empty",
            "target_name": "character name or empty",
            "character_emotions": {"Character Name": "vivid specific emotion description, not generic"},
            "active_forms": {"Character Name": "form_name from that character's forms list, or 'base' for default"},
        },
        "rules": [
            "DUAL REALITY: if beat says character experiences two worlds/times/memories simultaneously → visual_approach=dual_reality. suggested_action must describe BOTH worlds appearing in the same frame (ghost overlay, split panel, reflection, double exposure).",
            "FLASHBACK / CHILD: if beat references a childhood memory or past scene → visual_approach=flashback_child. suggested_action must describe showing the YOUNGER/CHILD version of the character in that past setting.",
            "ENVIRONMENT SHIFT: if beat mentions a completely different environment (modern city, coffee, cars, traffic, school, office) → visual_approach=environment_shift. Show that specific environment — do NOT stay in the current location.",
            "MENTAL BATTLE: if beat describes inner conflict, two forces, two versions of self → visual_approach=mental_battle. Show it as two shadow-selves, a storm of fragmented visions, or light-self vs dark-self facing each other.",
            "POV SHOT: if beat describes character noticing something specific (ceiling, object, sky) → visual_approach=pov_shot. Show what THEY SEE, not them looking.",
            "PHYSICAL SENSATION: if beat describes body feeling heavy/disconnected/throbbing → visual_approach=physical_sensation. Show visual distortion — blurred edges, warping environment, double-vision effect.",
            "DEFAULT: if none of the above apply → visual_approach=single_character. Still ground in physical activity + show subject of thought as visual element.",
            "character_emotions must be vivid: 'hollow stare, jaw slack' not 'sad'. 'eyes wide, hand pressed to mouth' not 'shocked'.",
            "Only include characters clearly present or strongly implied.",
            "active_forms: For each suggested character pick their active form name from their 'forms' list, or 'base' for their default state. Only use a non-base form when the beat EXPLICITLY shows that form in use. DEACTIVATION: if a transform was active but the beat shows the character resting, at home, eating, sleeping, or in a casual non-combat scene — use 'base'. Never infer a form from activity type alone; require explicit beat evidence.",
            "Location continuity: keep current location UNLESS beat clearly moves elsewhere or environment_shift applies.",
        ],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 500,
        "temperature": 0,
        "system": system,
        "messages": [{"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
    }
    try:
        r = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        text_parts = []
        for part in data.get("content", []):
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        plan = _extract_json_object("\n".join(text_parts))
        if isinstance(plan, dict):
            return plan
    except Exception:
        return None
    return None


def _call_claude_on_image_text(text_type: str, beat_text: str, action_line: str) -> Optional[str]:
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    if not api_key:
        return None
    try:
        import requests
    except Exception:
        return None

    system = (
        "You write short on-image text for manhwa panels. "
        "Return ONLY the text to place in the panel (no quotes, no markdown). "
        "Keep it short (max ~10 words). "
        "No profanity. No slurs."
    )
    user = {
        "text_type": text_type,
        "beat_text": beat_text,
        "action_line": action_line,
        "rules": [
            "If Speech Bubble: write natural dialogue that fits the moment.",
            "If Thought Bubble: inner monologue.",
            "If Narration Box: concise narration.",
            "If System Message: game-like system line.",
            "If SFX: stylized sound effect (e.g., 'BAM!', 'THUD!').",
        ],
    }
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": CLAUDE_MODEL,
        "max_tokens": 80,
        "temperature": 0.4,
        "system": system,
        "messages": [{"role": "user", "content": json.dumps(user, ensure_ascii=False)}],
    }
    try:
        r = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        text_parts = []
        for part in data.get("content", []):
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        out = clean_for_prompt("\n".join(text_parts))
        if len(out.split()) > 12:
            out = " ".join(out.split()[:12])
        return out.strip() or None
    except Exception:
        return None


def _infer_emotion_snippet(beat_text: str) -> str:
    low = (beat_text or "").lower()
    found = [w for w in EMOTION_WORDS if w in low]
    if found:
        return found[0]
    if any(x in low for x in ["smile", "grin", "smirks", "laugh"]):
        return "confident, faint smile"
    if any(x in low for x in ["cry", "tears", "sob"]):
        return "sad, teary-eyed"
    if any(x in low for x in ["shout", "yell", "scream"]):
        return "intense, mouth open mid-shout"
    if any(x in low for x in ["stare", "watch", "looks out"]):
        return "tense, focused gaze"
    if any(x in low for x in ["fight", "attack", "charge"]):
        return "determined, aggressive posture"
    return "focused, controlled expression"


def _heuristic_beat_plan(st: ProjectState, beat_text: str, beat_index: int) -> Dict[str, Any]:
    low = (beat_text or "").lower()
    suggested_characters = []
    for name in (st.characters or {}).keys():
        if name.lower() in low:
            suggested_characters.append(name)
    if not suggested_characters and st.characters:
        char_names = list(st.characters.keys())
        protagonist = char_names[0]
        # Single-character projects always use the protagonist
        # Multi-character: add protagonist when beat uses any first/third-person pronoun
        # Include " i'" to catch contractions: I'd, I've, I'm, I'll
        _pronouns = [" i ", " i'", " me ", " my ", " myself ", " he ", " she ", " his ", " her ", " him ", " they ", " their ", " them "]
        _low_padded = f" {low} "
        if any(p in _low_padded for p in _pronouns):
            suggested_characters = [protagonist]
        elif len(st.characters) == 1:
            suggested_characters = [protagonist]

    suggested_location = ""
    for loc_name in (st.locations or {}).keys():
        if loc_name.lower() in low:
            suggested_location = loc_name
            break
    if not suggested_location and st.locations:
        # Use the nearest previous beat's location for continuity instead of always defaulting to index 0
        for _prev in range(beat_index - 1, 0, -1):
            _prev_plan = (st.beat_plans or {}).get(_prev)
            if _prev_plan:
                _prev_loc = (_prev_plan.get("suggested_location") or "").strip()
                if _prev_loc and _prev_loc.lower() != "none":
                    suggested_location = _prev_loc
                    break
        if not suggested_location:
            suggested_location = list(st.locations.keys())[0]

    suggested_sub_location = "None"
    if suggested_location and suggested_location != "None":
        sub_choices = _sub_location_choices(st, suggested_location)
        for sub in sub_choices:
            if sub != "None" and sub.lower() in low:
                suggested_sub_location = sub
                break
        if suggested_sub_location == "None" and len(sub_choices) > 1:
            suggested_sub_location = sub_choices[1]

    scene_type = _classify_scene_type(beat_text)
    camera_choices = _camera_choices(scene_type)
    suggested_camera = camera_choices[(beat_index - 1) % len(camera_choices)] if camera_choices else CAMERA_TYPES[0]

    suggested_items = []
    for item_name in (st.items or {}).keys():
        if item_name.lower() in low:
            suggested_items.append(item_name)

    emo = _infer_emotion_snippet(beat_text)
    character_emotions = {name: emo for name in suggested_characters}
    attacker_name = suggested_characters[0] if scene_type == "ACTION" and len(suggested_characters) > 1 else ""
    target_name = suggested_characters[1] if scene_type == "ACTION" and len(suggested_characters) > 1 else ""
    action = clean_for_prompt(beat_text)
    if attacker_name and target_name:
        action = f"{attacker_name} striking {target_name} directly in front of {attacker_name}"
    action = _ensure_location_in_action(action, suggested_location)
    return {
        "suggested_location": suggested_location,
        "suggested_sub_location": suggested_sub_location,
        "scene_type": scene_type,
        "camera_type": suggested_camera,
        "suggested_perspective": suggested_camera,
        "suggested_characters": suggested_characters,
        "suggested_items": suggested_items,
        "suggested_action": action,
        "attacker_name": attacker_name,
        "target_name": target_name,
        "character_emotions": character_emotions,
        "outfit_overrides": {},
        "plan_source": "heuristic",
    }


def _sanitize_plan(st: ProjectState, plan: Dict[str, Any], beat_text: str) -> Dict[str, Any]:
    valid_chars = set((st.characters or {}).keys())
    valid_locs = set(["None"] + list((st.locations or {}).keys()))
    valid_items = set((st.items or {}).keys())

    loc = (plan.get("suggested_location") or "").strip()
    if not loc or loc not in valid_locs:
        loc_lower = loc.lower()
        # 1. Exact case-insensitive match (Claude returned correct name, different case)
        matched = next((v for v in valid_locs if v != "None" and v.lower() == loc_lower), None)
        if not matched and loc_lower:
            # 2. Claude returned a prefix of the stored key, e.g. "Training Ground" → "Training Ground (North)"
            matched = next((v for v in valid_locs if v != "None" and v.lower().startswith(loc_lower)), None)
        if not matched and loc_lower:
            # 3. Stored key starts with what Claude returned (same direction)
            matched = next((v for v in valid_locs if v != "None" and loc_lower.startswith(v.lower())), None)
        if not matched and loc_lower and len(loc_lower) >= 5:
            # 4. Substring match — one contains the other (handles "Training Hall" ↔ "The Training Hall")
            matched = next((v for v in valid_locs if v != "None" and (loc_lower in v.lower() or v.lower() in loc_lower)), None)
        loc = matched or "None"
    sub_loc = (plan.get("suggested_sub_location") or "").strip()
    if loc != "None":
        valid_subs = set(((st.locations.get(loc) or {}).get("sub_locations") or {}).keys())
        if sub_loc not in valid_subs:
            sub_loc = list(valid_subs)[0] if valid_subs else "None"
    else:
        sub_loc = "None"
    scene_type = clean_for_prompt(str(plan.get("scene_type") or _classify_scene_type(beat_text))).upper()
    if scene_type not in SCENE_TYPES:
        scene_type = _classify_scene_type(beat_text)
    camera_type = clean_for_prompt(str(plan.get("camera_type") or plan.get("suggested_perspective") or ""))
    if camera_type not in CAMERA_TYPES:
        camera_choices = _camera_choices(scene_type)
        camera_type = camera_choices[0] if camera_choices else CAMERA_TYPES[0]
    # Fuzzy-match character names — Claude sometimes returns names with different
    # casing, spacing, or minor spelling differences.
    def _match_char(name: str) -> str:
        if name in valid_chars:
            return name
        nl = name.lower()
        # exact case-insensitive
        m = next((v for v in valid_chars if v.lower() == nl), None)
        if m:
            return m
        # substring: "Yoon" matches "Yoon Dohan"
        if len(nl) >= 3:
            m = next((v for v in valid_chars if nl in v.lower() or v.lower() in nl), None)
        return m or ""
    chars = [r for r in (_match_char(c) for c in (plan.get("suggested_characters") or [])) if r]
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
    raw_emotions = plan.get("character_emotions") or {}
    if not isinstance(raw_emotions, dict):
        raw_emotions = {}
    emotions = {c: clean_for_prompt(str(raw_emotions.get(c) or "focused, controlled expression")) for c in chars}
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
        "outfit_overrides": {},
        "plan_source": clean_for_prompt(str(plan.get("plan_source") or "claude")) or "claude",
    }


def ensure_beat_plan(st: ProjectState, beat_index: int) -> Dict[str, Any]:
    existing = (st.beat_plans or {}).get(beat_index)
    if existing:
        return existing
    beat_text = st.beats[beat_index - 1] if 1 <= beat_index <= len(st.beats or []) else ""
    plan = _heuristic_beat_plan(st, beat_text, beat_index)
    if st.beat_plans is None:
        st.beat_plans = {}
    st.beat_plans[beat_index] = plan
    return plan


def _fill_missing_beat_plans(st: "ProjectState", beat_indices: List[int]) -> int:
    """Pre-plan any beats in beat_indices that lack a Claude-quality plan.

    Both truly missing plans AND heuristic fallbacks (plan_source="heuristic")
    are re-planned with Claude so every beat gets proper location/character context.
    Retries each chunk up to 2 times on failure before accepting a heuristic fallback.
    Saves the project to disk when any plans are improved, so good plans persist
    across runs and don't need to be re-planned every time.
    """
    def _needs_plan(i: int) -> bool:
        p = (st.beat_plans or {}).get(i)
        if not p:
            return True
        if p.get("plan_source") == "heuristic":
            return True
        loc = (p.get("suggested_location") or "").strip()
        return not loc or loc == "None"

    missing = [i for i in beat_indices if _needs_plan(i)]
    if not missing:
        return 0
    try:
        from build import _call_claude_batch_beat_plans, _sanitize_plan as _sp
    except Exception:
        return 0
    if st.beat_plans is None:
        st.beat_plans = {}

    import time as _time_mod

    PLAN_BATCH = 20
    MAX_RETRIES = 2
    RETRY_SLEEP = 4   # seconds between retries
    any_improved = False

    for b_start in range(0, len(missing), PLAN_BATCH):
        chunk = missing[b_start:b_start + PLAN_BATCH]
        start_idx = chunk[0]
        batch_beats = [st.beats[i - 1] for i in chunk if 1 <= i <= len(st.beats)]

        # Find the most recent Claude-quality prior location for continuity
        prior_loc = ""
        for prev_i in range(start_idx - 1, 0, -1):
            prev_plan = st.beat_plans.get(prev_i)
            if prev_plan and prev_plan.get("plan_source") != "heuristic":
                cand = (prev_plan.get("suggested_location") or "").strip()
                if cand and cand.lower() != "none":
                    prior_loc = cand
                    break

        # Retry the Claude call up to MAX_RETRIES times before accepting heuristic
        rows: Dict[int, Any] = {}
        for attempt in range(MAX_RETRIES + 1):
            try:
                result = _call_claude_batch_beat_plans(st, start_idx, batch_beats, prior_loc)
                if result:
                    rows = result
                    break
            except Exception:
                pass
            if attempt < MAX_RETRIES:
                _time_mod.sleep(RETRY_SLEEP)

        for rel_i, idx in enumerate(chunk):
            if idx < 1 or idx > len(st.beats):
                continue
            beat_text = st.beats[idx - 1]
            raw = rows.get(idx)
            if raw:
                plan = _sp(st, raw, beat_text)
                any_improved = True
            else:
                # Claude failed for this beat — keep existing plan if it's there,
                # otherwise fall back to heuristic (better than nothing)
                existing = st.beat_plans.get(idx)
                plan = existing if existing else _sp(st, _heuristic_beat_plan(st, beat_text, idx), beat_text)
            st.beat_plans[idx] = plan
            if plan.get("suggested_location") and plan["suggested_location"] != "None":
                prior_loc = plan["suggested_location"]

    # Persist improved plans so the next batch run doesn't need to re-plan them
    if any_improved:
        try:
            _save_project_json(st)
        except Exception:
            pass

    return len(missing)


def _get_saved_prompt(st: ProjectState, beat_index: int) -> Dict[str, Any]:
    return (st.image_prompts or {}).get(beat_index) or {}


def _emotion_text_from_plan(plan: Dict[str, Any]) -> str:
    emotions = plan.get("character_emotions") or {}
    return "; ".join([f"{k}: {v}" for k, v in emotions.items() if v])


def _outfit_text_from_plan(plan: Dict[str, Any]) -> str:
    # active_forms is the new system (shows non-base selections only)
    active = plan.get("active_forms") or {}
    if active:
        lines = [f"{k}: {v}" for k, v in active.items() if v and v != "base"]
        if lines:
            return "\n".join(lines)
    # Legacy fallback for plans generated before the forms system
    overrides = plan.get("outfit_overrides") or {}
    return "\n".join([f"{k}: {v}" for k, v in overrides.items() if v])


def _ensure_location_in_action(action: str, location_name: str) -> str:
    if not location_name or location_name == "None":
        return action
    if location_name.lower() in action.lower():
        return action
    return f"In the {location_name}, {action}"


def _compose_wilson_prompt(
    st: "ProjectState",
    camera_type: str,
    scene_type: str,
    chars: List[str],
    action_text: str,
    location_name: str,
    sub_location_name: str,
    text_type: str = "None",
    beat_index: int = 0,
    cast_appearance: Optional[Dict[str, str]] = None,
) -> str:
    """Build a K.D.Wilson-format prompt entirely in Python — no API call needed."""
    # 1. Shot bracket
    bracket = "(50mm Medium Shot eye-level)"
    cam_lower = (camera_type or "").lower()
    for key, val in WILSON_BRACKET_MAP.items():
        if cam_lower.startswith(key):
            bracket = val
            break

    # 2. Era + style
    era = WILSON_ERA_MAP.get((scene_type or "").upper(), "2020s modern shonen anime-style")

    # 3. Character description (cast library appearance > age-phase > DNA prompt)
    char_parts = []
    for name in chars:
        c = (st.characters or {}).get(name) or {}
        # Priority 1: cast template appearance from the library (actual photo reference)
        dna = ""
        if cast_appearance and name in cast_appearance:
            dna = clean_for_prompt(cast_appearance[name])
        # Priority 2: age-phase appearance
        if not dna and beat_index > 0:
            phase = age_phase_for_beat(st, name, beat_index)
            if phase and phase.get("appearance_prompt"):
                dna = clean_for_prompt(phase["appearance_prompt"])
        # Priority 3: story bible dna_prompt
        if not dna:
            dna = clean_for_prompt(str(c.get("dna_prompt") or ""))
        # Apply active form overlay (situational outfit / transformation)
        if beat_index > 0 and dna:
            _form = get_active_form_for_beat(st, name, beat_index)
            if _form:
                _ftype = (_form.get("type") or "outfit").lower()
                _fdesc = clean_for_prompt(_form.get("description") or "")
                if _fdesc:
                    if _ftype == "full_transform":
                        # Full body transformation — clothing completely hidden
                        dna = _fdesc + ", full body transformation, no clothing visible"
                    elif _ftype == "partial_transform":
                        # Layers on top of current outfit — clothing still visible
                        dna = dna + ", " + _fdesc + ", clothing still visible beneath"
                    else:
                        # Named outfit override
                        dna = dna + ", currently wearing: " + _fdesc
        if dna:
            char_parts.append(dna)

    loc_obj = (st.locations or {}).get(location_name) or {}
    sub_obj = (
        ((loc_obj.get("sub_locations") or {}).get(sub_location_name) or {})
        if sub_location_name and sub_location_name != "None"
        else {}
    )
    env_full = clean_for_prompt(
        str(sub_obj.get("bible_prompt") or loc_obj.get("bible_prompt") or location_name or "")
    )

    if not char_parts and st:
        # Last-resort: if beat text contains pronoun references, inject the protagonist's DNA
        # rather than defaulting to anonymous crowds
        _char_names = list((st.characters or {}).keys())
        if _char_names and beat_index > 0:
            _beat_raw = ((st.beats or [])[beat_index - 1] if beat_index <= len(st.beats or []) else "").lower()
            _action_low = (action_text or "").lower()
            _combined_low = f" {_beat_raw} {_action_low} "
            # Include " i'" to catch contractions: I'd, I've, I'm, I'll
            _pronouns = [" i ", " i'", " me ", " my ", " myself ", " he ", " she ", " his ", " her ", " him ", " they ", " their ", " them "]
            if len(_char_names) == 1 or any(p in _combined_low for p in _pronouns):
                _proto = _char_names[0]
                _c = (st.characters or {}).get(_proto) or {}
                _dna = clean_for_prompt(str(_c.get("dna_prompt") or ""))
                if _dna:
                    char_parts.append(_dna)
                    chars = [_proto]

    if not char_parts:
        # Distinguish pure environment/weather beats from actual crowd scenes.
        # If the beat has no human subject (weather, nature, abstract) use an
        # establishing-shot descriptor instead of forcing people into the frame.
        _env_subjects = {
            "rain", "snow", "wind", "storm", "fog", "mist", "thunder", "lightning",
            "darkness", "silence", "shadow", "light", "smoke", "fire", "flame",
            "sun", "moon", "sky", "cloud", "forest", "tree", "mountain", "river",
            "sea", "ocean", "wave", "earth", "ground", "night", "dawn", "dusk",
            "morning", "evening", "cold", "heat", "air", "blood trail", "trail",
        }
        _beat_check = ((st.beats or [])[beat_index - 1] if st and beat_index > 0 and beat_index <= len(st.beats or []) else action_text or "").lower()
        _first_word = (_beat_check.split()[0] if _beat_check.split() else "")
        _is_env_beat = any(subj in _beat_check[:60] for subj in _env_subjects) and not any(
            p in f" {_beat_check} " for p in [" i ", " i'", " he ", " she ", " his ", " her ", " him ", " they ", " we "]
        )
        if _is_env_beat:
            char_parts.append("cinematic establishing shot, no visible characters")
        else:
            char_parts.append("crowds of anonymous people, varied outfits")
    char_desc = char_parts[0] if len(char_parts) == 1 else " and ".join(char_parts)

    # 4. Action
    action = clean_for_prompt(action_text) if action_text else "standing still"

    # 5. Background — first sentence of env only
    env_short = (env_full.split(".")[0].strip() if "." in env_full else env_full[:150]) or "background environment"
    if env_short.lower() in ("none", ""):
        env_short = "background environment"
    if chars and len(chars) > 1:
        env_short += f", other figures blurred in background"

    # 6. Lighting
    lighting = WILSON_LIGHTING_MAP.get((scene_type or "").upper(), "dramatic rim lighting")

    # 7. Quality tags
    quality = "refined lineart, dynamic lighting, moody atmosphere, vibrant colors, clean anime rendering"

    # 8. Studio style
    studio = WILSON_STUDIO_MAP.get((scene_type or "").upper(), "Ufotable style")

    # 9. Text / end cap
    text_end = "NO TEXT, NO WATERMARK"

    return f"{bracket} of a {era} {char_desc}, {action}, {env_short}, {lighting}, {quality}, {studio}, {text_end}"


def compose_tab2_preview_prompt(
    st: ProjectState,
    scene_type: str,
    location_name: str,
    sub_location_name: str,
    camera_type: str,
    chars: List[str],
    items: List[str],
    action_text: str,
    emotion_notes: str,
    include_signature: bool,
    style: str,
    attacker_name: str = "",
    target_name: str = "",
    prompt_format: str = "wilson",
    beat_index: int = 0,
    cast_appearance: Optional[Dict[str, str]] = None,
) -> str:
    if (prompt_format or "").lower() == "wilson":
        return _compose_wilson_prompt(
            st=st,
            camera_type=camera_type,
            scene_type=scene_type,
            chars=chars,
            action_text=action_text,
            location_name=location_name,
            sub_location_name=sub_location_name,
            beat_index=beat_index,
            cast_appearance=cast_appearance,
        )
    lines = [style or DEFAULT_STYLE]
    if location_name and location_name != "None" and location_name in (st.locations or {}):
        loc_obj = st.locations.get(location_name) or {}
        sub_obj = ((loc_obj.get("sub_locations") or {}).get(sub_location_name) or {}) if sub_location_name and sub_location_name != "None" else {}
        env = clean_for_prompt(str(sub_obj.get("bible_prompt") or loc_obj.get("bible_prompt") or location_name))
        if env:
            lines.append(env)
    lines.append(CAMERA_SHOT_MAP.get(camera_type, camera_type))
    for name in chars:
        c = (st.characters or {}).get(name) or {}
        # Use cast appearance from library template if available
        dna = clean_for_prompt(cast_appearance.get(name, "") if cast_appearance else "") \
              or clean_for_prompt(str(c.get("dna_prompt") or ""))
        if dna:
            lines.append(dna)
    if emotion_notes:
        lines.append(clean_for_prompt(emotion_notes))
    for item_name in items:
        item = (st.items or {}).get(item_name) or {}
        bible = clean_for_prompt(str(item.get("bible_prompt") or ""))
        if bible:
            lines.append(bible)
    if action_text:
        lines.append(clean_for_prompt(action_text))
    lines.append(
        f"scene type: {scene_type.lower()}, "
        "bold expressive line art, deep shadow pools, dramatic high-contrast cel shading, "
        "vivid saturated color, cinematic asymmetric composition with foreground framing, "
        "large luminous eyes with sharp catchlights, 2D Korean manhwa webtoon illustration"
    )
    return clean_for_prompt("\n".join([x for x in lines if clean_for_prompt(x)]))


def _text_instruction(text_type: str, text_content: str) -> str:
    if not text_type or text_type == "None" or not text_content:
        return ""
    mapping = {
        "Speech Bubble": f'speech bubble with text: "{text_content}"',
        "Thought Bubble": f'thought bubble with text: "{text_content}"',
        "Narration Box": f'narration box with text: "{text_content}"',
        "System Message": f'system message overlay with text: "{text_content}"',
        "SFX (Sound Effect)": f'large stylized SFX text: "{text_content}"',
    }
    return mapping.get(text_type, "")


PROMPT_FORMAT_OPTIONS: List[str] = ["panel", "wilson"]
DEFAULT_PROMPT_FORMAT = "panel"


def _build_openai_final_prompt(
    st: ProjectState,
    beat_text: str,
    scene_type: str,
    location_name: str,
    sub_location_name: str,
    camera_type: str,
    chars: List[str],
    items: List[str],
    action_text: str,
    emotion_notes: str,
    include_signature: bool,
    style: str,
    base_prompt: str,
    text_type: str = "None",
    beat_index: int = 0,
    prompt_format: str = DEFAULT_PROMPT_FORMAT,
) -> Tuple[str, str]:
    beats_list = st.beats or []
    total_beats = len(beats_list)
    prev_beat_text = beats_list[beat_index - 2] if beat_index >= 2 else ""
    next_beat_text = beats_list[beat_index] if 0 < beat_index < total_beats else ""

    char_details = []
    for name in chars:
        c = (st.characters or {}).get(name) or {}
        fields = c.get("fields", {})
        detail: Dict[str, Any] = {
            "name": name,
            "dna_prompt": c.get("dna_prompt", ""),
            "hair": fields.get("hair", ""),
            "eyes": fields.get("eyes", ""),
            "outfit": fields.get("outfit", ""),
            "anchor": fields.get("anchor", ""),
            "build": fields.get("build", ""),
            "skin": fields.get("skin", ""),
        }
        # Inject age-phase appearance override when one exists for this beat
        if beat_index > 0:
            phase = age_phase_for_beat(st, name, beat_index)
            if phase and phase.get("appearance_prompt"):
                detail["age_appearance"] = phase["appearance_prompt"]
            # Inject active form (situational outfit/transformation)
            _form = get_active_form_for_beat(st, name, beat_index)
            if _form:
                _ftype = (_form.get("type") or "outfit").lower()
                _fdesc = _form.get("description") or ""
                if _fdesc:
                    if _ftype == "full_transform":
                        detail["active_form"] = f"FULL TRANSFORMATION — character body is completely: {_fdesc}. No clothing visible."
                    elif _ftype == "partial_transform":
                        detail["active_form"] = f"PARTIAL TRANSFORMATION — {_fdesc} visible extending from body. Clothing still showing underneath."
                    else:
                        detail["active_form"] = f"OUTFIT CHANGE — character is currently wearing: {_fdesc}"
        char_details.append(detail)
    loc_detail = None
    if location_name and location_name != "None" and location_name in (st.locations or {}):
        loc_obj = st.locations[location_name]
        sub_obj = ((loc_obj.get("sub_locations") or {}).get(sub_location_name) or {}) if sub_location_name and sub_location_name != "None" else {}
        loc_detail = {
            "name": location_name,
            "sub_location": sub_location_name,
            "bible_prompt": sub_obj.get("bible_prompt") or loc_obj.get("bible_prompt", ""),
        }
    item_details = []
    for name in items:
        it = (st.items or {}).get(name) or {}
        item_details.append({"name": name, "bible_prompt": it.get("bible_prompt", "")})
    payload = {
        "beat_index": beat_index,
        "total_beats": total_beats,
        "beat_text": beat_text,
        "previous_beat_text": prev_beat_text,
        "next_beat_text": next_beat_text,
        "scene_type": scene_type,
        "location": loc_detail,
        "camera_type": camera_type,
        "characters": char_details,
        "items": item_details,
        "action_text": action_text,
        "emotion_notes": emotion_notes,
        "text_type": text_type,
        "base_prompt": base_prompt,
        "prompt_format": prompt_format or DEFAULT_PROMPT_FORMAT,
    }
    improved, status = _call_openai_text_with_status(TAB2_COMPREHENSIVE_PROMPT_SYSTEM, payload, model=OPENAI_PROMPT_MODEL, max_output_tokens=1400)
    cleaned = clean_for_prompt(improved or "")
    # Wilson is a single-line format — only prose/token/core/hybrid/original need 4+ lines
    min_lines = 1 if (prompt_format or "").lower() == "wilson" else 4
    if cleaned and len(cleaned.splitlines()) >= min_lines:
        return cleaned, f"OpenAI {OPENAI_PROMPT_MODEL}: {status}"
    detail = status or f"no text returned from {OPENAI_PROMPT_MODEL}"
    # For Wilson, fall back to Python-built bracket prompt — never the raw base_seed
    if (prompt_format or "").lower() == "wilson":
        wilson = _compose_wilson_prompt(
            st=st, camera_type=camera_type, scene_type=scene_type,
            chars=chars, action_text=action_text, location_name=location_name,
            sub_location_name=sub_location_name, text_type=text_type, beat_index=beat_index,
        )
        return wilson, f"Wilson fallback (Python-built): {detail}"
    return clean_for_prompt(base_prompt), f"OpenAI fallback: {detail}"


def build_ai_prompt_preview(
    st: ProjectState,
    beat_label: str,
    scene_type_name: str,
    location_name: str,
    sub_location_name: str,
    camera_name: str,
    chars_sel: List[str],
    items_sel: List[str],
    action_line_val: str,
    emotion_notes_val: str,
    include_signature_val: bool,
    style_val: str,
    prompt_format: str = DEFAULT_PROMPT_FORMAT,
    use_openai_final: bool = True,
) -> Tuple[str, str]:
    text_type = "None"
    if not st or not st.beats:
        return "", "❌ No project yet. Build in Tab 1 first."
    idx = _beat_index_from_label(beat_label or "", len(st.beats))
    beat_text = st.beats[idx - 1] if 1 <= idx <= len(st.beats) else ""
    stype = clean_for_prompt(str(scene_type_name or _classify_scene_type(beat_text))).upper()
    if stype not in SCENE_TYPES:
        stype = _classify_scene_type(beat_text)
    camera = camera_name if camera_name in CAMERA_TYPES else (_camera_choices(stype)[0] if _camera_choices(stype) else CAMERA_TYPES[0])
    chars_use = chars_sel or []
    items_use = items_sel or []
    attacker_name = chars_use[0] if stype == "ACTION" and len(chars_use) > 1 else ""
    target_name = chars_use[1] if stype == "ACTION" and len(chars_use) > 1 else ""
    action_text = clean_for_prompt(action_line_val or beat_text)
    if stype == "ACTION" and attacker_name and target_name:
        action_text = f"{attacker_name} striking {target_name} directly in front of {attacker_name}"
    if sub_location_name and sub_location_name != "None" and location_name and location_name != "None":
        if sub_location_name.lower() not in action_text.lower():
            action_text = f"In the {sub_location_name} of the {location_name}, {action_text}"
    action_text = _rewrite_action_for_location(st, action_text, location_name)
    base_seed = compose_tab2_preview_prompt(
        st,
        stype,
        location_name,
        sub_location_name,
        camera,
        chars_use,
        items_use,
        action_text,
        emotion_notes_val or "",
        bool(include_signature_val),
        style_val or DEFAULT_STYLE,
        attacker_name,
        target_name,
        beat_index=idx,
    )
    # When AI polish is off, the preview should show exactly what FAL will receive —
    # the Python-built Wilson prompt — so preview always matches generation.
    if not use_openai_final:
        return clean_for_prompt(base_seed), "Wilson format (AI polish off — preview matches generation)"
    ai_prompt, ai_status = _build_openai_final_prompt(
        st,
        beat_text,
        stype,
        location_name,
        sub_location_name,
        camera,
        chars_use,
        items_use,
        action_text,
        emotion_notes_val or "",
        bool(include_signature_val),
        style_val or DEFAULT_STYLE,
        base_seed,
        text_type or "None",
        beat_index=idx,
        prompt_format=prompt_format or DEFAULT_PROMPT_FORMAT,
    )
    return clean_for_prompt(ai_prompt), ai_status


def refresh_lists_cb(st: ProjectState):
    if not st or not st.beats:
        return (
            1,
            "10",
            gr.Dropdown(choices=[], value=None),
            "",
            gr.Dropdown(choices=[], value=None),
            gr.Dropdown(choices=["None"], value="None"),
            gr.Dropdown(choices=SCENE_TYPES, value="EMOTION"),
            gr.Dropdown(choices=CAMERA_TYPES, value=CAMERA_TYPES[0]),
            gr.CheckboxGroup(choices=[], value=[]),
            gr.CheckboxGroup(choices=[], value=[]),
            "",
            "",
            "",
            gr.Dropdown(choices=["All beats"], value="All beats"),
            [],
            [],
            "",
            gr.Dropdown(choices=["None"], value="None"),
            "❌ No project yet. Build in Tab 1 first.",
            "",
            _format_session_cost(),
        )

    beat_choices = [f"{i+1:03d} — {b}" for i, b in enumerate(st.beats)]
    beat_filter_choices = ["All beats"] + [f"Beat {i+1:03d}" for i in range(len(st.beats))]
    gallery_paths = filter_gallery_cb(st, "All beats")
    # Generate missing thumbnails in background so the next load is much faster
    if st.image_paths:
        threading.Thread(target=_ensure_thumbs_bg, args=(list(st.image_paths),), daemon=True).start()
    first_plan = ensure_beat_plan(st, 1)
    loc_choices = _location_choices(st)
    loc_value = first_plan.get("suggested_location") or (loc_choices[0] if loc_choices else None)
    sub_choices = _sub_location_choices(st, loc_value)
    sub_value = first_plan.get("suggested_sub_location") or (sub_choices[0] if sub_choices else "None")
    scene_type = clean_for_prompt(str(first_plan.get("scene_type") or _classify_scene_type(st.beats[0]))).upper()
    if scene_type not in SCENE_TYPES:
        scene_type = "EMOTION"
    camera_choices = _camera_choices(scene_type)
    camera_value = first_plan.get("camera_type") or (camera_choices[0] if camera_choices else CAMERA_TYPES[0])
    first_beat_label = beat_choices[0] if beat_choices else None
    saved_prompt = _get_saved_prompt(st, 1)
    _saved_raw = clean_for_prompt(str(saved_prompt.get("prompt") or ""))
    # Only reuse a saved prompt if it is already in Wilson bracket format
    if _saved_raw and _saved_raw.lstrip().startswith("("):
        prompt_preview = _saved_raw
        prompt_status = f"Loaded precomputed prompt ({saved_prompt.get('source','none')})"
    else:
        prompt_preview = compose_tab2_preview_prompt(
            st,
            scene_type,
            loc_value,
            sub_value,
            camera_value,
            first_plan.get("suggested_characters") or [],
            first_plan.get("suggested_items") or [],
            first_plan.get("suggested_action") or st.beats[0],
            _emotion_text_from_plan(first_plan),
            True,
            DEFAULT_STYLE,
            first_plan.get("attacker_name") or "",
            first_plan.get("target_name") or "",
        )
        prompt_status = "Local composed prompt (press 'Refresh Lists' to run AI polish)"

    total = len(gallery_paths)
    img_note = f" | Images: {total}"

    # Resume from one past the highest generated beat (not the first gap, which is usually
    # an isolated miss in the middle rather than where the user wants to continue from)
    by_beat = st.images_by_beat or {}
    if by_beat:
        max_generated = max(by_beat.keys())
        next_beat = min(max_generated + 1, len(st.beats))
    else:
        next_beat = 1

    # For old projects with no saved cost, estimate from image count × default model rate
    if st.total_cost == 0.0 and len(st.image_paths) > 0:
        avg_cost = FAL_COST_MAP.get("fal-ai/z-image/turbo", 0.005)
        st.total_cost = round(len(st.image_paths) * avg_cost, 4)
        st.total_images = len(st.image_paths)
        _save_project_json(st)

    # Restore project cost so session display shows cumulative spend, not just this session
    global _SESSION_COST
    _SESSION_COST = {
        "total": st.total_cost,
        "images": st.total_images,
        "openai_calls": _SESSION_COST.get("openai_calls", 0),
    }

    return (
        next_beat,
        "10",
        gr.Dropdown(choices=beat_choices, value=beat_choices[0]),
        st.beats[0] if st.beats else "",
        gr.Dropdown(choices=loc_choices, value=loc_value),
        gr.Dropdown(choices=sub_choices, value=sub_value if sub_value in sub_choices else (sub_choices[0] if sub_choices else "None")),
        gr.Dropdown(choices=SCENE_TYPES, value=scene_type),
        gr.Dropdown(choices=CAMERA_TYPES, value=camera_value if camera_value in CAMERA_TYPES else CAMERA_TYPES[0]),
        gr.CheckboxGroup(choices=list(st.characters.keys()), value=first_plan.get("suggested_characters") or []),
        gr.CheckboxGroup(choices=list(st.items.keys()), value=first_plan.get("suggested_items") or []),
        first_plan.get("suggested_action") or "",
        _emotion_text_from_plan(first_plan),
        _outfit_text_from_plan(first_plan),
        gr.Dropdown(choices=beat_filter_choices, value="All beats"),
        gallery_paths,
        gallery_paths,
        "",
        gr.Dropdown(choices=_location_choices(st), value=loc_value if loc_value in _location_choices(st) else "None"),
        f"✅ Loaded: Beats={len(st.beats)} Chars={len(st.characters)} Locs={len(st.locations)} Items={len(st.items)} | Beat plans={len(getattr(st, 'beat_plans', {}) or {})}{img_note} | {prompt_status}",
        prompt_preview,
        _format_session_cost(),
    )


def on_select_location_cb(st: ProjectState, location_name: str):
    sub_choices = _sub_location_choices(st, location_name)
    return gr.Dropdown(choices=sub_choices, value=(sub_choices[0] if sub_choices else "None"))


def _beat_refs_preview_cb(st: "ProjectState", beat_label: str):
    """Return (gallery_images, html) showing which library images will influence this beat."""
    if not st or not beat_label:
        return [], "<p style='color:#888;font-size:12px'>Select a beat to see its library references.</p>"
    try:
        from character_library import resolve_scene_refs as _rsr, load_library as _ll
        idx = _beat_index_from_label(beat_label, len(st.beats) if st.beats else 0)
        if not idx:
            return [], ""
        plan = (st.beat_plans or {}).get(idx) or {}
        char_names = plan.get("suggested_characters") or []
        _ppp_n = _ppp(st)
        _page_num = (idx - 1) // _ppp_n
        _page_beats = range(_page_num * _ppp_n + 1, (_page_num + 1) * _ppp_n + 1)
        _page_bps = [st.beat_plans[b] for b in _page_beats if (st.beat_plans or {}).get(b)]
        if not char_names:
            for _bp in _page_bps:
                for _cn in (_bp.get("suggested_characters") or []):
                    if _cn not in char_names:
                        char_names.append(_cn)

        if not (st.character_cast and char_names):
            return [], "<p style='color:#888;font-size:12px'>No cast assigned — build the story first.</p>"

        _char_types = {
            name: (data.get("fields") or {}).get("character_type", "human")
            for name, data in (getattr(st, "characters", None) or {}).items()
        }
        _char_fields = {
            name: (data.get("fields") or {})
            for name, data in (getattr(st, "characters", None) or {}).items()
        }
        refs, prefix, gaps, _, _panel_ref_data = _rsr(
            char_names, st.character_cast,
            page_beat_plans=_page_bps,
            used_urls=set(),   # preview mode: ignore used tracking
            char_types=_char_types,
            char_fields=_char_fields,
        )

        lib = _ll()
        templates = lib.get("templates", {})
        cast_inv = {v: k for k, v in (st.character_cast or {}).items()}

        # Build O(1) reverse lookup: fal_url → (local_path, label_suffix)
        # Done once here instead of scanning all templates per ref URL
        _fal_to_local: dict = {}
        for _tid, _t in templates.items():
            _cname = cast_inv.get(_tid, _tid)
            if _t.get("fal_face_url"):
                _fal_to_local[_t["fal_face_url"]] = (_t.get("local_face", ""), f"{_cname} · Identity (face)")
            if _t.get("fal_body_url"):
                _fal_to_local[_t["fal_body_url"]] = (_t.get("local_body", _t.get("local_face", "")), f"{_cname} · Identity (body)")
            if _t.get("fal_url"):
                _fal_to_local[_t["fal_url"]] = (_t.get("local_face", _t.get("local_body", "")), None)

        images = []
        html_rows = []
        for r in refs:
            url = r.get("url", "")
            role = r.get("role", r.get("tag", "ref"))
            label = r.get("label", role)
            # O(1) lookup instead of scanning all templates
            local_path = ""
            if url in _fal_to_local:
                local_path, _lbl = _fal_to_local[url]
                if _lbl:
                    label = _lbl
            if local_path and os.path.exists(local_path):
                images.append((local_path, label))
            elif url:
                html_rows.append(f"<li style='color:#aaa'>{label} — uploaded (no local preview)</li>")

        # ── Classify refs by type for a cleaner display ───────────────────
        identity_images, setting_images = [], []
        identity_rows, setting_rows = [], []
        for img_tuple in images:
            lbl = img_tuple[1] if isinstance(img_tuple, tuple) else ""
            if "Identity" in lbl:
                identity_images.append(img_tuple)
            else:
                setting_images.append(img_tuple)
        for row in html_rows:
            if "Identity" in row:
                identity_rows.append(row)
            else:
                setting_rows.append(row)

        # ── Gap section — separate identity vs setting gaps ────────────────
        setting_gaps = [g for g in gaps if g.get("type") == "setting"]
        identity_gaps = [g for g in gaps if g.get("type") != "setting"]

        # Build HTML
        extra_html = f"<div style='font-size:12px;line-height:1.6'>"

        # Primary locked items: location + character
        loc = (plan.get("suggested_location") or "").strip()
        sub = (plan.get("suggested_sub_location") or "").strip()
        loc_display = f"{loc} › {sub}" if sub and sub != "None" else loc
        if loc_display:
            loc_ref_found = bool(setting_images or setting_rows)
            loc_icon = "✅" if loc_ref_found else "⚠️"
            loc_color = "#7ecb7e" if loc_ref_found else "#e8a838"
            extra_html += (
                f"<p style='margin:4px 0;color:{loc_color}'>"
                f"{loc_icon} <b>Location:</b> {loc_display}"
                + ("" if loc_ref_found else " — <i>no library image for this location</i>")
                + "</p>"
            )
            for g in setting_gaps:
                extra_html += f"<p style='margin:2px 0 2px 12px;color:#e8a838;font-size:11px'>→ Add a photo tagged '{g.get('needed','?')}' to the Library</p>"

        if char_names:
            chars_with_ref = set()
            for img_tuple in identity_images:
                lbl = img_tuple[1] if isinstance(img_tuple, tuple) else ""
                for cn in char_names:
                    if cn.lower() in lbl.lower():
                        chars_with_ref.add(cn)
            for g in identity_gaps:
                needed = g.get("needed", "")
                char_color = "#e8a838"
                extra_html += f"<p style='margin:2px 0;color:{char_color}'>⚠️ <b>Character:</b> {needed} — no library ref</p>"
            for cn in char_names:
                if cn in chars_with_ref:
                    extra_html += f"<p style='margin:2px 0;color:#7ecb7e'>✅ <b>Character:</b> {cn}</p>"

        if setting_rows:
            extra_html += "<ul style='margin:2px 0;padding-left:18px'>" + "".join(setting_rows) + "</ul>"
        if identity_rows:
            extra_html += "<ul style='margin:2px 0;padding-left:18px'>" + "".join(identity_rows) + "</ul>"

        extra_html += "</div>"

        all_images = setting_images + identity_images
        if not all_images and not setting_rows and not identity_rows and not gaps:
            return [], "<p style='color:#888;font-size:12px'>No library refs resolved — upload images to FAL in the Library tab.</p>"

        return all_images, extra_html
    except Exception as e:
        return [], f"<p style='color:#e07070;font-size:12px'>Ref preview error: {e}</p>"


def on_select_beat_fill_cb(st: ProjectState, beat_label: str):
    if not st or not st.beats:
        return (
            st,
            "",
            gr.Dropdown(choices=[], value=None),
            gr.Dropdown(choices=["None"], value="None"),
            gr.Dropdown(choices=SCENE_TYPES, value="EMOTION"),
            gr.Dropdown(choices=CAMERA_TYPES, value=CAMERA_TYPES[0]),
            [],
            [],
            "",
            "",
            "",
            "❌ Build first.",
            "",
        )
    idx = _beat_index_from_label(beat_label, len(st.beats))
    plan = ensure_beat_plan(st, idx)
    loc_choices = _location_choices(st)
    loc_value = plan.get("suggested_location") or (loc_choices[0] if loc_choices else None)
    sub_choices = _sub_location_choices(st, loc_value)
    sub_value = plan.get("suggested_sub_location") or (sub_choices[0] if sub_choices else "None")
    scene_type = clean_for_prompt(str(plan.get("scene_type") or _classify_scene_type(st.beats[idx - 1]))).upper()
    if scene_type not in SCENE_TYPES:
        scene_type = "EMOTION"
    camera_choices = _camera_choices(scene_type)
    camera_value = plan.get("camera_type") or (camera_choices[0] if camera_choices else CAMERA_TYPES[0])
    source = plan.get("plan_source", "unknown")
    saved_prompt = _get_saved_prompt(st, idx)
    _saved_raw = clean_for_prompt(str(saved_prompt.get("prompt") or ""))
    # Only reuse a saved prompt if it is already in Wilson bracket format
    if _saved_raw and _saved_raw.lstrip().startswith("("):
        prompt_preview = _saved_raw
        prompt_status = f"Loaded precomputed prompt ({saved_prompt.get('source','none')})"
    else:
        prompt_preview = compose_tab2_preview_prompt(
            st,
            scene_type,
            loc_value,
            sub_value,
            camera_value,
            plan.get("suggested_characters") or [],
            plan.get("suggested_items") or [],
            plan.get("suggested_action") or st.beats[idx - 1],
            _emotion_text_from_plan(plan),
            True,
            DEFAULT_STYLE,
            plan.get("attacker_name") or "",
            plan.get("target_name") or "",
        )
        prompt_status = f"Local composed prompt for Beat {idx:03d} ({source})"
    return (
        st,
        st.beats[idx - 1],
        gr.Dropdown(choices=loc_choices, value=loc_value),
        gr.Dropdown(choices=sub_choices, value=sub_value if sub_value in sub_choices else (sub_choices[0] if sub_choices else "None")),
        gr.Dropdown(choices=SCENE_TYPES, value=scene_type),
        gr.Dropdown(choices=CAMERA_TYPES, value=camera_value if camera_value in CAMERA_TYPES else CAMERA_TYPES[0]),
        plan.get("suggested_characters") or [],
        plan.get("suggested_items") or [],
        plan.get("suggested_action") or st.beats[idx - 1],
        _emotion_text_from_plan(plan),
        _outfit_text_from_plan(plan),
        f"✅ Beat {idx:03d} loaded ({source}). {prompt_status}",
        prompt_preview,
    )


def _sync_restored_images(st: ProjectState) -> None:
    """Check all_manifest_paths for files newly downloaded from cloud and add them to state."""
    if not st or not getattr(st, "all_manifest_paths", None):
        return
    manifest_beat = getattr(st, "manifest_beat_index", {}) or {}
    for img_path in st.all_manifest_paths:
        abs_p = os.path.abspath(img_path)
        if abs_p not in st.image_paths and os.path.isfile(abs_p):
            st.image_paths.append(abs_p)
            beat_idx = manifest_beat.get(img_path, 0) or manifest_beat.get(abs_p, 0)
            if beat_idx > 0:
                st.images_by_beat.setdefault(beat_idx, [])
                if abs_p not in st.images_by_beat[beat_idx]:
                    st.images_by_beat[beat_idx].append(abs_p)


def filter_gallery_cb(st: ProjectState, beat_filter: str):
    if not st:
        return []

    # Pick up any images that were downloaded from cloud since the project was loaded
    _sync_restored_images(st)

    def _to_thumbs(paths: List[str]) -> List[str]:
        # Gradio 6.x checks files against allowed_paths using absolute paths.
        # Always resolve to absolute so the security check passes regardless of CWD.
        out = []
        for p in paths:
            abs_p = os.path.abspath(p)
            if not os.path.isfile(abs_p):
                continue
            thumb = os.path.abspath(_get_thumb_path(p))
            out.append(thumb if os.path.isfile(thumb) else abs_p)
        return out

    if beat_filter == "All beats":
        # Sort by beat number so regenerated images stay in story order
        by_beat = st.images_by_beat or {}
        if by_beat:
            ordered: List[str] = []
            for beat_idx in sorted(by_beat.keys()):
                ordered.extend(by_beat[beat_idx])
        else:
            ordered = st.image_paths[:] if st.image_paths else []
        # Cap at GALLERY_PREVIEW_LIMIT (most recent) so the browser never downloads 900 MB
        limited = _limit_gallery(_normalize_gallery_list(ordered))
        return _to_thumbs(limited)
    m = re.match(r"^Beat\s+(\d+)", beat_filter or "")
    if not m:
        return _to_thumbs(_normalize_gallery_list(st.image_paths[:] if st.image_paths else []))
    idx = int(m.group(1))
    return _to_thumbs(_normalize_gallery_list(st.images_by_beat.get(idx, [])[:]))


def _parse_name_map(text: str) -> Dict[str, str]:
    out = {}
    for line in (text or "").splitlines():
        if ":" in line:
            a,b = line.split(":",1)
            a=a.strip(); b=clean_for_prompt(b)
            if a and b:
                out[a]=b
    return out


QUALITY_PRESETS: List[str] = [
    "🎭 NB2 Edit  nano-banana-2/edit  $0.162/img  (image-to-image · 4K + thinking · uses library refs)",
    "🐝 NB2       nano-banana-2       $0.080/img  (100 pages = 1,000 panels @ $8)",
    "🏆 NB Pro    nano-banana-pro     $0.150/img  (100 pages = 1,000 panels @ $15)",
    "⚡ Turbo     z-image/turbo       $0.005/img",
]
DEFAULT_QUALITY_PRESET = "🎭 NB2 Edit  nano-banana-2/edit  $0.162/img  (image-to-image · 4K + thinking · uses library refs)"

ASPECT_RATIO_OPTIONS: List[str] = ["9:16 (portrait)", "16:9 (landscape)", "1:1 (square)"]
DEFAULT_ASPECT_RATIO = "9:16 (portrait)"


def _parse_aspect_ratio(s: str) -> Optional[str]:
    """Extract raw ratio from option string e.g. '9:16 (portrait)' → '9:16'."""
    if not s:
        return None
    return s.split(" ")[0].strip() or None


RESOLUTION_OPTIONS: List[str] = [
    "Native (no upscale)",
    "2× ESRGAN  (~2K, +$0.001)",
    "4× ESRGAN  (~4K, +$0.002)",
]
DEFAULT_RESOLUTION = "4× ESRGAN  (~4K, +$0.002)"


def _parse_resolution(s: str):
    """Returns (skip_esrgan: bool, upscale_factor: int) from dropdown string."""
    sl = (s or "").lower()
    if "native" in sl or "no upscale" in sl:
        return True, 1
    if "4" in sl:
        return False, 4
    return False, 2


def _draw_text_overlay(img: "Image.Image", text_type: str, text_content: str) -> "Image.Image":
    """Draw a clean manga-style text overlay using PIL. Always readable — never model-rendered."""
    if not text_type or text_type == "None" or not (text_content or "").strip():
        return img
    try:
        from PIL import ImageDraw, ImageFont
        import textwrap as _tw
    except ImportError:
        return img

    text_content = (text_content or "").strip()
    W, H = img.size
    fs = max(14, W // 52)

    font = None
    for fp in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/run/current-system/sw/share/X11/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/nix/store/*/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]:
        try:
            font = ImageFont.truetype(fp, fs)
            break
        except Exception:
            pass
    if font is None:
        try:
            font = ImageFont.load_default(size=fs)
        except Exception:
            font = ImageFont.load_default()

    max_chars = max(18, int(W * 0.40 / max(1, fs * 0.55)))
    lines = _tw.fill(text_content, width=max_chars).splitlines() or [text_content]
    pad = max(8, fs // 2)
    lh = fs + pad // 2

    tmp_draw = ImageDraw.Draw(img.copy())
    try:
        max_line_w = max(tmp_draw.textlength(ln, font=font) for ln in lines)
    except Exception:
        max_line_w = max(len(ln) for ln in lines) * fs * 0.6

    box_w = min(int(max_line_w + pad * 2.5), int(W * 0.46))
    box_h = int(len(lines) * lh + pad * 2)

    img = img.copy()
    draw = ImageDraw.Draw(img, "RGBA")

    if text_type == "Narration Box":
        x, y = int(W * 0.03), int(H * 0.03)
        draw.rectangle([x, y, x + box_w, y + box_h], fill=(245, 240, 210, 240), outline=(15, 15, 15, 255), width=2)
        for i, ln in enumerate(lines):
            draw.text((x + pad, y + pad + i * lh), ln, fill=(10, 10, 10, 255), font=font)

    elif text_type == "Speech Bubble":
        x, y = int(W * 0.06), int(H * 0.04)
        draw.ellipse([x, y, x + box_w, y + box_h], fill=(255, 255, 255, 245), outline=(15, 15, 15, 255), width=2)
        tip_x = x + int(box_w * 0.25)
        tip_y = y + box_h + int(box_h * 0.38)
        draw.polygon([(x + int(box_w * 0.13), y + box_h - 3), (tip_x, tip_y), (x + int(box_w * 0.42), y + box_h - 3)],
                     fill=(255, 255, 255, 245), outline=(15, 15, 15, 255))
        for i, ln in enumerate(lines):
            draw.text((x + pad + 4, y + pad + i * lh), ln, fill=(10, 10, 10, 255), font=font)

    elif text_type == "Thought Bubble":
        x, y = int(W * 0.53), int(H * 0.04)
        try:
            draw.rounded_rectangle([x, y, x + box_w, y + box_h], radius=fs, fill=(232, 236, 255, 230), outline=(70, 70, 175, 255), width=2)
        except AttributeError:
            draw.rectangle([x, y, x + box_w, y + box_h], fill=(232, 236, 255, 230), outline=(70, 70, 175, 255), width=2)
        for i, ln in enumerate(lines):
            draw.text((x + pad, y + pad + i * lh), ln, fill=(25, 25, 90, 255), font=font)

    elif text_type == "SFX (Sound Effect)":
        sfx_fs = max(36, W // 15)
        sfx_font = None
        for fp in [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]:
            try:
                sfx_font = ImageFont.truetype(fp, sfx_fs)
                break
            except Exception:
                pass
        if sfx_font is None:
            try:
                sfx_font = ImageFont.load_default(size=sfx_fs)
            except Exception:
                sfx_font = font
        try:
            bb = draw.textbbox((0, 0), text_content, font=sfx_font)
            tw = bb[2] - bb[0]
        except Exception:
            tw = len(text_content) * sfx_fs * 0.6
        tx, ty = (W - int(tw)) // 2, H // 3
        for ox, oy in [(-3, -3), (3, -3), (-3, 3), (3, 3), (0, -3), (0, 3), (-3, 0), (3, 0)]:
            draw.text((tx + ox, ty + oy), text_content, fill=(200, 30, 30, 220), font=sfx_font)
        draw.text((tx, ty), text_content, fill=(255, 225, 20, 250), font=sfx_font)

    return img


def _quality_preset_to_params(preset: str):
    p = (preset or "").lower()
    if "nb pro" in p or "banana-pro" in p or "nano-banana-pro" in p:
        return "fal-ai/nano-banana-pro", None, 0.0
    # NB2 Edit must be checked before generic "nb2" / "banana-2" (it's a stricter match)
    elif "nb2 edit" in p or "cast-consistent" in p or "nano-banana-2/edit" in p:
        return "fal-ai/nano-banana-2/edit", None, 0.0
    elif "nb2" in p or "banana-2" in p or "nano-banana-2" in p:
        return "fal-ai/nano-banana-2", None, 0.0
    elif "turbo" in p:
        return "fal-ai/z-image/turbo", 8, 7.5
    else:
        return "fal-ai/nano-banana-2", None, 0.0


def _model_cost_html(preset: str) -> str:
    p = (preset or "").lower()
    if "nb pro" in p or "banana-pro" in p:
        name, base, note = "Nano Banana Pro", 0.150, "2K/4K: price increases — check fal.ai"
    elif "nb2 edit" in p or "cast-consistent" in p or "nano-banana-2/edit" in p:
        name, base, note = "NB2 Edit (cast-consistent)", 0.162, "4K res + high thinking included · cast required"
    elif "nb2" in p or "banana-2" in p:
        name, base, note = "Nano Banana 2", 0.080, "2K/4K: price increases — check fal.ai"
    else:
        name, base, note = "Z-Turbo", 0.005, "Fixed resolution, no 2K/4K option"
    c500 = base * 500
    return (
        f"<div style='background:#111827;border:1px solid #2d3748;border-radius:6px;"
        f"padding:7px 12px;font-size:12.5px;color:#cbd5e0;line-height:1.7'>"
        f"<b style='color:#68d391'>{name}</b> &nbsp;·&nbsp; "
        f"<b style='color:#fff'>${base:.3f}</b> per image &nbsp;·&nbsp; "
        f"<span style='color:#fbd38d'>500 images → <b>${c500:.2f}</b></span> &nbsp;·&nbsp; "
        f"<span style='color:#90cdf4'>{note}</span>"
        f"</div>"
    )


def _get_prev_page_panel_refs(
    st: "ProjectState",
    current_page_num: int,
    n_prev: int = 3,
) -> List[Dict[str, str]]:
    """Return FAL reference-image dicts for the last `n_prev` panels of page
    (current_page_num - 1), cropped from that page's most-recent generated image.

    Results are cached on st.prev_page_panel_urls so repeated regenerations of
    beats on the same page never re-upload.  Returns [] on any failure so the
    caller can always proceed without crashing.
    """
    if current_page_num < 1:
        return []

    prev_page_0 = current_page_num - 1  # 0-indexed previous page

    # ── Cache hit ────────────────────────────────────────────────────────────
    cache: Dict[int, List[str]] = getattr(st, "prev_page_panel_urls", None) or {}
    if prev_page_0 in cache:
        urls = cache[prev_page_0]
        if urls:
            _msg = f"[prev panels] Using {len(urls)} cached FAL URLs for page {prev_page_0 + 1}"
            print(_msg, flush=True)
            return [{"url": u, "tag": "character"} for u in urls]

    # ── Find the most-recent generated image for the previous page ───────────
    try:
        images_dir = ensure_dirs(st.project_dir)["images"]
    except Exception:
        return []

    prev_file_num = prev_page_0 + 1  # 1-indexed file prefix (e.g. page 0 → "001_")
    prefix = f"{prev_file_num:03d}_"
    try:
        candidates = sorted(
            [
                os.path.join(images_dir, fn)
                for fn in os.listdir(images_dir)
                if fn.startswith(prefix)
                and fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
                and os.path.isfile(os.path.join(images_dir, fn))
            ],
            key=os.path.getmtime,
            reverse=True,
        )
    except Exception:
        return []

    if not candidates:
        return []

    prev_img_path = candidates[0]  # most recent attempt of that page

    # ── Crop last n_prev panels from the 2-column grid ───────────────────────
    try:
        from PIL import Image as _PI

        im = _PI.open(prev_img_path).convert("RGB")
        w, h = im.size

        is_shorts = getattr(st, "build_mode", "Panel") == "Shorts"
        n_panels = SHORTS_PANELS_PER_PAGE if is_shorts else PANELS_PER_PAGE
        n_cols = 2
        n_rows = n_panels // n_cols
        pw = w // n_cols
        ph = h // n_rows

        # Panel numbers to extract: e.g. panels 8, 9, 10 from a 10-panel page
        first_panel = max(1, n_panels - n_prev + 1)
        urls: List[str] = []
        for panel_num in range(first_panel, n_panels + 1):
            panel_idx = panel_num - 1  # 0-indexed
            r = panel_idx // n_cols
            c = panel_idx % n_cols
            crop = im.crop((c * pw, r * ph, (c + 1) * pw, (r + 1) * ph))
            url = upload_pil_to_fal(crop)
            urls.append(url)
            print(
                f"[prev panels] Uploaded panel {panel_num}/{n_panels} of page {prev_page_0 + 1} → {url[:60]}",
                flush=True,
            )

        # ── Store in cache ────────────────────────────────────────────────────
        if not isinstance(getattr(st, "prev_page_panel_urls", None), dict):
            st.prev_page_panel_urls = {}
        st.prev_page_panel_urls[prev_page_0] = urls

        return [{"url": u, "tag": "character"} for u in urls]

    except Exception as _e:
        print(f"[prev panels] Failed to crop/upload: {_e}", flush=True)
        return []


def generate_cb(
    st: ProjectState,
    beat_label: str,
    location_name: str,
    sub_location_name: str,
    scene_type: str,
    camera_type: str,
    chars: List[str],
    items: List[str],
    action_line: str,
    emotion_notes: str,
    outfit_overrides_text: str,
    include_beat_text: bool,
    allow_silhouettes: bool,
    include_signature: bool,
    text_type: str,
    text_mode: str,
    on_image_text: str,
    style: str,
    negative: str,
    beat_filter_current: str,
    editable_prompt: str,
    final_prompt_text: str,
    use_openai_final_prompt: bool,
    enable_safety_checker: bool,
    quality_preset: str = DEFAULT_QUALITY_PRESET,
    prompt_format: str = DEFAULT_PROMPT_FORMAT,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    resolution: str = DEFAULT_RESOLUTION,
    style_ref_image=None,
    style_ref_tag: str = "style",
):
    if not st or not st.project_dir or not st.beats:
        return st, "❌ Build first in Tab 1.", None, [], clean_for_prompt(editable_prompt or ""), clean_for_prompt(final_prompt_text or ""), beat_filter_current, _format_session_cost()

    idx = _beat_index_from_label(beat_label, len(st.beats))
    beat_text = st.beats[idx - 1]
    stype = (scene_type or "").upper()
    if stype not in SCENE_TYPES:
        stype = _classify_scene_type(beat_text)
    camera_choices = _camera_choices(stype)
    if camera_type not in CAMERA_TYPES:
        camera_type = camera_choices[0] if camera_choices else CAMERA_TYPES[0]

    attacker_name = ""
    target_name = ""
    if stype == "ACTION" and len(chars or []) > 1:
        attacker_name = chars[0]
        target_name = chars[1]

    effective_action = clean_for_prompt(action_line or beat_text)
    outfit_overrides = _parse_name_map(outfit_overrides_text)
    if stype == "ACTION" and attacker_name and target_name:
        effective_action = f"{attacker_name} striking {target_name} directly in front of {attacker_name}"
    if sub_location_name and sub_location_name != "None" and location_name and location_name != "None":
        if sub_location_name.lower() not in effective_action.lower():
            effective_action = f"In the {sub_location_name} of the {location_name}, {effective_action}"
    effective_action = _rewrite_action_for_location(st, effective_action, location_name)

    if camera_type == "extreme impact close-up":
        if attacker_name and target_name:
            effective_action = f"extreme close-up of impact object from {attacker_name} smashing into {target_name}"
        else:
            effective_action = "extreme close-up of impact object smashing into target surface"

    # ── Clear per-generation step log ────────────────────────────────────────
    _last_gen_steps.clear()
    _last_gen_steps.append(f"[Beat {idx}] generation started")

    # ── Step A: Resolve library refs BEFORE building the prompt ──────────────
    # Character appearance from the cast template must feed INTO the prompt text
    # (not be bolted on afterwards), so the model knows what each character
    # actually looks like when it writes the scene description.
    _cast_refs: List[Dict[str, str]] = []
    _cast_prompt_prefix = ""
    _cast_appearance_overrides: Dict[str, str] = {}
    _ref_gaps: List[Dict] = []
    _panel_ref_data: Dict = {}
    _ppp_n = _ppp(st)
    _page_num = (idx - 1) // _ppp_n
    _page_beats_range = range(_page_num * _ppp_n + 1, (_page_num + 1) * _ppp_n + 1)
    _page_bps: List[Dict] = [st.beat_plans[b] for b in _page_beats_range
                              if b in (st.beat_plans or {})]
    _plan_chars: List[str] = []
    for _bp in _page_bps:
        for _cn in (_bp.get("suggested_characters") or []):
            if _cn not in _plan_chars:
                _plan_chars.append(_cn)
    if not _plan_chars:
        _plan_chars = ((st.beat_plans or {}).get(idx) or {}).get("suggested_characters") or []

    if st and getattr(st, "character_cast", None):
        try:
            from character_library import resolve_scene_refs as _resolve_refs
            from character_library import log_generation_gaps as _log_gaps

            # used_reference_urls now only contains SETTING urls (not char face/body),
            # so the resolver can always include character face refs for identity.
            _used_urls = set(getattr(st, "used_reference_urls", None) or [])
            _gen_char_types = {
                name: (data.get("fields") or {}).get("character_type", "human")
                for name, data in (getattr(st, "characters", None) or {}).items()
            }
            _gen_char_fields = {
                name: (data.get("fields") or {})
                for name, data in (getattr(st, "characters", None) or {}).items()
            }
            _cast_refs, _cast_prompt_prefix, _ref_gaps, _cast_appearance_overrides, _panel_ref_data = _resolve_refs(
                _plan_chars,
                st.character_cast,
                page_beat_plans=_page_bps,
                used_urls=_used_urls,
                url_usage=dict(getattr(st, "ref_url_usage", None) or {}),
                char_usage=dict(getattr(st, "char_ref_usage", None) or {}),
                char_types=_gen_char_types,
                char_fields=_gen_char_fields,
                max_total=12,
            )
            if _cast_refs:
                _msg = (
                    f"[Step A] Refs resolved: {len(_cast_refs)} total "
                    f"(page {_page_num + 1}, {len(_plan_chars)} chars, "
                    f"{len(_page_bps)} panels, "
                    f"{len(_cast_appearance_overrides)} appearance overrides)"
                )
                _last_gen_steps.append(_msg)
                print(_msg, flush=True)
            if _ref_gaps:
                _gaps_msg = f"[Step A gaps] No library ref for: {[g['needed'] for g in _ref_gaps]}"
                _last_gen_steps.append(_gaps_msg)
                print(_gaps_msg, flush=True)
                try:
                    _log_gaps(_ref_gaps, getattr(st, "project_id", ""))
                    _existing_gaps = list(getattr(st, "generation_gaps", None) or [])
                    _seen_gap_keys = {(g.get("type"), g.get("needed")) for g in _existing_gaps}
                    for _g in _ref_gaps:
                        _k = (_g.get("type"), _g.get("needed"))
                        if _k not in _seen_gap_keys:
                            _existing_gaps.append({**_g, "page": _page_num + 1})
                            _seen_gap_keys.add(_k)
                    st.generation_gaps = _existing_gaps
                except Exception:
                    pass
        except Exception as _cre:
            print(f"[generate_cb] cast ref resolver error: {_cre}", flush=True)

    # ── Block generation if any character is missing an identity reference ────
    # A panel without face refs for its characters cannot maintain identity
    # consistency. Hard-stop here so the user knows to fix the library gap
    # rather than generating an unrecognisable result.
    _char_id_gaps = [g for g in _ref_gaps if g.get("type") == "character"]
    if _char_id_gaps:
        _missing_names = ", ".join(g["needed"] for g in _char_id_gaps)
        print(f"[generate_cb] BLOCKED — missing character refs: {_missing_names}", flush=True)
        return (
            st,
            f"❌ Missing character references: {_missing_names}. "
            f"Go to the Library tab → assign cast templates with uploaded FAL images for these characters, then retry.",
            None,
            filter_gallery_cb(st, beat_filter_current),
            clean_for_prompt(editable_prompt or ""),
            clean_for_prompt(final_prompt_text or ""),
            beat_filter_current,
            _format_session_cost(),
        )

    # ── Step B: Build base prompt using cast appearance descriptions ──────────
    _last_gen_steps.append(
        f"[Step B] Building prompt — {len(_cast_appearance_overrides)} char appearances from cast templates"
    )
    base_prompt = compose_tab2_preview_prompt(
        st,
        stype,
        location_name,
        sub_location_name,
        camera_type,
        chars or [],
        items or [],
        effective_action,
        emotion_notes or "",
        bool(include_signature),
        style or DEFAULT_STYLE,
        attacker_name,
        target_name,
        beat_index=idx,
        cast_appearance=_cast_appearance_overrides or None,
    )

    if text_type == "🌶️ Auto (seasonal)":
        seasoned_type, seasoned_text = _auto_season_text(beat_text, idx, build_mode=getattr(st, 'build_mode', 'Panel'))
        text_type = seasoned_type
        if seasoned_text and not (on_image_text or "").strip():
            on_image_text = seasoned_text

    wants_text = bool(text_type) and text_type != "None"
    final_text = ""
    if wants_text:
        final_text = (on_image_text or "").strip()
        if (text_mode or "").strip().lower().startswith("auto") and not final_text:
            final_text = _call_claude_on_image_text(text_type, beat_text, effective_action) or ""
        final_text = clean_for_prompt(final_text)
    # Text is applied as PIL overlay AFTER generation — never ask the model to render words

    preview_prompt = clean_for_prompt(editable_prompt or base_prompt)
    existing_final_prompt = clean_for_prompt(final_prompt_text or "")
    if use_openai_final_prompt:
        prompt_source = "OpenAI final prompt"
        prompt, openai_status = _build_openai_final_prompt(
            st,
            beat_text,
            stype,
            location_name,
            sub_location_name,
            camera_type,
            chars or [],
            items or [],
            effective_action,
            emotion_notes or "",
            bool(include_signature),
            style or DEFAULT_STYLE,
            preview_prompt,
            "None",  # always NO TEXT in model prompt — PIL draws overlays cleanly after
            beat_index=idx,
            prompt_format=prompt_format or DEFAULT_PROMPT_FORMAT,
        )
    else:
        prompt_source = "AI prompt disabled"
        openai_status = "OpenAI AI prompt disabled"
        prompt = existing_final_prompt or preview_prompt
    # Panel format: override with precomputed page script if available
    if (prompt_format or DEFAULT_PROMPT_FORMAT) == "panel":
        _page_idx = (idx - 1) // _ppp(st)
        _page_script = (getattr(st, "page_scripts", None) or {}).get(_page_idx, "")
        if _page_script:
            # Swap out inline character appearance blocks with cast template
            # descriptions so the model never sees a conflicting dna_prompt
            # description sitting right next to the reference photos.
            if _cast_appearance_overrides:
                _page_script = _substitute_cast_in_script(_page_script, _cast_appearance_overrides)
            # The reference image header (built from cast templates) goes to the TOP
            # of the final prompt so the model sees who everyone is BEFORE it reads
            # the panel descriptions.  The separate CHARACTERS block is redundant
            # once the ref header is present — skip it to avoid conflicting text.
            # Items/creatures that have no identity ref still get a lock line.
            _item_locks = []
            for _iname, _idata in (getattr(st, "items", None) or {}).items():
                if _page_script and _iname.lower() not in _page_script.lower():
                    continue
                _ibible = ((_idata or {}).get("bible_prompt") or "").strip()
                if _ibible:
                    _item_locks.append(f"• {_iname}: {_ibible[:300]}")
            if _item_locks:
                _item_block = (
                    "CREATURES/ITEMS IN THIS PAGE — render exactly as described:\n"
                    + "\n".join(_item_locks) + "\n\n"
                )
                prompt = _item_block + strip_shot_labels(_page_script)
            else:
                prompt = strip_shot_labels(_page_script)
            prompt_source = f"panel page script (page {_page_idx + 1})"
            openai_status = "panel mode"
    neg = (negative or DEFAULT_NEGATIVE).strip()
    if not allow_silhouettes:
        if "silhouette" not in neg.lower():
            neg = (neg + ", silhouette, shadow person, background person").strip().strip(",")
    if camera_type == "extreme impact close-up":
        extra = "full body, portrait, selfie, distant shot, wide landscape view"
        neg = clean_for_prompt(f"{neg}, {extra}")

    fal_model, fal_steps, fal_guidance = _quality_preset_to_params(quality_preset or DEFAULT_QUALITY_PRESET)
    _ar_override = _parse_aspect_ratio(aspect_ratio or DEFAULT_ASPECT_RATIO)
    _skip_esrgan, _upscale = _parse_resolution(resolution or DEFAULT_RESOLUTION)

    # ── Inject "Use Image N (role only)." lines below each PANEL header ────────
    # Matches the prompt style in the example: each panel section starts with
    # explicit "Use Image N (role)." directives so the model knows exactly which
    # ref to apply and for what purpose — no ambiguous bracket annotations.
    # Mapping: signal_type → human-readable role description
    _ROLE_LABELS = {
        "camera": "camera/composition only — copy the exact shot angle, depth, and framing geometry; all people are invisible mannequins showing spatial arrangement only",
        "mood":   "lighting and atmosphere only — copy lighting direction, quality (hard vs soft), and color temperature; all people are invisible",
        "action": "body POSE SKELETON only — trace the joint positions, limb angles, body angle, and weight distribution ONLY. The reference figure's skin coverage, clothing material, props, setting decor, and expression are 100% invisible — do NOT copy them. Replace the figure with the story character wearing their own period-appropriate outfit with their own level of clothing coverage.",
    }
    _LOC_ROLE = "setting/architecture only — copy the background environment, architecture, and ambient light quality; all people are invisible"
    if _panel_ref_data and prompt:
        import re as _re
        _pm   = (_panel_ref_data or {}).get("panel_map") or {}
        _locs = (_panel_ref_data or {}).get("loc_refs") or []

        def _inject_line(m: "_re.Match") -> str:  # type: ignore[name-defined]
            _hdr  = m.group(1)           # e.g. "PANEL 3 [MCU]:"
            _pidx = int(m.group(2)) - 1  # 0-indexed
            _rest = m.group(3).strip()   # "Chen Ping kneeling..."
            _bucket = _pm.get(_pidx, {}) if isinstance(_pm.get(_pidx), dict) else {}
            _use_lines: list = []
            for _stype in ("camera", "mood", "action"):
                _nums = _bucket.get(_stype, [])
                if _nums:
                    _img_str = ", ".join(f"Image {r}" for r in _nums)
                    _use_lines.append(f"Use {_img_str} ({_ROLE_LABELS[_stype]}).")
            if _locs:
                _loc_img = ", ".join(f"Image {r}" for r in _locs)
                _use_lines.append(f"Use {_loc_img} ({_LOC_ROLE}).")
            if _use_lines:
                return _hdr + "\n" + "\n".join(_use_lines) + "\n\n" + _rest
            return m.group(0)

        prompt = _re.sub(
            r"(PANEL\s+(\d+)(?:\s*\[[^\]]*\])?\s*:)(.*)",
            _inject_line,
            prompt,
        )
        _last_gen_steps.append(
            f"[Step C-inline] Injected Use Image directives into {len(_pm)} panels"
        )

    # ── Append art direction sections ─────────────────────────────────────────
    # These go AFTER the panel descriptions so the model has already read the
    # action before seeing the quality/style/rules constraints.
    _div = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    _ART_DIRECTION = f"""
{_div}
GLOBAL QUALITY PRIORITIES
{_div}

Priority order:
1. Preserve each character's exact locked identity (face, hair, eyes, skin tone, outfit).
2. Apply the reference images exactly as directed inside each panel section.
3. Follow the camera angle, shot type, and character arrangement for each panel.
4. Render clear, readable story action in every panel.
5. Apply premium facial quality: refined anatomy, gradient irises, sharp eye highlights, individually rendered hair strands.
6. Simplify anonymous background people and minor props before touching the main characters.

{_div}
ART STYLE
{_div}

Premium high-budget Korean action-fantasy manhwa / webtoon.

Use:
- Exceptionally clean thick-to-thin linework
- Refined attractive facial anatomy with consistent proportions
- Detailed luminous gradient eyes with crisp catchlights
- Individually rendered hair strands
- Smooth professional cel shading with soft secondary shadows
- Natural fabric folds and detailed hands
- Cinematic depth with controlled motion effects
- Rich warm-versus-cool lighting contrast
- Bright, readable focal characters
- Polished 2D webtoon finish at high-resolution appearance
- Not photorealistic

{_div}
PANEL RULES
{_div}

- Keep every character, limb, and important object fully inside its panel.
- No characters or objects crossing white gutters between panels.
- No cropped heads or cut-off hands on main characters.
- No dialogue bubbles, speech captions, signs, logos, watermarks, or readable writing.
- Read left-to-right, top-to-bottom.

{_div}
NEGATIVE PROMPT
{_div}

identity drift, different face in each panel, changed hair color, changed eye color, changed skin tone, wrong age, redesigned outfit, copied reference people, copied reference clothing, face swapping, outfit swapping, extra limbs, fused fingers, broken anatomy, malformed hands, muddy shadows, dull eyes, excessive darkness, rough unfinished coloring, unreadable action, cluttered composition, characters crossing panel gutters, speech bubbles, readable text, watermark, logo, photorealism, blurry, low resolution, generic anime, inconsistent character proportions, shirtless when clothed in story, bare skin from reference bleeding through, smartphone, modern electronics, gym floor, gym equipment, modern props in historical setting, anachronistic objects from reference images, reference character's skin exposure level, reference character's clothing copied onto story character, reference props visible in panel.
"""
    if prompt and (prompt_format or DEFAULT_PROMPT_FORMAT) == "panel":
        prompt = prompt + _ART_DIRECTION

    # Place the reference-image header at the TOP of the final prompt so the model
    # sees who everyone is BEFORE reading the panel descriptions.  It goes after
    # OpenAI refinement so it isn't treated as prose to rewrite.
    if _cast_prompt_prefix and prompt:
        _last_gen_steps.append(
            f"[Step C] Prepending ref header ({len(_cast_refs)} refs) to prompt top"
        )
        prompt = _cast_prompt_prefix + "\n" + prompt

    # If NB2 Edit was requested but no refs resolved (cast not set, or no FAL URLs
    # uploaded yet), block entirely — silently downgrading to plain NB2 produces
    # images without identity consistency which is worse than a clear error.
    if fal_model == "fal-ai/nano-banana-2/edit" and not _cast_refs:
        print("[generate_cb] BLOCKED — NB2 Edit requested but no FAL refs available.", flush=True)
        return (
            st,
            "❌ No character references available for image-to-image generation. "
            "Go to the Library tab → assign cast templates → upload to FAL, then retry.",
            None,
            filter_gallery_cb(st, beat_filter_current),
            clean_for_prompt(editable_prompt or ""),
            clean_for_prompt(final_prompt_text or ""),
            beat_filter_current,
            _format_session_cost(),
        )

    # Omni / style reference — upload PIL image to FAL storage if provided
    # For NB2 Edit, cast refs take precedence; style ref is appended after
    _reference_images: Optional[List[Dict[str, str]]] = _cast_refs if _cast_refs else None

    # ── Previous-page panel refs (Panel/Shorts mode, page ≥ 2) ──────────────
    # Take the last 3 panels from the previous page's generated image and add
    # them as character references.  This anchors appearance across page
    # boundaries — especially helpful in long stories where character drift
    # accumulates.  Skipped on page 1 (no previous page exists) and in Normal
    # mode (individual beat images have no grid to crop).
    if _page_num > 0 and getattr(st, "build_mode", "Panel") in ("Panel", "Shorts"):
        _prev_refs = _get_prev_page_panel_refs(st, _page_num)
        if _prev_refs:
            if _reference_images is None:
                _reference_images = []
            _reference_images = list(_reference_images) + _prev_refs
            _last_gen_steps.append(
                f"[Step A+] Added {len(_prev_refs)} prev-page panel refs for continuity"
            )

    if style_ref_image is not None:
        try:
            _ref_url = upload_pil_to_fal(style_ref_image)
            _tag = style_ref_tag if style_ref_tag in ("style", "character", "face", "composition") else "style"
            if _reference_images is None:
                _reference_images = []
            _reference_images.append({"url": _ref_url, "tag": _tag})
            st.style_reference_url = _ref_url
        except Exception as _re:
            pass  # reference upload failure never blocks generation

    # NB2 Edit outputs at 768×1376 regardless of resolution param — run ESRGAN to reach full size
    _effective_skip_esrgan = _skip_esrgan
    try:
        img = call_fal_generate(prompt, neg, enable_safety_checker=enable_safety_checker, model=fal_model, num_inference_steps=fal_steps, guidance_scale=fal_guidance, aspect_ratio_override=_ar_override, skip_esrgan=_effective_skip_esrgan, upscale_factor=_upscale, reference_images=_reference_images)
    except Exception as e:
        return st, f"❌ Generate failed: {e} | {openai_status}", None, filter_gallery_cb(st, beat_filter_current), clean_for_prompt(editable_prompt or base_prompt), clean_for_prompt(prompt), beat_filter_current, _format_session_cost()

    # Text overlay removed

    # ── FAST LOCAL SAVE (lock held briefly — no slow uploads inside) ───────
    with _BATCH_SAVE_LOCK:
        out_dir = ensure_dirs(st.project_dir)["images"]
        beat_count = len((st.images_by_beat or {}).get(idx, [])) + 1
        if (prompt_format or DEFAULT_PROMPT_FORMAT) == "panel":
            file_num = (idx - 1) // _ppp(st) + 1
        else:
            file_num = idx
        fn = f"{file_num:03d}_{beat_count:02d}.{FAL_OUTPUT_EXT}"
        out_path = os.path.abspath(os.path.join(out_dir, fn))
        img.save(out_path)
        threading.Thread(target=_make_thumb, args=(out_path,), daemon=True).start()

        st.next_image_index = max(int(getattr(st, "next_image_index", 1) or 1), idx * 100 + beat_count + 1)
        st.image_paths.append(out_path)
        st.images_by_beat.setdefault(idx, []).append(out_path)

        # ── Usage-frequency counters ───────────────────────────────────────────
        # Global persistent counter (url_usage.json) accumulates across ALL
        # projects and ALL generations ever — this is the primary signal for
        # deprioritising over-used reference images in future selections.
        # Per-project st.ref_url_usage is kept as a same-session fast-path.
        if _cast_refs:
            _ref_urls = [_r.get("url", "") for _r in _cast_refs if _r.get("url")]
            if _ref_urls:
                from character_library import increment_global_url_usage as _inc_usage
                _inc_usage(_ref_urls)
            _url_usage = dict(getattr(st, "ref_url_usage", None) or {})
            for _u in _ref_urls:
                _url_usage[_u] = _url_usage.get(_u, 0) + 1
            st.ref_url_usage = _url_usage

        if _plan_chars:
            _ch_usage = dict(getattr(st, "char_ref_usage", None) or {})
            for _cn in _plan_chars:
                _ch_usage[_cn] = _ch_usage.get(_cn, 0) + 1
            st.char_ref_usage = _ch_usage

        # Only blacklist SETTING refs (tag=="style") across pages — character
        # face/body refs must be reused every page for identity consistency.
        if _cast_refs:
            _setting_urls = {_r["url"] for _r in _cast_refs
                             if _r.get("tag") == "style" and _r.get("url")}
            if _setting_urls:
                _prev_used = set(getattr(st, "used_reference_urls", None) or [])
                _prev_used.update(_setting_urls)
                st.used_reference_urls = list(_prev_used)
        # Keep all_manifest_paths + manifest_beat_index in sync so ZIP always
        # finds ALL generated images (not just the ones present at project load time)
        if not isinstance(getattr(st, "all_manifest_paths", None), list):
            st.all_manifest_paths = []
        if out_path not in st.all_manifest_paths:
            st.all_manifest_paths.append(out_path)
        if not isinstance(getattr(st, "manifest_beat_index", None), dict):
            st.manifest_beat_index = {}
        st.manifest_beat_index[out_path] = idx

        manifest = os.path.join(st.project_dir, "manifest.jsonl")
        entry = {
            "image_index": beat_count,
            "beat_index": idx,
            "beat_text": beat_text,
            "action_line": action_line,
            "scene_type": stype,
            "camera_type": camera_type,
            "sub_location": sub_location_name,
            "emotion_notes": emotion_notes,
            "outfit_overrides": outfit_overrides,
            "location": location_name,
            "perspective": camera_type,
            "characters": chars,
            "items": items,
            "prompt": prompt,
            "negative_prompt": neg,
            "image_path": out_path,
            "reference_images": [r.get("url", "") for r in (_cast_refs or []) if r.get("url")],
        }
        with open(manifest, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        new_gallery = filter_gallery_cb(st, beat_filter_current)
        _res = "4K" if fal_model == "fal-ai/nano-banana-2/edit" else "1K"
        _thinking = fal_model == "fal-ai/nano-banana-2/edit"  # edit always uses thinking_level="high"
        fal_cost_val = _calc_fal_cost(fal_model, resolution=_res, use_thinking=_thinking)
        openai_cost_val = OPENAI_PROMPT_COST if use_openai_final_prompt else 0.0
        st.total_cost = round(st.total_cost + fal_cost_val + openai_cost_val, 6)
        st.total_images += 1
        _save_project_json(st)
        _add_session_cost(fal_model, use_openai_final_prompt, resolution=_res, use_thinking=_thinking)
        cost_str = _format_session_cost()

    # ── SLOW CLOUD UPLOADS (outside lock — workers run truly in parallel) ──
    # Image first — must be in cloud before manifest references it.
    _cloud.upload_file_blocking(out_path)
    # Manifest is the recovery record. Multiple workers may upload it
    # concurrently; last writer wins, which is always a valid superset.
    _cloud.upload_file_blocking(manifest)

    return st, f"✅ Beat {idx:03d} → {fn} | {prompt_source} | {openai_status}", img, new_gallery, clean_for_prompt(prompt), clean_for_prompt(prompt), beat_filter_current, cost_str


def batch_generate_cb(
    batch_start: int,
    batch_size_str: str,
    st: ProjectState,
    beat_label: str,
    location_name: str,
    sub_location_name: str,
    scene_type_name: str,
    camera_name: str,
    chars_sel: List[str],
    items_sel: List[str],
    action_line_val: str,
    emotion_notes_val: str,
    outfit_overrides_val: str,
    include_beat_text: bool,
    allow_silhouettes: bool,
    include_signature: bool,
    text_type: str,
    text_mode: str,
    on_image_text: str,
    style: str,
    negative: str,
    beat_filter_current: str,
    editable_prompt: str,
    final_prompt_text: str,
    use_openai_final: bool,
    enable_safety_checker: bool,
    quality_preset: str = DEFAULT_QUALITY_PRESET,
    prompt_format: str = DEFAULT_PROMPT_FORMAT,
    aspect_ratio: str = DEFAULT_ASPECT_RATIO,
    resolution: str = DEFAULT_RESOLUTION,
    style_ref_image=None,
    style_ref_tag: str = "style",
    concurrency_str: str = "1",
    progress=gr.Progress(track_tqdm=False),
):
    """Generator — yields one 9-tuple per image (adds visible_gallery_state as 9th element).
    Size=1: single beat via beat_label. Size>1: batch from batch_start, cycling cameras."""
    import time as _time

    def _mk_status(msg: str, g=None, bt=""):
        gall = g or []
        return (st, msg, None, gall,
                clean_for_prompt(editable_prompt or ""),
                clean_for_prompt(final_prompt_text or ""),
                beat_filter_current, _format_session_cost(), gall, bt, gr.update(), gr.update())

    _BATCH_STOP_EVENT.clear()
    _BATCH_PAUSE_EVENT.clear()

    is_panel = (prompt_format or DEFAULT_PROMPT_FORMAT) == "panel"

    try:
        bs_str = str(batch_size_str).strip()
        if bs_str.lower() == "all":
            if is_panel and st and st.beats:
                batch_size = (len(st.beats) + _ppp(st) - 1) // _ppp(st)
            else:
                batch_size = len(st.beats) if (st and st.beats) else 999
        else:
            batch_size = int(bs_str)
    except Exception:
        batch_size = 10

    # ── SINGLE BEAT MODE (size=0 only — size=1 falls through to batch path) ────
    # NOTE: batch_size=1 from the dropdown means "generate 1 beat/page starting at From Beat/Page"
    # and should use the batch path so it respects batch_start_num, not beat_selector.
    if batch_size < 1:
        result = generate_cb(
            st, beat_label,
            location_name, sub_location_name, scene_type_name, camera_name,
            chars_sel, items_sel, action_line_val, emotion_notes_val, outfit_overrides_val,
            include_beat_text, allow_silhouettes, include_signature,
            text_type, text_mode, on_image_text,
            style, negative, beat_filter_current, editable_prompt, final_prompt_text,
            use_openai_final, enable_safety_checker, quality_preset, prompt_format, aspect_ratio, resolution,
            style_ref_image, style_ref_tag,
        )
        rl = list(result)
        new_gall = rl[3]
        rl.append(new_gall)  # visible_gallery_state
        bt = str(beat_label or "")
        bt = bt.split(" — ", 1)[1] if " — " in bt else bt
        rl.append(bt)   # beat_text_display
        latest_fname = os.path.basename(new_gall[-1]) if new_gall else ""
        rl.append(latest_fname)  # latest_image_name_display
        rl.append(gr.update())  # batch_start_num — no advance in single mode
        yield tuple(rl)
        return

    # ── BATCH MODE ─────────────────────────────────────────────────────────
    if not st or not st.beats:
        yield _mk_status("❌ Build first in Tab 1.")
        return

    if is_panel:
        # batch_start = PAGE number (1-based), batch_size = number of pages
        # 1 page = _ppp(st) beats → 1 page image
        _ppg = _ppp(st)
        n_pages_total = (len(st.beats) + _ppg - 1) // _ppg
        start_page = max(1, int(batch_start or 1))
        if start_page > n_pages_total:
            yield _mk_status(
                f"⚠️ From Page ({start_page}) is past the end of your story "
                f"({n_pages_total} pages total). Set 'From Page' to 1 to regenerate from the beginning, "
                f"or pick any page between 1 and {n_pages_total}."
            )
            return
        start_beat = (start_page - 1) * _ppg + 1
        end_beat = min(start_beat + batch_size * _ppg - 1, len(st.beats))
        # Clamp next_page so the counter never shoots past the story end
        _next_page = min(start_page + batch_size, n_pages_total)
    else:
        start_beat = max(1, int(batch_start or 1))
        end_beat = min(start_beat + batch_size - 1, len(st.beats))
        n_pages_total = 0
        _next_page = None

    beats_in_range = list(range(start_beat, end_beat + 1))
    # Panel mode: deduplicate — only generate ONE image per page group of 10 beats.
    # Do NOT require a page_scripts entry — that would silently skip redo of old pages.
    if is_panel:
        _seen_pgs: set = set()
        _panel_beats: List[int] = []
        for _b in beats_in_range:
            _pg = (_b - 1) // _ppp(st)
            if _pg not in _seen_pgs:
                _seen_pgs.add(_pg)
                _panel_beats.append(_b)
        beats_in_range = _panel_beats
    total = len(beats_in_range)

    wall_t0 = _time.time()
    times: List[float] = []
    last_beat_text = ""

    concurrency = max(1, int((concurrency_str or "1").split()[0]))

    # Pre-plan beats that have no plan, a heuristic fallback, OR a Claude plan
    # whose location was sanitized to "None" (happens when Claude returns a slightly
    # different location name that didn't match the stored key exactly).
    # All three cases produce bland "background environment" images.
    def _beat_needs_plan(i: int) -> bool:
        p = (st.beat_plans or {}).get(i)
        if not p:
            return True
        if p.get("plan_source") == "heuristic":
            return True
        loc = (p.get("suggested_location") or "").strip()
        if not loc or loc == "None":
            return True
        # Fourth bad-plan category: plan lists characters but NONE survive fuzzy
        # matching — the names on disk are garbage (pre-normalization-fix plans).
        _vc = set((st.characters or {}).keys())
        raw_chars = p.get("suggested_characters") or []
        if raw_chars and _vc:
            def _quick_snap(name: str) -> bool:
                if name in _vc:
                    return True
                nl = name.lower()
                if any(v.lower() == nl for v in _vc):
                    return True
                if len(nl) >= 3 and any(nl in v.lower() or v.lower() in nl for v in _vc):
                    return True
                return False
            if not any(_quick_snap(c) for c in raw_chars):
                return True
        return False

    weak_count = sum(1 for i in beats_in_range if _beat_needs_plan(i))
    if weak_count > 0:
        progress(0, desc=f"Pre-planning {weak_count} beats before generating…")
        yield _mk_status(f"🔄 Pre-planning {weak_count} beats with missing/heuristic plans (est. ~{max(5, weak_count // 2)}s)…")
        _fill_missing_beat_plans(st, beats_in_range)

    progress(0, desc=f"Starting batch — {total} images to generate… ({concurrency} worker{'s' if concurrency > 1 else ''})")

    def _build_beat_args(beat_idx: int):
        """Build generate_cb call args for one beat. Fast, serial, read-only on st."""
        if beat_idx < 1 or beat_idx > len(st.beats):
            return None
        beat_text = st.beats[beat_idx - 1]
        plan = ensure_beat_plan(st, beat_idx)
        beat_label_idx = f"{beat_idx:03d} — {beat_text}"
        loc = clean_for_prompt(str(plan.get("suggested_location") or "None")) or "None"
        sub_loc = clean_for_prompt(str(plan.get("suggested_sub_location") or "None")) or "None"
        scene_t = clean_for_prompt(str(plan.get("scene_type") or _classify_scene_type(beat_text))).upper()
        if scene_t not in SCENE_TYPES:
            scene_t = "EMOTION"
        camera_pool = _camera_choices(scene_t) or CAMERA_TYPES
        cam = camera_pool[(beat_idx - 1) % len(camera_pool)]
        # Fuzzy-match character names at the last step before generation — same logic
        # as _sanitize_plan, because plans on disk may predate the source normalization.
        _valid_chars_bg = set((st.characters or {}).keys())
        def _snap_char_bg(name: str) -> str:
            if name in _valid_chars_bg:
                return name
            nl = name.lower()
            m = next((v for v in _valid_chars_bg if v.lower() == nl), None)
            if m:
                return m
            if len(nl) >= 3:
                m = next((v for v in _valid_chars_bg if nl in v.lower() or v.lower() in nl), None)
            return m or ""
        chars_list = [r for r in (_snap_char_bg(c) for c in (plan.get("suggested_characters") or [])) if r]
        # Deduplicate
        _seen_c: set = set()
        chars_list = [c for c in chars_list if not (_seen_c.add(c) or c in _seen_c)]  # type: ignore[func-returns-value]
        items_list = [it for it in (plan.get("suggested_items") or []) if it in (st.items or {})]
        action = clean_for_prompt(str(plan.get("suggested_action") or beat_text))
        emotion_text = _emotion_text_from_plan(plan)
        return (beat_idx, beat_text, beat_label_idx, loc, sub_loc, scene_t, cam,
                chars_list, items_list, action, emotion_text)

    def _submit_beat(executor, beat_idx):
        """Pre-build args then submit generate_cb to the thread pool."""
        args = _build_beat_args(beat_idx)
        if args is None:
            return None, beat_idx
        (beat_idx, beat_text, beat_label_idx, loc, sub_loc, scene_t, cam,
         chars_list, items_list, action, emotion_text) = args
        future = executor.submit(
            generate_cb,
            st, beat_label_idx,
            loc, sub_loc, scene_t, cam,
            chars_list, items_list,
            action, emotion_text, "",
            include_beat_text, allow_silhouettes, include_signature,
            text_type, text_mode, "",
            style, negative, beat_filter_current,
            "", "",
            use_openai_final, enable_safety_checker, quality_preset, prompt_format, aspect_ratio, resolution,
            style_ref_image, style_ref_tag,
        )
        return future, beat_text

    if concurrency <= 1:
        # ── SEQUENTIAL PATH ────────────────────────────────────────────────
        for i, beat_idx in enumerate(beats_in_range):
            progress(i / total, desc=f"Image {i + 1}/{total} — Beat {beat_idx:03d}")
            if _BATCH_STOP_EVENT.is_set():
                _BATCH_STOP_EVENT.clear()
                g = filter_gallery_cb(st, beat_filter_current)
                yield (st,
                       f"⏹ Stopped after {i}/{total} images. "
                       f"Set 'From Beat' to {beat_idx} to resume.",
                       None, g, clean_for_prompt(editable_prompt or ""),
                       clean_for_prompt(final_prompt_text or ""),
                       beat_filter_current, _format_session_cost(), g, last_beat_text, gr.update(), beat_idx)
                return

            if _BATCH_PAUSE_EVENT.is_set():
                g = filter_gallery_cb(st, beat_filter_current)
                while _BATCH_PAUSE_EVENT.is_set():
                    yield (st,
                           f"⏸ Paused — beat {beat_idx}/{end_beat}. Click ▶ Resume to continue.",
                           None, g, clean_for_prompt(editable_prompt or ""),
                           clean_for_prompt(final_prompt_text or ""),
                           beat_filter_current, _format_session_cost(), g, last_beat_text, gr.update(), gr.update())
                    _time.sleep(0.4)

            args = _build_beat_args(beat_idx)
            if args is None:
                continue
            (beat_idx, beat_text, beat_label_idx, loc, sub_loc, scene_t, cam,
             chars_list, items_list, action, emotion_text) = args

            t0 = _time.time()
            result = generate_cb(
                st, beat_label_idx,
                loc, sub_loc, scene_t, cam,
                chars_list, items_list,
                action, emotion_text, "",
                include_beat_text, allow_silhouettes, include_signature,
                text_type, text_mode, "",
                style, negative, beat_filter_current,
                "", "",
                use_openai_final, enable_safety_checker, quality_preset, prompt_format, aspect_ratio, resolution,
                style_ref_image, style_ref_tag,
            )
            elapsed = _time.time() - t0
            times.append(elapsed)
            avg = sum(times) / len(times)

            st = result[0]
            rl = list(result)
            new_gall = rl[3]
            rl[1] = (
                f"[{i + 1}/{total}] Beat {beat_idx:03d}: {rl[1]} "
                f"| ⏱ {elapsed:.1f}s · avg {avg:.1f}s/img"
            )
            last_beat_text = beat_text
            latest_fname = os.path.basename(new_gall[-1]) if new_gall else ""
            rl.append(new_gall)
            rl.append(beat_text)
            rl.append(latest_fname)
            rl.append(gr.update())
            yield tuple(rl)

    else:
        # ── PARALLEL PATH (N concurrent FAL calls) ─────────────────────────
        import concurrent.futures as _cf

        # Pre-build all beat plans serially (fast, ensures plans are ready before threads start)
        beat_args_list = []
        for beat_idx in beats_in_range:
            args = _build_beat_args(beat_idx)
            if args is not None:
                beat_args_list.append(args)

        with _cf.ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_info: Dict[Any, tuple] = {}
            for args in beat_args_list:
                (beat_idx, beat_text, beat_label_idx, loc, sub_loc, scene_t, cam,
                 chars_list, items_list, action, emotion_text) = args
                future = executor.submit(
                    generate_cb,
                    st, beat_label_idx,
                    loc, sub_loc, scene_t, cam,
                    chars_list, items_list,
                    action, emotion_text, "",
                    include_beat_text, allow_silhouettes, include_signature,
                    text_type, text_mode, "",
                    style, negative, beat_filter_current,
                    "", "",
                    use_openai_final, enable_safety_checker, quality_preset, prompt_format, aspect_ratio, resolution,
                )
                future_to_info[future] = (beat_idx, beat_text)

            completed = 0
            for future in _cf.as_completed(future_to_info):
                beat_idx, beat_text = future_to_info[future]
                completed += 1
                progress(completed / total,
                         desc=f"Image {completed}/{total} — Beat {beat_idx:03d} ⚡{concurrency}×")

                if _BATCH_STOP_EVENT.is_set():
                    _BATCH_STOP_EVENT.clear()
                    for f in future_to_info:
                        f.cancel()
                    g = filter_gallery_cb(st, beat_filter_current)
                    yield (st,
                           f"⏹ Stopped after {completed}/{total} images.",
                           None, g, clean_for_prompt(editable_prompt or ""),
                           clean_for_prompt(final_prompt_text or ""),
                           beat_filter_current, _format_session_cost(), g, last_beat_text, gr.update(), beat_idx)
                    return

                try:
                    result = future.result()
                except Exception as e:
                    g = filter_gallery_cb(st, beat_filter_current)
                    yield _mk_status(f"❌ Beat {beat_idx:03d} failed: {e}", g=g)
                    continue

                elapsed_total = _time.time() - wall_t0
                st = result[0]
                rl = list(result)
                new_gall = rl[3]
                rl[1] = (
                    f"[{completed}/{total}] Beat {beat_idx:03d}: {rl[1]} "
                    f"| ⏱ {elapsed_total:.1f}s elapsed ⚡{concurrency}× workers"
                )
                last_beat_text = beat_text
                latest_fname = os.path.basename(new_gall[-1]) if new_gall else ""
                rl.append(new_gall)
                rl.append(beat_text)
                rl.append(latest_fname)
                rl.append(gr.update())
                yield tuple(rl)

    total_elapsed = _time.time() - wall_t0
    avg_final = total_elapsed / total if total else 0
    final_g = filter_gallery_cb(st, beat_filter_current)
    # Advance by ACTUAL pages generated (total), not batch_size — prevents counter overshoot
    if is_panel:
        next_start = min(start_page + total, n_pages_total)
    else:
        next_start = end_beat + 1
    progress(1.0, desc=f"✅ Done — {total} images in {total_elapsed:.0f}s")
    if is_panel:
        all_done = next_start >= n_pages_total
        if all_done:
            done_msg = f"✅ All {total} pages done in {total_elapsed:.0f}s (avg {avg_final:.1f}s/page). Story complete ({n_pages_total} pages total)."
        else:
            done_msg = (f"✅ Done — {total} pages in {total_elapsed:.0f}s (avg {avg_final:.1f}s/page). "
                        f"Next: page {next_start + 1} of {n_pages_total}.")
    else:
        if batch_size >= len(st.beats):
            done_msg = f"✅ All {total} beats done in {total_elapsed:.0f}s (avg {avg_final:.1f}s/img)."
        else:
            done_msg = (f"✅ Done — {total} imgs in {total_elapsed:.0f}s (avg {avg_final:.1f}s/img). "
                        f"Next: beats {next_start}–{next_start + batch_size - 1}.")
    yield (st,
           done_msg,
           None, final_g,
           clean_for_prompt(editable_prompt or ""),
           clean_for_prompt(final_prompt_text or ""),
           beat_filter_current, _format_session_cost(), final_g,
           last_beat_text,  # beat_text_display — keep last beat
           gr.update(),     # latest_image_name_display
           next_start)      # auto-advance batch_start_num


def _purge_images(st: ProjectState, original_paths: List[str]) -> List[str]:
    """Remove a list of original image paths from all state collections, disk, and cloud.
    Returns a list of errors (empty = full success). Also rewrites manifest.jsonl."""
    from cloud_storage import delete_files_bg, upload_file_bg
    errors: List[str] = []
    to_delete_cloud: List[str] = []
    basenames_removed: set = set()

    for orig in original_paths:
        base = os.path.splitext(os.path.basename(orig))[0]
        basenames_removed.add(base)

        # ── remove from all in-memory state ──────────────────────────────
        if orig in (st.image_paths or []):
            st.image_paths.remove(orig)
        for k in list((st.images_by_beat or {}).keys()):
            if orig in st.images_by_beat[k]:
                st.images_by_beat[k].remove(orig)
            if not st.images_by_beat[k]:
                st.images_by_beat.pop(k, None)
        if orig in (getattr(st, "all_manifest_paths", None) or []):
            st.all_manifest_paths.remove(orig)
        mbi = getattr(st, "manifest_beat_index", None) or {}
        mbi.pop(orig, None)
        st.manifest_beat_index = mbi

        # ── delete local files ────────────────────────────────────────────
        thumb = _get_thumb_path(orig)
        for path in [orig, thumb]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    to_delete_cloud.append(path)
                except Exception as e:
                    errors.append(f"{os.path.basename(path)}: {e}")

    # ── rewrite manifest.jsonl (filter out deleted basenames) ────────────
    if st.project_dir and basenames_removed:
        manifest_path = os.path.join(st.project_dir, "manifest.jsonl")
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r") as f:
                    lines = f.readlines()
                kept = []
                for line in lines:
                    stem = os.path.splitext(os.path.basename(line.split('"image_path"')[-1].split('"')[1] if '"image_path"' in line else "")).strip()[0] if '"image_path"' in line else "__keep__"
                    # simple: keep line if none of the deleted basenames appear in it
                    if not any(b in line for b in basenames_removed):
                        kept.append(line)
                with open(manifest_path, "w") as f:
                    f.writelines(kept)
                to_delete_cloud.append(manifest_path)
                upload_file_bg(manifest_path)
            except Exception as e:
                errors.append(f"manifest rewrite: {e}")

    # ── delete from cloud (fire-and-forget) ──────────────────────────────
    if to_delete_cloud:
        delete_files_bg(to_delete_cloud)

    return errors


def delete_by_index_cb(st: ProjectState, beat_filter: str, visible_gallery: List[Any], selected_index: Optional[int]):
    if not st:
        return st, _normalize_gallery_list(visible_gallery), "❌ No project.", None
    visible = _normalize_gallery_list(visible_gallery)
    if selected_index is None:
        return st, visible, "❌ Click an image in the gallery then press Delete.", None
    if not visible:
        return st, [], "❌ No images to delete in this view.", None
    if selected_index < 0 or selected_index >= len(visible):
        return st, visible, "❌ Invalid selection. Click the image again.", None

    # visible_gallery_state holds thumbnail paths — convert to original
    thumb_path = visible[selected_index]
    orig_path = _thumb_to_original(thumb_path)

    errors = _purge_images(st, [orig_path])
    msg = f"⚠️ Deleted (with errors: {'; '.join(errors)})" if errors else "✅ Deleted."
    new_gallery = filter_gallery_cb(st, beat_filter)
    return st, new_gallery, msg, None


def mass_delete_cb(st: ProjectState, beat_filter: str, visible_gallery: List[Any]):
    """Delete ALL images currently visible in the gallery (respects beat filter)."""
    if not st:
        return st, [], "❌ No project.", None
    visible = _normalize_gallery_list(visible_gallery)
    if not visible:
        return st, [], "❌ No images in current view.", None

    orig_paths = [_thumb_to_original(p) for p in visible]
    errors = _purge_images(st, orig_paths)
    n = len(orig_paths)
    msg = (f"⚠️ Deleted {n} images (errors: {'; '.join(errors)})"
           if errors else f"✅ Deleted {n} image(s).")
    new_gallery = filter_gallery_cb(st, beat_filter)
    return st, new_gallery, msg, None


def range_delete_cb(st: ProjectState, beat_filter: str, from_beat: Optional[int], to_beat: Optional[int]):
    """Delete all images whose beat index falls within [from_beat, to_beat] inclusive."""
    if not st:
        return st, [], "❌ No project.", None
    if from_beat is None or to_beat is None:
        return st, filter_gallery_cb(st, beat_filter), "❌ Enter both From and To beat numbers.", None
    lo = int(min(from_beat, to_beat))
    hi = int(max(from_beat, to_beat))

    # Collect all original paths whose beat index is in range.
    # images_by_beat keys are beat indices (int); image filenames start with zero-padded beat number.
    orig_paths: List[str] = []
    seen: set = set()
    images_by_beat = st.images_by_beat or {}
    for beat_idx, paths in images_by_beat.items():
        try:
            bidx = int(beat_idx)
        except (TypeError, ValueError):
            continue
        if lo <= bidx <= hi:
            for p in paths:
                if p not in seen:
                    seen.add(p)
                    orig_paths.append(p)

    # Also scan all_manifest_paths by filename prefix in case images_by_beat is stale.
    for p in (getattr(st, "all_manifest_paths", None) or []):
        if p in seen:
            continue
        stem = os.path.splitext(os.path.basename(p))[0]
        try:
            beat_part = int(stem.split("_")[0])
            if lo <= beat_part <= hi:
                orig_paths.append(p)
                seen.add(p)
        except (ValueError, IndexError):
            pass

    if not orig_paths:
        return st, filter_gallery_cb(st, beat_filter), f"⚠️ No images found for beats {lo}–{hi}.", None

    errors = _purge_images(st, orig_paths)
    n = len(orig_paths)
    msg = (f"⚠️ Deleted {n} images for beats {lo}–{hi} (errors: {'; '.join(errors)})"
           if errors else f"✅ Deleted {n} image(s) for beats {lo}–{hi}.")
    new_gallery = filter_gallery_cb(st, beat_filter)
    return st, new_gallery, msg, None


def _ai_rewrite_prompt(current_prompt: str, instruction: str,
                       character_context: str = "", beat_context: str = "") -> str:
    """Use Claude to rewrite an image prompt based on a natural language instruction.
    character_context: character DNA/appearance reference block.
    beat_context: the original beat/story text for this panel."""
    import anthropic
    client = anthropic.Anthropic()

    char_section = ""
    if character_context:
        char_section = f"\n\nCHARACTER APPEARANCE REFERENCE (use this to correctly describe characters):\n{character_context}"

    beat_section = ""
    if beat_context:
        beat_section = f"\n\nORIGINAL BEAT / STORY CONTEXT:\n{beat_context}"

    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=700,
        messages=[{
            "role": "user",
            "content": f"""You are an expert AI image prompt editor for a manhwa image generator.
{char_section}{beat_section}

CURRENT IMAGE PROMPT:
{current_prompt}

USER'S EDIT INSTRUCTION:
{instruction}

Rewrite the prompt to implement the user's instruction exactly, using the character appearance reference above to correctly describe any characters mentioned. Preserve all other visual elements, scene atmosphere, camera/shot type, style tokens, and manhwa render quality notes that the user did NOT mention changing. Return ONLY the rewritten prompt — no explanation, no preamble, no quotes."""
        }]
    )
    return msg.content[0].text.strip()


def _detect_beat_from_selection(st, vis, sel_idx) -> Optional[int]:
    """Detect beat index from a gallery selection. Returns None if not found.

    visible_gallery_state holds THUMBNAIL paths; manifest.jsonl stores ORIGINAL
    paths.  Always convert thumb → original before any path comparison.
    """
    visible = _normalize_gallery_list(vis)
    if sel_idx is None or not visible:
        return None
    try:
        sel = int(sel_idx)
    except Exception:
        return None
    if sel < 0 or sel >= len(visible):
        return None
    thumb_path = visible[sel]
    # Convert thumbnail path back to original so manifest lookup works.
    img_path = _thumb_to_original(thumb_path)

    # Fast path: in-memory manifest_beat_index avoids a disk read.
    if st:
        mbi = getattr(st, "manifest_beat_index", {}) or {}
        beat = mbi.get(img_path) or mbi.get(thumb_path)
        if beat:
            return beat

    # Slow path: scan manifest.jsonl on disk.
    if st and st.project_dir:
        manifest_path = os.path.join(st.project_dir, "manifest.jsonl")
        img_basename = os.path.splitext(os.path.basename(img_path))[0]
        try:
            if os.path.exists(manifest_path):
                with open(manifest_path, "r", encoding="utf-8") as mf:
                    for line in mf:
                        try:
                            entry = json.loads(line.strip())
                            entry_path = entry.get("image_path", "")
                            entry_base = os.path.splitext(os.path.basename(entry_path))[0]
                            if entry_path == img_path or entry_base == img_basename:
                                return int(entry.get("beat_index") or 0) or None
                        except Exception:
                            pass
        except Exception:
            pass
    # Last-resort: filename parsing — "001_02.png" → beat 1
    try:
        return int(os.path.basename(img_path).split("_")[0])
    except Exception:
        return None


def ai_edit_redo_cb(st, bf, vis, sel_idx, edit_instruction, *gen_args):
    """Rewrite prompt_debug via AI instruction, then regenerate the selected beat."""
    print(f"[ai_edit_redo_cb] fired — edit_instruction={repr((edit_instruction or '')[:80])!r}, sel_idx={sel_idx!r}, gen_args_len={len(gen_args)}", flush=True)
    # gen_args mirrors _gen_inputs: [state(0), beat_selector(1), loc(2), sub_loc(3),
    # scene_type(4), camera(5), chars(6), items(7), action(8), emotion(9), outfit(10),
    # include_beat(11), silhouette(12), signature(13), text_type(14), text_mode(15),
    # on_image_text(16), style(17), negative(18), beat_filter(19), prompt_debug(20),
    # comprehensive_prompt(21), use_openai_final(22), enable_safety(23), quality(24),
    # prompt_format(25), aspect_ratio(26), resolution(27)]
    _PROMPT_IDX = 20
    _USE_OPENAI_IDX = 22
    _PROMPT_FORMAT_IDX = 25

    # Yield a "starting" status immediately so we know the generator is alive
    yield (st, "⏳ Edit & Redo starting...", None, [], "", "", bf, _format_session_cost(), [], "", "", gr.update())

    # Wrap pre-yield setup in try/except — an exception here silently kills the generator
    try:
        g = filter_gallery_cb(st, bf) if st else []
    except Exception as _ge:
        print(f"[ai_edit_redo_cb] filter_gallery_cb error: {_ge}", flush=True)
        g = []
    current_prompt = gen_args[_PROMPT_IDX] if len(gen_args) > _PROMPT_IDX else ""
    prompt_format = gen_args[_PROMPT_FORMAT_IDX] if len(gen_args) > _PROMPT_FORMAT_IDX else ""
    is_panel = (prompt_format or DEFAULT_PROMPT_FORMAT) == "panel"
    print(f"[ai_edit_redo_cb] current_prompt_len={len(current_prompt)}, is_panel={is_panel}", flush=True)

    def _fail(msg):
        print(f"[ai_edit_redo_cb] _fail: {msg}", flush=True)
        tup = (st, msg, None, g, current_prompt, "", bf, _format_session_cost(), g, "", "", gr.update())
        return iter([tup])

    if not (edit_instruction or "").strip():
        yield from _fail("❌ Enter an edit instruction in the box below first.")
        return
    if not (current_prompt or "").strip():
        yield from _fail("❌ Click an image in the gallery first so its prompt loads into the box above.")
        return

    # Detect beat first so we can supply beat text as context to the rewriter
    try:
        beat_idx = _detect_beat_from_selection(st, vis, sel_idx)
    except Exception as _be:
        print(f"[ai_edit_redo_cb] _detect_beat exception: {_be}", flush=True)
        beat_idx = None
    print(f"[ai_edit_redo_cb] beat_idx={beat_idx}", flush=True)

    # Build character DNA reference so the rewriter knows what each character looks like
    character_context = ""
    try:
        if st and st.characters:
            lines = []
            for name, cdata in st.characters.items():
                dna = (cdata.get("dna_prompt") or "").strip()
                fields = cdata.get("fields") or {}
                hair = fields.get("hair", "")
                eyes = fields.get("eyes", "")
                outfit = fields.get("outfit", "")
                summary = ", ".join(p for p in [hair, eyes, outfit] if p)
                lines.append(f"- {name}: {dna or summary or '(no details)'}")
            character_context = "\n".join(lines)
    except Exception:
        pass

    # Pull original beat text for scene context (in panel mode use the whole page script)
    beat_context = ""
    try:
        if beat_idx and st and st.beats and 1 <= beat_idx <= len(st.beats):
            if is_panel:
                _ppg2 = _ppp(st)
                _page_idx = (beat_idx - 1) // _ppg2
                beat_context = (getattr(st, "page_scripts", None) or {}).get(_page_idx, "")
                if not beat_context:
                    _pb = (beat_idx - 1) // _ppg2 * _ppg2
                    beat_context = "\n".join(
                        st.beats[i] for i in range(_pb, min(_pb + _ppg2, len(st.beats)))
                    )
            else:
                beat_context = st.beats[beat_idx - 1]
    except Exception:
        pass

    print(f"[ai_edit_redo_cb] yielding rewrite status...", flush=True)
    yield (st, "✏️ Rewriting prompt with AI...", None, g, current_prompt, "",
           bf, _format_session_cost(), g, "", "", gr.update())

    print(f"[ai_edit_redo_cb] calling _ai_rewrite_prompt...", flush=True)
    try:
        new_prompt = _ai_rewrite_prompt(current_prompt, edit_instruction,
                                        character_context=character_context,
                                        beat_context=beat_context)
        print(f"[ai_edit_redo_cb] rewrite done, new_prompt_len={len(new_prompt)}", flush=True)
    except Exception as e:
        print(f"[ai_edit_redo_cb] rewrite exception: {e}", flush=True)
        yield from _fail(f"❌ AI rewrite failed: {e}")
        return

    if beat_idx is None:
        yield (st, "✅ Prompt rewritten. Click an image in the gallery, then click Edit & Redo to regenerate.",
               None, g, new_prompt, "", bf, _format_session_cost(), g, "", "", gr.update())
        return

    yield (st, "✅ Prompt rewritten — regenerating...", None, g, new_prompt, "",
           bf, _format_session_cost(), g, "", "", gr.update())

    beats_list = (st.beats or []) if st else []
    beat_label = (f"{beat_idx:03d} — {beats_list[beat_idx-1][:70]}"
                  if 1 <= beat_idx <= len(beats_list) else f"{beat_idx:03d}")

    # Panel mode: save the rewritten script into st.page_scripts so the panel override
    # in generate_cb uses the NEW script instead of silently restoring the old one.
    if is_panel and st:
        _page_idx = (beat_idx - 1) // _ppp(st)
        if not isinstance(getattr(st, "page_scripts", None), dict):
            st.page_scripts = {}
        st.page_scripts[_page_idx] = new_prompt

    new_gen_args = list(gen_args)
    new_gen_args[1] = beat_label            # beat_selector → detected beat
    new_gen_args[_PROMPT_IDX] = new_prompt  # inject rewritten prompt
    new_gen_args[_PROMPT_IDX + 1] = ""      # clear comprehensive_prompt mirror
    # Disable OpenAI rebuild so the rewritten prompt is used as-is
    new_gen_args[_USE_OPENAI_IDX] = False

    # Use size "0" → single-beat path → uses editable_prompt directly
    for _y in batch_generate_cb(beat_idx, "0", *new_gen_args):
        yield _y

    # Final gallery refresh — keep the user's existing filter, just make sure the new image shows
    _bf_cur = new_gen_args[19] if len(new_gen_args) > 19 else "All beats"
    _final_g = filter_gallery_cb(st, _bf_cur)
    print(f"[ai_edit_redo_cb] done — refreshing gallery with filter={_bf_cur!r}, {len(_final_g)} image(s)", flush=True)
    yield (st, f"✅ Beat {beat_idx:03d} regenerated",
           None, _final_g, new_prompt, "", _bf_cur, _format_session_cost(),
           _final_g, "", "", gr.update())


def mass_fix_cb(st, fix_from, fix_to, fix_instruction, *gen_args):
    """Rewrite stored prompts for a beat range via AI rule, then regenerate one image per beat."""
    _PROMPT_IDX   = 20
    _USE_OPENAI_IDX = 22
    bf = gen_args[19] if len(gen_args) > 19 else "All beats"
    g  = filter_gallery_cb(st, bf) if st else []

    def _status(msg):
        return (st, msg, None, g,
                gen_args[_PROMPT_IDX] if len(gen_args) > _PROMPT_IDX else "",
                "", bf, _format_session_cost(), g, "", "", gr.update())

    if not st or not st.beats:
        yield _status("❌ Load or build a project first.")
        return

    instruction = (fix_instruction or "").strip()
    if not instruction:
        yield _status("❌ Enter a fix instruction before applying.")
        return

    try:
        lo = max(1, int(fix_from or 1))
        hi = min(len(st.beats), int(fix_to or len(st.beats)))
    except (TypeError, ValueError):
        yield _status("❌ Enter valid beat numbers.")
        return

    if lo > hi:
        lo, hi = hi, lo

    beats_to_fix = list(range(lo, hi + 1))
    n = len(beats_to_fix)
    yield _status(f"🔄 Rewriting {n} prompts (beats {lo}–{hi}) in parallel…")

    # Build character DNA context once
    character_context = ""
    if st and st.characters:
        lines = []
        for name, cdata in st.characters.items():
            dna    = (cdata.get("dna_prompt") or "").strip()
            fields = cdata.get("fields") or {}
            summary = ", ".join(p for p in [fields.get("hair",""), fields.get("eyes",""), fields.get("outfit","")] if p)
            lines.append(f"- {name}: {dna or summary or '(no details)'}")
        character_context = "\n".join(lines)

    # Rewrite all prompts in parallel with Claude
    import concurrent.futures as _cf_mf
    rewritten: dict = {}

    def _do_rewrite(beat_idx: int):
        beat_ctx  = st.beats[beat_idx - 1] if 1 <= beat_idx <= len(st.beats) else ""
        # Use the stored Claude prompt — it was generated in the project's prompt
        # format (e.g. wilson) at build time, so it matches the project's style.
        stored    = (st.image_prompts or {}).get(beat_idx) or {}
        current_p = stored.get("prompt", "").strip() or beat_ctx
        try:
            new_p = _ai_rewrite_prompt(current_p, instruction,
                                       character_context=character_context,
                                       beat_context=beat_ctx)
            return beat_idx, new_p
        except Exception:
            return beat_idx, current_p   # keep original on failure

    with _cf_mf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(_do_rewrite, b): b for b in beats_to_fix}
        done = 0
        for fut in _cf_mf.as_completed(futs):
            b_idx, new_p = fut.result()
            rewritten[b_idx] = new_p
            done += 1

    # Persist rewritten prompts back into state so single-edit still sees them
    for b_idx, new_p in rewritten.items():
        bucket = (st.image_prompts or {}).get(b_idx) or {}
        bucket["prompt"] = new_p
        if not isinstance(st.image_prompts, dict):
            st.image_prompts = {}
        st.image_prompts[b_idx] = bucket

    yield _status(f"✅ Rewrote {len(rewritten)} prompts — regenerating images (1 per beat)…")

    beats_list = st.beats or []
    for i, b in enumerate(beats_to_fix):
        new_p = rewritten.get(b, "")
        if not new_p:
            continue
        beat_label = (f"{b:03d} — {beats_list[b-1][:70]}"
                      if 1 <= b <= len(beats_list) else f"{b:03d}")

        yield _status(f"🎨 Regenerating beat {b} ({i+1}/{n})…")

        new_gen_args = list(gen_args)
        new_gen_args[1]               = beat_label   # beat_selector
        new_gen_args[_PROMPT_IDX]     = new_p        # editable_prompt (prompt_debug)
        new_gen_args[_PROMPT_IDX + 1] = ""           # clear comprehensive_prompt mirror
        # Deliberately do NOT override use_openai_final — inherit the user's UI setting
        # so the same final-prompt pipeline (e.g. OpenAI → wilson format) runs as normal

        yield from batch_generate_cb(b, "0", *new_gen_args)

    yield _status(f"✅ Mass fix done — regenerated {n} image(s) across beats {lo}–{hi}.")


def redo_selected_cb(st, bf, vis, sel_idx, *gen_args):
    """Regenerate the selected gallery image using the current prompt_debug — detects beat from manifest."""
    g = filter_gallery_cb(st, bf) if st else []
    def _fail(msg):
        tup = (st, msg, None, g, "", "", bf, _format_session_cost(), g, "", "", gr.update())
        return iter([tup])

    beat_idx = _detect_beat_from_selection(st, vis, sel_idx)
    if beat_idx is None:
        yield from _fail("❌ Click an image in the gallery first, then click Redo.")
        return

    beats_list = (st.beats or []) if st else []
    beat_label = (f"{beat_idx:03d} — {beats_list[beat_idx - 1][:70]}"
                  if 1 <= beat_idx <= len(beats_list) else f"{beat_idx:03d}")

    # Override beat_selector (index 1); keep prompt_debug (index 20) as-is so edits are respected
    new_gen_args = (gen_args[0], beat_label) + gen_args[2:]
    _cur_prompt = new_gen_args[20] if len(new_gen_args) > 20 else ""
    _bf_cur = new_gen_args[19] if len(new_gen_args) > 19 else "All beats"
    # Use "0" → single-beat path → passes editable_prompt (prompt_debug) directly to generate_cb
    for _y in batch_generate_cb(beat_idx, "0", *new_gen_args):
        yield _y

    # Final gallery refresh — keep the user's existing filter
    _final_g = filter_gallery_cb(st, _bf_cur)
    print(f"[redo_selected_cb] done — refreshing gallery with filter={_bf_cur!r}, {len(_final_g)} image(s)", flush=True)
    yield (st, f"✅ Beat {beat_idx:03d} regenerated",
           None, _final_g, _cur_prompt, "", _bf_cur, _format_session_cost(),
           _final_g, "", "", gr.update())


import threading as _zip_threading

# ── Module-level ZIP build state (so gr.Timer can poll it) ───────────────────
_ZIP_STATE: dict = {
    "running": False,
    "done":    False,
    "html":    "",
    "status":  "",
    "count":   0,
    "total":   0,
    "gen_id":  0,
}


def zip_cb(st: ProjectState, zip_base_name: str = "untitled", zip_part: str = "All"):
    """Non-generator: starts zip in a background thread and returns immediately."""
    import time as _time
    if not st or not st.project_dir:
        return gr.update(), gr.update(value="❌ Nothing to zip.", visible=True), gr.update()

    base_name = clean_for_prompt(zip_base_name or "untitled").strip() or "untitled"
    safe_base = re.sub(r"[^a-zA-Z0-9_\-]+", "_", base_name).strip("_") or "untitled"
    stamp = datetime.now(ZoneInfo("America/Indiana/Indianapolis")).strftime("%Y%m%d")
    zip_filename = f"{safe_base}_{stamp}.zip"
    zip_path = os.path.join("/tmp", zip_filename)
    total_images = sum(len(_normalize_gallery_list(v)) for v in (st.images_by_beat or {}).values())

    gen_id = _ZIP_STATE["gen_id"] + 1
    _ZIP_STATE.update({"running": True, "done": False, "html": "",
                        "status": "🔄 Starting ZIP…", "count": 0,
                        "total": total_images, "gen_id": gen_id})

    result: dict = {"path": None, "status": None, "done": False, "count": 0}

    def _build():
        try:
            import zipfile as _zf
            import concurrent.futures as _zip_cf
            import glob as _glob
            import time as _t
            # Clean up old ZIPs in /tmp to free disk space before building a new one
            for _old_zip in _glob.glob("/tmp/*.zip"):
                try:
                    if _old_zip != zip_path and (_t.time() - os.path.getmtime(_old_zip)) > 60:
                        os.remove(_old_zip)
                except Exception:
                    pass
            if os.path.exists(zip_path):
                os.remove(zip_path)

            # ── Resolve image paths ────────────────────────────────────────────
            # Merge ALL three sources so nothing is missed:
            #  1. all_manifest_paths (set at project load, updated on generate)
            #  2. images_by_beat     (in-memory, always current for this session)
            #  3. manifest.jsonl on disk (fresh read catches any gap)
            seen_paths: set = set()
            all_img_paths: list = []
            manifest_by_path: dict = {}

            # Generation writes image state and manifest rows under this same
            # lock. Snapshot them together so a download can never include a
            # just-created image without its exact prompt record.
            with _BATCH_SAVE_LOCK:
                _snap_all_manifest_paths = list(st.all_manifest_paths or [])
                _snap_images_by_beat = {
                    b: list(_normalize_gallery_list(v))
                    for b, v in (st.images_by_beat or {}).items()
                }
                _snap_beat_index = dict(st.manifest_beat_index or {})
                _snap_manifest_rows = []
                _mf_path = os.path.join(st.project_dir, "manifest.jsonl")
                if os.path.isfile(_mf_path):
                    try:
                        with open(_mf_path, "r", encoding="utf-8") as _mf:
                            for _line in _mf:
                                _line = _line.strip()
                                if not _line:
                                    continue
                                try:
                                    _snap_manifest_rows.append(json.loads(_line))
                                except Exception:
                                    pass
                    except Exception:
                        pass

            def _resolve_to_source(p: str) -> str:
                """If p is a thumbnail path, resolve it back to the original full image."""
                if not p:
                    return p
                norm = p.replace("\\", "/")
                # Thumbnail paths live under …/thumbnails/…  and are always JPEG
                if "/thumbnails/" in norm:
                    # Swap the thumbnails dir segment for images
                    candidate_base = norm.replace("/thumbnails/", "/images/")
                    # Strip the thumbnail extension and try known image extensions
                    candidate_base = os.path.splitext(candidate_base)[0]
                    for _ext in (FAL_OUTPUT_EXT, "png", "jpg", "jpeg", "webp"):
                        candidate = f"{candidate_base}.{_ext}"
                        if os.path.isfile(candidate):
                            return candidate
                return p

            def _add(p: str) -> None:
                p = _resolve_to_source(p)
                if p and p not in seen_paths:
                    seen_paths.add(p)
                    all_img_paths.append(p)

            for p in _snap_all_manifest_paths:
                _add(p)

            for b in sorted(_snap_images_by_beat.keys()):
                for p in _snap_images_by_beat.get(b, []):
                    _add(p)

            # Fresh manifest snapshot catches images generated before the
            # in-memory path list was populated. Keep all indexing local to
            # this immutable export snapshot.
            for _e in _snap_manifest_rows:
                _p = _e.get("image_path", "")
                if _p:
                    _add(_p)
                    _resolved = _resolve_to_source(_p)
                    manifest_by_path[os.path.abspath(_resolved)] = _e
                    _snap_beat_index.setdefault(
                        _resolved, int(_e.get("beat_index", 0) or 0)
                    )

            # ── Download missing images from GCS (parallel, blocking) ──────────
            try:
                import cloud_storage as _cs_zip
                if _cs_zip.is_available() and all_img_paths:
                    missing = [p for p in all_img_paths if not os.path.isfile(p)]
                    if missing:
                        result["status"] = f"☁️ Fetching {len(missing)} images from cloud…"
                        bucket = _cs_zip._get_bucket()
                        def _dl_one(p: str) -> None:
                            try:
                                key  = p.replace("\\", "/").lstrip("./")
                                blob = bucket.blob(key)
                                if blob.exists():
                                    os.makedirs(os.path.dirname(p), exist_ok=True)
                                    blob.download_to_filename(p)
                            except Exception:
                                pass
                        with _zip_cf.ThreadPoolExecutor(max_workers=8) as _dl_ex:
                            list(_dl_ex.map(_dl_one, missing))
            except Exception:
                pass

            # ── Build ordered image_entries from manifest beat index ───────────
            # Filter to only files that exist on disk so deleted images don't
            # consume a numbering slot (which would leave gaps like 001_02.png
            # with no 001_01.png when 001_01 was deleted).
            beat_index = _snap_beat_index
            beat_to_imgs: dict = {}
            for p in all_img_paths:
                if not os.path.isfile(p):
                    continue  # skip deleted / missing images entirely
                b = beat_index.get(p, 0)
                beat_to_imgs.setdefault(b, []).append(p)
            image_entries = []
            for b in sorted(beat_to_imgs.keys()):
                for j, p in enumerate(beat_to_imgs[b], start=1):
                    image_entries.append((b, j, p))

            IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

            def _compress(fn: str) -> int:
                return _zf.ZIP_STORED if os.path.splitext(fn)[1].lower() in IMAGE_EXTS else _zf.ZIP_DEFLATED

            def _write_beat_maps(zfile, folder: str, rows: list) -> None:
                zfile.writestr(f"{folder}/beat_image_map.json", json.dumps(rows, indent=2, ensure_ascii=False))
                csv_lines = ["beat,image,sentence"] + [
                    f'{r["beat"]},"{r["image"]}","{str(r["sentence"]).replace(chr(34), chr(39))}"'
                    for r in rows
                ]
                txt_lines = [f'Beat {int(r["beat"]):03d} | {r["image"]} | {r["sentence"]}' for r in rows]
                zfile.writestr(f"{folder}/beat_image_map.csv", "\n".join(csv_lines))
                zfile.writestr(f"{folder}/beat_image_map.txt", "\n".join(txt_lines))

            def _manifest_for_path(path: str) -> dict:
                return manifest_by_path.get(os.path.abspath(_resolve_to_source(path)), {})

            def _numbered_prompts_md(entries: list, exported_names: list = None) -> str:
                lines = [
                    "# Numbered Prompts",
                    "",
                    "Prompts are numbered in the same order as the generated images in this export.",
                    "Each Prompt number corresponds to one paid image generation.",
                    "",
                ]
                for prompt_num, (b, j, p) in enumerate(entries, start=1):
                    entry = _manifest_for_path(p)
                    exported = (
                        exported_names[prompt_num - 1]
                        if exported_names and prompt_num <= len(exported_names)
                        else os.path.basename(p)
                    )
                    lines.extend([
                        f"## Prompt {prompt_num}",
                        "",
                        f"- Generated file: `{exported}`",
                        f"- Beat: {b}",
                        f"- Version: {j}",
                        "",
                        str(entry.get("prompt") or
                            "[Prompt unavailable: legacy image has no manifest prompt record.]"),
                        "",
                        "---",
                        "",
                    ])
                return "\n".join(lines)

            story_parts = getattr(st, "story_parts", []) or []
            total_beats = len(st.beats) if st.beats else 0

            def _beat_range_for_part(part):
                lo = int(part.get("start_beat") or 1)
                hi = part.get("end_beat")
                return lo, int(hi) if hi is not None else total_beats

            with _zf.ZipFile(zip_path, "w", allowZip64=True) as zfile:
                # ── Project metadata at root ───────────────────────────────────
                for root, _dirs, files in os.walk(st.project_dir):
                    rel_root = os.path.relpath(root, st.project_dir)
                    if rel_root.split(os.sep)[0] in ("images", "thumbnails"):
                        continue
                    for fn in files:
                        if fn.upper() in {"NUMBERED_PROMPTS.MD", "NUMBERED_PROMPTS.TXT"}:
                            continue
                        full = os.path.join(root, fn)
                        rel = os.path.relpath(full, st.project_dir)
                        zfile.write(full, rel, compress_type=_compress(fn))

                # ── Combined folder — all images ───────────────────────────────
                rows_combined = []
                combined_names = []
                for idx, (b, j, p) in enumerate(image_entries):
                    ext = os.path.splitext(p)[1].lstrip(".") or FAL_OUTPUT_EXT
                    ordered = f"{b:03d}_{j:02d}.{ext}"
                    if os.path.exists(p):
                        zfile.write(p, os.path.join("combined", "images", ordered), compress_type=_zf.ZIP_STORED)
                    beat_text = st.beats[b - 1] if 1 <= b <= len(st.beats) else ""
                    rows_combined.append({"beat": b, "image": ordered, "sentence": beat_text})
                    combined_names.append(f"combined/images/{ordered}")
                    result["count"] = idx + 1
                    _ZIP_STATE["count"] = idx + 1
                _write_beat_maps(zfile, "combined", rows_combined)
                numbered_prompts = _numbered_prompts_md(image_entries, combined_names)
                zfile.writestr("NUMBERED_PROMPTS.md", numbered_prompts)
                zfile.writestr("NUMBERED_PROMPTS.txt", numbered_prompts)
                zfile.writestr("combined/NUMBERED_PROMPTS.md", numbered_prompts)
                zfile.writestr("combined/NUMBERED_PROMPTS.txt", numbered_prompts)

                # ── Per-part folders (only when multiple parts exist) ──────────
                if len(story_parts) > 1:
                    for part in story_parts:
                        part_label = part.get("name", "Part")
                        safe_label = re.sub(r"[^a-zA-Z0-9_\-\s]+", "", part_label).strip() or "Part"
                        lo, hi = _beat_range_for_part(part)
                        part_entries = [(b, j, p) for b, j, p in image_entries if lo <= b <= hi]
                        rows_part = []
                        part_names = []
                        for b, j, p in part_entries:
                            ext = os.path.splitext(p)[1].lstrip(".") or FAL_OUTPUT_EXT
                            ordered = f"{b:03d}_{j:02d}.{ext}"
                            if os.path.exists(p):
                                zfile.write(p, os.path.join(safe_label, "images", ordered), compress_type=_zf.ZIP_STORED)
                            beat_text = st.beats[b - 1] if 1 <= b <= len(st.beats) else ""
                            rows_part.append({"beat": b, "image": ordered, "sentence": beat_text})
                            part_names.append(f"{safe_label}/images/{ordered}")
                        _write_beat_maps(zfile, safe_label, rows_part)
                        _part_prompts = _numbered_prompts_md(part_entries, part_names)
                        zfile.writestr(f"{safe_label}/NUMBERED_PROMPTS.md", _part_prompts)
                        zfile.writestr(f"{safe_label}/NUMBERED_PROMPTS.txt", _part_prompts)

                # ── Shorts mode: add pages/ + cropped panels/ ─────────────────
                if getattr(st, 'build_mode', 'Panel') == "Shorts" and image_entries:
                    try:
                        from PIL import Image as _PIL_Image
                        import numpy as _np
                        import io as _sio

                        def _find_gutter_cuts(arr1d_mean, arr1d_std, length, edge_skip=10, min_gap_frac=0.05):
                            """Return midpoints of gutter runs along one axis.
                            arr1d_mean / arr1d_std are per-row (or per-col) averages."""
                            is_gutter = (arr1d_std < 12) | (arr1d_mean > 220)
                            cuts = []
                            in_run = False
                            run_start = 0
                            for i in range(length):
                                if is_gutter[i] and not in_run:
                                    in_run = True
                                    run_start = i
                                elif not is_gutter[i] and in_run:
                                    in_run = False
                                    mid = (run_start + i) // 2
                                    if edge_skip < mid < length - edge_skip:
                                        cuts.append(mid)
                            # Merge cuts that are too close
                            min_gap = max(int(length * min_gap_frac), 20)
                            merged: list = []
                            for c in sorted(cuts):
                                if not merged or c - merged[-1] >= min_gap:
                                    merged.append(c)
                            return merged

                        def _crop_panels_from_page(img_path: str, n_panels: int = SHORTS_PANELS_PER_PAGE):
                            """Crop a panel page into individual panels.
                            1. Detects horizontal gutter rows → splits into row strips.
                            2. Within each row strip, detects vertical gutter columns →
                               splits into individual panels.
                            Gutters are detected as near-uniform (low std) or near-white
                            (high mean) lines. Falls back gracefully when nothing is found."""
                            # Preferred: border-aware cutter — snaps the expected grid to
                            # the real drawn panel borders and trims frames/gutters.
                            try:
                                from panel_cut import smart_cut_panels_path as _scp
                                _rows = max(1, (n_panels + 1) // 2)
                                _smart = _scp(img_path, n_rows=_rows, n_cols=2)
                                if len(_smart) >= 2:
                                    return _smart
                            except Exception:
                                pass
                            img = _PIL_Image.open(img_path).convert("RGB")
                            w, h = img.size
                            arr = _np.array(img, dtype=_np.float32)  # (H, W, 3)

                            # ── Step 1: horizontal cuts (row separators) ──────────────
                            row_mean = arr.mean(axis=(1, 2))   # shape (H,)
                            row_std  = arr.std(axis=(1, 2))    # shape (H,)
                            h_cuts = _find_gutter_cuts(row_mean, row_std, h, edge_skip=10, min_gap_frac=0.05)

                            min_panel_h = max(int(h * 0.05), 40)
                            row_bounds = [0] + h_cuts + [h]
                            row_slices = []
                            for i in range(len(row_bounds) - 1):
                                y0, y1 = row_bounds[i], row_bounds[i + 1]
                                if y1 - y0 >= min_panel_h:
                                    row_slices.append((y0, y1))

                            if not row_slices:
                                row_slices = [(0, h)]

                            # ── Step 2: vertical cuts within each row ─────────────────
                            panels = []
                            min_panel_w = max(int(w * 0.10), 40)
                            for y0, y1 in row_slices:
                                row_arr = arr[y0:y1, :, :]      # (row_h, W, 3)
                                col_mean = row_arr.mean(axis=(0, 2))  # shape (W,)
                                col_std  = row_arr.std(axis=(0, 2))   # shape (W,)
                                v_cuts = _find_gutter_cuts(col_mean, col_std, w, edge_skip=10, min_gap_frac=0.08)
                                col_bounds = [0] + v_cuts + [w]
                                for j in range(len(col_bounds) - 1):
                                    x0, x1 = col_bounds[j], col_bounds[j + 1]
                                    if x1 - x0 >= min_panel_w:
                                        # Inset ~1.5% per side so white gutter
                                        # slivers never survive on panel edges.
                                        _ix = max(4, int((x1 - x0) * 0.015))
                                        _iy = max(4, int((y1 - y0) * 0.015))
                                        panels.append(img.crop((x0 + _ix, y0 + _iy, x1 - _ix, y1 - _iy)))

                            if not panels:
                                panels = [img]
                            return panels

                        # Group by page (one composite image per _ppp(st) beats)
                        _zip_ppg = _ppp(st)
                        page_seen: set = set()
                        page_entries_ordered = []
                        for b, j, p in image_entries:
                            pg = (b - 1) // _zip_ppg
                            if pg not in page_seen:
                                page_seen.add(pg)
                                page_entries_ordered.append((pg, b, p))
                        shorts_entries = [(b, j, p) for _pg, b, p in page_entries_ordered
                                          for j in [next((ej for eb, ej, ep in image_entries if ep == p), 1)]]
                        shorts_names = [
                            f"pages/page_{pg + 1:03d}.{os.path.splitext(p)[1].lstrip('.') or FAL_OUTPUT_EXT}"
                            for pg, _b, p in page_entries_ordered
                        ]
                        _pages_prompts = _numbered_prompts_md(shorts_entries, shorts_names)
                        zfile.writestr("pages/NUMBERED_PROMPTS.md", _pages_prompts)
                        zfile.writestr("pages/NUMBERED_PROMPTS.txt", _pages_prompts)

                        global_panel_num = 0
                        for pg_idx, b, p in sorted(page_entries_ordered):
                            if not os.path.isfile(p):
                                continue
                            ext = os.path.splitext(p)[1].lstrip(".") or FAL_OUTPUT_EXT
                            # Add full composite page
                            zfile.write(p, f"pages/page_{pg_idx + 1:03d}.{ext}", compress_type=_zf.ZIP_STORED)
                            # Crop and add individual panels
                            try:
                                panels = _crop_panels_from_page(p)
                                for panel_img in panels:
                                    global_panel_num += 1
                                    buf = _sio.BytesIO()
                                    panel_img.save(buf, format="JPEG", quality=95)
                                    buf.seek(0)
                                    zfile.writestr(
                                        f"panels/panel_{global_panel_num:03d}.jpg",
                                        buf.read(),
                                        compress_type=_zf.ZIP_STORED,
                                    )
                            except Exception:
                                pass  # page already saved above; cropping failure is non-fatal
                        # ── Cover image — last page ───────────────────────────
                        cover_path = getattr(st, "cover_image_path", "")
                        if cover_path and os.path.isfile(cover_path):
                            cover_ext = os.path.splitext(cover_path)[1].lstrip(".") or "jpg"
                            zfile.write(
                                cover_path,
                                f"pages/page_cover.{cover_ext}",
                                compress_type=_zf.ZIP_STORED,
                            )
                            zfile.write(
                                cover_path,
                                f"cover.{cover_ext}",
                                compress_type=_zf.ZIP_STORED,
                            )
                    except Exception:
                        pass  # numpy/PIL failure is non-fatal

            result["path"] = zip_path
            result["status"] = f"✅ Zipped {zip_filename} — {len(image_entries)} images · {len(image_entries)} numbered prompts included"
        except Exception as exc:
            try:
                # The fallback archive must still contain a human-readable,
                # numbered prompt list. Prefer the already-built exact export
                # ordering; if failure happened earlier, use manifest order.
                if "image_entries" in locals() and "_numbered_prompts_md" in locals():
                    _fallback_prompts = _numbered_prompts_md(image_entries)
                else:
                    _fallback_lines = [
                        "# Numbered Prompts",
                        "",
                        "Prompts are numbered in manifest generation order.",
                        "",
                    ]
                    with _BATCH_SAVE_LOCK:
                        _fallback_rows = []
                        _fallback_mf = os.path.join(st.project_dir, "manifest.jsonl")
                        if os.path.isfile(_fallback_mf):
                            with open(_fallback_mf, "r", encoding="utf-8") as _fh:
                                for _line in _fh:
                                    try:
                                        _fallback_rows.append(json.loads(_line))
                                    except Exception:
                                        pass
                    for _num, _entry in enumerate(_fallback_rows, start=1):
                        _fallback_lines.extend([
                            f"## Prompt {_num}",
                            "",
                            f"- Generated file: `{os.path.basename(str(_entry.get('image_path') or ''))}`",
                            f"- Beat: {int(_entry.get('beat_index', 0) or 0)}",
                            "",
                            str(_entry.get("prompt") or
                                "[Prompt unavailable: legacy image has no manifest prompt record.]"),
                            "",
                            "---",
                            "",
                        ])
                    _fallback_prompts = "\n".join(_fallback_lines)
                # Rebuild the fallback directly so a pre-existing project file
                # named NUMBERED_PROMPTS.md cannot create a duplicate ZIP member.
                with _zf.ZipFile(zip_path, "w", allowZip64=True) as _fallback_zip:
                    for _root, _dirs, _files in os.walk(st.project_dir):
                        for _fn in _files:
                            if _fn.upper() in {"NUMBERED_PROMPTS.MD", "NUMBERED_PROMPTS.TXT"}:
                                continue
                            _full = os.path.join(_root, _fn)
                            _rel = os.path.relpath(_full, st.project_dir)
                            _fallback_zip.write(
                                _full, _rel,
                                compress_type=(
                                    _zf.ZIP_STORED
                                    if os.path.splitext(_fn)[1].lower() in
                                    {".png", ".jpg", ".jpeg", ".webp"}
                                    else _zf.ZIP_DEFLATED
                                ),
                            )
                    _fallback_zip.writestr("NUMBERED_PROMPTS.md", _fallback_prompts)
                    _fallback_zip.writestr("NUMBERED_PROMPTS.txt", _fallback_prompts)
                result["path"] = zip_path
                result["status"] = f"⚠️ Zipped (fallback order): {exc}"
            except Exception as exc2:
                result["path"] = ""
                result["status"] = f"❌ ZIP failed: {exc2}"
        finally:
            # Upload to cloud storage so any server instance can serve the download
            if result.get("path") and os.path.isfile(result["path"]):
                try:
                    import cloud_storage as _cs
                    if _cs.is_available():
                        result["status"] = (result.get("status") or "") + " · ☁️ uploading…"
                        ok = _cs.upload_zip_blocking(result["path"], zip_filename)
                        if ok:
                            result["status"] = result["status"].replace("· ☁️ uploading…", "· ☁️ backed up")
                except Exception:
                    pass
            # Write progress + final result into _ZIP_STATE so timer can read it
            if result.get("path") and os.path.isfile(result["path"]):
                size_mb = os.path.getsize(result["path"]) / (1024 * 1024)
                dl_html = (
                    f'<div style="padding:10px 14px;background:#1a1a2e;border:1px solid #27273e;'
                    f'border-left:4px solid #f97316;border-radius:8px;margin-top:4px;">'
                    f'<a href="/zip-download/{zip_filename}" download="{zip_filename}" '
                    f'style="color:#f97316;font-weight:700;font-size:14px;text-decoration:none;">'
                    f'📥 Download {zip_filename}</a>'
                    f'<span style="color:#6868a0;font-size:12px;margin-left:10px;">{size_mb:.1f} MB</span>'
                    f'</div>'
                )
                _ZIP_STATE["html"] = dl_html
            else:
                _ZIP_STATE["html"] = ""
            _ZIP_STATE["status"] = result.get("status", "❌ Unknown error")
            result["done"] = True

    def _wrapped_build():
        try:
            _build()
        finally:
            if _ZIP_STATE.get("gen_id") == gen_id:
                _ZIP_STATE.update({"running": False, "done": True})

    _zip_threading.Thread(target=_wrapped_build, daemon=True).start()
    # Return immediately — timer (zip_timer) will poll _ZIP_STATE every 2 s
    return gr.update(), gr.update(value="🚀 ZIP started — updates every 2 s…", visible=True), gr.update(active=True)


def _poll_zip():
    """Called by gr.Timer every 2 s; reads _ZIP_STATE and updates zip UI."""
    if not _ZIP_STATE.get("running") and not _ZIP_STATE.get("done"):
        return gr.update(), gr.update(), gr.update()   # idle — no-op

    is_done = _ZIP_STATE.get("done") and not _ZIP_STATE.get("running")
    timer_upd = gr.update(active=False) if is_done else gr.update()

    if is_done:
        html = _ZIP_STATE.get("html", "")
        status = _ZIP_STATE.get("status", "")
        return gr.update(value=html), gr.update(value=status, visible=True), timer_upd

    # Still running — show progress
    packed = _ZIP_STATE.get("count", 0)
    total  = _ZIP_STATE.get("total", 0)
    status_txt = _ZIP_STATE.get("status", "")
    msg = f"🔄 Building ZIP…  {packed} / {total} images packed" + (f" — {status_txt}" if status_txt else "")
    return gr.update(), gr.update(value=msg, visible=True), timer_upd


def _bg_worker(gen_inputs_tuple: tuple) -> None:
    """Run batch_generate_cb in a daemon thread, independent of any browser connection.
    Stores each yielded result in _BG_JOB so the UI timer can poll it."""
    global _BG_JOB
    try:
        for yielded in batch_generate_cb(*gen_inputs_tuple):
            with _BG_LOCK:
                if _BG_JOB.get("cancel"):
                    _BG_JOB["status"] = "⛔ Cancelled"
                    _BG_JOB["running"] = False
                    return
                # yielded: (state, gen_status, latest, gallery, prompt_debug,
                #            comprehensive_prompt, beat_filter, cost_display,
                #            visible_gallery_state, beat_text_display,
                #            latest_image_name_display, batch_start_num)
                _BG_JOB["latest_state"] = yielded[0]
                _BG_JOB["status"] = str(yielded[1] or "")
                _BG_JOB["latest_img"] = yielded[2]
                if len(yielded) > 4 and yielded[4]:
                    _BG_JOB["prompt_debug"] = str(yielded[4])
                if len(yielded) > 9 and yielded[9]:
                    _BG_JOB["beat_text"] = str(yielded[9])
                if len(yielded) > 10 and yielded[10]:
                    _BG_JOB["image_name"] = str(yielded[10])
                if len(yielded) > 11 and isinstance(yielded[11], (int, float)):
                    _BG_JOB["next_start"] = int(yielded[11])
    except Exception as exc:
        with _BG_LOCK:
            _BG_JOB["status"] = f"❌ Error: {exc}"
    finally:
        with _BG_LOCK:
            if _BG_JOB.get("running"):
                _BG_JOB["status"] = (_BG_JOB.get("status", "") + " ✅ Done").strip()
            _BG_JOB["running"] = False
            _notif_status = _BG_JOB.get("status", "")
        if "⛔" not in _notif_status and "❌" not in _notif_status:
            _send_ntfy_notification(_notif_status or "Generation complete!")


def _submit_bg_cb(*args):
    """Start background generation and return immediately.
    Safe to use even if wifi drops mid-generation — the thread keeps running."""
    global _BG_JOB
    with _BG_LOCK:
        if _BG_JOB.get("running"):
            return "⚠️ Already generating in background — press ⏹ Stop to cancel first."
        _BG_JOB["running"] = True
        _BG_JOB["cancel"] = False
        _BG_JOB["status"] = "🔄 Starting background generation…"
        _BG_JOB["next_start_consumed"] = False  # arm the one-shot beat-advance delivery
        _BG_JOB["latest_state"] = None  # clear stale state from previous job so session st is used until first yield
        _BG_JOB["gallery_delivered"] = False  # arm one-shot final gallery push after job ends
    _BATCH_STOP_EVENT.clear()
    _BATCH_PAUSE_EVENT.clear()
    t = threading.Thread(target=_bg_worker, args=(args,), daemon=True)
    t.start()
    return "🔄 Generating in background — safe to lose wifi. Gallery refreshes every 4s."


def _stop_bg_cb():
    """Cancel the running background job."""
    _BATCH_STOP_EVENT.set()
    with _BG_LOCK:
        _BG_JOB["cancel"] = True
        _BG_JOB["status"] = "⏹ Stopping…"
    return "⏹ Stopping…"


def _format_gaps_html(st) -> str:
    """Render accumulated generation gaps as a compact HTML summary for Tab 2."""
    gaps: List[Dict] = list(getattr(st, "generation_gaps", None) or [])
    if not gaps:
        return "<p style='color:#4a9e6b;font-size:12px;margin:4px 0'>✅ No reference gaps detected yet.</p>"

    char_gaps    = [g for g in gaps if g.get("type") == "character"]
    setting_gaps = [g for g in gaps if g.get("type") == "setting"]

    rows = []
    for g in char_gaps:
        name = g.get("needed") or g.get("description") or "?"
        page = g.get("page")
        pg_s = f" (p.{page})" if page else ""
        rows.append(
            f"<tr><td style='padding:2px 8px 2px 0;color:#f87171;white-space:nowrap'>👤 character</td>"
            f"<td style='padding:2px 0;color:#fca5a5'>{name}{pg_s}</td></tr>"
        )
    for g in setting_gaps:
        name = g.get("needed") or g.get("description") or "?"
        page = g.get("page")
        pg_s = f" (p.{page})" if page else ""
        rows.append(
            f"<tr><td style='padding:2px 8px 2px 0;color:#fbbf24;white-space:nowrap'>🏞 setting</td>"
            f"<td style='padding:2px 0;color:#fde68a'>{name}{pg_s}</td></tr>"
        )

    table = (
        "<div style='background:#1a1a2e;border:1px solid #6b3a3a;border-radius:6px;"
        "padding:8px 12px;margin:4px 0'>"
        f"<div style='font-size:12px;color:#f87171;margin-bottom:4px'>"
        f"⚠️ <b>{len(gaps)} reference gap(s)</b> — add these to your Library to fix</div>"
        "<table style='font-size:11px;width:100%;border-collapse:collapse'>"
        + "".join(rows) +
        "</table></div>"
    )
    return table


def _poll_bg_cb(st, bf):
    """Called by gr.Timer — returns latest job state so the UI stays live."""
    with _BG_LOCK:
        running = _BG_JOB.get("running", False)
        # Use the BG thread's state whenever it's set AND it belongs to the same project.
        # This handles the common race where the 148s FAL call completes between two
        # 4-second poll ticks: latest_state is set for only ~5ms while running=True,
        # so a poll tick almost never fires in that window. Using latest_state even
        # after running=False ensures the gallery picks up the new image.
        # latest_state is cleared to None at the START of each new job (in _submit_bg_cb)
        # so project-switching works: if the user resumes a different project, the
        # project_id mismatch causes fallback to the fresh session state st.
        _latest = _BG_JOB.get("latest_state")
        if _latest and st and getattr(_latest, "project_id", None) != getattr(st, "project_id", None):
            _latest = None  # user switched project — ignore stale job state
        live_state = _latest or st
        status = _BG_JOB.get("status", "")
        latest_img = _BG_JOB.get("latest_img")
        next_start = _BG_JOB.get("next_start")
        prompt_debug = _BG_JOB.get("prompt_debug", "")
        beat_text = _BG_JOB.get("beat_text", "")
        image_name = _BG_JOB.get("image_name", "")
        next_start_consumed = _BG_JOB.get("next_start_consumed", True)
        gallery_delivered   = _BG_JOB.get("gallery_delivered", True)

    # ── Fast no-op path for idle ticks ───────────────────────────────────────
    # The poll timer fires every 4 s even when nothing is generating.  Returning
    # the full ProjectState on every idle tick causes Gradio to serialize potentially
    # megabytes of JSON (beats, beat_plans, image_prompts, page_scripts …) and send
    # it through the SSE stream — this is what froze the browser's JS thread and
    # produced the "Page Unresponsive" dialog.
    #
    # When the job is idle AND all one-shot deliveries have already been made:
    #   - Run the cheap cloud-restore sync (mutates st in-place by reference).
    #   - If no new images appeared, return a complete no-op so Gradio does nothing.
    #   - If new images did appear, deliver one gallery update and the updated state.
    if not running and next_start_consumed and gallery_delivered:
        _before = len(getattr(live_state, "image_paths", None) or [])
        _sync_restored_images(live_state)
        _after  = len(getattr(live_state, "image_paths", None) or [])
        if _after == _before:
            # Truly idle — nothing changed. Deactivate timer to prevent Tab 2
            # from being force-rendered by timer ticks when nothing is generating.
            return (gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(active=False))  # deactivate poll_timer
        # New images restored from cloud — push one gallery update then go idle.
        _restored_paths = filter_gallery_cb(live_state, bf)
        _gal = _restored_paths if _restored_paths else gr.update()
        return (live_state, gr.update(), gr.update(), _gal, _gal,
                gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(active=False))  # deactivate poll_timer

    # ── Active path: batch is running (or one-shot deliveries still pending) ─
    gallery_bf = "All beats" if running else bf
    paths = filter_gallery_cb(live_state, gallery_bf)

    # batch_start_num / prompt / beat / image-name: only while running
    if running:
        next_start_update = gr.update(value=next_start) if next_start is not None else gr.update()
        latest_img_update = latest_img
        prompt_upd = prompt_debug
        beat_upd = beat_text
        name_upd = image_name
    else:
        # Deliver the final beat-advance exactly once after the job completes
        with _BG_LOCK:
            if not next_start_consumed and next_start is not None:
                _BG_JOB["next_start_consumed"] = True
                next_start_update = gr.update(value=next_start)
            else:
                next_start_update = gr.update()
        latest_img_update = gr.update()
        prompt_upd = gr.update()
        beat_upd = gr.update()
        name_upd = gr.update()

    # Gallery: push while running, or for exactly one final delivery after job ends
    if running:
        gallery_upd = paths if paths else gr.update()
        vis_upd     = paths if paths else gr.update()
    else:
        with _BG_LOCK:
            if not gallery_delivered:
                _BG_JOB["gallery_delivered"] = True
                gallery_upd = paths if paths else gr.update()
                vis_upd     = paths if paths else gr.update()
            else:
                gallery_upd = gr.update()
                vis_upd     = gr.update()

    gaps_html_upd = _format_gaps_html(live_state)

    # Keep timer active while generating; deactivate once fully idle (handled above).
    _timer_upd = gr.update(active=True) if running else gr.update(active=False)
    return (live_state, status, latest_img_update, gallery_upd, vis_upd,
            _format_session_cost(), next_start_update, _cloud.get_status(),
            prompt_upd, beat_upd, name_upd, gaps_html_upd, _timer_upd)


def build_tab2(state: gr.State, sync_token: gr.State = None):
    with gr.Tab("Tab 2 — Director & Generate"):

        # ── GENERATION CONTROL BAR ─────────────────────────────────────────
        with gr.Group(elem_classes=["gen-bar"]):
            with gr.Row():
                batch_start_num = gr.Number(label="From Page", value=1, minimum=1, precision=0, scale=1,
                                            info="Page number to start from. 1 page = 10 panels.")
                batch_size_dd = gr.Dropdown(
                    label="Batch Pages", choices=["1", "10", "25", "50", "100", "All"], value="50", scale=1,
                    info="Pages per run (1 page = 10 panels). Batch 10 = 100 panels."
                )
                concurrency_dd = gr.Dropdown(
                    label="⚡ Workers", choices=["1", "2", "3", "4", "6", "8", "10"], value="10", scale=1,
                    info="Parallel FAL calls. More workers = faster batches. Each worker runs independently."
                )
                pause_btn = gr.Button("⏸ Pause", variant="secondary", size="sm", scale=1)
                resume_btn = gr.Button("▶ Resume", variant="secondary", size="sm", scale=1)
                stop_btn = gr.Button("⏹ Stop", variant="stop", size="sm", scale=1)
            with gr.Row():
                beat_selector = gr.Dropdown(
                    label="Beat (size=1 mode)", choices=[], value=None, scale=1,
                    allow_custom_value=True,
                    info="Active only when Batch Size = 1."
                )
                beat_preview = gr.Textbox(
                    label="Beat Text", lines=2, interactive=False, scale=3,
                    placeholder="Beat text appears here after selecting a beat above."
                )
            with gr.Row():
                beat_refs_gallery = gr.Gallery(
                    label="📚 Library refs influencing this beat",
                    columns=6, rows=1, height=130, object_fit="cover",
                    show_label=True, interactive=False,
                    elem_id="beat_refs_gallery",
                )
                beat_refs_html = gr.HTML(
                    value="<p style='color:#888;font-size:12px;padding:8px'>Select a beat to see which library images will be used as references.</p>"
                )
            with gr.Row():
                gen_btn = gr.Button("🎨 Generate + Save", variant="primary", scale=2)
                build_ai_prompt_btn = gr.Button("✨ Build AI Prompt", variant="secondary", scale=1)
                use_openai_final = gr.Checkbox(label="AI polish at generate time", value=False, scale=1)
            with gr.Row():
                quality_preset = gr.Dropdown(label="Model", choices=QUALITY_PRESETS, value=DEFAULT_QUALITY_PRESET, scale=2)
                prompt_format_dd = gr.Dropdown(label="Prompt Format", choices=PROMPT_FORMAT_OPTIONS, value=DEFAULT_PROMPT_FORMAT, scale=1, interactive=True)
                aspect_ratio_dd = gr.Dropdown(label="Aspect Ratio", choices=ASPECT_RATIO_OPTIONS, value=DEFAULT_ASPECT_RATIO, scale=1, interactive=True)
                resolution_dd = gr.Dropdown(label="Resolution", choices=RESOLUTION_OPTIONS, value=DEFAULT_RESOLUTION, scale=1, interactive=True)
                regen_page_scripts_btn = gr.Button("⚡ Regen Page Scripts", variant="secondary", size="sm", scale=1)
            with gr.Accordion("📋 Review Page Scripts before generating (saves money — catch wrong characters here first)", open=False):
                page_scripts_display = gr.Textbox(
                    label="All Page Scripts — review these before spending on generation",
                    lines=18, interactive=False, max_lines=50,
                    placeholder="Click '⚡ Regen Page Scripts' or '📋 Load Scripts' below to preview exactly what will be sent to the image model for each page.\n\nUse this to catch wrong character ages, bad descriptions, or off-style panels BEFORE paying for generation.",
                )
                view_scripts_btn = gr.Button("📋 Load Page Scripts", variant="secondary", size="sm")
            model_cost_info = gr.HTML(value=_model_cost_html(DEFAULT_QUALITY_PRESET))
            with gr.Row():
                gen_status = gr.Textbox(label="Status", interactive=False, scale=4)
                cost_display = gr.Textbox(label="Session Cost", interactive=False, scale=1, value="💰 $0.0000  |  🖼 0 imgs", elem_classes=["cost-box"])
                cloud_status_display = gr.Textbox(label="Cloud Backup", interactive=False, scale=1, value=_cloud.get_status())
            gen_gaps_html = gr.HTML(
                value="<p style='color:#4a9e6b;font-size:12px;margin:4px 0'>✅ No reference gaps detected yet.</p>",
                label="",
            )
            with gr.Row():
                gen_log_box = gr.Textbox(
                    label="⚙️ Generation Steps Log — click Refresh after each generation to see the exact order of operations",
                    lines=6, interactive=False, max_lines=20,
                    placeholder="Click 'Refresh Log' after generating to see: Step A (refs resolved), Step B (prompt built), Step C (ref header prepended to top), then FAL call.\nThis confirms refs were chosen BEFORE the prompt was written.",
                )
                gen_log_refresh_btn = gr.Button("🔍 Refresh Log", variant="secondary", scale=0)
            with gr.Row():
                ntfy_topic_input = gr.Textbox(
                    label="🔔 Push Notifications (ntfy.sh topic)",
                    placeholder="e.g. manhwa-abc123  ← install the free ntfy app on your phone, subscribe to this topic",
                    scale=4, max_lines=1,
                )
                ntfy_test_btn = gr.Button("📲 Test Notify", variant="secondary", size="sm", scale=1)

        # ── GALLERY (near top so no scrolling to see results) ───────────────
        with gr.Row():
            with gr.Column(scale=1):
                latest = gr.Image(label="Latest Output (9:16)", type="pil")
                latest_image_name_display = gr.Textbox(
                    label="File", interactive=False, lines=1, max_lines=1,
                    placeholder="001_01.png",
                )
                beat_text_display = gr.Textbox(
                    label="Beat sentence", interactive=False, lines=2,
                    placeholder="The story beat this image came from appears here."
                )
            with gr.Column(scale=2):
                with gr.Row():
                    beat_filter_label = gr.HTML("<span style='font-size:13px;color:#aaa'>Filter:</span>")
                    reload_gallery_btn = gr.Button("🔄 Reload Gallery", variant="secondary", size="sm", scale=0)
                gallery = gr.Gallery(label="Saved Images", columns=4, height=540, object_fit="contain")
                with gr.Row():
                    selected_image_name = gr.Textbox(label="Selected", interactive=False, scale=3)
                    redo_btn = gr.Button("🔄 Redo", variant="secondary", scale=1)
                    delete_btn = gr.Button("🗑 Delete", variant="stop", scale=1)
                    gr.HTML("<div style='min-width:60px'></div>")
                    mass_delete_btn = gr.Button("🗑️ Delete All Visible", variant="stop", scale=1)
                with gr.Row():
                    gr.HTML("<div style='flex:1'></div>")
                    copy_prompt_btn = gr.Button("📋 Copy prompt", size="sm", scale=0)
                prompt_debug = gr.Textbox(
                    label="📋 Prompt for selected image (editable — tweak it, then Redo)",
                    lines=4,
                    interactive=True,
                    placeholder="Click any image to load its prompt here. Edit freely, then click 🔄 Redo to regenerate.",
                )
                ref_sources_gallery = gr.Gallery(
                    label="🖼 Reference images used (hover to enlarge — numbers match the prompt)",
                    columns=6,
                    height=160,
                    object_fit="cover",
                    show_label=True,
                    allow_preview=True,
                    preview=False,
                    elem_id="ref_sources_gallery",
                )
                with gr.Row():
                    gr.HTML("<div style='flex:1'></div>")
                    copy_reasoning_btn = gr.Button("📋 Copy reasoning", size="sm", scale=0)
                panel_reasoning_display = gr.Textbox(
                    label="💬 Why this panel was built this way",
                    lines=8,
                    max_lines=16,
                    interactive=False,
                    placeholder="Click any image to see a plain-English explanation of how each reference was used — which image set the shot framing, which provided the pose, which set the mood and lighting, and which gave the background setting.",
                )
                with gr.Row():
                    prompt_edit_instruction = gr.Textbox(
                        label="✏️ AI edit instruction",
                        placeholder='e.g. "remove the cameraman, make skin tone look natural"',
                        lines=4,
                        max_lines=12,
                        scale=4,
                    )
                    edit_redo_btn = gr.Button("✏️ Edit & Redo", variant="primary", scale=1)

        with gr.Accordion("🔧 Mass Fix (Beat Range)", open=False):
            gr.Markdown(
                "Rewrite the stored prompt for every beat in a range using your correction rules, "
                "then regenerate one new image per beat. The scene action is preserved — only your "
                "specified details are enforced.\n\n"
                "**Example:** *Keep David as an older man wearing black. Keep all children 3–6 years old, male, seated.*"
            )
            with gr.Row():
                mass_fix_from = gr.Number(label="From beat", value=1, minimum=1, precision=0, scale=1)
                mass_fix_to   = gr.Number(label="To beat",   value=10, minimum=1, precision=0, scale=1)
            mass_fix_instruction = gr.Textbox(
                label="Fix rules (applied to every prompt in the range)",
                placeholder='e.g. "Keep David as an older man in black robes. All children must be 3–6 years old, male, seated on the ground."',
                lines=4,
                max_lines=14,
            )
            mass_fix_btn = gr.Button("🔧 Apply Fix to Range", variant="primary")

        with gr.Row():
            beat_filter = gr.Dropdown(label="Filter by beat", choices=["All beats"], value="All beats", scale=1)
            zip_base_name = gr.Textbox(label="ZIP name", value="untitled", scale=1)
            zip_part_dd = gr.Dropdown(label="Part (ZIP filter)", choices=["All"], value="All", scale=1)
            zip_btn = gr.Button("📦 Zip", variant="secondary", scale=1)
            zip_file = gr.HTML(value="", scale=2)
        with gr.Row():
            range_from = gr.Number(label="Delete beats from", value=None, minimum=1, precision=0, scale=1)
            range_to   = gr.Number(label="to", value=None, minimum=1, precision=0, scale=1)
            range_delete_btn = gr.Button("🗑 Delete Beat Range", variant="stop", scale=1)
        delete_status = gr.Textbox(label="Status", interactive=False, lines=1)
        zip_status = gr.Textbox(label="Zip Status", interactive=False, lines=1, visible=True)
        zip_timer = gr.Timer(value=2, active=False)

        visible_gallery_state = gr.State([])
        selected_index_state = gr.State(None)

        # ── STYLE REFERENCE (OMNI REFERENCE) ──────────────────────────────
        with gr.Accordion("🎨 Style Reference (Omni Reference)", open=False):
            gr.Markdown(
                "Upload any panel or artwork — every image generated will match its visual style, "
                "color palette, and line art. Works with **Nano Banana 2** and **Nano Banana Pro** models.\n\n"
                "**Style** — match overall art style, palette, line weight  \n"
                "**Character** — keep a specific character's look consistent  \n"
                "**Composition** — match the framing and panel layout  \n"
                "**Face** — lock a specific face across panels"
            )
            with gr.Row():
                style_ref_image_input = gr.Image(
                    label="Reference Image",
                    type="pil",
                    height=220,
                    scale=1,
                )
                with gr.Column(scale=1):
                    style_ref_tag_dd = gr.Dropdown(
                        label="Reference Type",
                        choices=["style", "character", "composition", "face"],
                        value="style",
                        info="How the model should interpret this image",
                    )
                    gr.Markdown(
                        "**Tip:** Clear the image to remove the reference and go back to prompt-only generation. "
                        "Reference is uploaded fresh each time you generate — it is not stored permanently."
                    )

        # ── STYLE & SAFETY ─────────────────────────────────────────────────
        with gr.Accordion("⚙️ Style & Safety", open=False):
            with gr.Row():
                style = gr.Textbox(label="Style tokens", value=DEFAULT_STYLE, lines=3)
                negative = gr.Textbox(label="Negative prompt", value=DEFAULT_NEGATIVE, lines=3)
            enable_safety_checker = gr.Checkbox(label="Enable fal safety checker", value=False)

        gen_btn_bottom = gr.Button("🎨 Generate + Save", variant="primary")

        load_project_btn = gr.Button("🔄 Load / Reload Project into Tab 2", size="sm")
        refresh_status = gr.Textbox(label="Load Status", interactive=False, lines=1)
        comprehensive_prompt = gr.Textbox(label="AI Prompt mirror (internal)", lines=2, interactive=True, visible=False)

        # ── SCENE SETUP ────────────────────────────────────────────────────
        with gr.Accordion("🎬 Scene Setup", open=False):
            with gr.Row():
                location_dd = gr.Dropdown(label="Location", choices=[], value=None)
                sub_location_dd = gr.Dropdown(label="Sub-location", choices=["None"], value="None")
                scene_type_dd = gr.Dropdown(label="Scene Type", choices=SCENE_TYPES, value="EMOTION")
                camera_dd = gr.Dropdown(label="Camera Type", choices=CAMERA_TYPES, value=CAMERA_TYPES[0])
            with gr.Row():
                chars = gr.CheckboxGroup(label="Characters in Frame", choices=[])
                items = gr.CheckboxGroup(label="Items / Props", choices=[])
            action_line = gr.Textbox(label="Action Line", lines=2, placeholder="What is physically happening right now?")
            emotion_notes = gr.Textbox(label="Character Emotions", lines=3, placeholder="Specific expressions and body language...")
            outfit_overrides = gr.Textbox(label="Outfit Overrides  (Character: outfit description)", lines=2)
            with gr.Row():
                include_beat_text = gr.Checkbox(label="Include raw beat text in prompt", value=False)
                allow_silhouettes = gr.Checkbox(label="Allow background silhouettes", value=False)
                include_signature = gr.Checkbox(label="Include signature detail", value=True)

        # Text overlay removed — PIL text drawing disabled
        text_type = gr.State("None")
        text_mode = gr.State("Manual")
        on_image_text = gr.State("")

        # ── BULK LOCATION ──────────────────────────────────────────────────
        with gr.Accordion("📍 Bulk Location Fix", open=False):
            gr.Markdown("*See the full location map below — ⚠️ red rows have no location assigned. Use the controls to fix a range, then click Refresh Map.*")
            with gr.Row():
                bulk_start = gr.Number(label="Start Beat", value=1, precision=0)
                bulk_end = gr.Number(label="End Beat", value=1, precision=0)
                bulk_location = gr.Dropdown(label="Apply Location", choices=["None"], value="None")
                bulk_apply_btn = gr.Button("Apply to Range", variant="secondary")
            bulk_status = gr.Textbox(label="Bulk Status", interactive=False)
            with gr.Row():
                location_map_btn = gr.Button("🗺️ Refresh Map", variant="secondary", size="sm")
            location_map_html = gr.HTML(value="<p style='color:#6868a0;font-size:13px;padding:6px'>Load a project to see the location map.</p>", label="")

        # ── WIRING ─────────────────────────────────────────────────────────
        refresh_outputs = [
            batch_start_num, batch_size_dd, beat_selector, beat_preview,
            location_dd, sub_location_dd, scene_type_dd, camera_dd,
            chars, items, action_line, emotion_notes, outfit_overrides,
            beat_filter, gallery, visible_gallery_state, selected_image_name,
            bulk_location, refresh_status, prompt_debug,
            cost_display,
        ]

        load_project_btn.click(refresh_lists_cb, inputs=[state], outputs=refresh_outputs)
        load_project_btn.click(lambda: "", outputs=[comprehensive_prompt])

        beat_selector.change(on_select_beat_fill_cb, inputs=[state, beat_selector], outputs=[
            state, beat_preview, location_dd, sub_location_dd, scene_type_dd, camera_dd,
            chars, items, action_line, emotion_notes, outfit_overrides, refresh_status, prompt_debug,
        ])
        beat_selector.change(lambda: "", outputs=[comprehensive_prompt])
        beat_selector.change(_beat_refs_preview_cb, inputs=[state, beat_selector], outputs=[beat_refs_gallery, beat_refs_html])

        def _sync_beat_filter_from_selector(beat_label: str):
            """When a beat is selected in the generator dropdown, sync the gallery filter to it."""
            if not beat_label:
                return gr.update()
            # beat_label format: "001 — ECU: worn-out snea…"
            m = re.match(r"^(\d+)", beat_label.strip())
            if not m:
                return gr.update()
            num = int(m.group(1))
            return gr.update(value=f"Beat {num:03d}")

        # NOTE: auto-syncing beat_selector → beat_filter is intentionally disabled.
        # Gradio 6+ fires change events on programmatic updates, so selecting any
        # beat in the editor immediately filtered the gallery to only that beat,
        # making all other generated images appear to disappear.
        # Users can still change beat_filter manually to filter the gallery.

        def _location_change_and_rewrite(st, location_name, current_action):
            dd = on_select_location_cb(st, location_name)
            return dd, _rewrite_action_for_location(st, current_action, location_name)
        location_dd.change(_location_change_and_rewrite, inputs=[state, location_dd, action_line], outputs=[sub_location_dd, action_line])

        def _on_scene_type_change(scene_type_name):
            choices = _camera_choices(scene_type_name)
            return gr.Dropdown(choices=CAMERA_TYPES, value=(choices[0] if choices else CAMERA_TYPES[0]))
        scene_type_dd.change(_on_scene_type_change, inputs=[scene_type_dd], outputs=[camera_dd])

        def _apply_filter(st: ProjectState, bf: str):
            paths = filter_gallery_cb(st, bf)
            return paths, paths, None, "", "", ""
        beat_filter.change(_apply_filter, inputs=[state, beat_filter], outputs=[gallery, visible_gallery_state, selected_index_state, delete_status, selected_image_name, prompt_debug])

        def _build_panel_reasoning(prompt_text: str, beat_idx: int, st: ProjectState) -> str:
            """Return a plain-English explanation for ALL panels on this page."""
            import re as _re
            if not prompt_text or not beat_idx:
                return ""
            _nppp = _ppp(st)
            # Figure out which page this image is from
            _page_idx = (beat_idx - 1) // _nppp
            _first_beat_on_page = _page_idx * _nppp + 1  # 1-indexed beat for panel 1

            def _extract_nums_from_section(section: str, keyword: str) -> str:
                _hits = _re.findall(
                    r"Use\s+((?:Image\s+\d+(?:,\s*Image\s+\d+)*))\s*\(" + keyword,
                    section
                )
                return ", ".join(_hits) if _hits else ""

            _scene_explanations = {
                "ACTION":    "action scene — wide or dynamic framing works best to show the full movement",
                "EMOTION":   "emotional beat — tight framing on the face lands the feeling",
                "DIALOGUE":  "dialogue — medium or over-the-shoulder keeps both characters readable",
                "POWER":     "power reveal — low angle makes the subject imposing",
                "DISCOVERY": "discovery moment — close-up on the object before pulling to the character",
                "SUSPENSE":  "suspense — slow zoom into a detail or cut to eyes builds tension",
                "COMBAT":    "combat — dynamic angle and motion blur sell the impact",
            }

            # ── Helper: map a beat index back to its original story sentence ────
            _CAMERA_PREFIX = _re.compile(
                r'^(?:ECU|CU|MS|LS|POV|low[\s-]?angle|overhead|dutch[\s-]?tilt|'
                r'tracking|wide\s*shot|extreme\s*close[\s-]?up|bird[\s-]?eye|'
                r'aerial|over[\s-]?shoulder)\b',
                _re.IGNORECASE,
            )

            def _orig_excerpt(bi_1: int) -> str:
                """Return the original story sentence closest to beat bi_1 (1-indexed)."""
                raw = (getattr(st, "original_story", None) or getattr(st, "story", "") or "").strip()
                if not raw:
                    return ""
                sents = [s.strip() for s in _re.split(r'(?<=[.!?…])\s+', raw) if s.strip()]
                if not sents:
                    return ""
                all_beats = st.beats or []
                n_beats   = len(all_beats)
                if n_beats == 0:
                    return ""
                # Proportional index: beat bi_1 (1-based) → sentence j (0-based)
                frac    = (bi_1 - 1) / max(n_beats - 1, 1)
                sent_j  = min(int(round(frac * (len(sents) - 1))), len(sents) - 1)
                return sents[sent_j]

            parts = []
            page_label = f"PAGE {_page_idx + 1} — Editorial breakdown: why each panel was made"
            parts.append(page_label)
            parts.append("=" * len(page_label))

            for _pi in range(_nppp):
                _pnum = _pi + 1           # 1-indexed panel on page
                _bi   = _first_beat_on_page + _pi  # global beat index

                # --- Story beat (always use real beat sentence, not AI-generated action) ---
                _beat_raw = ""
                if st.beats and _bi >= 1 and _bi <= len(st.beats):
                    _beat_raw = st.beats[_bi - 1]

                # --- Beat plan signals ---
                _plan: dict = {}
                try:
                    _plan = ensure_beat_plan(st, _bi) or {}
                except Exception:
                    pass
                _camera_t = ((_plan.get("camera_type") or "")).strip()
                _scene_t  = ((_plan.get("scene_type") or "")).strip().upper()
                _emotions = _plan.get("character_emotions") or {}
                _location = ((_plan.get("suggested_location") or "")).strip()
                _chars    = _plan.get("suggested_characters") or []

                # --- Extract ref assignments from the panel section of the prompt ---
                _section = ""
                _sm = _re.search(
                    rf"(PANEL\s+{_pnum}\b.*?)(?=PANEL\s+\d+\b|━━━|GLOBAL QUALITY|$)",
                    prompt_text, _re.DOTALL
                )
                if _sm:
                    _section = _sm.group(1)

                _camera_ref  = _extract_nums_from_section(_section, "camera")
                _pose_ref    = _extract_nums_from_section(_section, "body POSE")
                _mood_ref    = _extract_nums_from_section(_section, "lighting")
                _setting_ref = _extract_nums_from_section(_section, "setting")

                # --- Build this panel's block ---
                parts.append("")
                parts.append(f"━━━ Panel {_pnum}  (beat {_bi}) ━━━")

                # Show the original story sentence this beat comes from
                _orig = _orig_excerpt(_bi)
                if _orig:
                    _orig_snip = _orig[:300] + ("…" if len(_orig) > 300 else "")
                    parts.append(f'📖 Story text:  \u201c{_orig_snip}\u201d')
                else:
                    parts.append("📖 Story text: (unavailable)")

                # Show the visual beat (camera direction + composition) — only in Shorts
                # mode where beats are AI-expanded and differ from the original sentence
                if _beat_raw and _CAMERA_PREFIX.match(_beat_raw):
                    _beat_snip = _beat_raw[:250] + ("…" if len(_beat_raw) > 250 else "")
                    parts.append(f'🎬 Visual direction: {_beat_snip}')
                elif _beat_raw and _beat_raw.strip() != (_orig or "").strip():
                    # Panel mode: beat IS the original sentence — no need to repeat
                    pass

                # Editorial why — camera choice
                _scene_explain = _scene_explanations.get(_scene_t, "")
                if _camera_t:
                    _why = f" — {_scene_explain}" if _scene_explain else ""
                    parts.append(f"Shot: {_camera_t}{_why}")

                # Emotion context
                if _emotions:
                    _emo_str = ", ".join(f"{c}: {e}" for c, e in list(_emotions.items())[:3])
                    parts.append(f"Feeling: {_emo_str}")

                # References
                _ref_lines = []
                if _camera_ref:
                    _ref_lines.append(f"  📐 {_camera_ref} → shot framing (angle/depth/geometry)")
                if _pose_ref:
                    _char_hint = f" for {_chars[0]}" if _chars else ""
                    _ref_lines.append(f"  🧍 {_pose_ref} → body pose skeleton only{_char_hint} (skin/clothing/props from reference are invisible)")
                if _mood_ref:
                    _ref_lines.append(f"  💡 {_mood_ref} → lighting quality and color temperature")
                if _setting_ref:
                    _ref_lines.append(f"  🏙 {_setting_ref} → background architecture — {_location or 'scene location'}")
                if _ref_lines:
                    parts.append("References used:")
                    parts.extend(_ref_lines)
                else:
                    parts.append("References: none matched — generated from text only")

            return "\n".join(parts)

        def _store_selection(evt: gr.SelectData, current_visible: List[Any], st: ProjectState):
            current = _normalize_gallery_list(current_visible)
            picked: Optional[int] = None
            if evt is not None and getattr(evt, "index", None) is not None:
                try:
                    picked = int(evt.index)
                except Exception:
                    pass
            val = getattr(evt, "value", None)
            if isinstance(val, str) and val in current:
                picked = current.index(val)
            if isinstance(val, (list, tuple)) and val and isinstance(val[0], str) and val[0] in current:
                picked = current.index(val[0])
            if picked is None or picked < 0 or picked >= len(current):
                return None, "", "", "", "", [], ""

            thumb_path = current[picked]
            fname = os.path.basename(thumb_path)
            # Gallery shows thumbnails — convert back to original path for manifest lookup
            img_path = _thumb_to_original(thumb_path)
            img_basename = os.path.splitext(os.path.basename(img_path))[0]

            prompt_text = ""
            beat_text = ""
            beat_idx_found = 0
            ref_images: List[str] = []

            if st and st.project_dir:
                manifest_path = os.path.join(st.project_dir, "manifest.jsonl")
                try:
                    if os.path.exists(manifest_path):
                        with open(manifest_path, "r", encoding="utf-8") as mf:
                            for line in mf:
                                try:
                                    entry = json.loads(line.strip())
                                    entry_path = entry.get("image_path", "")
                                    entry_base = os.path.splitext(os.path.basename(entry_path))[0]
                                    # Match by full path OR basename (handles thumb→original mismatch)
                                    if entry_path == img_path or entry_base == img_basename:
                                        raw_prompt = entry.get("prompt", "")
                                        beat_idx_found = int(entry.get("beat_index", 0) or 0)
                                        if beat_idx_found > 0 and st.beats and beat_idx_found <= len(st.beats):
                                            beat_text = f"Beat {beat_idx_found:03d} — {st.beats[beat_idx_found - 1]}"
                                        # Always show the saved prompt (page scripts don't start with "(" but are still valid)
                                        if raw_prompt.strip():
                                            prompt_text = raw_prompt
                                        else:
                                            plan = ensure_beat_plan(st, beat_idx_found) if beat_idx_found > 0 else {}
                                            stype = clean_for_prompt(str(plan.get("scene_type") or "EMOTION")).upper()
                                            if stype not in SCENE_TYPES:
                                                stype = "EMOTION"
                                            lc = _location_choices(st)
                                            lv = plan.get("suggested_location") or (lc[0] if lc else "")
                                            sc = _sub_location_choices(st, lv)
                                            sv = plan.get("suggested_sub_location") or (sc[0] if sc else "None")
                                            cc = _camera_choices(stype)
                                            cv = plan.get("camera_type") or (cc[0] if cc else CAMERA_TYPES[0])
                                            prompt_text = compose_tab2_preview_prompt(
                                                st, stype, lv, sv, cv,
                                                plan.get("suggested_characters") or [],
                                                plan.get("suggested_items") or [],
                                                plan.get("suggested_action") or (st.beats[beat_idx_found - 1] if beat_idx_found > 0 else ""),
                                                _emotion_text_from_plan(plan),
                                                True, DEFAULT_STYLE,
                                                plan.get("attacker_name") or "",
                                                plan.get("target_name") or "",
                                                beat_index=beat_idx_found,
                                            )
                                        ref_images = entry.get("reference_images") or []
                                        break
                                except Exception:
                                    pass
                except Exception:
                    pass

            # Fallback: beat text from in-memory manifest_beat_index (no disk read needed)
            if not beat_text and st:
                mbi = getattr(st, "manifest_beat_index", {}) or {}
                bi = mbi.get(img_path) or mbi.get(thumb_path)
                if not bi:
                    # Try basename match against manifest_beat_index keys
                    for k, v in mbi.items():
                        if os.path.splitext(os.path.basename(k))[0] == img_basename:
                            bi = v
                            break
                if bi and bi > 0 and st.beats and bi <= len(st.beats):
                    beat_text = f"Beat {bi:03d} — {st.beats[bi - 1]}"

            # Numbered gallery captions so "Image 3" in the prompt maps to the 3rd thumbnail
            ref_gallery_items = [(url, f"Image {i+1}") for i, url in enumerate(ref_images) if url]
            reasoning_text = _build_panel_reasoning(prompt_text, beat_idx_found, st)
            return picked, fname, prompt_text, beat_text, fname, ref_gallery_items, reasoning_text

        gallery.select(_store_selection, inputs=[visible_gallery_state, state], outputs=[selected_index_state, selected_image_name, prompt_debug, beat_text_display, latest_image_name_display, ref_sources_gallery, panel_reasoning_display])

        build_ai_prompt_btn.click(
            build_ai_prompt_preview,
            inputs=[state, beat_selector, scene_type_dd, location_dd, sub_location_dd, camera_dd, chars, items, action_line, emotion_notes, include_signature, style, prompt_format_dd, use_openai_final],
            outputs=[prompt_debug, refresh_status],
        )

        quality_preset.change(_model_cost_html, inputs=[quality_preset], outputs=[model_cost_info])

        def _format_page_scripts(st: ProjectState) -> str:
            scripts = getattr(st, "page_scripts", None) or {}
            if not scripts:
                return "No page scripts found. Run '⚡ Regen Page Scripts' first."
            generated_at = getattr(st, "page_scripts_generated_at", "") or ""
            header = f"📋 {len(scripts)} page scripts"
            if generated_at:
                header += f"  |  last regenerated {generated_at}"
            header += "\n" + "─" * 60
            lines = [header]
            for pi in sorted(scripts.keys()):
                script = scripts[pi] or ""
                lines.append(f"\n{'='*60}\nPAGE {pi + 1}\n{'='*60}\n{script}\n")
            return "\n".join(lines)

        regen_page_scripts_btn.click(
            regenerate_page_scripts_cb,
            inputs=[state],
            outputs=[state, gen_status],
        ).then(
            _format_page_scripts,
            inputs=[state],
            outputs=[page_scripts_display],
        )

        view_scripts_btn.click(
            _format_page_scripts,
            inputs=[state],
            outputs=[page_scripts_display],
        )

        def _on_format_change(fmt):
            is_p = fmt == "panel"
            return (
                gr.update(
                    label="From Page" if is_p else "From Beat",
                    info="Page number to start from. 1 page = 10 panels." if is_p else "First beat to generate in batch mode.",
                ),
                gr.update(
                    label="Batch Pages" if is_p else "Batch Size",
                    info="Pages per run (1 page = 10 panels). Batch 10 = 100 panels." if is_p else "Images per run. Size=1 → single beat. All → every beat.",
                ),
            )

        prompt_format_dd.change(
            _on_format_change,
            inputs=[prompt_format_dd],
            outputs=[batch_start_num, batch_size_dd],
        )

        _gen_inputs = [
            state, beat_selector, location_dd, sub_location_dd, scene_type_dd, camera_dd,
            chars, items, action_line, emotion_notes, outfit_overrides,
            include_beat_text, allow_silhouettes, include_signature,
            text_type, text_mode, on_image_text,
            style, negative, beat_filter, prompt_debug, comprehensive_prompt,
            use_openai_final, enable_safety_checker, quality_preset, prompt_format_dd, aspect_ratio_dd, resolution_dd,
            style_ref_image_input, style_ref_tag_dd,
        ]
        _batch_inputs = [batch_start_num, batch_size_dd] + _gen_inputs + [concurrency_dd]
        _gen_outputs = [state, gen_status, latest, gallery, prompt_debug, comprehensive_prompt, beat_filter, cost_display, visible_gallery_state, beat_text_display, latest_image_name_display, batch_start_num]

        # Timer polls the background job every 4 s — keeps gallery/status live after reconnect.
        # Starts INACTIVE so it never fires before the user clicks Generate. The timer is
        # activated by gen_btn / gen_btn_bottom clicks and deactivated by _poll_bg_cb when
        # the job goes idle. This prevents the timer from force-rendering Tab 2's 166 Svelte
        # components before the user has even clicked that tab.
        poll_timer = gr.Timer(value=4, active=False)
        poll_timer.tick(
            _poll_bg_cb,
            inputs=[state, beat_filter],
            outputs=[state, gen_status, latest, gallery, visible_gallery_state, cost_display, batch_start_num, cloud_status_display,
                     prompt_debug, beat_text_display, latest_image_name_display, gen_gaps_html, poll_timer],
        )

        # Generate kicks off a background thread — returns immediately so wifi drops don't kill it.
        # Also activates the poll timer so it starts monitoring the job.
        gen_btn.click(_submit_bg_cb, inputs=_batch_inputs, outputs=[gen_status]).then(
            lambda: gr.update(active=True), outputs=[poll_timer]
        )
        gen_btn_bottom.click(_submit_bg_cb, inputs=_batch_inputs, outputs=[gen_status]).then(
            lambda: gr.update(active=True), outputs=[poll_timer]
        )

        # Stop signals the background thread to exit after the current image
        stop_btn.click(_stop_bg_cb, outputs=[gen_status])

        # Pause / Resume toggle the shared threading.Event (no cancel needed)
        def _pause_cb():
            _BATCH_PAUSE_EVENT.set()
            return "⏸ Paused — click ▶ Resume to continue"

        def _resume_cb():
            _BATCH_PAUSE_EVENT.clear()
            return "▶ Resuming..."

        pause_btn.click(_pause_cb, outputs=[gen_status])
        resume_btn.click(_resume_cb, outputs=[gen_status])

        def _set_ntfy_topic_cb(topic: str):
            global _NTFY_TOPIC
            _NTFY_TOPIC = topic.strip()

        def _test_ntfy_cb(topic: str) -> str:
            global _NTFY_TOPIC
            _NTFY_TOPIC = topic.strip()
            if not _NTFY_TOPIC:
                return "⚠️ Enter a topic name first"
            _send_ntfy_notification("🧪 Test from Manhwa Generator — notifications are working!")
            return f"📲 Test sent to ntfy.sh/{_NTFY_TOPIC}"

        ntfy_topic_input.change(_set_ntfy_topic_cb, inputs=[ntfy_topic_input])
        ntfy_test_btn.click(_test_ntfy_cb, inputs=[ntfy_topic_input], outputs=[gen_status])
        gen_log_refresh_btn.click(
            lambda: "\n".join(_last_gen_steps) if _last_gen_steps else "(no generation run yet — generate a panel first)",
            inputs=[],
            outputs=[gen_log_box],
        )

        def _reload_gallery_from_disk_cb(st: ProjectState, bf: str):
            """Read the manifest.jsonl directly from disk so the gallery is always
            up-to-date regardless of whether the Gradio state was refreshed.
            Updates both the state (image_paths / images_by_beat) and the gallery."""
            if not st or not getattr(st, "project_dir", None):
                return gr.update(), gr.update(), gr.update(), "⚠️ Load a project first."
            import json as _json
            manifest_path = os.path.join(st.project_dir, "manifest.jsonl")
            if not os.path.isfile(manifest_path):
                return gr.update(), gr.update(), gr.update(), "⚠️ No manifest found."
            st.image_paths = []
            st.images_by_beat = {}
            st.all_manifest_paths = []
            st.manifest_beat_index = {}
            loaded = 0
            with open(manifest_path, "r", encoding="utf-8") as mf:
                for line in mf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = _json.loads(line)
                        img_path = entry.get("image_path", "")
                        if not img_path:
                            continue
                        img_path = os.path.abspath(img_path)
                        beat_idx = int(entry.get("beat_index", 0) or 0)
                        st.all_manifest_paths.append(img_path)
                        st.manifest_beat_index[img_path] = beat_idx
                        if os.path.isfile(img_path) and img_path not in st.image_paths:
                            st.image_paths.append(img_path)
                            if beat_idx > 0:
                                st.images_by_beat.setdefault(beat_idx, [])
                                if img_path not in st.images_by_beat[beat_idx]:
                                    st.images_by_beat[beat_idx].append(img_path)
                            loaded += 1
                    except Exception:
                        pass
            paths = filter_gallery_cb(st, bf or "All beats")
            return st, paths, paths, f"✅ Reloaded {loaded} images from disk ({len(paths)} shown)"

        def _sync_visible(st: ProjectState, bf: str):
            paths = filter_gallery_cb(st, bf)
            return paths, paths, None, "", "", ""

        reload_gallery_btn.click(
            _reload_gallery_from_disk_cb,
            inputs=[state, beat_filter],
            outputs=[state, gallery, visible_gallery_state, refresh_status],
        )

        delete_btn.click(delete_by_index_cb, inputs=[state, beat_filter, visible_gallery_state, selected_index_state], outputs=[state, gallery, delete_status, selected_index_state])
        delete_btn.click(_sync_visible, inputs=[state, beat_filter], outputs=[gallery, visible_gallery_state, selected_index_state, delete_status, selected_image_name, prompt_debug])

        mass_delete_btn.click(
            mass_delete_cb,
            inputs=[state, beat_filter, visible_gallery_state],
            outputs=[state, gallery, delete_status, selected_index_state],
            js="() => confirm('⚠️ Delete ALL visible images?\\n\\nThis is permanent and cannot be undone.\\n\\nAre you sure?')",
        ).then(
            _sync_visible,
            inputs=[state, beat_filter],
            outputs=[gallery, visible_gallery_state, selected_index_state, delete_status, selected_image_name, prompt_debug],
        )

        range_delete_btn.click(range_delete_cb, inputs=[state, beat_filter, range_from, range_to], outputs=[state, gallery, delete_status, selected_index_state])
        range_delete_btn.click(_sync_visible, inputs=[state, beat_filter], outputs=[gallery, visible_gallery_state, selected_index_state, delete_status, selected_image_name, prompt_debug])

        def _refresh_gallery_after_redo(st, bf):
            """Post-redo gallery sync — runs after the streaming generator finishes.
            The timer may have overwritten the gallery with stale state while the redo
            was in progress; this guarantees a final refresh from the updated state."""
            paths = filter_gallery_cb(st, bf)
            print(f"[_refresh_gallery_after_redo] refreshing gallery — {len(paths)} image(s)", flush=True)
            return paths, paths

        _redo_inputs = [state, beat_filter, visible_gallery_state, selected_index_state] + _gen_inputs
        redo_btn.click(redo_selected_cb, inputs=_redo_inputs, outputs=_gen_outputs).then(
            _refresh_gallery_after_redo, inputs=[state, beat_filter], outputs=[gallery, visible_gallery_state]
        )

        _ai_edit_inputs = [state, beat_filter, visible_gallery_state, selected_index_state, prompt_edit_instruction] + _gen_inputs
        edit_redo_btn.click(ai_edit_redo_cb, inputs=_ai_edit_inputs, outputs=_gen_outputs).then(
            _refresh_gallery_after_redo, inputs=[state, beat_filter], outputs=[gallery, visible_gallery_state]
        )

        _mass_fix_inputs = [state, mass_fix_from, mass_fix_to, mass_fix_instruction] + _gen_inputs
        mass_fix_btn.click(mass_fix_cb, inputs=_mass_fix_inputs, outputs=_gen_outputs)

        # Copy-to-clipboard buttons — pure client-side JS, no server round-trip
        copy_prompt_btn.click(
            None,
            inputs=[prompt_debug],
            outputs=[],
            js="(text) => { if (text) navigator.clipboard.writeText(text); }",
        )
        copy_reasoning_btn.click(
            None,
            inputs=[panel_reasoning_display],
            outputs=[],
            js="(text) => { if (text) navigator.clipboard.writeText(text); }",
        )

        def _update_zip_parts(st):
            parts = getattr(st, "story_parts", []) or []
            choices = ["All"] + [p["name"] for p in parts]
            return gr.Dropdown(choices=choices, value="All")

        if sync_token is not None:
            sync_token.change(_update_zip_parts, inputs=[state], outputs=[zip_part_dd])

        zip_btn.click(zip_cb, inputs=[state, zip_base_name, zip_part_dd], outputs=[zip_file, zip_status, zip_timer])
        zip_timer.tick(_poll_zip, inputs=[], outputs=[zip_file, zip_status, zip_timer])

        def _bulk_apply_location(st, start_b, end_b, loc_name):
            if not st or not st.beats:
                return st, "❌ Build first.", gr.Dropdown(choices=_location_choices(st), value=loc_name or "None"), _location_overview_html(st)
            start = max(1, int(start_b or 1)); end = min(len(st.beats), int(end_b or start))
            if end < start:
                start, end = end, start
            for idx in range(start, end + 1):
                plan = ensure_beat_plan(st, idx)
                plan["suggested_location"] = loc_name or "None"
                sub_choices = _sub_location_choices(st, loc_name)
                if plan.get("suggested_sub_location") not in sub_choices:
                    plan["suggested_sub_location"] = sub_choices[0] if sub_choices else "None"
                scene_type = clean_for_prompt(str(plan.get("scene_type") or _classify_scene_type(st.beats[idx - 1]))).upper()
                cam_choices = _camera_choices(scene_type)
                if plan.get("camera_type") not in cam_choices:
                    plan["camera_type"] = cam_choices[0] if cam_choices else CAMERA_TYPES[0]
                plan["suggested_perspective"] = plan["camera_type"]
                plan["suggested_action"] = _rewrite_action_for_location(st, plan.get("suggested_action") or st.beats[idx-1], loc_name)
                st.beat_plans[idx] = plan
            _save_project_json(st)
            return st, f"✅ Updated beats {start:03d}-{end:03d} → {loc_name}", gr.Dropdown(choices=_location_choices(st), value=loc_name or "None"), _location_overview_html(st)
        bulk_apply_btn.click(_bulk_apply_location, inputs=[state, bulk_start, bulk_end, bulk_location], outputs=[state, bulk_status, bulk_location, location_map_html])
        location_map_btn.click(_location_overview_html, inputs=[state], outputs=[location_map_html])
        load_project_btn.click(_location_overview_html, inputs=[state], outputs=[location_map_html])

        if sync_token is not None:
            sync_token.change(refresh_lists_cb, inputs=[state], outputs=refresh_outputs)
            sync_token.change(lambda: "", outputs=[comprehensive_prompt])
            sync_token.change(_location_overview_html, inputs=[state], outputs=[location_map_html])

        # ── Direct state-change watcher: refresh Tab 2 when the project changes ──
        # sync_token.change can miss the updated state in Gradio's event chain;
        # watching state directly (guarded by project_dir) is more reliable.
        _tab2_proj_dir = gr.State(None)

        def _maybe_refresh_tab2(st, current_dir):
            """Full refresh only when the loaded project actually changes."""
            new_dir = getattr(st, "project_dir", None) if st else None
            if new_dir and new_dir != current_dir:
                results = refresh_lists_cb(st)
                return (new_dir,) + tuple(results)
            return (current_dir,) + tuple(gr.update() for _ in refresh_outputs)

        state.change(
            _maybe_refresh_tab2,
            inputs=[state, _tab2_proj_dir],
            outputs=[_tab2_proj_dir] + refresh_outputs,
        )

        return prompt_format_dd, gallery
