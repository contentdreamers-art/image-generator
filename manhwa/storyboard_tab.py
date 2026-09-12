"""
storyboard_tab.py — Manual Director / Storyboard tab.

Flow (top-to-bottom):
  1. Story name + story input + Parse
  2. Characters — HTML cards + dropdown assignment
  3. Recently Used — quick re-select already-assigned images
  4. Library search — full-width big gallery + selected image bar
  5. Panel Director — page/panel nav · beat · 5 ref slots
  6. Output — big image · prompt · refs gallery · ZIP
"""
from __future__ import annotations
import glob, os, re, time, threading, zipfile
from typing import Any, Dict, List, Optional, Tuple

# ── Global pause/resume state for Auto-Run ───────────────────────────────────
_GEN_PAUSED = threading.Event()   # set() = paused, clear() = running
_GEN_PAUSED.clear()
_GEN_RUNNING = threading.Event()  # guards against two Auto-Run batches at once
_SB_SAVE_LOCK = threading.Lock()  # serialises storyboard.json writes during parallel generation

import gradio as gr

# ─────────────────────────────────────────────────────────────────────────────
N_SLOTS      = 5
SLOT_CODES   = ["face",       "camera",     "mood",       "action",     "location"]
SLOT_LABELS  = ["A · face",   "B · camera", "C · mood",   "D · pose",   "E · location"]

_ROLE_SHORT = {
    "face":     "identity only — lock this exact face shape, hair color/style, eye color, and skin tone in every panel. Do not redesign or age this character between panels.",
    "camera":   "camera/composition only — copy the exact shot angle, depth, and character framing. All people are invisible mannequins — never copy their face, hair, clothing, skin tone, or identity.",
    "mood":     "mood/atmosphere only — borrow the lighting quality, color palette, and emotional tone. All people are invisible mannequins — never copy their face, hair, clothing, skin tone, or identity.",
    "action":   "action/scene energy only — match the compositional energy and visual intensity. All people are invisible mannequins — never copy their face, hair, clothing, skin tone, or identity.",
    "location": "ALL PANELS setting/architecture only. Use ONLY the background environment, architecture, and ambient lighting. All people in this image are invisible — never copy their face, clothing, or identity.",
}

_ROLE_PANEL = {
    "camera":   "camera/composition only — copy the exact shot angle, depth, and framing geometry; all people are invisible mannequins showing spatial arrangement only",
    "mood":     "lighting and atmosphere only — copy lighting direction, quality (hard vs soft), and color temperature; all people are invisible",
    "action":   "body POSE SKELETON only — trace the joint positions, limb angles, body angle, and weight distribution ONLY. The reference figure's skin coverage, clothing material, props, setting decor, and expression are 100% invisible — do NOT copy them. Replace the figure with the story character wearing their own period-appropriate outfit.",
    "location": "setting/architecture only — copy the background environment, architecture, and ambient light quality; all people are invisible",
}

_SB_OUT = "/tmp/storyboard_output"
_DIV    = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

_ART_STYLE = f"""
{_DIV} ART STYLE {_DIV}
Premium high-budget Korean action-fantasy manhwa / webtoon. Use:
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
"""

_PANEL_RULES = f"""
{_DIV} PANEL RULES {_DIV}
- Keep every character, limb, and important object fully inside its panel.
- No characters or objects crossing white gutters between panels.
- No cropped heads or cut-off hands on main characters.
- No dialogue bubbles, speech captions, signs, logos, watermarks, or readable writing.
- Read left-to-right, top-to-bottom.
"""

_NEG = (
    "identity drift, different face in each panel, changed hair color, changed eye color, "
    "changed skin tone, wrong age, redesigned outfit, copied reference people, copied reference clothing, "
    "face swapping, outfit swapping, extra limbs, fused fingers, broken anatomy, malformed hands, "
    "muddy shadows, dull eyes, excessive darkness, rough unfinished coloring, unreadable action, "
    "cluttered composition, characters crossing panel gutters, speech bubbles, readable text, "
    "watermark, logo, photorealism, blurry, low resolution, generic anime, inconsistent character proportions, "
    "shirtless when clothed in story, bare skin from reference bleeding through, "
    "anachronistic objects from reference images, reference character's skin exposure level, "
    "reference character's clothing copied onto story character, reference props visible in panel."
)


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────

def _empty_sb(page_size: int = 6) -> Dict[str, Any]:
    return {
        "story":         "",
        "world_context": "",
        "project_id":    "",
        "story_name":    "",
        "beats":         [],
        "beat_chars":    {},
        "n_beats":       0,
        "page_size":     page_size,
        "cur_page":      1,
        "cur_panel":     1,
        "cast":             {},        # {char_name: template_id or None}
        "char_class":       {},        # {char_name: class_name}  e.g. {"Mei": "farmer"}
        "char_appearance":  {},        # {char_name: freetext appearance note} — user-editable, injected into every prompt
        "selected_tid":  None,
        "selected_char": None,     # character currently targeted for cast assignment
        "page_refs":     {},
        "recent_tids":   [],        # recently used template IDs for quick re-use
        "last_prompt":   "",
        "page_prompts":  {},        # {str(page_num): prompt} — per-page, cleared on re-parse
        "page_prompt_versions": {}, # {str(page_num): [{path,prompt,model,version}]} immutable generation snapshots
    }


def _page_versions(sb: Dict, pg: Any) -> List[str]:
    """Return all generated paths for page `pg` (handles both old str and new list format)."""
    data = (sb.get("generated_pages") or {}).get(str(pg))
    if data is None:
        return []
    if isinstance(data, list):
        return [p for p in data if p and os.path.isfile(p)]
    return [data] if (data and os.path.isfile(data)) else []


def _latest_page_img(sb: Dict, pg: Any) -> Optional[str]:
    """Most recently generated image for page `pg`, or None."""
    versions = _page_versions(sb, pg)
    return versions[-1] if versions else None

# Pages 1-9: 1 beat per panel (cinematic pacing).
# Pages 10+: 2 beats per panel (halves page count for long stories, keeps cost manageable).
_PACK_AFTER = 9

def _panel_beat_idx(pg: int, pn: int, pz: int) -> int:
    """First (or only) beat index (1-based) for this panel under the 2-beat packing rule."""
    if pg <= _PACK_AFTER:
        return (pg - 1) * pz + pn
    return _PACK_AFTER * pz + (pg - _PACK_AFTER - 1) * pz * 2 + (pn - 1) * 2 + 1

def _panel_beat_text(beats: List[str], pg: int, pn: int, pz: int,
                     override_idx: int = 0) -> str:
    """Beat text for a panel — 1 beat on pages 1-9, 2 beats joined on pages 10+.
    override_idx: use this beat index instead of computing (for manually reassigned panels)."""
    bidx = override_idx or _panel_beat_idx(pg, pn, pz)
    if not (0 < bidx <= len(beats)):
        return f"Panel {pn}."
    b1 = beats[bidx - 1]
    if pg <= _PACK_AFTER or override_idx:
        return b1
    # Pack second beat if it exists
    b2 = beats[bidx] if bidx < len(beats) else ""
    return (b1 + "  /  " + b2) if b2 else b1

def _n_pages(sb: Dict) -> int:
    n, pz = sb.get("n_beats", 0), max(1, sb.get("page_size", 6))
    if n <= 0:
        return 0
    boundary = _PACK_AFTER * pz
    if n <= boundary:
        return (n + pz - 1) // pz
    return _PACK_AFTER + ((n - boundary) + pz * 2 - 1) // (pz * 2)

def _get_panel(sb: Dict, p: int, n: int) -> Dict:
    return ((sb.get("page_refs") or {}).get(str(p), {}).get("panels") or {}).get(str(n), {})

def _set_panel(sb: Dict, p: int, n: int, data: Dict) -> None:
    sb.setdefault("page_refs", {}).setdefault(str(p), {}).setdefault("panels", {})[str(n)] = data

def _panel_slots(panel: Dict) -> List[Optional[Dict]]:
    r = list(panel.get("refs") or [])
    while len(r) < N_SLOTS:
        r.append(None)
    return r[:N_SLOTS]

def _beat_choices(sb: Dict) -> List[str]:
    out = ["— auto —"]
    for i, b in enumerate(sb.get("beats") or [], 1):
        lbl = b[:80] + "…" if len(b) > 80 else b
        out.append(f"{i}: {lbl}")
    return out

def _add_recent(sb: Dict, tid: str) -> None:
    if not tid:
        return
    recent = list(sb.get("recent_tids") or [])
    if tid in recent:
        recent.remove(tid)
    recent.insert(0, tid)
    sb["recent_tids"] = recent[:24]


# ─────────────────────────────────────────────────────────────────────────────
# Library helpers
# ─────────────────────────────────────────────────────────────────────────────

def _template_info(tid: Optional[str]) -> Tuple[Optional[str], str]:
    if not tid:
        return None, "—"
    try:
        import character_library as lib
        t = lib.load_library().get("templates", {}).get(tid)
        if not t:
            return None, "(unknown)"
        img = lib.image_path(tid, "face") or lib.image_path(tid, "body")
        return (img if img and os.path.isfile(img) else None), (t.get("name") or tid)
    except Exception:
        return None, "(?)"

_POSE_CLOTH_PATTERNS = [
    # sitting / standing / lying positions
    r'[^.]*\b(?:sit[s]?|sitting|seated|stand[s]?|standing|lie[s]?|lying|recline[s]?|reclining|'
    r'crouche[s]?|crouching|kneel[s]?|kneeling|perch(?:es)?|leaning|posed?)\b[^.]*\.',
    # wearing / dressed / clothing
    r'[^.]*\bwearing\b[^.]*\.',
    r'[^.]*\b(?:dressed|clothed)\s+in\b[^.]*\.',
    # shot description ("Shown full body three quarter front sitting…")
    r'Shown\b[^.]*\.',
    # setting
    r'Setting[:\s][^.]*\.',
    # "against a … background"
    r'against\s+[^.]*\bbackground\b[^.]*',
    r'against\s+(?:a\s+)?(?:soft|warm|neutral|minimal|plain|simple)[^.]*',
    # ── Expression / emotional state ─────────────────────────────────────────
    # Face DNA must describe GEOMETRY only (hair, eyes, skin tone).
    # These patterns strip smiles, emotional moods, and gaze direction so the
    # scene beat text — not the reference photo — controls each panel's expression.
    r'\s*convey[s]?\s+[^.]*',               # "conveys a contemplative demeanor…"
    r'\s*express(?:es?|ing)\s+[^.]*',        # "expressing sadness…"
    r'\bwith\s+quiet\s+\w+[^.]*',            # "with quiet introspection"
    r'[^.]*\bradiating\b[^.]*\.?',           # "radiating joy and youthful warmth"
    r'[^.]*\byouthful\s+warmth\b[^.]*\.?',
    r'[^.]*\b(?:contemplative|melancholic|pensive|wistful|demeanor|introspect\w*)\b[^.]*\.',
    r'[^.]*\bgaze[sd]?\s+(?:directly\s+)?(?:forward|at\s+(?:the\s+)?viewer)\b[^.]*',
    r'[^.]*directed\s+toward\s+the\s+viewer[^.]*',
    r'\brosy\s+cheeks?[^,\.]*[,\.]?\s*',    # "rosy cheeks" (expression indicator)
    r'[^.]*\bsmil(?:e[sd]?|ing)\b[^.]*\.',  # "warm, genuine smile…"
    r'[^.]*\bwarm,?\s+genuine\b[^.]*\.',    # "warm, genuine smile/warmth"
    r'[^.]*\bradiates?\s+(?:joy|warmth|happiness|confidence)\b[^.]*\.',
    r'[^.]*\b(?:joy(?:ful(?:ly)?)?)\b[^.]*\.',   # joy / joyful sentences
]

def _strip_pose_clothing(text: str) -> str:
    """Strip position, clothing, setting, and shot-description sentences.
    Keeps only face / hair / eyes / complexion / expression content."""
    import re as _re
    for pat in _POSE_CLOTH_PATTERNS:
        text = _re.sub(pat, '', text, flags=_re.IGNORECASE)
    text = _re.sub(r'\s+', ' ', text).strip().strip('.')
    return text


def _face_only_dna(tid: Optional[str]) -> str:
    """Like _template_dna but returns ONLY face-geometry descriptors.
    Strips pose, clothing, setting, and background so the model cannot copy
    the reference photo's outfit or position onto the story character."""
    full = _template_dna(tid)
    if not full:
        return ""
    stripped = _strip_pose_clothing(full)
    if len(stripped) >= 20:
        return stripped
    # Fallback: assemble from structured ai_analysis atoms
    try:
        import character_library as _cl2
        t = (_cl2.load_library().get("templates") or {}).get(tid or "", {})
        ai = t.get("ai_analysis") or {}
        parts: List[str] = []
        for key in ("hair", "eye_description", "skin_tone", "complexion"):
            v = (ai.get(key) or t.get(key) or "").strip()
            if v:
                parts.append(v)
        if parts:
            return ". ".join(parts)
    except Exception:
        pass
    return stripped


def _template_dna(tid: Optional[str]) -> str:
    """Return a rich visual description for this template, for use in prompts.

    Priority: dna_prompt → ai_description → ai_analysis fields assembled into prose → summary.
    """
    if not tid:
        return ""
    try:
        import character_library as lib
        t = lib.load_library().get("templates", {}).get(tid)
        if not t:
            return ""
        # Prefer hand-written DNA prompt
        explicit = (t.get("dna_prompt") or t.get("ai_description") or "").strip()
        if explicit:
            return explicit
        # Assemble from structured ai_analysis fields
        ai = t.get("ai_analysis") or {}
        if not ai:
            # Fall back to top-level fields saved alongside template
            ai = t
        parts: List[str] = []
        summary = (ai.get("summary") or t.get("summary") or "").strip()
        hair    = (ai.get("hair") or t.get("hair") or "").strip()
        eyes    = (ai.get("eye_description") or t.get("eye_description") or "").strip()
        cloth   = (ai.get("clothing_description") or t.get("clothing_description") or "").strip()
        feats   = ai.get("distinctive_features") or t.get("distinctive_features") or []
        pose_d  = ai.get("pose") or {}
        framing = (pose_d.get("framing") or "").strip() if isinstance(pose_d, dict) else ""
        orient  = (pose_d.get("orientation") or "").strip() if isinstance(pose_d, dict) else ""
        stance  = (pose_d.get("body_stance") or "").strip() if isinstance(pose_d, dict) else ""
        setting_d = ai.get("setting") or {}
        loc_type  = (setting_d.get("location_type") or "").strip() if isinstance(setting_d, dict) else ""
        lighting_d = ai.get("lighting") or {}
        light_desc = ""
        if isinstance(lighting_d, dict):
            light_desc = (lighting_d.get("overall_mood") or lighting_d.get("quality") or "").strip()

        if summary:
            parts.append(summary)
        else:
            # Build from atoms when no summary
            char_parts = []
            if hair:   char_parts.append(f"hair: {hair}")
            if eyes:   char_parts.append(f"eyes: {eyes}")
            if cloth:  char_parts.append(f"wearing {cloth}")
            if feats:  char_parts.append(f"notable features: {', '.join(feats[:4])}")
            if char_parts:
                parts.append(". ".join(char_parts) + ".")
        # Append pose / setting context when not already in summary
        if framing and framing.lower() not in (summary or "").lower():
            pose_str = " ".join(filter(None, [framing, orient, stance])).strip()
            if pose_str:
                parts.append(f"Shown {pose_str}.")
        if loc_type and loc_type.lower() not in (summary or "").lower():
            env_str = loc_type
            if light_desc:
                env_str += f", {light_desc} lighting"
            parts.append(f"Setting: {env_str}.")
        return " ".join(parts).strip()
    except Exception:
        return ""

def _template_tags_str(tid: Optional[str]) -> str:
    """Return comma-separated tags for use in reference role labels."""
    if not tid:
        return ""
    try:
        import character_library as lib
        t = lib.load_library().get("templates", {}).get(tid)
        if not t:
            return ""
        tags = t.get("tags") or []
        return ", ".join(tags[:6]) if tags else ""
    except Exception:
        return ""

# ---------------------------------------------------------------------------
# Synonym map for library search.
# Each key expands to itself + listed alternatives when used as a query token.
# Lookup is bidirectional: searching "child" also matches entries tagged "kid".
# ---------------------------------------------------------------------------
_SEARCH_SYNONYMS: Dict[str, List[str]] = {
    # age / youth
    "kid":        ["child", "children", "young", "little", "youth", "boy", "girl"],
    "kids":       ["child", "children", "young", "little", "youth"],
    "child":      ["kid", "kids", "children", "young", "little", "youth"],
    "children":   ["kid", "kids", "child", "young", "little", "youth"],
    "little":     ["child", "kid", "young", "small"],
    "young":      ["youth", "kid", "child", "teen", "juvenile"],
    "youth":      ["young", "kid", "child", "teen", "juvenile"],
    "teen":       ["teenager", "adolescent", "young", "youth", "juvenile"],
    "teenager":   ["teen", "adolescent", "young", "youth"],
    "adolescent": ["teen", "teenager", "young", "youth"],
    "old":        ["elderly", "aged", "senior", "ancient", "elder"],
    "elderly":    ["old", "aged", "senior", "elder", "ancient"],
    "elder":      ["elderly", "old", "aged", "senior"],
    "aged":       ["old", "elderly", "senior"],
    "senior":     ["old", "elderly", "elder", "aged"],
    "adult":      ["grown", "mature", "man", "woman"],
    "mature":     ["adult", "grown"],
    # gender
    "guy":        ["man", "male", "boy", "fellow"],
    "gal":        ["woman", "female", "girl"],
    "lady":       ["woman", "female", "girl", "noble"],
    "gentleman":  ["man", "male", "noble"],
    "boy":        ["male", "young", "lad", "kid", "child"],
    "girl":       ["female", "young", "lass", "kid", "child"],
    "man":        ["male", "guy", "adult"],
    "woman":      ["female", "lady", "adult"],
    "male":       ["man", "boy", "guy"],
    "female":     ["woman", "girl", "lady"],
    # emotion / expression
    "sad":        ["crying", "tears", "weeping", "sorrowful", "melancholy", "grief"],
    "crying":     ["sad", "tears", "weeping", "grief", "sob"],
    "tears":      ["crying", "sad", "weeping"],
    "happy":      ["cheerful", "joyful", "smiling", "gleeful", "laughing"],
    "cheerful":   ["happy", "joyful", "smiling", "gleeful"],
    "joyful":     ["happy", "cheerful", "smiling"],
    "angry":      ["rage", "furious", "wrath", "fierce", "mad", "wrathful"],
    "rage":       ["angry", "furious", "wrath", "fierce"],
    "furious":    ["angry", "rage", "fierce", "wrath"],
    "scared":     ["afraid", "frightened", "fearful", "terror"],
    "afraid":     ["scared", "frightened", "fearful"],
    "frightened": ["scared", "afraid", "fearful", "terror"],
    "gentle":     ["kind", "tender", "soft", "caring", "sweet"],
    "tender":     ["gentle", "kind", "soft", "caring"],
    "proud":      ["confident", "dignified", "regal", "noble"],
    "confident":  ["proud", "bold", "strong", "determined"],
    # combat / action
    "fight":      ["combat", "battle", "fighting", "warrior", "brawl", "duel"],
    "fighting":   ["fight", "combat", "battle", "warrior", "brawl"],
    "battle":     ["fight", "combat", "war", "fighting", "warrior"],
    "combat":     ["fight", "battle", "fighting", "warrior"],
    "war":        ["battle", "combat", "military", "soldier", "army"],
    "sword":      ["blade", "katana", "weapon", "saber"],
    "magic":      ["mage", "sorcerer", "wizard", "spell", "arcane", "mystical"],
    "run":        ["running", "chase", "fleeing", "sprint"],
    "running":    ["run", "chase", "sprint", "fleeing"],
    # archetypes / roles
    "soldier":    ["warrior", "knight", "fighter", "guard", "military"],
    "warrior":    ["soldier", "knight", "fighter", "combat"],
    "knight":     ["warrior", "soldier", "fighter", "armored"],
    "mage":       ["wizard", "sorcerer", "witch", "magic", "arcane"],
    "wizard":     ["mage", "sorcerer", "magic", "arcane"],
    "witch":      ["mage", "sorcerer", "magic", "arcane"],
    "priest":     ["cleric", "monk", "holy", "divine"],
    "cleric":     ["priest", "monk", "holy", "divine"],
    "monk":       ["priest", "cleric", "holy"],
    "noble":      ["aristocrat", "lord", "lady", "royalty", "nobleman", "noblewoman"],
    "royalty":    ["noble", "king", "queen", "prince", "princess", "royal"],
    "king":       ["royalty", "ruler", "monarch", "lord"],
    "queen":      ["royalty", "ruler", "monarch", "lady"],
    "prince":     ["royalty", "noble", "young"],
    "princess":   ["royalty", "noble", "young", "girl"],
    "peasant":    ["farmer", "villager", "commoner", "servant"],
    "farmer":     ["peasant", "villager", "commoner"],
    "villager":   ["peasant", "farmer", "commoner"],
    "commoner":   ["peasant", "farmer", "villager"],
    "servant":    ["maid", "attendant", "slave", "commoner"],
    "scholar":    ["student", "academic", "learned", "researcher"],
    # appearance
    "dark":       ["black", "shadow", "night", "gloomy"],
    "light":      ["white", "bright", "pale", "blonde"],
    "beautiful":  ["pretty", "gorgeous", "lovely", "elegant"],
    "pretty":     ["beautiful", "lovely", "cute", "gorgeous"],
    "cute":       ["adorable", "sweet", "pretty", "charming"],
    "adorable":   ["cute", "sweet", "charming"],
    "handsome":   ["attractive", "beautiful", "good-looking"],
    "tall":       ["towering", "giant", "large"],
    "small":      ["little", "tiny", "short", "petite"],
    "tiny":       ["small", "little", "petite", "child"],
    # setting / scene
    "group":      ["crowd", "multiple", "family", "team", "trio", "duo"],
    "crowd":      ["group", "multiple", "people", "gathering"],
    "family":     ["group", "mother", "father", "parent", "child"],
    "mother":     ["mom", "parent", "woman", "caregiver"],
    "mom":        ["mother", "parent", "woman"],
    "father":     ["dad", "parent", "man"],
    "dad":        ["father", "parent", "man"],
    # creatures
    "snake":      ["serpent", "scaled", "reptile", "viper", "cobra"],
    "serpent":    ["snake", "scaled", "reptile", "creature"],
    "dragon":     ["serpent", "scaled", "creature", "beast", "monster"],
    "beast":      ["monster", "creature", "demon", "animal"],
    "monster":    ["beast", "creature", "demon", "supernatural"],
    # composition
    "onlookers":  ["crowd", "group", "bystanders", "watching", "staring"],
    "bystanders": ["onlookers", "crowd", "group", "watching"],
    "watching":   ["staring", "looking", "onlookers", "crowd", "bystanders"],
    "staring":    ["watching", "looking", "intense", "onlookers"],
    "alone":      ["solo", "isolated", "lone", "solitary", "lonely"],
    "lonely":     ["alone", "solo", "isolated", "solitary", "melancholy"],
    "lone":       ["alone", "solo", "isolated", "solitary"],
    "solitary":   ["alone", "solo", "isolated", "lone"],
}

def _expand_search_token(tok: str) -> List[str]:
    """Return tok plus all its known synonyms (deduplicated, lowercase)."""
    extras = _SEARCH_SYNONYMS.get(tok, [])
    seen: set = {tok}
    expanded = [tok]
    for s in extras:
        if s not in seen:
            seen.add(s)
            expanded.append(s)
    return expanded


def _search_library(query: str) -> Tuple[List, List[str]]:
    """Hybrid semantic + tag search.

    1. Semantic search via embeddings (OpenAI text-embedding-3-small) when the
       index is available — understands natural language, synonyms, and concepts.
    2. Tag/name/category search always runs in parallel — catches exact matches
       the embedding might miss.
    3. Results merged: semantic score weighted 3×, tag score normalised as
       secondary signal. Top 80 returned.

    Falls back gracefully to tag-only if embeddings not yet built.
    """
    import re as _re
    if not (query or "").strip():
        return [], []

    try:
        import character_library as lib
        import semantic_search as sem

        templates_data = lib.load_library().get("templates", {})

        # ── 1. Semantic search: embeddings top-100 → LLM re-rank → top-20 ───
        sem_scores: Dict[str, float] = {}
        if sem.is_ready():
            try:
                for tid, score in sem.search_with_rerank(query, first_pass=100, final=20):
                    sem_scores[tid] = score
            except Exception:
                pass

        # ── 2. Tag / name / category search ───────────────────────────────
        raw_tokens   = [t.strip().lower() for t in query.strip().lower().split() if t.strip()]
        token_groups = [_expand_search_token(tok) for tok in raw_tokens]

        tag_scored: Dict[str, Tuple[int, str, str]] = {}  # tid → (score, img, name)
        for tid, t in templates_data.items():
            if not (t.get("local_face") or t.get("local_body")):
                continue
            name_lower = (t.get("name") or "").lower()
            tags_list  = [tg.lower() for tg in (t.get("tags") or [])]
            tags_set   = set(tags_list)
            cat_lower  = (t.get("category") or "").lower()

            total = 0
            matched = True
            for i, group in enumerate(token_groups):
                primary   = raw_tokens[i]
                tok_score = 0
                if primary in tags_set:
                    tok_score += 3
                elif any(syn in tags_set for syn in group[1:]):
                    tok_score += 2
                elif any(any(syn in tg for tg in tags_list) for syn in group):
                    tok_score += 1
                if any(_re.search(rf'\b{_re.escape(syn)}', name_lower) for syn in group):
                    tok_score += 2
                if any(syn in cat_lower for syn in group):
                    tok_score += 1
                if tok_score == 0:
                    matched = False
                    break
                total += tok_score

            if matched and total > 0:
                img = lib.image_path(tid, "face") or lib.image_path(tid, "body")
                if img and os.path.isfile(img):
                    tag_scored[tid] = (total, img, t.get("name") or tid)

        # ── 3. Merge ───────────────────────────────────────────────────────
        all_tids = set(sem_scores) | set(tag_scored)
        scored: List[Tuple[float, str, str, str]] = []

        for tid in all_tids:
            sem_s = sem_scores.get(tid, 0.0)
            tag_s, img, name = tag_scored.get(tid, (0, None, None))  # type: ignore[assignment]

            if img is None:
                # Only in sem results — look up image
                t = templates_data.get(tid)
                if not t:
                    continue
                img = lib.image_path(tid, "face") or lib.image_path(tid, "body")
                name = t.get("name") or tid
                if not img or not os.path.isfile(img):
                    continue

            # Combined score: semantic primary (0–3), tag secondary (normalised)
            combined = sem_s * 3.0 + (tag_s / 10.0)
            scored.append((combined, tid, img, name))

        scored.sort(key=lambda x: -x[0])
        results = [(img, nm) for _, _, img, nm in scored[:80]]
        ids     = [tid for _, tid, _, _ in scored[:80]]
        return results, ids
    except Exception:
        return [], []

def _get_fal_url(tid: str) -> str:
    """Return a FAL-hosted URL for tid. Uploads the local image on-demand if not yet uploaded."""
    try:
        from character_library import get_fal_urls, get_template, update_template, image_path
        fu, bu = get_fal_urls(tid)
        if fu or bu:
            return fu or bu
        # Not uploaded yet — do it now synchronously
        t = get_template(tid)
        if not t:
            return ""
        from build import upload_pil_to_fal
        from PIL import Image as _PILImg
        for img_type in ("face", "body"):
            local = t.get(f"local_{img_type}") or image_path(tid, img_type)
            if local and os.path.isfile(local):
                try:
                    pil = _PILImg.open(local).convert("RGB")
                    url = upload_pil_to_fal(pil)
                    update_template(tid, **{f"fal_{img_type}_url": url})
                    return url
                except Exception:
                    pass
        return ""
    except Exception:
        return ""

def _recent_gallery(sb: Dict) -> Tuple[List, List[str]]:
    recent = sb.get("recent_tids") or []
    imgs, ids = [], []
    for tid in recent:
        p, nm = _template_info(tid)
        if p:
            imgs.append((p, nm))
            ids.append(tid)
    return imgs, ids


def _cast_quick_gallery(sb: Dict) -> Tuple[List, List[str]]:
    """Face thumbnails for all cast members — beat chars first, rest of cast after.
    Used for the inline quick-assign panel in the beat area."""
    cur   = sb.get("cur_page", 1)
    pn    = sb.get("cur_panel", 1)
    pz    = max(1, sb.get("page_size", 6))
    panel = _get_panel(sb, cur, pn)
    bidx  = panel.get("beat_idx") or _panel_beat_idx(cur, pn, pz)
    beat_chars = list(panel.get("chars") or (sb.get("beat_chars") or {}).get(str(bidx), []))
    cast  = sb.get("cast") or {}

    imgs: List = []
    names: List[str] = []

    def _add(cname: str, in_beat: bool) -> None:
        tid = cast.get(cname)
        prefix = "" if in_beat else "• "
        if tid:
            p, _ = _template_info(tid)
            if p and os.path.isfile(p):
                imgs.append((p, f"{prefix}{cname} ✓"))
                names.append(cname)
                return
        # No face ref image — skip gallery entirely (None crashes Gradio).
        # Character is still reachable via the cast dropdown.

    seen: set = set()
    for cname in beat_chars:
        _add(cname, True); seen.add(cname)
    for cname in cast:
        if cname not in seen:
            _add(cname, False); seen.add(cname)

    return imgs, names


def _build_timeline_gallery(sb: Dict) -> List[Tuple[str, str]]:
    """All generated pages/versions as (path, label) sorted by page then version."""
    gen_pages = sb.get("generated_pages") or {}
    result: List[Tuple[str, str]] = []
    for k in sorted(gen_pages.keys(), key=lambda x: int(x) if x.isdigit() else 0):
        versions = _page_versions(sb, k)
        for i, path in enumerate(versions):
            label = f"P{k}" if len(versions) == 1 else f"P{k} v{i + 1}"
            result.append((path, label))
    return result


# Keyword → emotional expression override for the scene beat
_EMOTION_MAP: List[Tuple[List[str], str]] = [
    (["grave", "grieve", "grief", "mourn", "funeral", "bury", "buried",
      "burial", "dead", "died", "death", "gone", "weep", "wail", "sob", "tears"],
     "grief-stricken — face crumpled with sorrow, tears or red eyes, expression of raw devastation. NO smiling."),
    (["devastated", "collapsed", "collapse", "battered", "trembling", "shaking", "hollow", "numb"],
     "devastated and broken — hollow empty stare, expression of shock and heartbreak. NO smiling."),
    (["afraid", "scared", "terror", "terrified", "dread", "fear", "horror", "pale with fright"],
     "wide-eyed terror — pale, open mouth, trembling. NO smiling."),
    (["rage", "furious", "fury", "enraged", "wrath", "livid", "seething", "snarling"],
     "burning with rage — jaw clenched, eyes fierce and cold, brow furrowed. NO smiling."),
    (["tense", "urgent", "danger", "dangerous", "threat", "hunting", "hunts", "wariness"],
     "tense and alert — jaw set, eyes sharp and scanning. Neutral, no smile."),
    (["determined", "resolute", "departs", "sets out", "strides", "ready for", "prepared for"],
     "fierce determination — eyes forward and focused, jaw set, no hesitation. Neutral serious expression."),
    (["solemn", "somber", "grim", "silent", "still", "quiet dread", "grave matter"],
     "solemn and grim — mouth a thin line, eyes heavy. NO smiling."),
    (["desperate", "last hope", "no choice", "forced to", "cornered", "helpless"],
     "desperate intensity — eyes wide and strained, expression of urgency. NO smiling."),
]

def _scene_emotion(beat_text: str) -> str:
    """Return an emotional expression override derived from the beat text.
    Empty string if the scene has no strong emotional signal."""
    low = (beat_text or "").lower()
    for keywords, expression in _EMOTION_MAP:
        if any(kw in low for kw in keywords):
            return expression
    return ""


# ── Shot-notation expander ─────────────────────────────────────────────────────
# Map shot abbreviations to framing *descriptions* — avoid the label names
# (e.g. "LONG SHOT") entirely so the model doesn't render them as caption text.
_SHOT_EXPAND = {
    "ECU": "extreme close-up framing, subject fills entire frame, only eyes/mouth visible,",
    "BCU": "big close-up framing, face fills the frame,",
    "CU":  "close-up framing, face and neck fill the frame,",
    "MCU": "medium close-up framing, chest and face visible,",
    "MS":  "medium shot framing, waist-up view,",
    "MLS": "medium long shot framing, thighs-up view,",
    "LS":  "long shot framing, full body visible with surrounding environment,",
    "ELS": "extreme long shot framing, tiny figures dwarfed by vast environment,",
    "WS":  "wide shot framing, full environment with characters visible,",
    "OTS": "over-the-shoulder shot, camera behind one character looking at another,",
    "POV": "point-of-view framing, seen through the character's eyes,",
    "INT": "interior setting,",
    "EXT": "exterior setting,",
}

def _expand_shot(text: str) -> str:
    """Replace shot abbreviations (ECU:, MS:, LS:, etc.) with framing descriptions
    so the model renders the correct camera angle without printing label text."""
    import re
    for code, desc in _SHOT_EXPAND.items():
        text = re.sub(
            rf'(?mi)^{re.escape(code)}\s*:[ \t]*',
            desc + " ",
            text,
        )
    return text


# ── Beat-based panel suggestions ──────────────────────────────────────────────
def _compute_all_suggestions(sb: Dict) -> None:
    """Pre-compute 20 suggestions for every unique beat index, store in sb['panel_suggestions'].
    Pure local search — no network calls. Called after parse/enrich and on project load."""
    beats = sb.get("beats") or []
    pz    = max(1, sb.get("page_size", 6))
    n_pg  = _n_pages(sb)
    seen: set = set()
    sugg: Dict[str, List[str]] = {}
    for page in range(1, n_pg + 1):
        for pn in range(1, pz + 1):
            panel = _get_panel(sb, page, pn)
            bidx  = panel.get("beat_idx") or _panel_beat_idx(page, pn, pz)
            if bidx in seen or bidx < 1 or bidx > len(beats):
                continue
            seen.add(bidx)
            _, ids = _suggest_for_beat(beats[bidx - 1], n=20)
            sugg[str(bidx)] = ids
    sb["panel_suggestions"] = sugg


# Cache: beat_text → expanded keyword set (avoids repeated Claude calls)
_SUGGEST_KW_CACHE: Dict[str, set] = {}

def _expand_beat_keywords(beat_text: str) -> set:
    """Ask Claude to expand beat text into semantic keywords incl. synonyms.
    Falls back to local extraction if Claude is unavailable."""
    import re as _re
    _stops = {
        "the","and","for","are","but","not","you","all","can","her","his","one",
        "had","him","its","out","was","our","she","who","did","how","each","said",
        "with","this","that","from","have","been","will","their","they","into",
        "also","than","then","when","where","only","some","what","which","while",
        "panel","beat","draw","use","image","copy","match","face","show","shot",
        "close","medium","long","wide","interior","exterior","over","shoulder",
    }
    key = beat_text.strip()
    if key in _SUGGEST_KW_CACHE:
        return _SUGGEST_KW_CACHE[key]

    # Seed from literal text first
    local = set(
        w.lower() for w in _re.findall(r'\b[a-z]{3,}\b', beat_text.lower())
        if w.lower() not in _stops
    )

    # Try Claude expansion
    expanded = set(local)
    try:
        from build import _call_claude_json
        result = _call_claude_json(
            system="You are a visual keyword expander for manhwa image search. Return ONLY valid JSON.",
            user_payload={
                "beat": beat_text[:300],
                "json_schema": {"keywords": ["list of strings"]},
                "rules": [
                    "Extract every visual concept in the beat sentence.",
                    "For each key concept add 2-4 synonyms or visually-related terms — "
                    "e.g. kneel→bow,crouch,prostrate,surrender,submit; "
                    "fear→terror,dread,horror,panic; "
                    "reach→extend,stretch,grasp,touch; "
                    "softening→gentle,tender,kind,warm,compassion.",
                    "Include: character names, actions, poses, emotions, settings, objects, "
                    "atmosphere, lighting mood.",
                    "All terms lowercase, 3+ characters, no stopwords.",
                    "Return 25-50 terms total as a flat list.",
                ]
            },
        )
        kws = [w.lower().strip() for w in (result.get("keywords") or [])
               if w and len(w.strip()) >= 3 and w.strip().lower() not in _stops]
        expanded.update(kws)
    except Exception:
        pass

    _SUGGEST_KW_CACHE[key] = expanded
    return expanded


def _suggest_for_beat(beat_text: str, n: int = 20) -> Tuple[List, List[str]]:
    """Score library images by semantic keyword overlap with beat text, return top n."""
    words = _expand_beat_keywords(beat_text)
    if not words:
        return [], []
    try:
        import character_library as _clib
        scored = []
        for tid, t in _clib.load_library().get("templates", {}).items():
            lf = _clib.image_path(tid, "face")
            lb = _clib.image_path(tid, "body")
            img = (lf if (lf and os.path.isfile(lf)) else
                   lb if (lb and os.path.isfile(lb)) else "")
            if not img:
                continue
            hay = " ".join(filter(None, [
                (t.get("name") or "").lower(),
                " ".join(t.get("tags") or []),
                (t.get("ai_description") or t.get("description") or "").lower(),
            ]))
            score = sum(1 for w in words if w in hay)
            if score > 0:
                scored.append((score, tid, img, t.get("name") or tid))
        scored.sort(key=lambda x: -x[0])
        top = scored[:n]
        return [(img, nm) for _, _, img, nm in top], [tid for _, tid, _, _ in top]
    except Exception:
        return [], []


# ─────────────────────────────────────────────────────────────────────────────
# Character extraction  (no ProjectState required — calls Claude directly)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_char_names(story: str) -> List[str]:
    """Permissive extraction: named characters AND unnamed roles (Boy, Sister, Rival, etc.)."""
    try:
        from build import _call_claude_json, CLAUDE_MODEL
        result = _call_claude_json(
            system="You are a story character extractor. Return ONLY valid JSON. No markdown.",
            user_payload={
                "story": story[:8000],
                "json_schema": {"characters": [{"name": "character identifier"}]},
                "rules": [
                    "Include EVERY character who participates in scenes, takes actions, speaks, "
                    "or is directly interacted with — even minor recurring ones.",
                    "For properly named characters use their name (e.g. Alex, Elena, Kazimir).",
                    "For unnamed protagonists referred to as 'he', 'the boy', 'I', or 'MC', "
                    "use the story's clearest identifier: 'Boy', 'Protagonist', 'Girl', 'MC', etc.",
                    "For unnamed secondary characters use their story role as the identifier: "
                    "'Sister', 'Teacher', 'Rival', 'Mother', 'Friend', 'Boss', etc.",
                    "DO NOT include generic passing crowds unless they appear in multiple scenes.",
                    "If genuinely no character can be identified, return [{\"name\": \"Protagonist\"}].",
                ],
            },
            max_tokens=800,
            model=CLAUDE_MODEL,
        )
        if result and result.get("characters"):
            names = [str(c.get("name", "")).strip() for c in result["characters"]]
            names = [n for n in names if n and len(n) >= 2]
            if names:
                print(f"[storyboard] characters found: {names}", flush=True)
                return names
    except Exception as e:
        print(f"[storyboard] char extraction error: {e}", flush=True)

    # Regex fallback
    try:
        from build import extract_characters, extract_locations
        locs  = extract_locations(story)
        names = extract_characters(story, locs)
        if names:
            print(f"[storyboard] regex fallback: {names}", flush=True)
            return names
    except Exception:
        pass

    return ["Protagonist"]


def _extract_beat_chars(story: str, beats: List[str], known_chars: List[str]) -> Dict[str, List[str]]:
    """Ask Claude which characters appear in each beat. Returns {beat_1idx_str: [names]}."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {}
    try:
        from build import _call_claude_batch_beat_plans, ProjectState as _PS, \
            _sanitize_character_payload, _call_claude_names_only, _sanitize_location_payload
        _st = _PS()
        _st.story  = story
        _st.beats  = beats
        # Give Claude character context
        try:
            nd = _call_claude_names_only(_st, story)
            if nd:
                _st.characters = _sanitize_character_payload(_st, story, nd)
                _st.locations  = _sanitize_location_payload(nd)
        except Exception:
            _st.characters = {n: {"dna_prompt": n, "fields": {}, "forms": []} for n in known_chars}
            _st.locations  = {}

        bc: Dict[str, List[str]] = {}
        for bs in range(0, len(beats), 20):
            plans = _call_claude_batch_beat_plans(
                _st, bs + 1, beats[bs: bs + 20],
                prior_location="", prior_active_forms=None,
            ) or {}
            for bidx, plan in plans.items():
                flat = [
                    c for c in ((plan or {}).get("suggested_characters") or [])
                    if c and c.lower() not in ("none", "")
                ]
                if flat:
                    bc[str(bidx)] = flat
        return bc
    except Exception as e:
        print(f"[storyboard] beat char plans failed: {e}", flush=True)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Beat parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_story(story: str, page_size: int, world_ctx: str = "") -> Tuple[Dict, str]:
    from build import split_beats
    raw = split_beats(story.strip())
    if page_size == 6:
        try:
            from build import _expand_shorts_beats_with_claude, _validate_shorts_beats_with_claude
            words  = len((story or "").split())
            target = 30 if words < 100 else 35 if words < 250 else 45 if words < 500 else 60 if words < 800 else 75
            # Prepend world context so Claude's beat expansion is world-aware
            story_for_claude = story
            if world_ctx.strip():
                story_for_claude = (
                    f"[WORLD CONTEXT — read this first to understand the story's setting, "
                    f"power system, terminology, and character roles. Use it to write beats "
                    f"that reference correct in-world details (ranks, locations, abilities, etc.)]\n"
                    f"{world_ctx.strip()}\n\n"
                    f"[STORY — expand ONLY this into visual beats]\n"
                    f"{story}"
                )
            raw    = _validate_shorts_beats_with_claude(
                story,  # pass original story for validation so beat count isn't confused by preamble
                _expand_shorts_beats_with_claude(story_for_claude, raw, target_beats=min(target, 90))
            )
            msg = f"✅ {len(raw)} micro-beats (AI expanded for Shorts)."
        except Exception as e:
            msg = f"⚠️ AI expansion failed: {e}. Using {len(raw)} local beats."
    else:
        msg = f"✅ {len(raw)} beats split."
    sb = _empty_sb(page_size)
    sb["story"] = story
    sb["beats"] = raw
    sb["n_beats"] = len(raw)
    return sb, msg


def _generate_chars_dna(char_names: List[str], story: str, existing_dna: Dict) -> Dict[str, str]:
    """Ask Claude to write a concise visual appearance description for every character
    that doesn't already have a DNA entry. Returns {name: description} for new ones."""
    if not char_names or not os.environ.get("ANTHROPIC_API_KEY"):
        return {}
    need = [n for n in char_names if n and not existing_dna.get(n)]
    if not need:
        return {}
    try:
        from build import _call_claude_json
        result = _call_claude_json(
            system="You are a manhwa character designer. Return ONLY valid JSON. No markdown.",
            user_payload={
                "story_excerpt": story[:3000],
                "characters": need,
                "json_schema": {"characters": [{"name": "string", "appearance": "string"}]},
                "rules": [
                    "For each character, write a concise visual appearance description (1-2 sentences) "
                    "covering: approximate age, hair color & style, eye color, build, and any key "
                    "distinguishing features or clothing mentioned or implied in the story.",
                    "If the story gives no clues, invent consistent details that fit a manhwa aesthetic.",
                    "Keep each description under 60 words.",
                    "Use descriptive visual language suitable for an image-generation prompt.",
                ]
            },
        )
        out: Dict[str, str] = {}
        for entry in (result.get("characters") or []):
            nm = (entry.get("name") or "").strip()
            desc = (entry.get("appearance") or "").strip()
            if nm and desc:
                out[nm] = desc
        return out
    except Exception:
        return {}


def _auto_assign_classes(
    char_names: List[str],
    story: str,
    existing_classes: Dict,
    world_ctx: str = "",
) -> Dict[str, str]:
    """Ask Claude to match each character to an EXISTING clothing class in the library.

    Priority rules (enforced in the prompt sent to Claude):
      1. Pick the best-fitting name from EXISTING_CLASSES (fuzzy OK — "Village Chief" → "chief").
      2. Only invent a NEW name if no existing class fits at all.
         New names are returned as-is but flagged in the caller for user review.

    Only assigns characters not already manually assigned.
    Returns {char_name: class_name} for new assignments.
    """
    if not char_names or not os.environ.get("ANTHROPIC_API_KEY"):
        return {}
    need = [n for n in char_names if n and not existing_classes.get(n)]
    if not need:
        return {}
    try:
        from build import _call_claude_json
        import character_library as _cl_ac

        # Make sure default classes (incl. modern-* and seeded medieval DNA) exist
        # BEFORE reading the class list — otherwise a fresh library gives Claude
        # an empty list and it invents unseeded names instead.
        try:
            _cl_ac.ensure_default_classes()
        except Exception:
            pass
        # Fetch the current library classes so Claude can pick from them
        known_classes = sorted(_cl_ac.get_clothing_classes().keys())

        ctx_parts = []
        if world_ctx.strip():
            ctx_parts.append(f"WORLD BUILDING:\n{world_ctx.strip()}")
        ctx_parts.append(f"STORY:\n{story[:2500]}")
        combined_ctx = "\n\n".join(ctx_parts)

        result = _call_claude_json(
            system="You are a manhwa story analyst. Return ONLY valid JSON. No markdown.",
            user_payload={
                "context": combined_ctx,
                "characters": need,
                "existing_clothing_classes": known_classes,
                "json_schema": {"assignments": [{"name": "string", "class": "string", "is_new": "bool"}]},
                "rules": [
                    "IMPORTANT: existing_clothing_classes is the AUTHORITATIVE list of clothing classes "
                    "already defined in the user's library. Always prefer a name from that list.",
                    "ERA CONTEXT: First decide the story's time period from the WORLD BUILDING and STORY. "
                    "If the setting is modern/contemporary (cities, cars, phones, offices, guns, schools), "
                    "you MUST pick the 'modern-' prefixed classes (modern-noble, modern-warrior, modern-guard, etc.). "
                    "If the setting is medieval/fantasy/historical (kingdoms, swords, magic, villages), "
                    "pick the un-prefixed classes (noble, warrior, guard, etc.). "
                    "Never mix eras unless a specific character is explicitly from a different era "
                    "(e.g. a time traveler or isekai protagonist still wearing modern clothes).",
                    "Match semantically, not just lexically — 'Village Chief' should map to 'chief' if "
                    "'chief' is in the list; 'spirit-beast-hunter' should map to 'hunter' or 'warrior' "
                    "if those exist.",
                    "Only set is_new=true if the character's role is genuinely unlike any existing class. "
                    "When in doubt, pick the closest existing class and set is_new=false.",
                    "If existing_clothing_classes is empty, infer sensible names from the story "
                    "(e.g. 'noble', 'farmer', 'warrior') and set is_new=true for all.",
                    "Assign each character exactly ONE class. Use lowercase with hyphens for multi-word.",
                    "If genuinely uncertain, use 'commoner' for medieval settings or 'modern-commoner' "
                    "for modern settings (set is_new=false unless that class is absent).",
                    "Return an entry for every character in the input list.",
                ],
            },
        )
        out: Dict[str, str] = {}
        for entry in (result.get("assignments") or []):
            nm     = (entry.get("name") or "").strip()
            cls    = (entry.get("class") or "").strip().lower()
            is_new = bool(entry.get("is_new", False))
            if nm and cls and nm in need:
                # Double-check: if Claude marked it new but it closely matches an existing
                # class, override to the existing one (case-insensitive exact match)
                if is_new and cls in known_classes:
                    is_new = False
                out[nm] = cls
                out[f"__new__{nm}"] = is_new   # type: ignore[assignment]
        return out
    except Exception:
        return {}


def _enrich(sb: Dict) -> Tuple[Dict, str]:
    story = sb.get("story", "")
    beats = sb.get("beats") or []

    # Phase 1 — extract character names from story text
    char_names = _extract_char_names(story)
    cast = dict(sb.get("cast") or {})
    for nm in char_names:
        if nm not in cast:
            cast[nm] = None
    sb["cast"] = cast

    if not beats:
        return sb, f"✅ {len(cast)} characters found. No beats to analyse."

    # Phase 2 — per-beat character hints
    bc = _extract_beat_chars(story, beats, list(cast.keys()))
    sb["beat_chars"] = bc

    # Merge any new names from beat plans
    for chars in bc.values():
        for c in chars:
            if c not in cast:
                cast[c] = None
    sb["cast"] = cast

    # Phase 3 — generate text DNA for characters with no library reference
    existing_dna = dict(sb.get("char_dna") or {})
    new_dna = _generate_chars_dna(list(cast.keys()), story, existing_dna)
    existing_dna.update(new_dna)
    sb["char_dna"] = existing_dna

    # Phase 4 — auto-assign social classes (only for characters not already assigned)
    existing_char_class = dict(sb.get("char_class") or {})
    world_ctx = (sb.get("world_context") or "").strip()
    raw_assignments = _auto_assign_classes(list(cast.keys()), story, existing_char_class, world_ctx=world_ctx)
    if raw_assignments:
        # Separate real assignments from the __new__ flag sentinels
        real_assignments = {k: v for k, v in raw_assignments.items() if not k.startswith("__new__")}
        new_flags        = {k[len("__new__"):]: v for k, v in raw_assignments.items() if k.startswith("__new__")}
        existing_char_class.update(real_assignments)
        sb["char_class"] = existing_char_class
        # Only register a blank placeholder for genuinely NEW class names
        # (i.e. ones Claude invented because nothing in the library fit).
        # Classes that were matched from the existing library are NOT registered again.
        try:
            import character_library as _cl_tmp
            known = _cl_tmp.get_clothing_classes()
            for char_nm, cls_name in real_assignments.items():
                is_new = bool(new_flags.get(char_nm, True))
                if is_new and cls_name and cls_name not in known:
                    _cl_tmp.set_clothing_class(cls_name, "", "")
        except Exception:
            pass

    # Phase 4.5 — auto-build char_appearance from char_dna + clothing class DNA
    # Never overwrites a manually-saved appearance note.
    try:
        import character_library as _cl_ap
        char_appearance = dict(sb.get("char_appearance") or {})
        all_char_class  = sb.get("char_class") or {}
        all_char_dna    = sb.get("char_dna") or {}
        for cname in cast:
            if char_appearance.get(cname):
                continue  # manual override — preserve it
            # DRIFT FIX: characters with an assigned face reference must NOT get an
            # auto-generated appearance note. The note would silently override the
            # actual face-image DNA at generation time (user_note > face ref DNA),
            # making the model average a Claude-invented face with the real reference.
            if cast.get(cname):
                continue
            parts = []
            dna = (all_char_dna.get(cname) or "").strip()
            if dna:
                parts.append(dna)
            cls_name = all_char_class.get(cname)
            if cls_name:
                clothing_dna = _cl_ap.get_class_clothing_dna(cls_name, seed_key=cname)
                if clothing_dna:
                    parts.append(clothing_dna)
            if parts:
                char_appearance[cname] = " ".join(parts)
                # Mark as auto-generated so it can be safely cleared later
                # if the user assigns a face reference (real image > invented text)
                sb.setdefault("char_appearance_auto", {})[cname] = True
        sb["char_appearance"] = char_appearance
    except Exception:
        pass

    # Phase 5 — cinematic shot recipes + color script (one Claude call for all beats)
    try:
        _assign_recipes_and_colors(sb)
    except Exception:
        pass

    return sb, f"✅ {sb['n_beats']} beats · {_n_pages(sb)} pages · {len(cast)} characters found."


def _assign_recipes_and_colors(sb: Dict) -> None:
    """Tag every beat with a cinematic shot recipe + assign a color script per story phase.
    One Claude call for the whole story. Results stored in sb['beat_recipes'] and
    sb['beat_colors'] (both keyed by str beat index, 1-based). Manual per-panel
    overrides (panel['recipe']) always win at generation time."""
    beats = sb.get("beats") or []
    if not beats:
        return
    # Always drop old assignments first — beats may have changed, and stale
    # recipe/color direction attached to the wrong beats is worse than none.
    sb.pop("beat_recipes", None)
    sb.pop("beat_colors", None)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return
    from build import _call_claude_json
    import character_library as _cl_sr

    recipes = _cl_sr.get_shot_recipes()
    recipe_guide = {name: r.get("when", "") for name, r in recipes.items()}

    result = _call_claude_json(
        system=("You are a manhwa cinematography director. Return ONLY valid JSON. No markdown. "
                "You choose the best camera/lighting recipe for each story beat and design "
                "a color script that gives the story professional visual phases."),
        user_payload={
            "beats": {str(i + 1): b[:160] for i, b in enumerate(beats)},
            "recipes": recipe_guide,
            "world_context": (sb.get("world_context") or "")[:300],
            "json_schema": {
                "recipes": {"<beat_number>": "<recipe_name>"},
                "color_phases": [{"from": "int (first beat)", "to": "int (last beat)",
                                  "hue": "short color-mood phrase, e.g. 'cold electric blue wash, dungeon gloom'"}],
            },
            "rules": [
                "Assign EVERY beat exactly one recipe name from the recipes dict (keys only).",
                "Vary the shots — never use the same recipe more than 2 beats in a row.",
                "Match the recipe to the beat's dramatic function, not just its literal content.",
                "color_phases: divide the story into 3-7 contiguous phases. Each phase gets ONE dominant "
                "hue/mood phrase that reflects its emotional register (e.g. warm gold for family flashback, "
                "blood red for oppression, cold blue for the dungeon, ash grey for betrayal).",
                "Phases must cover all beats with no gaps and no overlaps.",
            ],
        },
        max_tokens=4000,
    ) or {}

    valid = set(recipes.keys())
    nb = len(beats)
    beat_recipes: Dict[str, str] = {}
    for k, v in (result.get("recipes") or {}).items():
        try:
            bi = int(k)
        except (TypeError, ValueError):
            continue
        v = (v or "").strip()
        if 1 <= bi <= nb and v in valid:
            beat_recipes[str(bi)] = v
    if beat_recipes:
        sb["beat_recipes"] = beat_recipes

    beat_colors: Dict[str, str] = {}
    for ph in (result.get("color_phases") or []):
        try:
            lo, hi = int(ph.get("from")), int(ph.get("to"))
            hue = (ph.get("hue") or "").strip()[:120]
            if hue:
                for bi in range(max(1, lo), min(nb, hi) + 1):
                    beat_colors.setdefault(str(bi), hue)  # first phase wins on overlap
        except Exception:
            continue
    if beat_colors:
        sb["beat_colors"] = beat_colors


# ─────────────────────────────────────────────────────────────────────────────
# HTML renderers
# ─────────────────────────────────────────────────────────────────────────────

def _beats_html(sb: Dict) -> str:
    beats = sb.get("beats") or []
    bc    = sb.get("beat_chars") or {}
    if not beats:
        return "<p style='color:#555;padding:4px'>Beat list appears here after parsing.</p>"
    pz = max(1, sb.get("page_size", 6))
    out = ["<div style='font-size:12px;line-height:1.65;max-height:180px;overflow-y:auto;padding:2px 0'>"]
    for i, b in enumerate(beats, 1):
        if (i - 1) % pz == 0:
            out.append(f"<div style='color:#f6ad55;font-size:10px;margin:5px 0 2px'>── PAGE {(i-1)//pz+1} ──</div>")
        chars = bc.get(str(i), [])
        ch    = f" <span style='color:#90cdf4;font-size:10px'>👤{', '.join(chars)}</span>" if chars else ""
        out.append(f"<div style='margin:1px 0'><b style='color:#718096'>{i}.</b> <span style='color:#cbd5e0'>{b}</span>{ch}</div>")
    out.append("</div>")
    return "".join(out)


def _img_b64(path: Optional[str]) -> str:
    """Return a base64 data-URI for a local image file, or empty string."""
    if not path or not os.path.isfile(path):
        return ""
    try:
        import base64 as _b64
        with open(path, "rb") as f:
            return "data:image/png;base64," + _b64.b64encode(f.read()).decode()
    except Exception:
        return ""


def _cast_html(sb: Dict) -> str:
    cast = sb.get("cast") or {}
    if not cast:
        return "<p style='color:#555;font-size:12px;padding:4px'>No characters found yet — parse a story first.</p>"
    sel        = sb.get("selected_char")
    char_class = sb.get("char_class") or {}
    cards = []
    for cname, tid in cast.items():
        path, tname = _template_info(tid)
        border = "border:2px solid #f6ad55" if cname == sel else "border:1px solid #4a5568"
        b64 = _img_b64(path)
        if b64:
            img_tag = f"<img src='{b64}' style='width:72px;height:82px;object-fit:cover;border-radius:4px;display:block'>"
        else:
            img_tag = "<div style='width:72px;height:82px;background:#2d3748;border-radius:4px;display:flex;align-items:center;justify-content:center;color:#4a5568;font-size:24px'>?</div>"
        assigned  = f"<div style='color:#68d391;font-size:9px'>{tname}</div>" if tid else "<div style='color:#718096;font-size:9px'>no face ref</div>"
        cls_name  = char_class.get(cname)
        cls_badge = (f"<div style='color:#f6ad55;font-size:9px'>👗{cls_name}</div>" if cls_name
                     else "<div style='color:#4a5568;font-size:9px'>no class</div>")
        cards.append(
            f"<div style='display:inline-block;text-align:center;margin:4px;padding:6px 4px;"
            f"background:#1a202c;border-radius:6px;{border};min-width:80px;max-width:90px;vertical-align:top'>"
            f"{img_tag}"
            f"<div style='color:#e2e8f0;font-size:11px;margin-top:3px;font-weight:600;"
            f"overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:80px'>{cname}</div>"
            f"{assigned}{cls_badge}</div>"
        )
    return "<div style='display:flex;flex-wrap:wrap;gap:0;padding:2px'>" + "".join(cards) + "</div>"


# ─────────────────────────────────────────────────────────────────────────────
# Panel output helpers
# ─────────────────────────────────────────────────────────────────────────────

def _slots_html(slots: List) -> str:
    """Compact HTML grid for 5 ref slots (replaces 5×gr.Image + 5×gr.Textbox)."""
    parts = []
    padded = list(slots) + [None] * (N_SLOTS - len(slots))
    for slabel, slot in zip(SLOT_LABELS, padded[:N_SLOTS]):
        if slot and slot.get("tid"):
            p, nm = _template_info(slot["tid"])
            b64 = _img_b64(p)
            if b64:
                img = (f'<img src="{b64}" style="width:100%;height:70px;'
                       f'object-fit:cover;border-radius:4px;display:block">')
            else:
                img = '<div style="height:70px;background:#1a2035;border-radius:4px;display:flex;align-items:center;justify-content:center;color:#4a5568;font-size:10px">no img</div>'
            label = (nm or "—")[:28]
        else:
            img = ('<div style="height:70px;background:#1a2035;border-radius:4px;'
                   'display:flex;align-items:center;justify-content:center;'
                   'color:#4a5568;font-size:10px">empty</div>')
            label = "—"
        parts.append(
            f'<div style="flex:1;min-width:86px;padding:0 3px">'
            f'<div style="font-size:10px;color:#a0aec0;font-weight:600;margin-bottom:2px">{slabel}</div>'
            f'{img}'
            f'<div style="font-size:10px;color:#718096;margin-top:2px;'
            f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{label}</div>'
            f'</div>'
        )
    return f'<div style="display:flex;gap:4px;padding:2px 0">{"".join(parts)}</div>'


def _slots_html_from_sb(sb: Dict) -> str:
    """Generate slot grid HTML from current storyboard panel."""
    cur, pn = sb.get("cur_page", 1), sb.get("cur_panel", 1)
    panel   = _get_panel(sb, cur, pn)
    return _slots_html(_panel_slots(panel))


def _pg_set_html(path: Optional[str], name: Optional[str]) -> str:
    """Compact HTML display for the page setting ref (replaces gr.Image + gr.Textbox)."""
    b64 = _img_b64(path)
    if b64:
        return (f'<div style="display:flex;align-items:center;gap:8px;padding:2px 0">'
                f'<img src="{b64}" style="height:54px;width:54px;object-fit:cover;'
                f'border-radius:4px;flex-shrink:0">'
                f'<span style="font-size:11px;color:#a0aec0">{name or "—"}</span>'
                f'</div>')
    return '<div style="font-size:11px;color:#4a5568;padding:2px 0">— auto (none assigned)</div>'


def _load_panel(sb: Dict):
    """10 items: panel_nav_lbl, beat_dd, beat_txt, char_html, beat_add_dd, slots_html,
    suggest_gallery, suggest_ids, cast_quick_gallery, cast_quick_names"""
    cur_page  = sb.get("cur_page",  1)
    cur_panel = sb.get("cur_panel", 1)
    pz        = max(1, sb.get("page_size", 6))
    beats     = sb.get("beats") or []
    bc        = sb.get("beat_chars") or {}
    np        = _n_pages(sb)

    nav   = (f"Panel {cur_panel}/{pz}  ·  Page {cur_page}/{np}" if beats else "Parse a story first")
    panel = _get_panel(sb, cur_page, cur_panel)
    slots = _panel_slots(panel)
    bidx  = panel.get("beat_idx") or _panel_beat_idx(cur_page, cur_panel, pz)
    btxt  = _panel_beat_text(beats, cur_page, cur_panel, pz, override_idx=panel.get("beat_idx") or 0)

    bchoices = _beat_choices(sb)
    sel      = next((c for c in bchoices if c.startswith(f"{bidx}:")), bchoices[0])

    auto_ch = bc.get(str(bidx), [])
    man_ch  = panel.get("chars") or []
    shown   = list(dict.fromkeys(man_ch if man_ch else auto_ch))
    if shown:
        tags     = "".join(
            f"<span style='background:#2d3748;border:1px solid #4a5568;border-radius:3px;"
            f"padding:1px 7px;margin:2px;font-size:12px;color:#e2e8f0'>{c}</span>"
            for c in shown
        )
        char_html = f"<div style='padding:2px 0'>👤 {tags}</div>"
    else:
        char_html = "<div style='color:#555;font-size:12px;padding:2px 0'>No characters detected for this beat.</div>"

    # Suggestions pre-computed during parse/enrich; fall back to on-demand if missing
    sugg_tids = (sb.get("panel_suggestions") or {}).get(str(bidx))
    if sugg_tids is None:
        _, sugg_tids = _suggest_for_beat(btxt, n=20)
    sugg_imgs, sugg_ids = [], []
    for tid in sugg_tids:
        p, nm = _template_info(tid)
        if p:
            sugg_imgs.append((p, nm))
            sugg_ids.append(tid)

    cqg, cqn = _cast_quick_gallery(sb)
    # Choices for beat_add_dd: cast members NOT already shown in this beat
    add_choices = [c for c in (sb.get("cast") or {}) if c not in set(shown)]
    return [
        gr.update(value=nav),
        gr.update(choices=bchoices, value=sel),
        gr.update(value=btxt),
        gr.update(value=char_html),
        gr.update(choices=add_choices, value=None),  # beat_add_dd
        gr.update(value=_slots_html(slots)),           # slots_html (replaces 5×Image + 5×Textbox)
        gr.Gallery(value=sugg_imgs), sugg_ids, gr.Gallery(value=cqg), cqn,
    ]


def _load_page_header(sb: Dict):
    """2 items: page_lbl, pg_set_html"""
    cur = sb.get("cur_page", 1)
    np  = _n_pages(sb)
    lbl = f"Page {cur}/{np}  ·  {sb.get('page_size', 6)} panels/page" if sb.get("n_beats") else "—"
    pd  = (sb.get("page_refs") or {}).get(str(cur), {})
    pp, pn = _template_info(pd.get("page_setting"))
    return [gr.update(value=lbl), gr.update(value=_pg_set_html(pp, pn))]


def _cast_choices(sb: Dict) -> List[str]:
    return list((sb.get("cast") or {}).keys())


# ─────────────────────────────────────────────────────────────────────────────
# Storyboard persistence (save / load from projects dir)
# ─────────────────────────────────────────────────────────────────────────────

_SB_PROJECTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "projects")

def _sb_project_dir(project_id: str) -> str:
    return os.path.join(_SB_PROJECTS_DIR, project_id)

def _sb_save(sb: Dict) -> str:
    """Persist current storyboard state to disk. Returns project_id.
    Thread-safe: parallel Auto-Run workers may save concurrently."""
    import json as _json, time as _time, uuid as _uuid
    pid = sb.get("project_id") or ""
    if not pid:
        pid = _time.strftime("%Y%m%d_%H%M%S") + "_sb_" + _uuid.uuid4().hex[:6]
        sb["project_id"] = pid
    proj_dir = _sb_project_dir(pid)
    os.makedirs(proj_dir, exist_ok=True)
    # Save storyboard state (exclude non-serialisable Gradio objects)
    with _SB_SAVE_LOCK:
        final_path = os.path.join(proj_dir, "storyboard.json")
        tmp_path   = final_path + ".tmp"
        for _try in range(3):
            try:
                safe = {k: v for k, v in sb.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
                # Atomic write: full dump to temp file, then rename — a crash or
                # concurrent-mutation error mid-dump never corrupts the real file.
                with open(tmp_path, "w", encoding="utf-8") as f:
                    _json.dump(safe, f, indent=2, ensure_ascii=False)
                os.replace(tmp_path, final_path)
                break
            except RuntimeError:
                # dict mutated mid-serialisation by another worker — retry
                if _try == 2:
                    raise
                _time.sleep(0.1)
    # Also write a minimal project.json so the Projects tab can list it
    pj_path = os.path.join(proj_dir, "project.json")
    if not os.path.exists(pj_path):
        with open(pj_path, "w", encoding="utf-8") as f:
            _json.dump({
                "project_id":   pid,
                "project_name": sb.get("story_name") or pid,
                "story":        (sb.get("story") or "")[:500],
                "source":       "storyboard",
            }, f, indent=2, ensure_ascii=False)
    return pid

def _sb_list() -> List[Tuple[str, str]]:
    """Return [(project_id, display_name), ...] for all storyboard projects."""
    import json as _json
    out = []
    if not os.path.isdir(_SB_PROJECTS_DIR):
        return out
    for pid in sorted(os.listdir(_SB_PROJECTS_DIR), reverse=True):
        sb_path = os.path.join(_SB_PROJECTS_DIR, pid, "storyboard.json")
        if not os.path.isfile(sb_path):
            continue
        try:
            with open(sb_path, "r", encoding="utf-8") as f:
                data = _json.load(f)
            name = data.get("story_name") or pid
            out.append((pid, f"{name}  [{pid[:16]}]"))
        except Exception:
            out.append((pid, pid))
    return out

def _sb_load(project_id: str) -> Optional[Dict]:
    """Load storyboard state from disk. Returns None on failure."""
    import json as _json
    sb_path = os.path.join(_SB_PROJECTS_DIR, project_id, "storyboard.json")
    if not os.path.isfile(sb_path):
        return None
    try:
        with open(sb_path, "r", encoding="utf-8") as f:
            return _json.load(f)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Panel reasoning  ("Why this panel was built this way")
# ─────────────────────────────────────────────────────────────────────────────

def _build_panel_why(sb: Dict) -> str:
    """Plain-English breakdown of every panel on the current page."""
    import re as _re
    cur  = sb.get("cur_page",  1)
    pz   = max(1, sb.get("page_size", 6))
    np   = _n_pages(sb)
    beats = sb.get("beats") or []
    story = sb.get("story", "")

    # Split story into sentences for proportional lookup
    sents = [s.strip() for s in _re.split(r'(?<=[.!?…])\s+', story) if s.strip()] if story else []

    def _story_excerpt(beat_1idx: int) -> str:
        if not sents or not beats:
            return ""
        frac   = (beat_1idx - 1) / max(len(beats) - 1, 1)
        sent_j = min(int(round(frac * (len(sents) - 1))), len(sents) - 1)
        s = sents[sent_j]
        return (s[:280] + "…") if len(s) > 280 else s

    header = f"PAGE {cur}/{np} — editorial breakdown"
    parts  = [header, "=" * len(header)]

    cast = sb.get("cast") or {}
    assigned_cast = [(nm, tid) for nm, tid in cast.items() if tid]
    if assigned_cast:
        parts.append("")
        parts.append("🎭 Cast face refs assigned:")
        for nm, tid in assigned_cast:
            _, tname = _template_info(tid)
            parts.append(f"  {nm} → {tname}")

    for pi in range(pz):
        pn      = pi + 1
        bidx    = _panel_beat_idx(cur, pn, pz)
        panel   = _get_panel(sb, cur, pn)
        real_bidx = panel.get("beat_idx") or bidx
        btxt    = _panel_beat_text(beats, cur, pn, pz, override_idx=panel.get("beat_idx") or 0)

        parts.append("")
        parts.append(f"━━━ Panel {pn}  (beat {real_bidx}) ━━━")

        orig = _story_excerpt(real_bidx)
        if orig:
            parts.append(f'📖 Story:  "{orig}"')
        if btxt and btxt.strip() != orig.strip():
            parts.append(f"🎬 Beat:  {btxt[:240]}")

        # Slot refs
        slots = _panel_slots(panel)
        ref_lines = []
        for si, slot in enumerate(slots):
            if slot and slot.get("tid"):
                _, nm = _template_info(slot["tid"])
                ref_lines.append(f"  {SLOT_LABELS[si]}: {nm}")
        if ref_lines:
            parts.append("Refs assigned:")
            parts.extend(ref_lines)
        else:
            parts.append("Refs: none — panel generated from text only.")

    return "\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Generation  (rich Tab-2-style prompt)
# ─────────────────────────────────────────────────────────────────────────────

def _fire_fal(sb: Dict, cur: int, quality_preset: str,
              prompt: str, capped_refs: List[Dict], ref_gallery: List, n_all: int,
              ) -> Tuple[Optional[str], str, str, List]:
    """Actually call FAL with the final prompt + refs. Saves result to project dir."""
    # Persist ref gallery paths so they can be restored on page refresh / reload
    sb["last_ref_gallery"] = [(p, lbl) for p, lbl in ref_gallery if p]
    sb.setdefault("page_ref_galleries", {})[str(cur)] = [(p, lbl) for p, lbl in ref_gallery if p]
    try:
        from build import call_fal_generate
        from director import _quality_preset_to_params
        model, steps, guidance = _quality_preset_to_params(quality_preset)
        pil = call_fal_generate(
            prompt, "",
            model=model, num_inference_steps=steps, guidance_scale=guidance,
            skip_esrgan=False, upscale_factor=2,
            reference_images=capped_refs or None,
        )
        pid = sb.get("project_id") or ""
        if pid:
            pages_dir = os.path.join(_SB_PROJECTS_DIR, pid, "pages")
            os.makedirs(pages_dir, exist_ok=True)
            version = len(_page_versions(sb, cur)) + 1
            path = os.path.join(pages_dir, f"page{cur:03d}_v{version:02d}.png")
        else:
            os.makedirs(_SB_OUT, exist_ok=True)
            path = os.path.join(_SB_OUT, f"page{cur:03d}_{int(time.time())}.png")
        pil.save(path, format="PNG")
        # Preserve the exact prompt/model used for this specific generated
        # version. page_prompts[cur] is editable/latest-only and cannot safely
        # represent older downloaded versions.
        prompt_versions = sb.setdefault("page_prompt_versions", {}).setdefault(str(cur), [])
        prompt_versions.append({
            "path": path,
            "prompt": prompt,
            "model": quality_preset,
            "version": version if pid else len(prompt_versions) + 1,
        })
        # Store as a list so multiple versions per page are preserved
        existing = sb.setdefault("generated_pages", {}).get(str(cur))
        if existing is None:
            sb["generated_pages"][str(cur)] = [path]
        elif isinstance(existing, list):
            existing.append(path)
        else:
            sb["generated_pages"][str(cur)] = [existing, path]
        return path, f"✅ Page {cur} generated · {n_all} unique refs sent.", prompt, ref_gallery
    except Exception as e:
        import traceback; traceback.print_exc()
        return None, f"❌ {e}", prompt, ref_gallery


def _generate_page(sb: Dict, quality_preset: str,
                   send_to_fal: bool = True,
                   prompt_override: Optional[str] = None,
                   ) -> Tuple[Optional[str], str, str, List]:
    cur, pz      = sb.get("cur_page", 1), max(1, sb.get("page_size", 6))
    beats        = sb.get("beats") or []
    cast         = sb.get("cast") or {}
    pd           = (sb.get("page_refs") or {}).get(str(cur), {})
    world_ctx    = (sb.get("world_context") or "").strip()
    if not beats:
        return None, "❌ Parse a story first.", "", []

    # ── Step 1: collect every tid needed for this page ────────────────────────
    # Maps tid → list of (role, panel_number) so we know what each tid is used for.
    needed_tids: Dict[str, List[Tuple[str, int]]] = {}

    pg_loc_tid = pd.get("page_setting")
    if pg_loc_tid:
        needed_tids.setdefault(pg_loc_tid, []).append(("location", 0))

    for cname, tid in cast.items():
        if tid:
            needed_tids.setdefault(tid, []).append(("face_cast", 0))

    for pi in range(pz):
        pn    = pi + 1
        panel = _get_panel(sb, cur, pn)
        for slot in _panel_slots(panel):
            if slot and slot.get("tid"):
                needed_tids.setdefault(slot["tid"], []).append((slot["role"], pn))

    # ── Step 2: parallel-upload any tids that don't yet have a FAL URL ────────
    import concurrent.futures as _cf2
    from character_library import get_fal_urls, get_template, update_template, image_path as _img_path

    def _ensure_fal_url(tid: str) -> Tuple[str, str]:
        """Return (tid, fal_url). Uploads if needed."""
        try:
            fu, bu = get_fal_urls(tid)
            if fu or bu:
                return tid, (fu or bu)
            t = get_template(tid)
            if not t:
                return tid, ""
            from build import upload_pil_to_fal
            from PIL import Image as _PILImg
            for img_type in ("face", "body"):
                local = t.get(f"local_{img_type}") or _img_path(tid, img_type)
                if local and os.path.isfile(local):
                    try:
                        pil = _PILImg.open(local).convert("RGB")
                        url = upload_pil_to_fal(pil)
                        update_template(tid, **{f"fal_{img_type}_url": url})
                        return tid, url
                    except Exception:
                        pass
        except Exception:
            pass
        return tid, ""

    tid_url: Dict[str, str] = {}
    tids_to_check = list(needed_tids.keys())
    if tids_to_check:
        _up_ex = _cf2.ThreadPoolExecutor(max_workers=4)
        try:
            futs = {_up_ex.submit(_ensure_fal_url, tid): tid for tid in tids_to_check}
            try:
                for fut in _cf2.as_completed(futs, timeout=90):
                    try:
                        t, u = fut.result()
                        tid_url[t] = u
                    except Exception:
                        pass
            except _cf2.TimeoutError:
                # Collect whatever completed within the timeout and move on
                for fut, tid in futs.items():
                    if fut.done():
                        try:
                            t, u = fut.result()
                            tid_url[t] = u
                        except Exception:
                            pass
        finally:
            _up_ex.shutdown(wait=False, cancel_futures=True)

    # ── Step 3: build ref lists using the now-available URLs ──────────────────
    all_refs: List[Dict]           = []
    ref_role_lines: List[str]      = []
    tid_to_ref: Dict[str, int]     = {}
    face_tids:  set                = set()
    panel_map:  Dict[int, Dict]    = {}
    loc_ref_nums: List[int]        = []
    ref_local_paths: List[Tuple[str, str]] = []

    def _add(url: str, tag: str, role_line: str,
             local_path: Optional[str] = None, label: str = "", tid: Optional[str] = None) -> int:
        if tid and tid in tid_to_ref:
            return tid_to_ref[tid]
        all_refs.append({"url": url, "tag": tag})
        ref_role_lines.append(role_line)
        ref_local_paths.append((local_path or "", label or role_line[:50]))
        n = len(all_refs)
        if tid:
            tid_to_ref[tid] = n
        return n

    def _url(tid: str) -> str:
        return tid_url.get(tid, "")

    # Page-level location
    if pg_loc_tid:
        u = _url(pg_loc_tid)
        if u:
            p, nm = _template_info(pg_loc_tid)
            dna   = _template_dna(pg_loc_tid)
            tags  = _template_tags_str(pg_loc_tid)
            env_desc = dna if dna else (f"[{tags}]" if tags else "the environment in this image")
            num   = _add(
                u, "composition",
                f"Image {{N}} — ALL PANELS setting/architecture only. "
                f"{env_desc} "
                f"Use ONLY the background environment, architecture, and ambient lighting from this image. "
                f"All people in this image are invisible — never copy their face, clothing, or identity.",
                p, f"🏙 Page setting: {nm}", pg_loc_tid,
            )
            loc_ref_nums.append(num)

    # Cast face refs
    cast_char_to_num: Dict[str, int] = {}
    for cname, tid in cast.items():
        if not tid:
            continue
        u = _url(tid)
        if not u:
            continue
        p, nm  = _template_info(tid)
        # Face-only: strip pose/clothing/setting from the reference description
        # so the model locks the face geometry without copying the reference outfit
        dna   = _face_only_dna(tid)
        dna_s = f" {dna}" if dna else ""
        num   = _add(
            u, "face",
            f"Image {{N}} — {cname} identity only (face).{dna_s} "
            f"Shown bust shot three quarter front view standing or seated, relaxed. "
            f"Setting: neutral background, studio-like environment, warm lighting. "
            f"Lock this exact face shape, hair color/style, eye color, and skin tone in every panel. "
            f"Do not redesign or age this character between panels.",
            p, f"🎭 {cname} face", tid,
        )
        face_tids.add(tid)
        cast_char_to_num[cname] = num

    # Per-panel slot refs
    def _slot_role_line(role: str, pn: int, nm: str, dna: str, tags: str,
                        tid: Optional[str] = None) -> str:
        # For face slots, always use face-only DNA (strips pose/clothing/setting)
        if role == "face" and tid:
            face_dna  = _face_only_dna(tid)
            dna_s = f" {face_dna}" if face_dna else ""
        else:
            dna_s = f" {dna}" if dna else (f" [{tags}]" if tags else "")
        if role == "face":
            return (
                f"Image {{N}} — Panel {pn} character identity only (face).{dna_s} "
                f"Shown bust shot three quarter front view standing or seated, relaxed. "
                f"Setting: neutral background, studio-like environment, warm lighting. "
                f"Lock this exact face shape, hair color/style, eye color, and skin tone in every panel. "
                f"Do not redesign or age this character between panels."
            )
        if role == "location":
            return (
                f"Image {{N}} — Panel {pn} setting/architecture only.{dna_s} "
                f"Use ONLY the background environment, architecture, and ambient lighting. "
                f"All people in this image are invisible — never copy their face, clothing, or identity."
            )
        if role == "camera":
            return (
                f"Image {{N}} — Panel {pn} camera framing only.{dna_s} "
                f"Match the shot type, camera angle, distance, and composition from this image. "
                f"Do not copy faces, clothing, or identities of people shown."
            )
        if role == "mood":
            return (
                f"Image {{N}} — Panel {pn} mood/lighting reference only.{dna_s} "
                f"Reproduce the color palette, lighting quality, shadow direction, and emotional atmosphere. "
                f"Do not copy faces, clothing, or identities of people shown."
            )
        if role == "action":
            return (
                f"Image {{N}} — Panel {pn} body POSE SKELETON only — trace the joint positions, "
                f"limb angles, body angle, and weight distribution ONLY. "
                f"The reference figure's skin coverage, clothing material, props, setting decor, "
                f"and expression are 100% invisible — do NOT copy them. "
                f"Replace the figure with the story character wearing their own period-appropriate outfit."
            )
        return f"Image {{N}} — Panel {pn} {_ROLE_SHORT.get(role, role)}.{dna_s}"

    for pi in range(pz):
        pn    = pi + 1
        panel = _get_panel(sb, cur, pn)
        slots = _panel_slots(panel)
        bkt: Dict[str, List[int]] = {}
        ploc: List[int] = []
        for slot in slots:
            if not slot or not slot.get("tid"):
                continue
            tid, role = slot["tid"], slot["role"]
            u = _url(tid)
            if not u:
                continue
            p, nm   = _template_info(tid)
            dna     = _template_dna(tid)
            tags    = _template_tags_str(tid)
            fal_tag = "face" if role == "face" else "composition"
            if role == "face":
                if tid not in face_tids:
                    num = _add(u, fal_tag, _slot_role_line(role, pn, nm, dna, tags, tid=tid),
                               p, f"🎭 P{pn} face: {nm}", tid)
                    face_tids.add(tid)
                else:
                    num = tid_to_ref[tid]
                bkt.setdefault("face", []).append(num)
            elif role == "location":
                num = _add(u, fal_tag, _slot_role_line(role, pn, nm, dna, tags),
                           p, f"🏙 P{pn} location: {nm}", tid)
                ploc.append(num)
            else:
                icon = "📐" if role == "camera" else "💡" if role == "mood" else "🧍"
                num = _add(u, fal_tag, _slot_role_line(role, pn, nm, dna, tags),
                           p, f"{icon} P{pn} {role}: {nm}", tid)
                bkt.setdefault(role, []).append(num)
        if bkt or ploc:
            panel_map[pi] = {**bkt, **({"location": ploc} if ploc else {})}

    if not all_refs:
        return None, "❌ No refs assigned. Add characters to Cast or assign images to panel slots.", "", []

    # ── Fix {N} placeholders to real numbers ─────────────────────────────────
    ref_lines_final = [line.replace("{N}", str(i)) for i, line in enumerate(ref_role_lines, 1)]

    # ── Build REFERENCE ROLES header ─────────────────────────────────────────
    hdr_lines = [f"{_DIV} REFERENCE ROLES {_DIV}"] + ref_lines_final
    if face_tids:
        hdr_lines += [
            "",
            "FACE LOCK RULE: Every character's face must be an exact copy of their identity image in "
            "every single panel — hair color, hair style, eye color, eye shape, face shape, skin tone, "
            "and all distinguishing features. Do not invent, age, or redesign any character.",
            "MANNEQUIN RULE: People shown in camera/mood/pose/action/location references are completely "
            "invisible mannequins — never copy their face, hair, skin tone, clothing, accessories, or identity. "
            "For body-pose references treat the figure as a skeleton wireframe only: copy joint positions and "
            "limb angles, then dress the skeleton in the story character's own period-appropriate outfit.",
            "LANGUAGE: All text, signs, and captions must be in ENGLISH.",
        ]

    # ── Art style line ────────────────────────────────────────────────────────
    rows    = (pz + 1) // 2
    art_line = (
        f"Korean manhwa webtoon art style, bold black ink outlines with thick-thin weight variation, "
        f"hard-edged cel shading flat color fills sharp shadow cutoffs, vivid saturated palette warm-cool contrast, "
        f"large luminous eyes gradient iris sharp catchlight, 9:16 vertical page, {pz} panels in a "
        f"2-column {rows}-row grid, white gutters, left-right top-bottom reading, clean 2D Korean webtoon "
        f"illustration not photorealistic, ALL figures and objects fully contained within their panel "
        f"boundaries — no character or limb crosses a panel border."
    )

    # ── Per-panel blocks ──────────────────────────────────────────────────────
    panel_lines = []
    for pi in range(pz):
        pn    = pi + 1
        panel = _get_panel(sb, cur, pn)
        bidx  = panel.get("beat_idx") or _panel_beat_idx(cur, pn, pz)
        btxt  = _expand_shot(_panel_beat_text(beats, cur, pn, pz, override_idx=panel.get("beat_idx") or 0)) or f"Panel {pn}."
        chars = panel.get("chars") or (sb.get("beat_chars") or {}).get(str(bidx), [])
        if not chars:
            # First-person narration ("I", "my") never names the protagonist, so
            # beat_chars misses them — inject the protagonist so their DNA/face
            # lock still lands in the panel block (prevents face drift).
            _known = list((sb.get("char_dna") or {}).keys()) or list(cast.keys())
            if _known and re.search(r"\b(I|I'm|I'd|I'll|me|my|mine|myself)\b", btxt):
                chars = [_known[0]]
        bkt   = panel_map.get(pi, {})
        loc_n = bkt.get("location") or loc_ref_nums

        ref_parts = []
        for stype in ("camera", "mood", "action"):
            ns = bkt.get(stype, [])
            if ns:
                ref_parts.append(f"Use {', '.join(f'Image {x}' for x in ns)} ({_ROLE_PANEL[stype]}).")
        if loc_n:
            ref_parts.append(f"Use {', '.join(f'Image {x}' for x in loc_n)} ({_ROLE_PANEL['location']}).")
        # Add face ref reminder for each character in this panel — with explicit DNA
        for cname in chars:
            fn = cast_char_to_num.get(cname)
            if fn:
                tid_ref      = cast.get(cname)
                dna_explicit = (_face_only_dna(tid_ref) or "").strip()
                dna_clause   = f" {dna_explicit}." if dna_explicit else ""
                ref_parts.append(
                    f"Draw {cname} matching Image {fn} face exactly —{dna_clause} "
                    f"Lock every feature: do NOT change hair color, eye color, skin tone, or face shape."
                )

        char_lines = []
        char_dna_map        = sb.get("char_dna") or {}
        char_class_map      = sb.get("char_class") or {}
        char_appearance_map = sb.get("char_appearance") or {}
        for cname in chars:
            tid      = cast.get(cname)
            cls_name = char_class_map.get(cname)
            # Priority: user-written appearance note > face ref DNA > AI-generated char DNA
            user_note = (char_appearance_map.get(cname) or "").strip()
            if user_note:
                face_desc = user_note
            elif tid:
                # Face-only DNA — strips pose/clothing so reference outfit never bleeds in
                face_desc = _face_only_dna(tid)
                # Skin tone is the most-missed attribute — ensure it's explicit even if DNA stripped it
                if "skin" not in face_desc.lower():
                    try:
                        import character_library as _cl_sk
                        _t_sk = (_cl_sk.load_library().get("templates") or {}).get(tid, {})
                        _ai_sk = _t_sk.get("ai_analysis") or {}
                        if isinstance(_ai_sk, str):
                            import ast as _ast_sk
                            try: _ai_sk = _ast_sk.literal_eval(_ai_sk)
                            except: _ai_sk = {}
                        _skin = (_ai_sk.get("skin_tone") or _ai_sk.get("complexion") or
                                 _t_sk.get("skin_tone") or "").strip()
                        if _skin:
                            face_desc = (f"{face_desc}. {_skin}" if face_desc else _skin)
                    except Exception:
                        pass
            else:
                face_desc = char_dna_map.get(cname, "")

            # Clothing comes from the assigned clothing class, never from the reference photo
            cloth_override = ""
            if cls_name:
                try:
                    import character_library as _cl
                    # seed_key=character name → the SAME outfit variation every panel
                    # and every page (was random per call = outfit drift mid-page)
                    cloth_dna = _cl.get_class_clothing_dna(cls_name, seed_key=cname)
                    if cloth_dna:
                        cloth_override = f"{cls_name} clothing: {cloth_dna}"
                except Exception:
                    pass

            # Build: face appearance + explicit clothing (class beats reference photo)
            if face_desc and cloth_override:
                dna = f"{face_desc}. {cloth_override}"
            elif cloth_override:
                dna = cloth_override
            else:
                dna = face_desc

            char_lines.append(f"{cname} — {dna}" if dna else cname)

        # Scene emotion: override reference-image expression with scene-appropriate feeling
        emotion = _scene_emotion(btxt)

        block = f"PANEL {pn} : "
        if ref_parts:
            block += "\n".join(ref_parts) + "\n"
        if char_lines:
            block += "\n".join(char_lines) + "\n"
        if emotion and chars:
            cnames = " & ".join(chars[:4])
            block += (
                f"EXPRESSION OVERRIDE — {cnames}: {emotion} "
                f"The reference image provides face shape/hair/eyes ONLY — "
                f"do NOT copy the reference photo's smile or mood expression.\n"
            )
        # Cinematic shot recipe + color script (panel override > auto-assigned per beat)
        recipe_name = (panel.get("recipe") or "").strip() or (sb.get("beat_recipes") or {}).get(str(bidx), "")
        if recipe_name:
            try:
                import character_library as _cl_rp
                _rp = (_cl_rp.get_shot_recipes().get(recipe_name) or {}).get("prompt", "").strip()
                if _rp:
                    block += f"SHOT STYLE: {_rp[:450]}\n"
            except Exception:
                pass
        color_hue = (sb.get("beat_colors") or {}).get(str(bidx), "")
        if color_hue:
            block += f"COLOR MOOD: dominant palette for this panel — {color_hue}\n"
        block += btxt
        panel_lines.append(block)

    # ── Character sheet — verbatim appearance DNA for every known character ──
    # Anchors identity even when a panel has no face-ref image (prevents the
    # protagonist's face drifting from page to page).
    _sheet_lines: List[str] = []
    _dna_all  = sb.get("char_dna") or {}
    _appr_all = sb.get("char_appearance") or {}
    _sheet_names = list(dict.fromkeys(list(_dna_all.keys()) + list(cast.keys())))[:6]
    for _nm in _sheet_names:
        _desc = (_appr_all.get(_nm) or "").strip() or (_dna_all.get(_nm) or "").strip()
        if _desc:
            _sheet_lines.append(f"• {_nm}: {_desc}")
    sheet_block = ""
    if _sheet_lines:
        sheet_block = (
            f"\n{_DIV} CHARACTER SHEET {_DIV}\n"
            "Draw each character below EXACTLY as described, IDENTICALLY in every panel "
            "and identical to previous pages — same face, hair color/style, eye color, "
            "skin tone, and build. Never redesign or age them.\n"
            + "\n".join(_sheet_lines)
        )

    # ── Assemble full prompt ──────────────────────────────────────────────────
    full_prompt = (
        "\n".join(hdr_lines)
        + sheet_block
        + "\n" + art_line
        + "\n" + "\n".join(panel_lines)
        + f"\n{_DIV} GLOBAL QUALITY PRIORITIES {_DIV}\n"
        "1. FACE IDENTITY — reproduce each character's exact face from their reference image in every panel.\n"
        "2. EXPRESSION — each character's expression MUST match the scene beat, NOT the reference photo. "
        "Grief scenes: grief-stricken and devastated. Action scenes: focused and intense. "
        "NEVER draw a smiling face when the beat describes sorrow, grief, fear, or tension.\n"
        "3. Apply all reference images exactly as instructed in each panel block.\n"
        "4. Follow camera angle, shot type, and character arrangement per panel.\n"
        "5. Render clear, readable story action.\n"
        "6. Premium facial quality: refined anatomy, gradient irises, sharp eye highlights, individual hair strands.\n"
        "7. Simplify background figures before touching main characters.\n"
        + _ART_STYLE
        + _PANEL_RULES
        + f"\n{_DIV} NEGATIVE PROMPT {_DIV}\n{_NEG}"
    )

    sb["last_prompt"] = full_prompt
    sb.setdefault("page_prompts", {})[str(cur)] = full_prompt  # per-page so navigation shows right prompt

    # ── Prev-page panel crops (page ≥ 2) ─────────────────────────────────────
    # Crop the last 3 panels from the previous page's generated image to anchor
    # character appearance across page boundaries and show them in the ref gallery.
    _prev_page_crops: List = []   # list of (PIL crop, tmp_path, label)
    if cur > 1:
        _prev_gen = (sb.get("generated_pages") or {}).get(str(cur - 1))
        if _prev_gen:
            _prev_path = (_prev_gen[-1] if isinstance(_prev_gen, list) else _prev_gen)
            if _prev_path and os.path.isfile(str(_prev_path)):
                try:
                    from PIL import Image as _PPI
                    _pim   = _PPI.open(_prev_path).convert("RGB")
                    _ncols = 2
                    _nrows = (pz + _ncols - 1) // _ncols
                    _cw    = _pim.width  // _ncols
                    _ch    = _pim.height // _nrows
                    _n_take = min(3, pz)
                    for _pi in range(pz - _n_take, pz):
                        _r    = _pi // _ncols; _cc = _pi % _ncols
                        _crop = _pim.crop((_cc*_cw, _r*_ch, (_cc+1)*_cw, (_r+1)*_ch))
                        _tmp  = f"/tmp/sb_pp{cur-1}_pn{_pi+1}.png"
                        _crop.save(_tmp)
                        _lbl  = f"🔗 Prev page {cur-1} panel {_pi+1}"
                        ref_local_paths.append((_tmp, _lbl))
                        _prev_page_crops.append((_crop, _tmp, _lbl))
                except Exception as _pex:
                    print(f"[prev page crops] {_pex}", flush=True)

    # Build numbered reference gallery — "Image 1 — 🎭 Protagonist face" etc.
    ref_gallery = [
        (p, f"Image {i+1} — {lbl}")
        for i, (p, lbl) in enumerate(ref_local_paths)
        if p and os.path.isfile(p)
    ]

    # Cap refs: face first, then others, max 20
    _MAX_REFS = 20
    capped_refs = all_refs
    if len(all_refs) > _MAX_REFS:
        face_refs  = [r for r in all_refs if r.get("tag") == "face"]
        other_refs = [r for r in all_refs if r.get("tag") != "face"]
        capped_refs = (face_refs + other_refs)[:_MAX_REFS]

    if not send_to_fal:
        # Prompt-only mode — show gallery with prev-page crops, skip FAL upload
        sb["last_ref_gallery"] = [(p, lbl) for p, lbl in ref_gallery if p]
        sb.setdefault("page_ref_galleries", {})[str(cur)] = [(p, lbl) for p, lbl in ref_gallery if p]
        return None, f"📋 Prompt ready · {len(all_refs)} refs prepared. Edit if needed, then click ▶ Send to FAL.", full_prompt, ref_gallery

    # Upload prev-page crops to FAL and append to capped_refs for continuity anchoring
    if _prev_page_crops:
        try:
            from build import upload_pil_to_fal as _ufal
            for _crop_pil, _tmp, _lbl in _prev_page_crops:
                _url = _ufal(_crop_pil)
                capped_refs = list(capped_refs) + [{"url": _url, "tag": "character"}]
        except Exception as _upe:
            print(f"[prev page FAL upload] {_upe}", flush=True)

    return _fire_fal(sb, cur, quality_preset, prompt_override or full_prompt, capped_refs, ref_gallery,
                     len(all_refs) + len(_prev_page_crops))


def _sb_zip(story_name: str, sb: Optional[Dict] = None) -> Tuple[Optional[str], str]:
    """Build a rich ZIP for the storyboard:
    - pages/          — full page images, ordered page_001.png … page_NNN.png
    - panels/         — individual panel crops (gutter-detected)
    - beat_page_map.* — JSON + TXT mapping each page → its beats + beat text
    """
    # ── Collect ordered page paths ────────────────────────────────────────────
    page_entries: List[Tuple[int, str]] = []  # (page_num, path)
    if sb:
        gen_pages = sb.get("generated_pages") or {}
        for pg_key in sorted(gen_pages.keys(), key=lambda k: int(k) if k.isdigit() else 0):
            pg_num = int(pg_key) if pg_key.isdigit() else (len(page_entries) + 1)
            for vi, p in enumerate(_page_versions(sb, pg_key)):
                page_entries.append((pg_num, vi + 1, p))
    # Fallback: scan legacy /tmp output dir
    if not page_entries:
        for i, p in enumerate(sorted(glob.glob(os.path.join(_SB_OUT, "*.png"))), start=1):
            page_entries.append((i, p))
    if not page_entries:
        return None, "❌ No generated images to zip yet."

    beats    = list(sb.get("beats") or []) if sb else []
    pz       = max(1, (sb or {}).get("page_size", 6))
    nm       = re.sub(r"[^\w\-]", "_", (story_name or "storyboard").strip()) or "storyboard"
    zp       = f"/tmp/sb_{nm}_{int(time.time())}.zip"

    # ── Gutter-based panel crop helper ───────────────────────────────────────
    def _crop_panels(img_path: str, n_cols: int = 2, n_rows: int = None):
        """Crop a page into individual panels.
        Preferred: border-aware cutter (snaps grid cuts to the real drawn
        panel borders, trims frames/gutters off the edges).
        Fallback: uniform grid with inset."""
        rows = n_rows or (pz + 1) // 2
        try:
            from panel_cut import smart_cut_panels_path
            smart = smart_cut_panels_path(img_path, n_rows=rows, n_cols=n_cols)
            if len(smart) >= 2:
                return smart
        except Exception as _sce:
            print(f"[smart panel cut] {_sce} — falling back to grid", flush=True)
        try:
            from PIL import Image as _PI
            im  = _PI.open(img_path).convert("RGB")
            w, h = im.size
            cols = n_cols
            panel_w = w // cols
            panel_h = h // rows
            # Inset each crop slightly so white gutter slivers never survive
            # on the panel edges (cut ~1.5% deeper on every side).
            ix = max(6, int(panel_w * 0.015))
            iy = max(6, int(panel_h * 0.015))
            crops = []
            for r in range(rows):
                for c in range(cols):
                    x0 = c * panel_w
                    y0 = r * panel_h
                    x1 = x0 + panel_w
                    y1 = y0 + panel_h
                    crops.append(im.crop((x0 + ix, y0 + iy, x1 - ix, y1 - iy)))
            return crops
        except Exception:
            return []

    # ── Build beat map data ───────────────────────────────────────────────────
    import json as _json
    import io   as _io
    map_rows: List[Dict] = []
    txt_rows: List[str]  = []
    prompt_rows: List[str] = [
        "# Numbered Prompts",
        "",
        "Prompts are numbered in the same order as the generated page images in this archive.",
        "Each Prompt number corresponds to one paid multi-panel page generation.",
        "",
    ]
    prompt_versions = (sb or {}).get("page_prompt_versions") or {}

    def _prompt_for_version(pg_num: int, ver_num: int, path: str) -> Tuple[str, str]:
        """Return the immutable prompt/model for this exact page version.
        Legacy projects may only have a latest-page prompt; use it only when
        there is exactly one exported version, never for ambiguous versions."""
        records = prompt_versions.get(str(pg_num)) or []
        norm_path = os.path.abspath(str(path))
        for rec in records:
            if os.path.abspath(str(rec.get("path") or "")) == norm_path:
                return str(rec.get("prompt") or ""), str(rec.get("model") or "")
        for rec in records:
            if int(rec.get("version") or 0) == ver_num:
                return str(rec.get("prompt") or ""), str(rec.get("model") or "")
        if _ver_totals.get(pg_num, 1) == 1:
            return str(((sb or {}).get("page_prompts") or {}).get(str(pg_num), "")), ""
        return "", ""

    # Count versions per page so we can suffix filenames only when needed
    _ver_totals: Dict[int, int] = {}
    for _pn, _vi, _pp in page_entries:
        _ver_totals[_pn] = _ver_totals.get(_pn, 0) + 1

    with zipfile.ZipFile(zp, "w", zipfile.ZIP_STORED) as zf:
        panel_idx = 0
        for prompt_num, (pg_num, ver_num, path) in enumerate(page_entries, start=1):
            ext     = os.path.splitext(path)[1].lstrip(".") or "png"
            if _ver_totals.get(pg_num, 1) > 1:
                pg_name = f"page_{pg_num:03d}_v{ver_num}.{ext}"
            else:
                pg_name = f"page_{pg_num:03d}.{ext}"
            zf.write(path, f"pages/{pg_name}")
            exact_prompt, exact_model = _prompt_for_version(pg_num, ver_num, path)
            prompt_rows.extend([
                f"## Prompt {prompt_num}",
                "",
                f"- Generated file: `pages/{pg_name}`",
                f"- Storyboard page: {pg_num}",
                f"- Version: {ver_num}",
                *([f"- Model preset: {exact_model}"] if exact_model else []),
                "",
                exact_prompt or "[Prompt unavailable: legacy generated version has no immutable prompt record.]",
                "",
                "---",
                "",
            ])

            # Beats + panel crops — only emit once per page (for the first version)
            if ver_num == 1:
                page_beats = []
                for pn in range(1, pz + 1):
                    bi   = _panel_beat_idx(pg_num, pn, pz)
                    btxt = _panel_beat_text(beats, pg_num, pn, pz)
                    page_beats.append({"beat": bi, "text": btxt})

                map_rows.append({
                    "page":   pg_num,
                    "file":   f"pages/{pg_name}",
                    "beats":  page_beats,
                })
                for pb in page_beats:
                    if pb["text"]:
                        txt_rows.append(f"Page {pg_num:03d} | Beat {pb['beat']:03d} | {pb['text']}")

            # Individual panel crops (all versions)
            v_suffix = f"_v{ver_num}" if _ver_totals.get(pg_num, 1) > 1 else ""
            crops = _crop_panels(path)
            for ci, crop_img in enumerate(crops, start=1):
                panel_idx += 1
                label = f"panel_{pg_num:03d}_{ci:02d}{v_suffix}"
                buf = _io.BytesIO()
                crop_img.save(buf, format="PNG")
                zf.writestr(f"panels/{label}.png", buf.getvalue())

        # Beat-page map files
        zf.writestr(
            "beat_page_map.json",
            _json.dumps(map_rows, indent=2, ensure_ascii=False),
        )
        zf.writestr(
            "beat_page_map.txt",
            "\n".join(txt_rows) if txt_rows else "(no beats)",
        )
        zf.writestr("NUMBERED_PROMPTS.md", "\n".join(prompt_rows))

    return zp, f"✅ {len(page_entries)} pages + {panel_idx} panel crops zipped."


_NONHUMAN_KEYWORDS = frozenset([
    "spirit", "beast", "ancient", "serpent", "dragon", "demon", "god", "deity",
    "creature", "monster", "snake", "wolf", "fox", "phoenix", "hydra", "bird",
    "tiger", "lion", "bear", "cat", "dog", "horse", "ghost", "wraith", "shade",
    "elemental", "golem", "undead", "zombie", "skeleton", "specter", "spectre",
])

def _is_nonhuman(name: str, sb: Dict) -> bool:
    """Return True if this character is a spirit, animal, or non-human creature."""
    name_l = name.lower()
    # Check name words against keyword list
    words = set(name_l.replace("-", " ").split())
    if words & _NONHUMAN_KEYWORDS:
        return True
    # Check their assigned clothing class
    cls = (sb.get("char_class") or {}).get(name, "").lower()
    if cls and any(k in cls for k in _NONHUMAN_KEYWORDS):
        return True
    return False


def _check_prompt_coverage(prompt: str, sb: Dict) -> str:
    """Grade how explicitly each named HUMAN cast character's appearance is described in the prompt.
    Non-human characters (spirits, animals, creatures) are skipped — they don't wear clothes
    and may not have conventional hair/eyes/skin."""
    import re as _rer
    cast = sb.get("cast") or {}
    if not cast or not prompt:
        return ""
    lines = []
    skipped = []
    for name in sorted(cast.keys()):
        if _is_nonhuman(name, sb):
            skipped.append(name)
            continue
        if name.lower() not in prompt.lower():
            continue
        idx = prompt.lower().find(name.lower())
        ctx = prompt[max(0, idx - 30): min(len(prompt), idx + 500)].lower()
        checks = {
            "hair":     bool(_rer.search(r"hair|strand|locks|brunette|blonde|bun|ponytail", ctx)),
            "eyes":     bool(_rer.search(r"eye|iris|gaze|pupils", ctx)),
            "skin":     bool(_rer.search(r"skin|complexion|pale|tan|dark skin|olive|fair", ctx)),
            "clothing": bool(_rer.search(r"wear|cloth|robe|armor|shirt|dress|hanbok|outfit|garment|fabric|jacket", ctx)),
        }
        score  = sum(checks.values())
        grade  = "✅" if score == 4 else "⚠️" if score >= 2 else "❌"
        missing = [k for k, v in checks.items() if not v]
        note   = f" ← missing: {', '.join(missing)}" if missing else ""
        lines.append(f"{grade} {name}: {score}/4{note}")
    if skipped:
        lines.append(f"⬜ skipped (non-human): {', '.join(skipped)}")
    if not lines:
        return "(no human cast characters found in this prompt)"
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Tab builder
# ─────────────────────────────────────────────────────────────────────────────

def build_storyboard_tab(state: gr.State) -> None:  # noqa: C901
    from director import QUALITY_PRESETS, DEFAULT_QUALITY_PRESET

    with gr.Tab("🎬 Storyboard"):
        sb         = gr.State(_empty_sb())
        search_ids = gr.State([])
        recent_ids = gr.State([])

        # ── 1. STORY ─────────────────────────────────────────────────────────
        # Load existing storyboard
        _sb_entries = _sb_list()
        _sb_choices = [label for _, label in _sb_entries]
        _sb_id_map  = {label: pid for pid, label in _sb_entries}
        with gr.Row():
            load_sb_dd  = gr.Dropdown(
                label="📂 Load saved storyboard", choices=_sb_choices, value=None,
                scale=4, interactive=True,
            )
            load_sb_btn   = gr.Button("📂 Load", variant="secondary", scale=1, min_width=80)
            refresh_sb_btn = gr.Button("🔄", variant="secondary", scale=0, min_width=44)
        sb_id_map_state = gr.State(_sb_id_map)

        gr.Markdown("---")

        with gr.Row():
            with gr.Column(scale=3):
                story_name_box = gr.Textbox(
                    label="Story name (used for ZIP filename and project save)",
                    placeholder="My Story", max_lines=1, lines=1,
                )
                story_box = gr.Textbox(
                    label="Story", placeholder="Paste your story here…",
                    lines=6, max_lines=20,
                )
                world_context_box = gr.Textbox(
                    label="🌍 World context  (optional — describe the setting, era, tone, magic system, etc.)",
                    placeholder="e.g. Modern-day Seoul where a hidden ranking system assigns grades to people's hidden power levels. "
                                "High-schoolers wear ranked badges. The protagonist just discovered he is unranked…",
                    lines=3, max_lines=8,
                )
            with gr.Column(scale=1, min_width=180):
                page_size_dd = gr.Dropdown(
                    label="Panels per page",
                    choices=["6  (Shorts — AI beats)", "10  (Panel mode)"],
                    value="6  (Shorts — AI beats)",
                )
                parse_btn      = gr.Button("▶ Parse Story", variant="primary")
                re_parse_btn   = gr.Button("🔄 Regenerate Beats", variant="secondary",
                                           elem_id="re_parse_btn")
                with gr.Row():
                    sb_pages_dd   = gr.Dropdown(
                        label="📄 Pages per run", choices=["1", "5", "10", "20", "50", "100"],
                        value="10", scale=1, interactive=True,
                    )
                    sb_workers_dd = gr.Dropdown(
                        label="⚡ Workers", choices=["1", "2", "3", "4", "6"],
                        value="2", scale=1, interactive=True,
                    )
                with gr.Row():
                    auto_run_btn    = gr.Button("🚀 Auto-Run Full Story", variant="primary", scale=3)
                    pause_run_btn   = gr.Button("⏸ Pause", variant="secondary", scale=1)
                auto_run_status = gr.Textbox(label="Auto-run progress", interactive=False,
                                             lines=2, value="", placeholder="Progress appears here…")
                parse_status   = gr.Textbox(label="", interactive=False, lines=2, value="")

        beats_html_box = gr.HTML(
            value="<p style='color:#555;padding:4px'>Beat list appears here after parsing.</p>",
        )
        story_display = gr.Textbox(
            label="📖 Full story",
            interactive=False, lines=4, max_lines=12,
            placeholder="Original story text shown here after parsing.",
            value="",
        )

        gr.Markdown("---")

        # ── 2. CHARACTERS ────────────────────────────────────────────────────
        gr.Markdown(
            "### 🎭 Characters *(global face refs)*\n"
            "<span style='font-size:12px;color:#888'>"
            "After parsing, characters appear as cards below. Select a character from the dropdown, "
            "search the library, click an image to select it, then click **← Assign** to set their face ref.</span>"
        )
        cast_html_box = gr.HTML(
            value="<p style='color:#555;font-size:12px;padding:4px'>No characters found yet — parse a story first.</p>",
        )
        with gr.Row():
            cast_char_dd  = gr.Dropdown(
                label="Character to assign", choices=[], value=None,
                interactive=True, scale=3,
            )
            cast_set_btn  = gr.Button("← Assign face ref", variant="secondary", scale=2)
            cast_clr_btn  = gr.Button("× Clear",           size="sm",           scale=1)

        # ── Class assignment ──────────────────────────────────────────────────
        def _clothing_class_choices():
            import character_library as _cl
            classes = _cl.get_clothing_classes()
            return ["— none —"] + sorted(classes.keys())

        with gr.Row():
            cls_char_dd  = gr.Dropdown(
                label="Character", choices=[], value=None,
                interactive=True, scale=2,
            )
            cls_class_dd = gr.Dropdown(
                label="Clothing class  (farmer / noble / knight …)",
                choices=_clothing_class_choices(),
                value="— none —", interactive=True, scale=3,
            )
            cls_set_btn  = gr.Button("← Assign class", variant="secondary", scale=2)
            cls_clr_btn  = gr.Button("× Remove class", size="sm",           scale=1)

        gr.Markdown(
            "#### ✏️ Character Appearance\n"
            "<span style='font-size:12px;color:#888'>"
            "Auto-built from this character's DNA + assigned clothing class. "
            "Edit to override — or clear and save to reset back to auto.</span>"
        )
        char_look_box = gr.Textbox(
            label="Appearance (auto-built — edit to override, clear + save to reset)",
            placeholder="Select a character above — their auto-built description appears here.",
            lines=3, max_lines=6, interactive=True,
        )
        with gr.Row():
            save_look_btn = gr.Button("💾 Save / reset", variant="secondary", scale=2)
            look_status   = gr.Textbox(label="", interactive=False, lines=1, scale=3, container=False)

        gr.Markdown("---")

        # ── 3. RECENTLY USED ─────────────────────────────────────────────────
        gr.Markdown(
            "### 🕐 Recently Used  "
            "<span style='font-size:12px;color:#888'>Click any image below to re-select it for assignment.</span>"
        )
        recent_gallery = gr.Gallery(
            label="", value=[], columns=12, rows=1,
            height=120, object_fit="cover", allow_preview=True,
        )

        gr.Markdown("---")

        # ── 4. LIBRARY SEARCH ─────────────────────────────────────────────────
        gr.Markdown("### 🔍 Library Search")
        with gr.Row():
            search_box = gr.Textbox(
                label="", placeholder="forest · sword · old man · Chen Ping · fighting stance…",
                scale=5, container=False,
            )
            search_btn = gr.Button("Search", variant="secondary", scale=1, min_width=90)

        search_gallery = gr.Gallery(
            label="", value=[], columns=8, rows=5,
            height=460, object_fit="cover", allow_preview=True,
        )

        with gr.Row():
            selected_preview = gr.Image(
                label="Selected", height=72, interactive=False,
                show_label=True, min_width=80, scale=0,
            )
            selected_name = gr.Textbox(
                value="Click any image to select it — then use ← Assign or ← Set to place it",
                interactive=False, label="", lines=1, container=False, scale=6,
            )

        gr.Markdown("---")

        # ── 5. PANEL DIRECTOR ─────────────────────────────────────────────────
        gr.Markdown("### 🎬 Panel Director")

        with gr.Row():
            prev_page_btn  = gr.Button("◄ Page",  size="sm", scale=1, min_width=80)
            page_lbl       = gr.Textbox(value="—", interactive=False, label="", scale=2, container=False)
            next_page_btn  = gr.Button("Page ►",  size="sm", scale=1, min_width=80)
            gr.HTML("<div style='width:20px'></div>")
            panel_prev_btn = gr.Button("◄ Panel", size="sm", scale=1, min_width=80)
            panel_nav_lbl  = gr.Textbox(value="—", interactive=False, label="", scale=2, container=False)
            panel_next_btn = gr.Button("Panel ►", size="sm", scale=1, min_width=80)

        with gr.Row(equal_height=False):
            with gr.Column(scale=2, min_width=260):
                beat_dd = gr.Dropdown(
                    label="Map panel → story beat",
                    info="Which beat does this panel illustrate?",
                    choices=["— auto —"], value="— auto —",
                )
                beat_txt = gr.Textbox(
                    label="Full beat text (editable — change it, then Save)",
                    interactive=True, lines=5, max_lines=12, value="",
                )
                save_beat_btn = gr.Button("💾 Save Beat", size="sm", variant="secondary")
                char_html_box = gr.HTML(
                    value="<div style='color:#555;font-size:12px'>Characters in this beat appear here after parsing.</div>",
                )
                with gr.Row():
                    beat_add_dd  = gr.Dropdown(
                        label="Add character to this beat",
                        info="Pick a cast member not yet in this beat",
                        choices=[], value=None, interactive=True, scale=3,
                    )
                    beat_add_btn = gr.Button("＋ Add",   size="sm", variant="secondary", scale=1)
                    beat_rm_btn  = gr.Button("－ Remove", size="sm", variant="secondary", scale=1)
                gr.Markdown(
                    "<span style='font-size:12px;color:#a0aec0'>🎭 **In this beat** — click a face to select that character</span>"
                )
                cast_quick_gallery = gr.Gallery(
                    label="", value=[], columns=5, height=120,
                    interactive=False, allow_preview=False, object_fit="cover",
                )
                cast_quick_names = gr.State([])
                gr.Markdown("##### 📍 Page Setting *(location, optional)*")
                pg_set_html = gr.HTML(value='<div style="font-size:11px;color:#4a5568;padding:2px 0">— auto (none assigned)</div>')
                with gr.Row():
                    pg_set_btn   = gr.Button("← Set", size="sm", scale=2)
                    pg_set_clear = gr.Button("×",     size="sm", scale=1)

            with gr.Column(scale=3):
                gr.Markdown(
                    "**Panel Ref Slots** — select an image above, then click **← Set**\n\n"
                    "_A face · B camera · C mood · D pose · E location_"
                )
                slots_html = gr.HTML(value='<div style="color:#4a5568;font-size:11px;padding:4px 0">No slots set yet.</div>')
                slot_set_btns, slot_clear_btns = [], []
                with gr.Row():
                    for si, slabel in enumerate(SLOT_LABELS):
                        with gr.Column(min_width=80):
                            gr.Markdown(f"<span style='font-size:10px;color:#a0aec0;font-weight:600'>{slabel}</span>")
                            with gr.Row():
                                ss = gr.Button("← Set", size="sm", scale=2)
                                sc = gr.Button("×",     size="sm", scale=1)
                            slot_set_btns.append(ss); slot_clear_btns.append(sc)
                auto_fill_btn = gr.Button("🤖 Auto-fill All Slots", variant="secondary", size="sm")
                auto_fill_status = gr.Textbox(value="", interactive=False, label="", container=False,
                                              placeholder="Auto-fill will pick A–E refs from your library…")

                # ── Panel Suggestions — in the empty space below slot buttons ──
                gr.HTML("<hr style='border-color:#2d3748;margin:8px 0'>")
                with gr.Row():
                    with gr.Column(scale=4, min_width=0):
                        gr.Markdown(
                            "**💡 Panel Suggestions**\n\n"
                            "<span style='font-size:11px;color:#888'>20 picks for this beat — click any to select</span>"
                        )
                    with gr.Column(scale=1, min_width=140):
                        get_suggest_btn = gr.Button("🔍 Refresh", variant="secondary", size="sm")
                suggest_gallery = gr.Gallery(
                    label="", value=[], columns=4,
                    height=340, object_fit="cover", allow_preview=True,
                )

        suggest_ids = gr.State([])

        gr.Markdown("---")

        # ── 6. GENERATE + OUTPUT ──────────────────────────────────────────────
        gr.Markdown("### ⚡ Generate")
        with gr.Row():
            quality_dd = gr.Dropdown(
                label="Model", choices=QUALITY_PRESETS, value=DEFAULT_QUALITY_PRESET, scale=3,
            )
            gen_btn    = gr.Button("⚡ Generate Page", variant="primary", scale=2)
        auto_gen_chk = gr.Checkbox(
            label="Auto-generate — send directly to FAL on click",
            value=False,
            info="Uncheck to review and edit the prompt before sending.",
        )
        send_fal_btn = gr.Button(
            "▶ Send to FAL", variant="primary", visible=True,
        )
        gen_status = gr.Textbox(label="Status", interactive=False, lines=1, value="")

        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                gen_image = gr.Image(label="Generated Page", interactive=False, height=600, type="filepath")
            with gr.Column(scale=3):
                # ── Appearance coverage — above prompt so issues are visible before editing ──
                with gr.Row():
                    re_check_btn     = gr.Button("🔍 Re-check", variant="secondary", size="sm")
                    patch_prompt_btn = gr.Button("🔧 Patch missing terms", variant="primary", size="sm")
                check_log_box = gr.Textbox(
                    label="⏱ Status",
                    interactive=False, lines=2, max_lines=3, value="",
                    placeholder="Status and timing will appear here…",
                )
                prompt_check_box = gr.Textbox(
                    label="✅ Appearance coverage",
                    interactive=True, lines=6, max_lines=20, value="",
                    placeholder="Panel-by-panel results appear here after Re-check or Patch.",
                )
                gr.Markdown(
                    "<span style='font-size:11px;color:#888'>"
                    "The tool reads the prompt panel-by-panel, spots which characters are missing hair / eyes / skin / clothing, "
                    "and **Patch** auto-injects their appearance notes to fix it. "
                    "Repeat until everything is ✅, then click ▶ Send to FAL.</span>"
                )
                prompt_box = gr.Textbox(
                    label="📋 Prompt sent to model",
                    interactive=True, lines=14, max_lines=40, value="",
                    placeholder="Full prompt appears here. Edit freely before clicking Send to FAL.",
                )
                ref_sources_gallery = gr.Gallery(
                    label="🖼 Reference images used (numbers match prompt)",
                    columns=6, rows=2, height=180,
                    object_fit="cover", allow_preview=True, value=[],
                )
                panel_why_box = gr.Textbox(
                    label="💬 Why this page was built this way",
                    interactive=False, lines=10, max_lines=30, value="",
                    placeholder="Editorial breakdown appears here after generating.",
                )

        # ── 6b. STORY TIMELINE ────────────────────────────────────────────────
        gr.Markdown("### 📖 Story Timeline")
        story_timeline = gr.Gallery(
            label="All generated pages — click any page for its beat details",
            value=[], columns=6, height=220,
            interactive=False, allow_preview=True, object_fit="contain",
        )
        timeline_info = gr.Textbox(
            label="Page details",
            interactive=False, lines=3, value="",
            placeholder="Click a page above to see its beat descriptions.",
        )

        gr.Markdown("---")

        # ── 7. DOWNLOAD ───────────────────────────────────────────────────────
        gr.Markdown("### 📦 Download")
        with gr.Row():
            zip_btn  = gr.Button("📦 Zip All Generated Pages", variant="secondary", scale=2)
            zip_file = gr.File(label="Download ZIP", scale=3, interactive=False)
        zip_status = gr.Textbox(label="", interactive=False, lines=1, value="")

        # ── Output lists ──────────────────────────────────────────────────────
        _panel_out   = ([panel_nav_lbl, beat_dd, beat_txt, char_html_box, beat_add_dd,
                         slots_html]
                        + [suggest_gallery, suggest_ids, cast_quick_gallery, cast_quick_names])
        _page_out    = [page_lbl, pg_set_html]
        _all_refresh = (_page_out + _panel_out
                        + [gen_image, prompt_box, ref_sources_gallery,
                           story_timeline])

        # ══════════════════════════════════════════════════════════════════════
        # Callbacks
        # ══════════════════════════════════════════════════════════════════════

        def _do_parse(story, psize_choice, sname, world_ctx):
            pz = 6 if "6" in str(psize_choice) else 10
            if not (story or "").strip():
                return _empty_sb(pz), "", "⚠️ Paste a story first.", ""
            sb_new, msg = _parse_story(story, pz, world_ctx=(world_ctx or "").strip())
            sb_new["story_name"]    = (sname or "").strip() or "Untitled"
            sb_new["world_context"] = (world_ctx or "").strip()
            _sb_save(sb_new)  # auto-save after parsing
            return sb_new, _beats_html(sb_new), msg, (story or "")

        def _do_re_parse(sb_val, psize_choice):
            """Re-run parse + enrich on the existing story, preserving cast/page refs/generated images."""
            story = (sb_val.get("raw_story") or "").strip()
            if not story:
                return (sb_val, _beats_html(sb_val),
                        "⚠️ No story text found — paste one in the Story box first.", "")
            pz       = 6 if "6" in str(psize_choice) else 10
            world_ctx = (sb_val.get("world_context") or "").strip()
            # fields to carry forward from the old storyboard
            keep = {k: sb_val[k] for k in (
                "cast", "clothing_classes", "char_appearance", "char_class",
                "page_refs", "generated_pages", "page_prompts", "page_prompt_versions", "page_ref_galleries",
                "story_name", "world_context", "last_prompt", "last_ref_gallery",
            ) if k in sb_val}
            sb_new, msg = _parse_story(story, pz, world_ctx=world_ctx)
            sb_new.update(keep)   # restore preserved fields
            _sb_save(sb_new)
            return sb_new, _beats_html(sb_new), msg, story

        def _do_enrich(sb_val):
            sb_val, msg = _enrich(sb_val)
            _compute_all_suggestions(sb_val)   # pre-compute for all panels (local search, fast)
            choices = _cast_choices(sb_val)
            default = choices[0] if choices else None
            if default:
                sb_val["selected_char"] = default
            _sb_save(sb_val)  # persist cast + beat_chars + suggestions after enrichment
            return (
                sb_val,
                _beats_html(sb_val),
                msg,
                _cast_html(sb_val),
                gr.update(choices=choices, value=default),
            )

        def _refresh_all(sb_val):
            cur = sb_val.get("cur_page", 1)
            saved_img = _latest_page_img(sb_val, cur)
            # Show this page's prompt, not the globally-last one (prevents wrong-page Send to FAL)
            last_prompt = (sb_val.get("page_prompts") or {}).get(str(cur), "")
            _pg_refs = (sb_val.get("page_ref_galleries") or {}).get(str(cur)) \
                       or sb_val.get("last_ref_gallery") or []
            last_refs = [
                (p, lbl) for p, lbl in _pg_refs
                if p and os.path.isfile(str(p))
            ]
            tl = _build_timeline_gallery(sb_val)
            return _load_page_header(sb_val) + _load_panel(sb_val) + [
                gr.update(value=saved_img), last_prompt, gr.Gallery(value=last_refs),
                gr.Gallery(value=tl),          # story_timeline
            ]

        def _update_gen_extras(sb_val):
            """After generation: refresh the story timeline gallery."""
            tl = _build_timeline_gallery(sb_val)
            return gr.Gallery(value=tl)

        def _timeline_click(evt: gr.SelectData, sb_val):
            """Show beat descriptions for the clicked story-timeline entry."""
            import re as _re2
            gallery = _build_timeline_gallery(sb_val)
            if evt.index >= len(gallery):
                return ""
            _, label = gallery[evt.index]
            m = _re2.match(r"P(\d+)", label)
            if not m:
                return ""
            pg_num = int(m.group(1))
            pz     = max(1, sb_val.get("page_size", 6))
            beats  = sb_val.get("beats") or []
            page_beats = [
                _panel_beat_text(beats, pg_num, pn, pz)
                for pn in range(1, pz + 1)
            ]
            lines  = [f"• {b}" for b in page_beats if b]
            ver_info = ""
            if " v" in label:
                ver_info = f" ({label.split(' ', 1)[1]})"
            return f"📄 Page {pg_num}{ver_info}:\n" + ("\n".join(lines) if lines else "(no beat text)")

        def _quick_cast_click(evt: gr.SelectData, sb_val, cq_names):
            """Clicking a face in the Beat Characters gallery selects that character."""
            if not cq_names or evt.index >= len(cq_names):
                return sb_val, "No character selected"
            cname = cq_names[evt.index]
            sb_val["selected_char"] = cname
            return sb_val, f"▶ {cname} selected — assign any library image as their face reference"

        def _nav(sb_val, direction: str):
            pz = max(1, sb_val.get("page_size", 6)); np = _n_pages(sb_val)
            if   direction == "pp": sb_val["cur_page"]  = max(1, sb_val.get("cur_page",  1) - 1); sb_val["cur_panel"] = 1
            elif direction == "np": sb_val["cur_page"]  = min(np, sb_val.get("cur_page", 1) + 1); sb_val["cur_panel"] = 1
            elif direction == "pn": sb_val["cur_panel"] = max(1, sb_val.get("cur_panel", 1) - 1)
            elif direction == "nn": sb_val["cur_panel"] = min(pz, sb_val.get("cur_panel", 1) + 1)
            return sb_val

        def _do_search(q):
            imgs, ids = _search_library(q)
            return gr.Gallery(value=imgs), ids

        def _do_get_suggestions(sb_val):
            """Suggest 20 library images that best match the current panel's beat."""
            cur  = sb_val.get("cur_page", 1)
            pn   = sb_val.get("cur_panel", 1)
            pz   = max(1, sb_val.get("page_size", 6))
            panel = _get_panel(sb_val, cur, pn)
            bidx  = panel.get("beat_idx") or _panel_beat_idx(cur, pn, pz)
            beats = sb_val.get("beats") or []
            beat  = _panel_beat_text(beats, cur, pn, pz, override_idx=panel.get("beat_idx") or 0)
            imgs, ids = _suggest_for_beat(beat, n=20)
            return gr.Gallery(value=imgs), ids

        def _select_image(tid: str, sb_val):
            """Common handler: set selected_tid, update preview bar."""
            sb_val["selected_tid"] = tid
            p, nm = _template_info(tid)
            return sb_val, p, nm

        def _gallery_click(evt: gr.SelectData, sb_val, ids):
            if not ids or evt.index >= len(ids): return sb_val, None, "—"
            return _select_image(ids[evt.index], sb_val)

        def _recent_click(evt: gr.SelectData, sb_val, rids):
            if not rids or evt.index >= len(rids): return sb_val, None, "—"
            return _select_image(rids[evt.index], sb_val)

        def _cast_char_change(sb_val, chosen):
            sb_val["selected_char"] = chosen
            return sb_val

        def _cast_assign(sb_val):
            tid  = sb_val.get("selected_tid")
            cname = sb_val.get("selected_char")
            if not tid:
                return sb_val, _cast_html(sb_val)
            if not cname:
                chars = list((sb_val.get("cast") or {}).keys())
                if not chars: return sb_val, _cast_html(sb_val)
                cname = chars[0]; sb_val["selected_char"] = cname
            sb_val.setdefault("cast", {})[cname] = tid
            # DRIFT FIX: a face ref is now the identity authority for this character.
            # Drop any AUTO-generated appearance note (invented before the face existed)
            # so it can't override the real face-image DNA at generation time.
            # User-authored overrides (not flagged auto) are preserved.
            if (sb_val.get("char_appearance_auto") or {}).get(cname):
                (sb_val.get("char_appearance") or {}).pop(cname, None)
                sb_val["char_appearance_auto"].pop(cname, None)
            _add_recent(sb_val, tid)
            _sb_save(sb_val)  # auto-save cast assignments
            rim, rid = _recent_gallery(sb_val)
            return sb_val, _cast_html(sb_val), gr.Gallery(value=rim), rid

        def _cast_clear_fn(sb_val):
            cname = sb_val.get("selected_char")
            if cname and cname in (sb_val.get("cast") or {}):
                sb_val["cast"][cname] = None
            _sb_save(sb_val)
            return sb_val, _cast_html(sb_val)

        def _cls_assign(sb_val, cname, cls_name):
            if not cname:
                return sb_val, _cast_html(sb_val)
            if not cls_name or cls_name == "— none —":
                sb_val.setdefault("char_class", {}).pop(cname, None)
            else:
                sb_val.setdefault("char_class", {})[cname] = cls_name.strip().lower()
            _sb_save(sb_val)
            return sb_val, _cast_html(sb_val)

        def _cls_clear(sb_val, cname):
            if cname:
                sb_val.setdefault("char_class", {}).pop(cname, None)
            _sb_save(sb_val)
            return sb_val, _cast_html(sb_val)

        def _pg_set(sb_val):
            tid = sb_val.get("selected_tid")
            if not tid:
                return sb_val, _pg_set_html(None, None), gr.Gallery(value=[]), []
            sb_val.setdefault("page_refs", {}).setdefault(str(sb_val.get("cur_page", 1)), {})["page_setting"] = tid
            _add_recent(sb_val, tid)
            _sb_save(sb_val)
            rim, rid = _recent_gallery(sb_val)
            p, nm = _template_info(tid)
            return sb_val, _pg_set_html(p, nm), gr.Gallery(value=rim), rid

        def _pg_clear(sb_val):
            sb_val.setdefault("page_refs", {}).setdefault(str(sb_val.get("cur_page", 1)), {}).pop("page_setting", None)
            _sb_save(sb_val)
            return sb_val, _pg_set_html(None, None)

        def _slot_set(sb_val, si):
            tid = sb_val.get("selected_tid")
            if not tid:
                return sb_val, _slots_html_from_sb(sb_val), gr.Gallery(value=[]), []
            cur, pn = sb_val.get("cur_page", 1), sb_val.get("cur_panel", 1)
            panel = _get_panel(sb_val, cur, pn); slots = _panel_slots(panel)
            slots[si] = {"tid": tid, "role": SLOT_CODES[si]}
            panel["refs"] = slots; _set_panel(sb_val, cur, pn, panel)
            _add_recent(sb_val, tid)
            _sb_save(sb_val)
            rim, rid = _recent_gallery(sb_val)
            return sb_val, _slots_html_from_sb(sb_val), gr.Gallery(value=rim), rid

        def _slot_clear(sb_val, si):
            cur, pn = sb_val.get("cur_page", 1), sb_val.get("cur_panel", 1)
            panel = _get_panel(sb_val, cur, pn); slots = _panel_slots(panel)
            slots[si] = None; panel["refs"] = slots; _set_panel(sb_val, cur, pn, panel)
            _sb_save(sb_val)
            return sb_val, _slots_html_from_sb(sb_val)

        def _do_auto_fill_slots(sb_val):
            """Pick best A-E refs from the library using Claude, then fill all slots."""
            try:
                from build import _call_claude_json, CLAUDE_FAST_MODEL
                import character_library as _cl

                cur, pn = sb_val.get("cur_page", 1), sb_val.get("cur_panel", 1)
                pz      = max(1, sb_val.get("page_size", 6))
                panel   = _get_panel(sb_val, cur, pn)
                bidx    = panel.get("beat_idx") or _panel_beat_idx(cur, pn, pz)

                beat_chars = list(panel.get("chars") or (sb_val.get("beat_chars") or {}).get(str(bidx), []))
                beats      = sb_val.get("beats") or []
                beat_text  = _panel_beat_text(beats, cur, pn, pz, override_idx=panel.get("beat_idx") or 0)
                cast       = sb_val.get("cast") or {}
                world_ctx  = (sb_val.get("world_context") or "").strip()

                if not beat_text:
                    return sb_val, _slots_html_from_sb(sb_val), "⚠️ No beat text — navigate to a panel first."

                lib_data  = _cl.load_library()
                templates = lib_data.get("templates") or {}
                if not templates:
                    return sb_val, _slots_html_from_sb(sb_val), "⚠️ Library is empty — add images first."

                # Build concise candidate list for Claude
                candidates = []
                for tid, t in templates.items():
                    tags = t.get("tags") or []
                    if isinstance(tags, str):
                        try:    tags = eval(tags)
                        except: tags = [tags]
                    scene = t.get("scene_suitability") or []
                    if isinstance(scene, str):
                        try:    scene = eval(scene)
                        except: scene = [scene]
                    candidates.append({
                        "tid":     tid,
                        "name":    t.get("name") or t.get("auto_name") or tid,
                        "cat":     t.get("category") or "other",
                        "tags":    ", ".join(str(x) for x in tags[:10]),
                        "summary": (t.get("summary") or t.get("auto_name") or "")[:100],
                        "scenes":  ", ".join(str(x) for x in scene[:5]),
                    })

                # Slot A — use cast mapping directly (no Claude needed)
                slots    = _panel_slots(panel)
                face_tid = cast.get(beat_chars[0]) if beat_chars else None
                if face_tid:
                    slots[0] = {"tid": face_tid, "role": SLOT_CODES[0]}

                # Slot D (pose) is the priority — only add E (location) for establishing shots.
                # B (camera) and C (mood) are never auto-filled; they over-constrain the model.
                _SYS = (
                    "You are a manhwa panel art director selecting reference images.\n"
                    "Given a story beat, pick AT MOST ONE reference slot:\n"
                    "  D = pose / body language / action reference "
                    "(PRIORITY — use this for almost every panel that has character action or emotion)\n"
                    "  E = location / setting / background reference "
                    "(only for pure establishing shots with no character action)\n\n"
                    "Rules:\n"
                    "- Almost always pick D. Only pick E when the beat is purely environment with no action.\n"
                    "- If picking D, set E to null. If picking E, set D to null.\n"
                    "- Use ONLY tids from the candidates list. Set to null if no match is strong.\n\n"
                    'Return ONLY valid JSON: {"D":"tid_or_null","E":"tid_or_null"}'
                )
                user_payload = {
                    "beat":          beat_text,
                    "characters":    ", ".join(beat_chars) if beat_chars else "none",
                    "world_context": world_ctx[:300],
                    "candidates":    candidates,
                }
                result = _call_claude_json(_SYS, user_payload, max_tokens=600, model=CLAUDE_FAST_MODEL)

                # Apply Claude's picks — max 2 slots total (A face + D pose or E location)
                filled = []
                if face_tid:
                    filled.append(f"A → {beat_chars[0]}")

                # Always clear B and C (never auto-fill camera/mood)
                slots[1] = None  # B camera
                slots[2] = None  # C mood
                slot_map = {"D": 3, "E": 4}
                for key, si in slot_map.items():
                    tid = (result or {}).get(key)
                    if tid and tid in templates:
                        slots[si] = {"tid": tid, "role": SLOT_CODES[si]}
                        nm = (templates[tid].get("name") or templates[tid].get("auto_name") or tid)[:22]
                        filled.append(f"{key} → {nm}")
                    else:
                        slots[si] = None

                panel["refs"] = slots
                _set_panel(sb_val, cur, pn, panel)
                _sb_save(sb_val)
                status = ("✅ " + " · ".join(filled)) if filled else "⚠️ No good matches found in library."
                return sb_val, _slots_html_from_sb(sb_val), status

            except Exception as e:
                import traceback; traceback.print_exc()
                return sb_val, _slots_html_from_sb(sb_val), f"❌ {e}"

        def _auto_fill_page_batch(sb_val, pg):
            """ONE Claude call fills D/E refs for ALL panels on the page — 6× cheaper than per-panel calls."""
            from build import _call_claude_json, CLAUDE_FAST_MODEL
            import character_library as _cl

            pz        = max(1, sb_val.get("page_size", 6))
            beats     = sb_val.get("beats") or []
            cast      = sb_val.get("cast") or {}
            world_ctx = (sb_val.get("world_context") or "").strip()

            lib_data  = _cl.load_library()
            templates = lib_data.get("templates") or {}

            # Build candidate list (shared across all panels)
            candidates = []
            for tid, t in templates.items():
                tags  = t.get("tags") or []
                if isinstance(tags, str):
                    try:    tags = eval(tags)
                    except: tags = [tags]
                scene = t.get("scene_suitability") or []
                if isinstance(scene, str):
                    try:    scene = eval(scene)
                    except: scene = [scene]
                candidates.append({
                    "tid":     tid,
                    "name":    (t.get("name") or t.get("auto_name") or tid)[:30],
                    "cat":     t.get("category") or "other",
                    "tags":    ", ".join(str(x) for x in tags[:8]),
                    "summary": (t.get("summary") or "")[:80],
                    "scenes":  ", ".join(str(x) for x in scene[:4]),
                })

            # Collect per-panel info
            panel_infos = []
            for pn in range(1, pz + 1):
                panel      = _get_panel(sb_val, pg, pn)
                bidx       = panel.get("beat_idx") or _panel_beat_idx(pg, pn, pz)
                beat_text  = _panel_beat_text(beats, pg, pn, pz, override_idx=panel.get("beat_idx") or 0) or f"Panel {pn}"
                beat_chars = list(panel.get("chars") or (sb_val.get("beat_chars") or {}).get(str(bidx), []))
                panel_infos.append({"pn": pn, "beat": beat_text,
                                    "chars": ", ".join(beat_chars) if beat_chars else "none",
                                    "beat_chars": beat_chars, "bidx": bidx})

            _SYS = (
                "You are a manhwa panel art director. For each panel pick AT MOST ONE ref slot:\n"
                "  D = pose/body language/action (PRIORITY — use for any panel with character action or emotion)\n"
                "  E = location/setting/background (only for pure establishing shots, no character action)\n\n"
                "Rules: almost always pick D. Set both null only when no candidate fits.\n"
                "Use ONLY tids from the candidates list.\n\n"
                'Return ONLY JSON: {"p1":{"D":"tid_or_null","E":"tid_or_null"},'
                '"p2":{...},...} — one key per panel number.'
            )
            result = _call_claude_json(
                _SYS,
                {"world_context": world_ctx[:200],
                 "panels": [{"p": pi["pn"], "beat": pi["beat"], "chars": pi["chars"]}
                            for pi in panel_infos],
                 "candidates": candidates},
                max_tokens=1200, model=CLAUDE_FAST_MODEL,  # haiku — cheaper; slot-filling doesn't need Sonnet
            ) or {}

            for pi in panel_infos:
                pn         = pi["pn"]
                panel      = _get_panel(sb_val, pg, pn)
                slots      = _panel_slots(panel)
                beat_chars = pi["beat_chars"]

                # A — face from cast (always direct)
                face_tid = cast.get(beat_chars[0]) if beat_chars else None
                if face_tid:
                    slots[0] = {"tid": face_tid, "role": SLOT_CODES[0]}
                slots[1] = None   # B camera — never auto-filled
                slots[2] = None   # C mood   — never auto-filled

                picks = result.get(f"p{pn}") or {}
                for key, si in {"D": 3, "E": 4}.items():
                    tid = picks.get(key)
                    slots[si] = {"tid": tid, "role": SLOT_CODES[si]} if (tid and tid in templates) else None

                panel["refs"] = slots
                _set_panel(sb_val, pg, pn, panel)
            _sb_save(sb_val)

        def _do_auto_run_story(sb_val, quality, workers_str="2", pages_str="10"):
            """Parallel full-pipeline generator: batch-fill → prompt → check → patch → FAL.
            Runs up to N workers concurrently and generates only the next `pages_str`
            un-generated pages, then stops. Works for any page_size (6 or 10 panels)."""
            beats   = sb_val.get("beats") or []
            n_pages = _n_pages(sb_val)
            _nf     = None         # zip_file placeholder
            _tl     = gr.update()  # gallery "no change" for intermediate status yields

            def _y(sb, img, msg, tl=None):
                """Yield helper — uses _tl (no-op) unless tl is explicitly passed."""
                return sb, img, msg, (tl if tl is not None else _tl), _nf, msg

            if _GEN_RUNNING.is_set():
                yield *_y(sb_val, None, "⚠️ An Auto-Run batch is already in progress — wait for it to finish."),
                return
            if not beats:
                yield *_y(sb_val, None, "⚠️ Parse a story first.", gr.Gallery(value=[])),; return
            if n_pages < 1:
                yield *_y(sb_val, None, "⚠️ No pages detected.", gr.Gallery(value=[])),; return
            cast = sb_val.get("cast") or {}
            has_refs = any(v for v in cast.values() if v)
            if not has_refs:
                yield *_y(sb_val, None, "ℹ️ No face refs — generating with text descriptions only."),

            try:
                workers = max(1, min(6, int(str(workers_str).strip())))
            except Exception:
                workers = 2
            try:
                limit = max(1, int(str(pages_str).strip()))
            except Exception:
                limit = 10

            # Shared nested containers MUST exist before per-worker shallow copies,
            # otherwise each worker's setdefault would create its own private dict.
            sb_val.setdefault("generated_pages", {})
            sb_val.setdefault("page_refs", {})
            sb_val.setdefault("page_prompts", {})
            sb_val.setdefault("page_prompt_versions", {})
            sb_val.setdefault("page_ref_galleries", {})

            pending = []
            for pg in range(1, n_pages + 1):
                ex_img = _latest_page_img(sb_val, pg)
                if not (ex_img and os.path.isfile(ex_img)):
                    pending.append(pg)
            batch        = pending[:limit]
            done_already = n_pages - len(pending)

            if not batch:
                tl = gr.Gallery(value=_build_timeline_gallery(sb_val))
                yield *_y(sb_val, _latest_page_img(sb_val, n_pages),
                          f"✅ All {n_pages} pages already generated — nothing to do.", tl),
                return

            _GEN_PAUSED.clear()   # make sure we start unpaused
            total = len(batch)
            yield *_y(sb_val, None,
                      f"🚀 Generating {total} page(s) — {done_already}/{n_pages} already done · "
                      f"{workers} worker(s) · pages {batch[0]}–{batch[-1]}…"),

            import concurrent.futures as _cf
            from build import _call_claude_json, CLAUDE_FAST_MODEL as _CHEAP

            def _process_page(pg):
                """Whole per-page pipeline. Runs in a worker thread.
                Uses a shallow copy of sb so each worker has its own cur_page while
                nested dicts (page_refs, generated_pages) stay shared."""
                pv = dict(sb_val)
                pv["cur_page"], pv["cur_panel"] = pg, 1
                try:
                    _auto_fill_page_batch(pv, pg)
                except Exception as _e:
                    print(f"[auto-run] batch-fill p{pg}: {_e}", flush=True)
                _, _, prompt, _ = _generate_page(pv, quality, send_to_fal=False)
                if not (prompt or "").strip():
                    return pg, None, "no prompt built"
                # Free local coverage check; Haiku patch only on hard misses
                if "❌" in _check_prompt_coverage(prompt, pv):
                    try:
                        _pr = _call_claude_json(
                            system=(
                                "You are a panel-by-panel manhwa prompt editor. "
                                "AUTHORITY: Each character's 'full_appearance' is the ONLY source of truth. "
                                "Inject VERBATIM full_appearance text for any panel missing hair, eyes, skin, "
                                "or clothing. Only modify panels that have gaps.\n"
                                'Return JSON only: {"patched_prompt":"...complete prompt..."}'
                            ),
                            user_payload={"prompt": prompt, "characters": _build_char_refs(pv)},
                            max_tokens=8000, model=_CHEAP,
                        )
                        if _pr and _pr.get("patched_prompt"):
                            prompt = _pr["patched_prompt"].strip()
                    except Exception as _e:
                        print(f"[auto-run] patch p{pg}: {_e}", flush=True)
                img_p, msg, _, _ = _generate_page(pv, quality,
                                                  send_to_fal=True, prompt_override=prompt)
                if not img_p:
                    return pg, None, (msg or "generation failed")
                try:
                    _sb_save(sb_val)
                except Exception as _e:
                    print(f"[auto-run] save p{pg}: {_e}", flush=True)
                return pg, img_p, None

            # ── Chain scheduling: split batch into contiguous runs, one per worker.
            # Each chain generates its pages IN ORDER, so the prev-page continuity
            # crops (page N anchors from page N-1's image) stay intact within a
            # chain. Only the chain-start pages (workers-1 of them) lose that
            # anchor — the best trade-off between speed and visual continuity.
            import math as _math
            import queue as _quu
            n_chains = max(1, min(workers, total))
            chunk    = _math.ceil(total / n_chains)
            chains   = [batch[i * chunk:(i + 1) * chunk] for i in range(n_chains)]
            chains   = [c for c in chains if c]

            events = _quu.Queue()   # thread-safe: workers push, generator drains

            def _run_chain(pages):
                for pg in pages:
                    while _GEN_PAUSED.is_set():
                        time.sleep(0.5)
                    events.put(("start", pg, None))
                    try:
                        _pg, img_p, err = _process_page(pg)
                    except Exception as _e:
                        img_p, err = None, str(_e)
                    events.put(("done", pg, img_p) if img_p else ("fail", pg, err))

            _GEN_RUNNING.set()
            ex   = _cf.ThreadPoolExecutor(max_workers=len(chains))
            futs = [ex.submit(_run_chain, ch) for ch in chains]
            done_n      = 0
            fail_n      = 0
            last_img    = None
            paused_note = False
            try:
                while (done_n + fail_n) < total:
                    try:
                        kind, pg, payload = events.get(timeout=1)
                    except _quu.Empty:
                        if _GEN_PAUSED.is_set() and not paused_note:
                            paused_note = True
                            yield *_y(sb_val, last_img,
                                      "⏸ Paused — in-flight pages finish, no new pages start. "
                                      "Click Resume to continue."),
                        elif not _GEN_PAUSED.is_set() and paused_note:
                            paused_note = False
                            yield *_y(sb_val, last_img, "▶️ Resumed…"),
                        # Safety: if every chain thread ended but counts don't add
                        # up (worker crashed before reporting), bail out.
                        if all(f.done() for f in futs) and events.empty():
                            break
                        continue
                    if kind == "start":
                        yield *_y(sb_val, last_img,
                                  f"🎨 Page {pg} started — {done_n}/{total} done…"),
                    elif kind == "done":
                        done_n  += 1
                        last_img = payload
                        tl = gr.Gallery(value=_build_timeline_gallery(sb_val))
                        yield *_y(sb_val, last_img,
                                  f"✅ Page {pg} done — {done_n}/{total} complete.", tl),
                    else:
                        fail_n += 1
                        yield *_y(sb_val, last_img, f"❌ Page {pg} failed: {payload}"),
            finally:
                _GEN_RUNNING.clear()
                ex.shutdown(wait=False, cancel_futures=True)

            try:
                _sb_save(sb_val)
            except Exception:
                pass

            # ── Auto-zip ──────────────────────────────────────────────────────
            tl_final = gr.Gallery(value=_build_timeline_gallery(sb_val))
            summary  = f"🎉 Batch done — {done_n}/{total} generated"
            if fail_n:
                summary += f", {fail_n} failed"
            remaining = len(pending) - total
            if remaining > 0:
                summary += f" · {remaining} page(s) still un-generated (run again to continue)"
            yield *_y(sb_val, last_img, f"📦 {summary} — creating ZIP…", tl_final),
            try:
                sname        = sb_val.get("story_name") or "storyboard"
                zip_p, zmsg  = _sb_zip(sname, sb=sb_val)
                msg          = f"{summary} · {zmsg}"
                yield sb_val, last_img, msg, tl_final, zip_p, msg
            except Exception as _e:
                yield *_y(sb_val, last_img, f"⚠️ ZIP failed: {_e}", tl_final),

        def _beat_change(sb_val, choice):
            if choice and choice != "— auto —":
                try:
                    bidx = int(choice.split(":")[0].strip())
                    cur, pn = sb_val.get("cur_page", 1), sb_val.get("cur_panel", 1)
                    panel = _get_panel(sb_val, cur, pn); panel["beat_idx"] = bidx
                    _set_panel(sb_val, cur, pn, panel)
                    _sb_save(sb_val)
                except Exception: pass
            return sb_val

        def _do_generate(sb_val, quality, auto_gen):
            """Build prompt (always). If auto_gen, also fire FAL immediately."""
            img, msg, prompt, refs = _generate_page(
                sb_val, quality, send_to_fal=bool(auto_gen),
            )
            for tid, v in (sb_val.get("cast") or {}).items():
                if v: _add_recent(sb_val, v)
            rim, rid = _recent_gallery(sb_val)
            why = _build_panel_why(sb_val)
            return sb_val, img, msg, prompt, refs, gr.Gallery(value=rim), rid, why

        def _do_send_to_fal_prompt(sb_val, quality, prompt_text):
            """Send the (possibly edited) prompt_box text directly to FAL, keeping the user's edits."""
            if not (prompt_text or "").strip():
                return sb_val, None, "⚠️ Prompt is empty — click Generate Page first to build it.", prompt_text, [], gr.Gallery(value=[]), [], ""
            # Run generate with prompt_override so refs are re-collected (fast — already uploaded)
            # but the prompt text the user edited is used verbatim.
            img, msg, returned_prompt, refs = _generate_page(
                sb_val, quality, send_to_fal=True, prompt_override=prompt_text.strip(),
            )
            for tid, v in (sb_val.get("cast") or {}).items():
                if v: _add_recent(sb_val, v)
            rim, rid = _recent_gallery(sb_val)
            why = _build_panel_why(sb_val)
            # Keep the user's edited prompt text visible, not the rebuilt one
            return sb_val, img, msg, prompt_text, refs, gr.Gallery(value=rim), rid, why

        def _toggle_auto_gen(checked):
            """Show/hide Send to FAL button based on checkbox state."""
            return gr.update(visible=not checked)

        def _do_save_beat(sb_val, new_text):
            """Write the edited beat text back into sb['beats'] and persist."""
            cur  = sb_val.get("cur_page", 1)
            pz   = max(1, sb_val.get("page_size", 6))
            pn   = sb_val.get("cur_panel", 1)
            panel = _get_panel(sb_val, cur, pn)
            bidx  = panel.get("beat_idx") or _panel_beat_idx(cur, pn, pz)
            beats = list(sb_val.get("beats") or [])
            if 0 < bidx <= len(beats) and new_text:
                beats[bidx - 1] = new_text.strip()
                sb_val["beats"] = beats
            _sb_save(sb_val)
            return sb_val, "✅ Beat saved."

        def _do_zip(sb_val, name):
            zp, msg = _sb_zip(name or (sb_val.get("story_name") or sb_val.get("story", "")[:30]) or "storyboard", sb=sb_val)
            return zp, msg

        def _do_load_sb(choice):
            """
            Load a saved storyboard — restores everything in one shot.
            Returns: sb, beats_html, story_box, world_ctx, story_name,
                     parse_status, story_display,
                     cast_html, cast_char_dd_update,
                     recent_gallery_update, recent_ids, gen_image,
                     prompt_box, ref_sources_gallery
            """
            def _fail(msg):
                empty = _empty_sb()
                rim, rid = _recent_gallery(empty)
                return (empty, "", "", "", "", msg, "",
                        _cast_html(empty),
                        gr.update(choices=[], value=None),
                        gr.Gallery(value=rim), rid, None,
                        "", gr.Gallery(value=[]))

            if not choice:
                return _fail("⚠️ No storyboard selected.")
            fresh_map = {label: pid for pid, label in _sb_list()}
            pid = fresh_map.get(choice)
            if not pid:
                return _fail("⚠️ Project not found — try refreshing the list.")
            data = _sb_load(pid)
            if not data:
                return _fail("❌ Could not load storyboard.")

            # Recompute suggestions if missing (older saves won't have them)
            if not data.get("panel_suggestions"):
                _compute_all_suggestions(data)

            choices = _cast_choices(data)
            default = choices[0] if choices else None
            if default and not data.get("selected_char"):
                data["selected_char"] = default
            rim, rid = _recent_gallery(data)

            cur       = data.get("cur_page", 1)
            saved_img = _latest_page_img(data, cur)
            n_gen     = sum(len(_page_versions(data, k)) for k in (data.get("generated_pages") or {}))
            _pg_refs = (data.get("page_ref_galleries") or {}).get(str(cur)) \
                       or data.get("last_ref_gallery") or []
            last_refs = [
                (p, lbl) for p, lbl in _pg_refs
                if p and os.path.isfile(str(p))
            ]
            return (
                data,
                _beats_html(data),
                data.get("story", ""),
                data.get("world_context", ""),
                data.get("story_name", ""),
                f"✅ Loaded '{data.get('story_name', pid)}' — "
                f"{data.get('n_beats', 0)} beats, "
                f"{len([v for v in (data.get('cast') or {}).values() if v])} face refs assigned"
                + (f", {n_gen} page(s) already generated." if n_gen else "."),
                data.get("story", ""),
                _cast_html(data),
                gr.update(choices=choices, value=default),
                gr.Gallery(value=rim),
                rid,
                saved_img,
                (data.get("page_prompts") or {}).get(str(data.get("cur_page", 1)), ""),
                gr.Gallery(value=last_refs),
            )

        def _do_refresh_sb_list():
            entries = _sb_list()
            choices = [label for _, label in entries]
            msg = f"✅ {len(choices)} saved storyboard(s)" if choices else "No saved storyboards yet."
            return gr.update(choices=choices, value=None), msg

        def _update_world_ctx(sb_val, ctx):
            sb_val["world_context"] = (ctx or "").strip()
            _sb_save(sb_val)
            return sb_val

        # ── Character appearance note ─────────────────────────────────────────
        def _auto_char_look(sb_val, cname):
            """Build appearance string from char_dna + clothing class DNA (no manual override)."""
            parts = []
            dna = (sb_val.get("char_dna") or {}).get(cname, "")
            if dna:
                parts.append(dna.strip())
            cls_name = (sb_val.get("char_class") or {}).get(cname)
            if cls_name:
                try:
                    import character_library as _cl_ll
                    clothing = _cl_ll.get_class_clothing_dna(cls_name, seed_key=cname)
                    if clothing:
                        parts.append(clothing.strip())
                except Exception:
                    pass
            return " ".join(parts)

        def _load_char_look(sb_val, cname):
            """Return saved appearance override, or auto-build from char_dna + clothing class."""
            if not cname:
                return ""
            manual = (sb_val.get("char_appearance") or {}).get(cname, "")
            return manual if manual else _auto_char_look(sb_val, cname)

        def _save_char_look(sb_val, cname, text):
            if not cname:
                return sb_val, "⚠️ Select a character first."
            text = (text or "").strip()
            appearance = sb_val.setdefault("char_appearance", {})
            if text:
                appearance[cname] = text
                # User-authored — remove the auto flag so it's never auto-cleared
                (sb_val.get("char_appearance_auto") or {}).pop(cname, None)
                msg = f"✅ Override saved for {cname}."
            else:
                appearance.pop(cname, None)   # clear = revert to auto-built
                (sb_val.get("char_appearance_auto") or {}).pop(cname, None)
                msg = f"↺ Reset to auto-built for {cname}."
            _sb_save(sb_val)
            return sb_val, msg

        # ── Beat add / remove ─────────────────────────────────────────────────
        def _beat_char_html(sb_val):
            """Re-render the character tag HTML for the current panel."""
            cur   = sb_val.get("cur_page", 1)
            pn    = sb_val.get("cur_panel", 1)
            pz    = max(1, sb_val.get("page_size", 6))
            panel = _get_panel(sb_val, cur, pn)
            bidx  = panel.get("beat_idx") or _panel_beat_idx(cur, pn, pz)
            bc    = sb_val.get("beat_chars") or {}
            man_ch  = panel.get("chars") or []
            auto_ch = bc.get(str(bidx), [])
            shown   = list(dict.fromkeys(man_ch if man_ch else auto_ch))
            if shown:
                tags = "".join(
                    f"<span style='background:#2d3748;border:1px solid #4a5568;border-radius:3px;"
                    f"padding:1px 7px;margin:2px;font-size:12px;color:#e2e8f0'>{c}</span>"
                    for c in shown
                )
                return f"<div style='padding:2px 0'>👤 {tags}</div>", shown
            return "<div style='color:#555;font-size:12px;padding:2px 0'>No characters for this beat.</div>", shown

        def _do_beat_add(sb_val, cname):
            """Add a cast member to the current panel's character list."""
            if not cname:
                return sb_val, *_beat_char_html(sb_val)[0:1], gr.update(), gr.Gallery(value=[]), []
            cur   = sb_val.get("cur_page", 1)
            pn    = sb_val.get("cur_panel", 1)
            pz    = max(1, sb_val.get("page_size", 6))
            panel = _get_panel(sb_val, cur, pn)
            bidx  = panel.get("beat_idx") or _panel_beat_idx(cur, pn, pz)
            bc    = sb_val.get("beat_chars") or {}
            auto_ch = list(bc.get(str(bidx), []))
            man_ch  = list(panel.get("chars") or auto_ch)
            if cname not in man_ch:
                man_ch.append(cname)
            panel["chars"] = man_ch
            _set_panel(sb_val, cur, pn, panel)
            _sb_save(sb_val)
            html, shown = _beat_char_html(sb_val)
            add_choices  = [c for c in (sb_val.get("cast") or {}) if c not in set(shown)]
            cqg, cqn     = _cast_quick_gallery(sb_val)
            return sb_val, html, gr.update(choices=add_choices, value=None), gr.Gallery(value=cqg), cqn

        def _do_beat_rm(sb_val, cname):
            """Remove a cast member from the current panel's character list."""
            if not cname:
                return sb_val, *_beat_char_html(sb_val)[0:1], gr.update(), gr.Gallery(value=[]), []
            cur   = sb_val.get("cur_page", 1)
            pn    = sb_val.get("cur_panel", 1)
            pz    = max(1, sb_val.get("page_size", 6))
            panel = _get_panel(sb_val, cur, pn)
            bidx  = panel.get("beat_idx") or _panel_beat_idx(cur, pn, pz)
            bc    = sb_val.get("beat_chars") or {}
            auto_ch = list(bc.get(str(bidx), []))
            man_ch  = list(panel.get("chars") or auto_ch)
            if cname in man_ch:
                man_ch.remove(cname)
            panel["chars"] = man_ch if man_ch else []
            _set_panel(sb_val, cur, pn, panel)
            _sb_save(sb_val)
            html, shown  = _beat_char_html(sb_val)
            add_choices  = [c for c in (sb_val.get("cast") or {}) if c not in set(shown)]
            cqg, cqn     = _cast_quick_gallery(sb_val)
            return sb_val, html, gr.update(choices=add_choices, value=None), gr.Gallery(value=cqg), cqn

        def _build_char_refs(sb_val):
            """Build the character reference dict sent to Claude for check/patch calls.
            Uses _load_char_look (respects manual overrides) — same source as the prompt builder."""
            cast       = sb_val.get("cast") or {}
            char_class = sb_val.get("char_class") or {}
            refs = {}
            for name, tid in cast.items():
                face_ref_desc = ""
                if tid:
                    try:
                        face_ref_desc = (_face_only_dna(tid) or "").strip()
                    except Exception:
                        pass
                refs[name] = {
                    "full_appearance":  _load_char_look(sb_val, name).strip(),  # same source as prompt builder
                    "class":            char_class.get(name, ""),
                    "is_nonhuman":      _is_nonhuman(name, sb_val),
                    "face_ref_desc":    face_ref_desc,   # library image DNA — may conflict with profile
                }
            return refs

        def _fmt_panel_report(result):
            """Format panel-by-panel report from Claude into the display string."""
            lines  = []
            panels = result.get("panels") or []
            for panel_block in panels:
                pnum  = panel_block.get("panel", "?")
                chars = panel_block.get("characters") or []
                if not chars:
                    continue
                parts = []
                for char in chars:
                    name    = char.get("name", "?")
                    missing = char.get("missing") or []
                    covered = char.get("covered") or []
                    score   = len(covered)
                    grade   = "✅" if not missing else ("⚠️" if score >= 2 else "❌")
                    note    = f" ← missing: {', '.join(missing)}" if missing else ""
                    parts.append(f"{grade} {name}{note}")
                lines.append(f"Panel {pnum}: " + " | ".join(parts))
            skipped = result.get("skipped_nonhuman") or []
            if skipped:
                lines.append(f"⬜ non-human skipped: {', '.join(skipped)}")
            return "\n".join(lines) if lines else "(no human cast found in any panel)"

        def _ts():
            from datetime import datetime
            return datetime.now().strftime("%H:%M:%S")

        _CHECK_SYSTEM = (
            "You are a panel-by-panel manhwa prompt auditor.\n\n"
            "The prompt is divided into PANEL 1, PANEL 2, ... blocks.\n\n"
            "AUTHORITY: Each character's 'full_appearance' field is THE SINGLE SOURCE OF TRUTH "
            "for how that character must look. ONLY check against full_appearance — ignore face_ref_desc.\n\n"
            "YOUR TASK: For each PANEL block in the prompt:\n"
            "1. Identify which named human characters appear IN THAT PANEL's text (skip is_nonhuman: true).\n"
            "2. For each character found in that panel, check all 4 categories against that panel's "
            "text ONLY, using ONLY the full_appearance profile:\n"
            "   - hair: specific hair color/style from full_appearance must appear in this panel's text\n"
            "   - eyes: specific eye description from full_appearance must appear in this panel's text\n"
            "   - skin: specific skin tone or build from full_appearance must appear in this panel's text\n"
            "   - clothing: specific clothing description from full_appearance must appear in this panel's text\n\n"
            "STRICT RULE: Generic words do NOT count. 'Finer robes' ≠ 'dark earth-tone robes with ornate embroidery'. "
            "'Black hair' ≠ 'graying black hair slicked back'. The full_appearance's SPECIFIC wording "
            "must appear in that panel's text block.\n\n"
            "Return JSON only — no markdown:\n"
            '{"panels": [{"panel": 1, "characters": [{"name": "Lin", "covered": ["hair","eyes","skin","clothing"], "missing": [], "skipped": false}]}], '
            '"skipped_nonhuman": ["Snake Spirit"]}'
        )

        def _run_check_call(prompt, sb_val):
            """Core check logic — returns report string. Call this directly; use _do_prompt_check for UI generator."""
            from build import _call_claude_json, CLAUDE_FAST_MODEL
            char_refs = _build_char_refs(sb_val)
            result = _call_claude_json(
                system=_CHECK_SYSTEM,
                user_payload={"prompt": prompt, "characters": char_refs},
                max_tokens=3500,
                model=CLAUDE_FAST_MODEL,
            )
            if not result or "panels" not in result:
                return "❌ Coverage check failed — try again."
            return _fmt_panel_report(result)

        def _do_prompt_check(prompt, sb_val):
            """Generator — yields (log, report) tuples: live status to log box, results to coverage box."""
            try:
                prompt = (prompt or "").strip()
                if not prompt:
                    yield "⬆ Generate a prompt first.", ""
                    return
                cast = sb_val.get("cast") or {}
                if not cast:
                    yield "No cast found — parse a story first.", ""
                    return

                # Pre-flight: detect cast members with no face ref assigned — these will
                # drift visually regardless of how good the text coverage is.
                no_ref = [cname for cname, tid in cast.items()
                          if not tid and not _is_nonhuman(cname, sb_val)]

                char_refs = _build_char_refs(sb_val)
                num_chars = sum(1 for v in char_refs.values() if not v.get("is_nonhuman"))
                yield f"⏱ {_ts()} — Checking {num_chars} characters (text + face refs)…", ""

                report = _run_check_call(prompt, sb_val)

                # Prepend face-ref warning so it's the first thing the user sees
                if no_ref:
                    warning = (
                        "⚠️ NO FACE REF — these characters have no library image assigned and "
                        "will look different every generation:\n"
                        + "\n".join(f"  • {n}" for n in no_ref)
                        + "\n→ Go to Library tab, assign an image, then come back.\n\n"
                    )
                    report = warning + report

                yield f"✔ Done {_ts()}", report
            except Exception as e:
                yield f"❌ Error: {e}", ""

        def _do_patch_and_check(prompt, sb_val):
            """Generator — yields (prompt, log, report) tuples: live status, updated prompt, coverage results."""
            # Yield immediately with the current prompt so Gradio never resets prompt_box to
            # empty — this prevents the "restarts app" symptom caused by a pre-first-yield error.
            prompt = (prompt or "").strip()
            yield prompt, f"⏱ {_ts()} — Starting patch…", ""
            try:
                if not prompt:
                    yield prompt, "⬆ Generate a prompt first.", ""
                    return
                cast = sb_val.get("cast") or {}
                if not cast:
                    yield prompt, "No cast found — parse a story first.", ""
                    return

                from build import _call_claude_json, CLAUDE_FAST_MODEL

                char_refs = _build_char_refs(sb_val)
                num_chars = sum(1 for v in char_refs.values() if not v.get("is_nonhuman"))
                yield prompt, f"⏱ {_ts()} — Step 1/2: Patching {num_chars} characters…", ""

                result = _call_claude_json(
                    system=(
                        "You are a panel-by-panel manhwa prompt editor.\n\n"
                        "The prompt is divided into PANEL 1, PANEL 2, ... blocks.\n\n"
                        "AUTHORITY: Each character's 'full_appearance' field is THE SINGLE SOURCE OF TRUTH "
                        "for how that character must look. The 'face_ref_desc' field is only a photo label — "
                        "it may be wrong. NEVER copy wording from face_ref_desc into the prompt. "
                        "ALL injected appearance text must come VERBATIM from full_appearance only.\n\n"
                        "STEP 1 — AUDIT EACH PANEL: For each PANEL block, identify which named human characters "
                        "appear in it (skip is_nonhuman: true). For each character in that panel, check all 4 "
                        "categories (hair, eyes, skin, clothing) against ONLY that panel's own text, "
                        "using ONLY the full_appearance profile as the standard. "
                        "Generic words do NOT count — the full_appearance's specific phrasing must be present.\n\n"
                        "STEP 2 — PATCH EACH PANEL: For every panel where a character is missing categories, "
                        "inject the VERBATIM full_appearance wording for those missing categories directly into "
                        "that panel's text, right after the character's name or description in that panel. "
                        "Example: panel says 'Chief gestures to guards' → patch to: "
                        "'Chief — graying black hair slicked back, small greedy eyes, rotund build, "
                        "dark earth-tone robes with ornate embroidery — gestures to guards'\n\n"
                        "CRITICAL RULES:\n"
                        "- Only modify panel blocks that have missing categories. Do not touch other panels.\n"
                        "- Do not paraphrase. Copy exact full_appearance wording.\n"
                        "- If a character appears in multiple panels with missing details, patch ALL of them.\n"
                        "- The patched_prompt MUST differ from the input if anything was missing.\n\n"
                        "Return JSON only — no markdown:\n"
                        '{"patched_prompt": "...complete prompt with per-panel injections..."}'
                    ),
                    user_payload={"prompt": prompt, "characters": char_refs},
                    max_tokens=8000,
                    model=CLAUDE_FAST_MODEL,
                )
                if not result:
                    yield prompt, "❌ Patch call failed — try again.", ""
                    return

                patched = (result.get("patched_prompt") or prompt).strip()
                changed = patched != prompt
                yield patched, f"⏱ {_ts()} — Step 2/2: Patch {'applied' if changed else 'not needed — running re-check'}. Verifying…", ""

                coverage = _run_check_call(patched, sb_val)
                yield patched, f"✔ Done {_ts()}", coverage
            except Exception as e:
                yield prompt, f"❌ Error: {e}", ""

        # ══════════════════════════════════════════════════════════════════════
        # Wire events
        # ══════════════════════════════════════════════════════════════════════

        parse_btn.click(
            _do_parse, [story_box, page_size_dd, story_name_box, world_context_box],
            [sb, beats_html_box, parse_status, story_display],
        ).then(
            _do_enrich, [sb],
            [sb, beats_html_box, parse_status, cast_html_box, cast_char_dd],
        ).then(
            _refresh_all, [sb], _all_refresh,
        )

        # Regenerate beats for an already-loaded project (preserves cast/page refs/generated images)
        re_parse_btn.click(
            _do_re_parse, [sb, page_size_dd],
            [sb, beats_html_box, parse_status, story_display],
        ).then(
            _do_enrich, [sb],
            [sb, beats_html_box, parse_status, cast_html_box, cast_char_dd],
        ).then(
            _refresh_all, [sb], _all_refresh,
        )

        # Load saved storyboard — full restore in one chain
        _load_sb_outputs = [
            sb, beats_html_box, story_box, world_context_box, story_name_box,
            parse_status, story_display,
            cast_html_box, cast_char_dd,
            recent_gallery, recent_ids, gen_image,
            prompt_box, ref_sources_gallery,
        ]
        load_sb_btn.click(
            _do_load_sb, [load_sb_dd], _load_sb_outputs,
        ).then(
            _refresh_all, [sb], _all_refresh,
        )

        refresh_sb_btn.click(_do_refresh_sb_list, [], [load_sb_dd, parse_status])

        # Sync world context live into sb (debounced save inside)
        world_context_box.change(_update_world_ctx, [sb, world_context_box], [sb])

        # Page navigation
        for btn, d in [(prev_page_btn, "pp"), (next_page_btn, "np")]:
            btn.click(lambda s, _d=d: _nav(s, _d), [sb], [sb]).then(
                _refresh_all, [sb], _all_refresh).then(
                lambda s: _build_panel_why(s), [sb], [panel_why_box])

        # Panel navigation
        for btn, d in [(panel_prev_btn, "pn"), (panel_next_btn, "nn")]:
            btn.click(lambda s, _d=d: _nav(s, _d), [sb], [sb]).then(
                _load_panel, [sb], _panel_out).then(
                lambda s: _build_panel_why(s), [sb], [panel_why_box])

        # Search
        search_btn.click(_do_search, [search_box], [search_gallery, search_ids])
        search_box.submit(_do_search, [search_box], [search_gallery, search_ids])

        # Library gallery click → select
        search_gallery.select(_gallery_click, [sb, search_ids], [sb, selected_preview, selected_name])

        # Recent gallery click → select
        recent_gallery.select(_recent_click, [sb, recent_ids], [sb, selected_preview, selected_name])

        # Panel suggestions
        get_suggest_btn.click(_do_get_suggestions, [sb], [suggest_gallery, suggest_ids])
        suggest_gallery.select(_gallery_click, [sb, suggest_ids], [sb, selected_preview, selected_name])

        # Cast assignment
        cast_char_dd.change(_cast_char_change, [sb, cast_char_dd], [sb])
        cast_char_dd.change(_load_char_look,   [sb, cast_char_dd], [char_look_box])
        cast_set_btn.click(
            _cast_assign, [sb],
            [sb, cast_html_box, recent_gallery, recent_ids],
        )
        cast_clr_btn.click(_cast_clear_fn, [sb], [sb, cast_html_box])
        save_look_btn.click(_save_char_look, [sb, cast_char_dd, char_look_box], [sb, look_status])

        # Class assignment — sync char dropdown with cast on every cast update
        def _sync_cls_char_dd(sb_val):
            return gr.Dropdown(choices=_cast_choices(sb_val), value=None)

        def _refresh_class_dd():
            return gr.Dropdown(choices=_clothing_class_choices())

        sb.change(_sync_cls_char_dd, [sb], [cls_char_dd])
        cls_set_btn.click(_cls_assign, [sb, cls_char_dd, cls_class_dd], [sb, cast_html_box])
        cls_clr_btn.click(_cls_clear,  [sb, cls_char_dd],               [sb, cast_html_box])

        # Page setting
        pg_set_btn.click(
            _pg_set, [sb],
            [sb, pg_set_html, recent_gallery, recent_ids],
        )
        pg_set_clear.click(_pg_clear, [sb], [sb, pg_set_html])

        # Slot set/clear
        for si in range(N_SLOTS):
            slot_set_btns[si].click(
                lambda s, _i=si: _slot_set(s, _i),
                [sb],
                [sb, slots_html, recent_gallery, recent_ids],
            )
            slot_clear_btns[si].click(
                lambda s, _i=si: _slot_clear(s, _i),
                [sb], [sb, slots_html],
            )

        # Auto-fill slots (single panel)
        auto_fill_btn.click(
            _do_auto_fill_slots, [sb], [sb, slots_html, auto_fill_status],
        )

        # Auto-run full story
        auto_run_btn.click(
            _do_auto_run_story, [sb, quality_dd, sb_workers_dd, sb_pages_dd],
            [sb, gen_image, gen_status, story_timeline, zip_file, auto_run_status],
        )

        def _toggle_pause():
            if _GEN_PAUSED.is_set():
                _GEN_PAUSED.clear()
                return gr.update(value="⏸ Pause", variant="secondary")
            else:
                _GEN_PAUSED.set()
                return gr.update(value="▶ Resume", variant="primary")

        pause_run_btn.click(_toggle_pause, [], [pause_run_btn])

        # Beat dropdown
        beat_dd.change(_beat_change, [sb, beat_dd], [sb]).then(_load_panel, [sb], _panel_out)

        # Save edited beat
        save_beat_btn.click(_do_save_beat, [sb, beat_txt], [sb, parse_status])

        # Auto-gen checkbox → show/hide Send to FAL button
        auto_gen_chk.change(_toggle_auto_gen, [auto_gen_chk], [send_fal_btn])

        # Beat Characters quick gallery — clicking a face selects that character
        cast_quick_gallery.select(
            _quick_cast_click, [sb, cast_quick_names], [sb, selected_name]
        )

        # Beat add / remove
        _beat_mod_out = [sb, char_html_box, beat_add_dd, cast_quick_gallery, cast_quick_names]
        beat_add_btn.click(_do_beat_add, [sb, beat_add_dd], _beat_mod_out)
        beat_rm_btn.click( _do_beat_rm,  [sb, beat_add_dd], _beat_mod_out)

        # Appearance coverage — re-check and patch buttons in the Generate section
        re_check_btn.click(_do_prompt_check, [prompt_box, sb], [check_log_box, prompt_check_box])
        patch_prompt_btn.click(_do_patch_and_check, [prompt_box, sb], [prompt_box, check_log_box, prompt_check_box])

        # Story Timeline — clicking a page shows its beat descriptions
        story_timeline.select(_timeline_click, [sb], [timeline_info])

        _gen_outputs = [sb, gen_image, gen_status, prompt_box, ref_sources_gallery, recent_gallery, recent_ids, panel_why_box]

        # Generate Page — builds prompt; fires FAL only if auto_gen is checked
        gen_btn.click(
            _do_generate, [sb, quality_dd, auto_gen_chk],
            _gen_outputs,
        ).then(
            _update_gen_extras, [sb], [story_timeline]
        ).then(
            _do_prompt_check, [prompt_box, sb], [check_log_box, prompt_check_box]
        )

        # Send to FAL — takes the (possibly edited) prompt_box and dispatches it
        send_fal_btn.click(
            _do_send_to_fal_prompt, [sb, quality_dd, prompt_box],
            _gen_outputs,
        ).then(
            _update_gen_extras, [sb], [story_timeline]
        ).then(
            _do_prompt_check, [prompt_box, sb], [check_log_box, prompt_check_box]
        )

        # ZIP
        zip_btn.click(_do_zip, [sb, story_name_box], [zip_file, zip_status])
