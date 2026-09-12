"""
semantic_search.py — Embedding-based semantic search for the character library.

Uses OpenAI text-embedding-3-small (~$0.002 per 1,800 templates one-time).
Embeddings stored as compressed numpy file for near-instant load.

Usage:
    import semantic_search as sem
    sem.ensure_current()           # embed any new templates
    results = sem.search("angry crowd staring at a boy", top_k=20)
    # → [(template_id, cosine_score), ...]
"""

import os
import numpy as np
import reasoning_provider as _rp
from typing import List, Tuple, Optional

_DIR   = os.path.join(os.path.dirname(__file__), "character_library")
_NPZ   = os.path.join(_DIR, "embeddings.npz")
_MODEL = "text-embedding-3-small"
_BATCH = 500   # OpenAI max per request

# In-process cache: (ids_list, float32 matrix)
_cache_ids:    Optional[List[str]]    = None
_cache_matrix: Optional[np.ndarray]  = None
_cache_mtime:  float                  = 0.0


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_cache() -> Tuple[List[str], np.ndarray]:
    """Return (ids, matrix) from disk, using in-process cache."""
    global _cache_ids, _cache_matrix, _cache_mtime
    if os.path.exists(_NPZ):
        mtime = os.path.getmtime(_NPZ)
    else:
        mtime = 0.0

    if _cache_ids is not None and mtime <= _cache_mtime:
        return _cache_ids, _cache_matrix  # type: ignore[return-value]

    if not os.path.exists(_NPZ):
        _cache_ids    = []
        _cache_matrix = np.empty((0, 1536), dtype=np.float32)
        _cache_mtime  = 0.0
        return _cache_ids, _cache_matrix

    data          = np.load(_NPZ, allow_pickle=True)
    _cache_ids    = data["ids"].tolist()
    _cache_matrix = data["matrix"].astype(np.float32)
    _cache_mtime  = mtime
    return _cache_ids, _cache_matrix


def _save(ids: List[str], matrix: np.ndarray) -> None:
    global _cache_ids, _cache_matrix, _cache_mtime
    os.makedirs(_DIR, exist_ok=True)
    tmp = _NPZ + ".tmp.npz"   # numpy appends .npz unless the name already ends with it
    np.savez_compressed(tmp,
                        ids=np.array(ids, dtype=object),
                        matrix=matrix.astype(np.float32))
    os.replace(tmp, _NPZ)
    _cache_ids    = ids
    _cache_matrix = matrix.astype(np.float32)
    _cache_mtime  = os.path.getmtime(_NPZ)


def _embed_texts(texts: List[str]) -> np.ndarray:
    """Call OpenAI embeddings API in batches. Returns float32 ndarray."""
    from openai import OpenAI
    client = OpenAI()
    vecs: List[List[float]] = []
    for i in range(0, len(texts), _BATCH):
        batch = texts[i : i + _BATCH]
        resp  = client.embeddings.create(model=_MODEL, input=batch)
        vecs.extend(e.embedding for e in resp.data)
    return np.array(vecs, dtype=np.float32)


def _template_text(t: dict) -> str:
    """Build searchable text from a template record."""
    parts: List[str] = []
    name = (t.get("name") or "").strip()
    # Skip bare IDs like "O1234" — they add no signal
    if name and not (len(name) <= 6 and name[0] in "OQo"):
        parts.append(name)
    tags = t.get("tags") or []
    if tags:
        parts.append(", ".join(tags))
    ai      = t.get("ai_analysis") or {}
    summary = (ai.get("summary") or t.get("summary") or "").strip()
    if summary:
        parts.append(summary[:400])
    return ". ".join(parts) or (t.get("template_id") or "unknown")


# ── Public API ────────────────────────────────────────────────────────────────

def is_ready() -> bool:
    """True if the index exists and has at least one entry."""
    ids, _ = _load_cache()
    return len(ids) > 0


def index_size() -> int:
    ids, _ = _load_cache()
    return len(ids)


def ensure_current(progress_cb=None) -> int:
    """
    Embed any templates not yet in the index.
    Only processes templates that have a local image.
    Returns count of newly embedded templates.
    """
    import character_library as cl

    ids_existing, _ = _load_cache()
    existing_set     = set(ids_existing)

    lib       = cl.load_library()
    templates = lib.get("templates", {})

    to_embed: dict = {
        tid: t
        for tid, t in templates.items()
        if tid not in existing_set and t.get("local_face")
    }
    if not to_embed:
        return 0

    tids  = list(to_embed.keys())
    texts = [_template_text(to_embed[tid]) for tid in tids]

    if progress_cb:
        progress_cb(0, len(tids), f"Embedding {len(tids)} templates…")

    new_vecs = _embed_texts(texts)

    if ids_existing:
        all_ids    = list(ids_existing) + tids
        all_matrix = np.vstack([_cache_matrix, new_vecs])  # type: ignore[arg-type]
    else:
        all_ids    = tids
        all_matrix = new_vecs

    _save(all_ids, all_matrix)

    if progress_cb:
        progress_cb(len(tids), len(tids), f"✅ {len(tids)} templates indexed")

    return len(tids)


def search(query: str, top_k: int = 100) -> List[Tuple[str, float]]:
    """
    Embedding-based first pass. Returns [(template_id, cosine_similarity), ...] sorted descending.
    Use search_with_rerank() for the full approach-3 pipeline.
    """
    if not (query or "").strip():
        return []

    ids, matrix = _load_cache()
    if not ids:
        return []

    q_arr  = _embed_texts([query])[0]
    q_arr /= (np.linalg.norm(q_arr) + 1e-9)

    norms   = np.linalg.norm(matrix, axis=1, keepdims=True)
    normed  = matrix / (norms + 1e-9)
    scores  = normed @ q_arr          # shape (N,)

    top_idx = np.argsort(-scores)[: top_k * 3]   # over-fetch, filter below

    results: List[Tuple[str, float]] = []
    for i in top_idx:
        score = float(scores[i])
        if score < 0.15:
            break
        results.append((ids[i], score))
        if len(results) >= top_k:
            break

    return results


def rerank(query: str, candidates: List[Tuple[str, float]], top_k: int = 20) -> List[Tuple[str, float]]:
    """
    LLM re-rank pass (Claude Haiku): reads the top candidates and picks the best top_k.

    Sends a compact description of each candidate (name + tags + summary snippet) to a
    cheap, fast LLM. The LLM understands composition type, mood, action, and visual concept —
    not just keyword tags — so "angry crowd watching a boy" finds the right group composition
    even if the tags say "tense" and "outdoor" without "crowd".

    Cost: ~$0.001–0.003 per search (Claude Haiku, ~1,500 tokens input).
    """
    import json, re as _re
    import character_library as cl

    if not candidates:
        return []
    if len(candidates) <= top_k:
        return candidates

    # Build compact single-line descriptions for each candidate
    lines: List[str] = []
    for i, (tid, _score) in enumerate(candidates):
        t = cl.get_template(tid)
        if not t:
            lines.append(f"{i+1}. [{tid}]")
            continue
        name    = (t.get("name") or tid)[:50]
        tags    = ", ".join((t.get("tags") or [])[:12])
        ai      = t.get("ai_analysis") or {}
        summary = (ai.get("summary") or t.get("summary") or "")[:100]
        line    = f"{i+1}. [{tid}] {name}"
        if tags:
            line += f" | {tags}"
        if summary:
            line += f" | {summary}"
        lines.append(line)

    prompt = (
        f'You select visual reference images for a Korean manhwa comic.\n\n'
        f'The user is looking for: "{query}"\n\n'
        f'Here are {len(candidates)} candidate images '
        f'(format: #. [ID] Name | tags | description):\n\n'
        + "\n".join(lines)
        + f'\n\nPick the {top_k} images that BEST match the search intent. '
        f'Think about: composition type (crowd vs solo), mood, action, pose, setting, visual concept. '
        f'Return ONLY a JSON array of IDs in order from best to worst. '
        f'Example: ["O612", "O1390", "O657"]\n'
        f'Return nothing else.'
    )

    try:
        if _rp.is_deepseek_mode():
            text, _status = _rp.call_text(
                "You rank visual reference candidates for a Korean manhwa project. Return only the requested JSON array.",
                prompt,
                max_tokens=400,
                temperature=0,
            )
            text = (text or "").strip()
        else:
            import anthropic
            client  = anthropic.Anthropic()
            msg     = client.messages.create(
                model      = "claude-haiku-4-5",
                max_tokens = 400,
                messages   = [{"role": "user", "content": prompt}],
            )
            text = (msg.content[0].text or "").strip()
        m    = _re.search(r'\[.*?\]', text, _re.DOTALL)
        if m:
            ranked_ids      = json.loads(m.group())
            score_map       = {tid: s for tid, s in candidates}
            seen: set       = set()
            results: List[Tuple[str, float]] = []
            # LLM-ordered results get a descending rank score
            for rank, tid in enumerate(ranked_ids):
                if tid in score_map and tid not in seen:
                    results.append((tid, 1.0 - rank * 0.02))
                    seen.add(tid)
                if len(results) >= top_k:
                    break
            # Pad with top embedding results if LLM returned fewer than top_k
            for tid, s in candidates:
                if tid not in seen and len(results) < top_k:
                    results.append((tid, s * 0.4))
                    seen.add(tid)
            return results
    except Exception:
        pass

    # Fallback: no re-rank, return embedding top_k
    return candidates[:top_k]


def search_with_rerank(query: str, first_pass: int = 100, final: int = 20) -> List[Tuple[str, float]]:
    """
    Full Approach 3 pipeline:
      1. Embeddings → top `first_pass` candidates (fast cosine similarity)
      2. LLM re-rank → best `final` results (Claude Haiku understands intent)

    Returns [(template_id, rank_score), ...] sorted best-first.
    Falls back gracefully to embedding-only if re-rank fails.
    """
    candidates = search(query, top_k=first_pass)
    if not candidates:
        return []
    return rerank(query, candidates, top_k=final)
