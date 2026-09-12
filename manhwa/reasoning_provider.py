"""reasoning_provider.py — global reasoning provider switch.

DeepSeek mode (default):
  - deepseek-flash (DeepSeek V4.1 Flash) for all text/JSON/vision reasoning.

Normal mode:
  - Existing Claude / OpenAI call sites stay unchanged.

This module deliberately does not handle image generation (FAL / Nano Banana).
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import time
from typing import Any, Dict, Optional, Tuple

DEEPSEEK_MODEL = os.getenv("DEEPSEEK_REASONING_MODEL", "deepseek-flash").strip() or "deepseek-flash"
DEEPSEEK_API_URL = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions").strip()
_REASONING_MODE = (os.getenv("REASONING_MODE", "deepseek") or "deepseek").strip().lower()
if _REASONING_MODE not in {"deepseek", "normal"}:
    _REASONING_MODE = "deepseek"


def get_mode() -> str:
    return _REASONING_MODE


def is_deepseek_mode() -> bool:
    return _REASONING_MODE == "deepseek"


def set_mode(mode: str) -> str:
    global _REASONING_MODE
    raw = (mode or "").strip().lower()
    if raw.startswith("deepseek") or "deepseek" in raw:
        _REASONING_MODE = "deepseek"
    elif raw.startswith("normal") or "normal" in raw:
        _REASONING_MODE = "normal"
    else:
        raise ValueError(f"Unknown reasoning mode: {mode}")
    return _REASONING_MODE


def _deepseek_key() -> str:
    return (
        os.getenv("DEEPSEEK_API_KEY")
        or os.getenv("DEEPSEEK_KEY")
        or os.getenv("DEEPSEEK_API_TOKEN")
        or ""
    ).strip()


def has_text_provider(normal_provider: str = "anthropic") -> bool:
    """Provider-aware availability check used by legacy Claude/OpenAI call sites."""
    if is_deepseek_mode():
        return bool(_deepseek_key())
    if normal_provider == "openai":
        return bool((os.getenv("OPENAI_API_KEY") or "").strip())
    return bool((os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY") or "").strip())


def active_model_label(normal_label: str = "Normal") -> str:
    return "DeepSeek V4.1 Flash" if is_deepseek_mode() else normal_label


def mode_status_markdown() -> str:
    if is_deepseek_mode():
        if _deepseek_key():
            return "### ⚡ ACTIVE: DeepSeek V4.1 Flash\nAll reasoning, prompt writing, checks, and vision analysis use **deepseek-flash**."
        return "### ⚠️ ACTIVE: DeepSeek V4.1 Flash — API KEY MISSING\nAdd **DEEPSEEK_API_KEY** to the runtime secrets before generating."
    return "### 🧠 ACTIVE: Normal Mode\nUses the existing **Claude + GPT-4.1 mini** reasoning stack."


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    raw = text.strip()
    raw = re.sub(r"^\x60\x60\x60(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*\x60\x60\x60$", "", raw)
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _deepseek_request(
    messages: list,
    max_tokens: int = 4000,
    temperature: float = 0.0,
    json_mode: bool = False,
    timeout: int = 180,
) -> Tuple[Optional[str], str]:
    key = _deepseek_key()
    if not key:
        return None, "missing DEEPSEEK_API_KEY (DeepSeek V4.1 Flash)"

    try:
        import requests
    except Exception as exc:
        return None, f"requests unavailable: {type(exc).__name__}: {exc}"

    payload: Dict[str, Any] = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    last_error = ""
    for attempt in range(3):
        try:
            r = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=(15, timeout))
        except Exception as exc:
            return None, f"DeepSeek request failed: {type(exc).__name__}: {exc}"

        if r.status_code == 429 and attempt < 2:
            try:
                wait = int(r.headers.get("Retry-After", 0)) or (5 * (2 ** attempt))
            except Exception:
                wait = 5 * (2 ** attempt)
            time.sleep(min(wait, 30))
            continue

        try:
            data = r.json()
        except Exception:
            data = {}

        if r.status_code >= 400:
            err = data.get("error") if isinstance(data, dict) else None
            if isinstance(err, dict):
                last_error = str(err.get("message") or err.get("type") or err)
            else:
                last_error = (getattr(r, "text", "") or f"HTTP {r.status_code}")[:400]
            return None, f"DeepSeek HTTP {r.status_code}: {last_error}"

        try:
            choice = (data.get("choices") or [])[0]
            message = choice.get("message") or {}
            text = message.get("content")
            if text:
                return str(text).strip(), f"ok (DeepSeek V4.1 Flash / {DEEPSEEK_MODEL})"
        except Exception:
            pass

        return None, "DeepSeek returned no final text"

    return None, last_error or "DeepSeek request failed"


def call_text(
    system: str,
    user: Any,
    max_tokens: int = 4000,
    temperature: float = 0.0,
) -> Tuple[Optional[str], str]:
    user_text = user if isinstance(user, str) else json.dumps(user, ensure_ascii=False)
    messages = [
        {"role": "system", "content": str(system or "")},
        {"role": "user", "content": user_text},
    ]
    return _deepseek_request(messages, max_tokens=max_tokens, temperature=temperature, json_mode=False)


def call_json(
    system: str,
    user_payload: Any,
    max_tokens: int = 4000,
    temperature: float = 0.0,
) -> Tuple[Optional[Dict[str, Any]], str]:
    user_text = user_payload if isinstance(user_payload, str) else json.dumps(user_payload, ensure_ascii=False)
    messages = [
        {"role": "system", "content": str(system or "") + "\nReturn valid JSON only."},
        {"role": "user", "content": user_text},
    ]
    text, status = _deepseek_request(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        json_mode=True,
    )
    return _extract_json(text or ""), status


def call_vision_json(
    pil_img,
    prompt: str,
    max_tokens: int = 1536,
) -> Tuple[Optional[Dict[str, Any]], str]:
    """Use DeepSeek V4.1 Flash native multimodal input for image analysis."""
    from PIL import Image as _PILImage

    img = pil_img.convert("RGB")
    max_edge = 768
    w, h = img.size
    if max(w, h) > max_edge:
        scale = max_edge / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), _PILImage.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=75)
    b64 = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"

    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_url, "detail": "high"}},
            {"type": "text", "text": str(prompt or "") + "\nReturn valid JSON only."},
        ],
    }]
    text, status = _deepseek_request(
        messages,
        max_tokens=max_tokens,
        temperature=0.0,
        json_mode=True,
    )
    return _extract_json(text or ""), status
