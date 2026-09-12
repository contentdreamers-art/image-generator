"""
Content-aware manhwa panel cutter.

Strategy: the page layout is known (2 columns x N rows), but the AI never
draws panels at exactly uniform sizes — so a blind uniform grid cuts
mid-panel, leaves slivers of neighbours, and sometimes merges two panels.

This cutter detects the REAL panel borders (solid black frame lines and
pure-white gutters that span the page), then snaps each expected grid cut
to the nearest detected border. Result: cuts land exactly on the drawn
borders, captions/speech bubbles inside panels are never split off, and
each crop is edge-trimmed so no black frame or white sliver survives.

Used by both the storyboard ZIP cutter and the Director cut-ZIP feature.
Callers fall back to their old logic if this returns [].
"""

from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

# ── Tunables ──────────────────────────────────────────────────────────────────
_DARK       = 70    # pixel value below this = "border black"
_LIGHT      = 210   # pixel value above this = "gutter white"
_WHITE_MEAN = 238   # a pure-white gutter line: mean above this …
_WHITE_STD  = 8     # … and nearly uniform
_MIN_BAND   = 2     # separator band must be at least this many px thick
_SNAP_FRAC  = 0.45  # snap window: ±45% of one expected panel span
_TRIM_CAP_FRAC     = 0.12   # never trim more than 12% per side
_SAFETY_INSET_FRAC = 0.004  # final inset after trimming (~0.4%, min 3px)


def _separator_mask(gray: np.ndarray, axis: int) -> np.ndarray:
    """Which rows (axis=1) / columns (axis=0) are separator lines.

    A line is a separator when it is either:
    • a pure-white gutter  (uniform, near-white), or
    • a black border line  (≥50% dark pixels and ≥95% of pixels are either
      dark or light — border black + gutter white, no artwork in between).
    """
    mean = gray.mean(axis=axis)
    std  = gray.std(axis=axis)
    dark_frac  = (gray < _DARK ).mean(axis=axis)
    light_frac = (gray > _LIGHT).mean(axis=axis)

    white_gutter = (mean > _WHITE_MEAN) & (std < _WHITE_STD)
    black_border = (dark_frac >= 0.50) & ((dark_frac + light_frac) >= 0.95)
    return white_gutter | black_border


def _band_centers(mask: np.ndarray, edge_skip: int) -> List[int]:
    """Centers of separator bands (runs of consecutive separator lines),
    ignoring bands hugging the outer image edge."""
    centers: List[int] = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if (j - i) >= _MIN_BAND:
                mid = (i + j) // 2
                if edge_skip < mid < n - edge_skip:
                    centers.append(mid)
            i = j
        else:
            i += 1
    return centers


def _snap_cuts(candidates: List[int], n_cuts: int, length: int) -> List[int]:
    """Snap the n_cuts expected uniform cut positions to the nearest detected
    border band. Falls back to the uniform position when no band is near."""
    if n_cuts <= 0:
        return []
    span = length / (n_cuts + 1)
    tol  = span * _SNAP_FRAC
    used: set = set()
    cuts: List[int] = []
    for k in range(1, n_cuts + 1):
        expect = k * span
        best, best_d = None, tol
        for ci, c in enumerate(candidates):
            if ci in used:
                continue
            d = abs(c - expect)
            if d <= best_d:
                best, best_d = ci, d
        if best is not None:
            used.add(best)
            cuts.append(candidates[best])
        else:
            cuts.append(int(round(expect)))
    cuts = sorted(set(cuts))
    return cuts


def _trim_edges(gray: np.ndarray, box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    """Walk each edge inward past leftover border/gutter lines, then apply a
    small safety inset."""
    x0, y0, x1, y1 = box
    sub = gray[y0:y1, x0:x1]
    h, w = sub.shape
    if h < 20 or w < 20:
        return box

    row_sep = _separator_mask(sub, axis=1)
    col_sep = _separator_mask(sub, axis=0)

    cap_y = max(2, int(h * _TRIM_CAP_FRAC))
    cap_x = max(2, int(w * _TRIM_CAP_FRAC))

    t = 0
    while t < cap_y and row_sep[t]:
        t += 1
    b = 0
    while b < cap_y and row_sep[h - 1 - b]:
        b += 1
    l = 0
    while l < cap_x and col_sep[l]:
        l += 1
    r = 0
    while r < cap_x and col_sep[w - 1 - r]:
        r += 1

    iy = max(3, int(h * _SAFETY_INSET_FRAC))
    ix = max(3, int(w * _SAFETY_INSET_FRAC))

    nx0, ny0 = x0 + l + ix, y0 + t + iy
    nx1, ny1 = x1 - r - ix, y1 - b - iy
    if nx1 - nx0 < 20 or ny1 - ny0 < 20:   # over-trimmed → keep original box
        return box
    return (nx0, ny0, nx1, ny1)


def smart_cut_panels(img: Image.Image,
                     n_rows: Optional[int] = None,
                     n_cols: int = 2) -> List[Image.Image]:
    """Cut a manhwa page into panels, snapping the expected n_rows x n_cols
    grid to detected borders. Returns panels in reading order
    (top-to-bottom, left-to-right). Returns [] when detection is impossible."""
    rgb  = img.convert("RGB")
    gray = np.asarray(rgb.convert("L"), dtype=np.float32)
    H, W = gray.shape
    if H < 100 or W < 100:
        return []

    if not n_rows or n_rows < 1:
        n_rows = 1
    n_cols = max(1, n_cols)

    # ── Horizontal cuts: snap expected row boundaries to real borders ────────
    row_mask   = _separator_mask(gray, axis=1)
    row_bands  = _band_centers(row_mask, edge_skip=max(10, H // 50))
    h_cuts     = _snap_cuts(row_bands, n_rows - 1, H)
    row_bounds = [0] + h_cuts + [H]

    boxes: List[Tuple[int, int, int, int]] = []
    for i in range(len(row_bounds) - 1):
        y0, y1 = row_bounds[i], row_bounds[i + 1]
        if y1 - y0 < 40:
            continue
        # ── Vertical cut(s) within this strip, snapped the same way ─────────
        strip     = gray[y0:y1, :]
        col_mask  = _separator_mask(strip, axis=0)
        col_bands = _band_centers(col_mask, edge_skip=max(10, W // 50))
        v_cuts    = _snap_cuts(col_bands, n_cols - 1, W)
        col_bounds = [0] + v_cuts + [W]
        for j in range(len(col_bounds) - 1):
            x0, x1 = col_bounds[j], col_bounds[j + 1]
            if x1 - x0 >= 40:
                boxes.append((x0, y0, x1, y1))

    # ── Trim borders/gutters off every edge, then crop ───────────────────────
    panels: List[Image.Image] = []
    for box in boxes:
        x0, y0, x1, y1 = _trim_edges(gray, box)
        if x1 - x0 >= 40 and y1 - y0 >= 40:
            panels.append(rgb.crop((x0, y0, x1, y1)))
    return panels


def smart_cut_panels_path(img_path: str,
                          n_rows: Optional[int] = None,
                          n_cols: int = 2) -> List[Image.Image]:
    """Convenience wrapper: open by path; [] on any failure."""
    try:
        return smart_cut_panels(Image.open(img_path), n_rows=n_rows, n_cols=n_cols)
    except Exception:
        return []
