"""Non-learned baselines, so the model numbers have context.

* ``projection_table``: classic whitespace-projection splitter on the image (rows = horizontal
  ink bands, columns = vertical gaps across the whole table). No headers, no spans.
* ``textbox_cluster_table``: clusters the (oracle) text boxes into rows by vertical overlap
  and into columns by horizontal overlap. No headers, no spans.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from docintel.geometry import Box
from docintel.structure import Table, Word, grid_table


def ink_mask(image: Image.Image, threshold: int = 180) -> NDArray[np.bool_]:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    return gray < threshold


def _runs(flags: NDArray[np.bool_]) -> list[tuple[int, int]]:
    """Half-open [start, end) runs of True values."""
    runs = []
    start = None
    for i, f in enumerate(flags.tolist()):
        if f and start is None:
            start = i
        elif not f and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(flags)))
    return runs


def rule_lines(mask: NDArray[np.bool_], min_frac: float = 0.6) -> tuple[int, int]:
    """Count (horizontal, vertical) ruling lines: pixel rows/cols that are mostly ink."""
    if mask.size == 0:
        return 0, 0
    h_flags = mask.mean(axis=1) >= min_frac
    v_flags = mask.mean(axis=0) >= min_frac
    return len(_runs(h_flags)), len(_runs(v_flags))


def _bands(profile: NDArray[np.bool_], min_gap: int) -> list[tuple[int, int]]:
    """Ink runs merged across gaps narrower than ``min_gap``."""
    runs = _runs(profile)
    merged: list[tuple[int, int]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


def projection_table(
    image: Image.Image, words: list[Word], row_gap: int = 2, col_gap: int = 10
) -> Table:
    mask = ink_mask(image)
    if mask.size == 0 or not mask.any():
        return Table(0, 0, [])
    # Erase ruling lines so they do not glue bands together.
    clean = mask.copy()
    clean[mask.mean(axis=1) >= 0.6, :] = False
    clean[:, mask.mean(axis=0) >= 0.6] = False
    rows = _bands(clean.any(axis=1), row_gap)
    cols = _bands(clean.any(axis=0), col_gap)
    h, w = mask.shape
    row_boxes: list[Box] = [(0.0, float(s), float(w), float(e)) for s, e in rows]
    col_boxes: list[Box] = [(float(s), 0.0, float(e), float(h)) for s, e in cols]
    return grid_table(row_boxes, col_boxes, words)


def _cluster(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for lo, hi in sorted(intervals):
        if merged and lo < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def textbox_cluster_table(words: list[Word]) -> Table:
    if not words:
        return Table(0, 0, [])
    rows = _cluster([(w.bbox[1], w.bbox[3]) for w in words])
    cols = _cluster([(w.bbox[0], w.bbox[2]) for w in words])
    x1 = min(w.bbox[0] for w in words)
    x2 = max(w.bbox[2] for w in words)
    y1 = min(w.bbox[1] for w in words)
    y2 = max(w.bbox[3] for w in words)
    row_boxes: list[Box] = [(x1, lo, x2, hi) for lo, hi in rows]
    col_boxes: list[Box] = [(lo, y1, hi, y2) for lo, hi in cols]
    return grid_table(row_boxes, col_boxes, words)
