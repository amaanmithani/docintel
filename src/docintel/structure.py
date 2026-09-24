"""Model-agnostic table structure: detections -> grid -> cells -> HTML / CSV.

Nothing here imports torch; the Table Transformer wrapper (``tatr.py``) only produces
``Detection`` objects and everything downstream is plain Python so it can be unit tested.
"""

from __future__ import annotations

import csv
import html as html_lib
import io
from dataclasses import dataclass, field

from docintel.geometry import Box, area, center, intersection, overlap_1d, union

ROW = "table row"
COLUMN = "table column"
HEADER = "table column header"
SPAN = "table spanning cell"
TABLE = "table"


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    bbox: Box


@dataclass(frozen=True)
class Word:
    """A piece of text with its box: an OCR word or a dataset-provided cell text box."""

    text: str
    bbox: Box


@dataclass
class Cell:
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1
    header: bool = False
    words: list[Word] = field(default_factory=list)

    @property
    def text(self) -> str:
        ordered = sorted(self.words, key=lambda w: (round(center(w.bbox)[1] / 4), w.bbox[0]))
        return " ".join(w.text for w in ordered if w.text)


@dataclass
class Table:
    n_rows: int
    n_cols: int
    cells: list[Cell]
    row_boxes: list[Box] = field(default_factory=list)
    col_boxes: list[Box] = field(default_factory=list)
    n_header_rows: int = 0

    @property
    def n_spanning(self) -> int:
        return sum(1 for c in self.cells if c.rowspan > 1 or c.colspan > 1)

    def to_html(self) -> str:
        by_row: dict[int, list[Cell]] = {}
        for c in self.cells:
            by_row.setdefault(c.row, []).append(c)
        n_head = self.n_header_rows
        parts = ["<table>"]
        for section, rows in (("thead", range(n_head)), ("tbody", range(n_head, self.n_rows))):
            if len(rows) == 0:
                continue
            parts.append(f"<{section}>")
            for r in rows:
                parts.append("<tr>")
                for c in sorted(by_row.get(r, []), key=lambda c: c.col):
                    attrs = ""
                    if c.colspan > 1:
                        attrs += f' colspan="{c.colspan}"'
                    if c.rowspan > 1:
                        attrs += f' rowspan="{c.rowspan}"'
                    parts.append(f"<td{attrs}>{html_lib.escape(c.text)}</td>")
                parts.append("</tr>")
            parts.append(f"</{section}>")
        parts.append("</table>")
        return "".join(parts)

    def to_grid(self) -> list[list[str]]:
        """Dense grid; spanned positions repeat nothing (left empty) except the origin."""
        grid = [["" for _ in range(self.n_cols)] for _ in range(self.n_rows)]
        for c in self.cells:
            grid[c.row][c.col] = c.text
        return grid

    def to_csv(self) -> str:
        buf = io.StringIO()
        csv.writer(buf, lineterminator="\n").writerows(self.to_grid())
        return buf.getvalue()


@dataclass(frozen=True)
class PostprocessConfig:
    row_threshold: float = 0.5
    col_threshold: float = 0.5
    span_threshold: float = 0.5
    header_threshold: float = 0.5
    # PubTables-1M "canonicalizes" headers by merging blank cells into row-spanning cells;
    # PubTabNet does not. When set, a header span whose text sits inside one row is split.
    split_header_rowspans: bool = False


def _dedupe(boxes: list[Detection], axis: int) -> list[Detection]:
    """Sort along an axis (1 = rows by y, 0 = columns by x) and drop heavy overlaps."""
    kept: list[Detection] = []
    for d in sorted(boxes, key=lambda d: -d.score):
        lo, hi = d.bbox[axis], d.bbox[axis + 2]
        clash = False
        for k in kept:
            ov = overlap_1d(lo, hi, k.bbox[axis], k.bbox[axis + 2])
            smaller = min(hi - lo, k.bbox[axis + 2] - k.bbox[axis])
            if smaller <= 0 or ov / smaller > 0.5:
                clash = True
                break
        if not clash:
            kept.append(d)
    return sorted(kept, key=lambda d: d.bbox[axis] + d.bbox[axis + 2])


def _covered(lo: float, hi: float, bands: list[Box], axis: int, frac: float = 0.5) -> list[int]:
    out = []
    for i, b in enumerate(bands):
        size = b[axis + 2] - b[axis]
        if size > 0 and overlap_1d(lo, hi, b[axis], b[axis + 2]) / size >= frac:
            out.append(i)
    return out


def assign_words(
    words: list[Word], row_boxes: list[Box], col_boxes: list[Box]
) -> dict[tuple[int, int], list[Word]]:
    """Map each word to the grid slot (row, col) it overlaps most (nearest centre if none)."""
    slots: dict[tuple[int, int], list[Word]] = {}
    if not row_boxes or not col_boxes:
        return slots
    for w in words:
        cx, cy = center(w.bbox)
        r_ov = [overlap_1d(w.bbox[1], w.bbox[3], b[1], b[3]) for b in row_boxes]
        c_ov = [overlap_1d(w.bbox[0], w.bbox[2], b[0], b[2]) for b in col_boxes]
        if max(r_ov) > 0:
            r = max(range(len(r_ov)), key=lambda i: r_ov[i])
        else:
            r = min(range(len(row_boxes)), key=lambda i: abs(center(row_boxes[i])[1] - cy))
        if max(c_ov) > 0:
            c = max(range(len(c_ov)), key=lambda i: c_ov[i])
        else:
            c = min(range(len(col_boxes)), key=lambda i: abs(center(col_boxes[i])[0] - cx))
        slots.setdefault((r, c), []).append(w)
    return slots


def grid_table(
    row_boxes: list[Box],
    col_boxes: list[Box],
    words: list[Word],
    spans: list[Box] | None = None,
    header_box: Box | None = None,
    split_header_rowspans: bool = False,
) -> Table:
    """Build a Table from row/column bands, optional spanning-cell and header boxes, and words."""
    n_rows, n_cols = len(row_boxes), len(col_boxes)
    if n_rows == 0 or n_cols == 0:
        return Table(0, 0, [], row_boxes, col_boxes)
    slots = assign_words(words, row_boxes, col_boxes)
    owner: dict[tuple[int, int], Cell] = {}
    cells: list[Cell] = []

    header_rows: set[int] = set()
    if header_box is not None:
        header_rows = set(_covered(header_box[1], header_box[3], row_boxes, axis=1))
        # A header must start at the top and be contiguous.
        n = 0
        while n in header_rows:
            n += 1
        header_rows = set(range(n))

    for sb in spans or []:
        rows = _covered(sb[1], sb[3], row_boxes, axis=1)
        cols = _covered(sb[0], sb[2], col_boxes, axis=0)
        if not rows or not cols:
            continue
        r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
        if (r1 - r0 + 1) * (c1 - c0 + 1) < 2:
            continue
        if any((r, c) in owner for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)):
            continue
        # Never let a span straddle the header/body boundary.
        in_head = {r in header_rows for r in range(r0, r1 + 1)}
        if len(in_head) > 1:
            continue
        cell_words = [
            w for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) for w in slots.get((r, c), [])
        ]
        if split_header_rowspans and r0 in header_rows and r1 > r0:
            text_rows = {
                r for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if slots.get((r, c))
            }
            if len(text_rows) <= 1:
                if c1 == c0:
                    continue  # plain vertical merge of blank header cells: drop the span
                keep_row = min(text_rows) if text_rows else r0
                r0 = r1 = keep_row
                cell_words = [w for c in range(c0, c1 + 1) for w in slots.get((keep_row, c), [])]
        cell = Cell(r0, c0, r1 - r0 + 1, c1 - c0 + 1, r0 in header_rows, cell_words)
        cells.append(cell)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                owner[(r, c)] = cell

    for r in range(n_rows):
        for c in range(n_cols):
            if (r, c) not in owner:
                cell = Cell(r, c, header=r in header_rows, words=list(slots.get((r, c), [])))
                cells.append(cell)
                owner[(r, c)] = cell
    cells.sort(key=lambda c: (c.row, c.col))
    return Table(n_rows, n_cols, cells, row_boxes, col_boxes, len(header_rows))


def build_table(
    detections: list[Detection],
    words: list[Word],
    config: PostprocessConfig | None = None,
) -> Table:
    """Turn raw structure detections plus text boxes into a Table."""
    cfg = config or PostprocessConfig()
    rows = _dedupe([d for d in detections if d.label == ROW and d.score >= cfg.row_threshold], 1)
    cols = _dedupe([d for d in detections if d.label == COLUMN and d.score >= cfg.col_threshold], 0)
    if not rows or not cols:
        return Table(0, 0, [])
    tables = [d for d in detections if d.label == TABLE]
    extent = (
        max(tables, key=lambda d: d.score).bbox if tables else union([d.bbox for d in rows + cols])
    )
    # Rows span the table width and columns its height; this removes ragged detections.
    x1, x2 = min(c.bbox[0] for c in cols), max(c.bbox[2] for c in cols)
    y1, y2 = min(r.bbox[1] for r in rows), max(r.bbox[3] for r in rows)
    x1, x2 = min(x1, extent[0]), max(x2, extent[2])
    y1, y2 = min(y1, extent[1]), max(y2, extent[3])
    row_boxes: list[Box] = [(x1, r.bbox[1], x2, r.bbox[3]) for r in rows]
    col_boxes: list[Box] = [(c.bbox[0], y1, c.bbox[2], y2) for c in cols]
    headers = [d for d in detections if d.label == HEADER and d.score >= cfg.header_threshold]
    header_box = max(headers, key=lambda d: d.score).bbox if headers else None
    spans = sorted(
        (d for d in detections if d.label == SPAN and d.score >= cfg.span_threshold),
        key=lambda d: -d.score,
    )
    return grid_table(
        row_boxes,
        col_boxes,
        words,
        spans=[s.bbox for s in spans],
        header_box=header_box,
        split_header_rowspans=cfg.split_header_rowspans,
    )


def words_inside(words: list[Word], box: Box, frac: float = 0.5) -> list[Word]:
    return [
        w for w in words if area(w.bbox) > 0 and intersection(w.bbox, box) / area(w.bbox) >= frac
    ]
