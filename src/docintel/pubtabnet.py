"""PubTabNet annotation handling: structure tokens + cell tokens -> HTML, grid stats, text boxes."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any

from docintel.structure import Word

_FORMAT_TAG = re.compile(r"^</?[a-z]+>$")


@dataclass(frozen=True)
class PtnCell:
    tokens: tuple[str, ...]
    bbox: tuple[float, float, float, float] | None

    @property
    def text(self) -> str:
        return "".join(t for t in self.tokens if not _FORMAT_TAG.match(t)).strip()


@dataclass(frozen=True)
class PtnAnnotation:
    structure: tuple[str, ...]
    cells: tuple[PtnCell, ...]


@dataclass(frozen=True)
class GridStats:
    n_rows: int
    n_cols: int
    n_cells: int
    n_spanning: int
    n_header_rows: int


def parse_annotation(raw: str | dict[str, Any]) -> PtnAnnotation:
    """Parse the ``html`` field (a dict, JSON, or a Python-literal string as in the HF mirror)."""
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = ast.literal_eval(raw)
    else:
        data = raw
    cells = []
    for c in data["cells"]:
        bbox = c.get("bbox")
        cells.append(
            PtnCell(
                tuple(c.get("tokens", [])),
                tuple(float(v) for v in bbox) if bbox else None,  # type: ignore[arg-type]
            )
        )
    return PtnAnnotation(tuple(data["structure"]["tokens"]), tuple(cells))


def to_html(ann: PtnAnnotation, with_content: bool = True) -> str:
    """Fill cell text into the structure tokens (formatting tags dropped)."""
    out: list[str] = ["<table>"]
    idx = 0
    for tok in ann.structure:
        out.append(tok)
        # A cell's content goes right after '<td>' or after the '>' closing '<td colspan=..'.
        if tok in ("<td>", ">"):
            if with_content and idx < len(ann.cells):
                text = ann.cells[idx].text
                out.append(text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))
            idx += 1
    out.append("</table>")
    return "".join(out)


def text_boxes(ann: PtnAnnotation) -> list[Word]:
    """Non-empty cells with boxes, used as 'oracle OCR' words for the structure pipeline."""
    return [Word(c.text, c.bbox) for c in ann.cells if c.bbox is not None and c.text]


_SPAN_RE = re.compile(r'(colspan|rowspan)="(\d+)"')


def grid_stats(ann: PtnAnnotation) -> GridStats:
    """Row/column counts from the structure tokens, resolving row/col spans."""
    occupied: set[tuple[int, int]] = set()
    row = -1
    col = 0
    n_cells = n_spanning = 0
    n_header_rows = 0
    in_head = False
    pending: dict[str, int] = {}
    toks = list(ann.structure)
    i = 0
    while i < len(toks):
        tok = toks[i]
        if tok == "<thead>":
            in_head = True
        elif tok == "</thead>":
            in_head = False
        elif tok == "<tr>":
            row += 1
            col = 0
            if in_head:
                n_header_rows += 1
        elif tok in ("<td>", "<td"):
            pending = {"colspan": 1, "rowspan": 1}
            if tok == "<td":
                i += 1
                while i < len(toks) and toks[i] != ">":
                    for name, val in _SPAN_RE.findall(toks[i]):
                        pending[name] = int(val)
                    i += 1
            while (row, col) in occupied:
                col += 1
            for dr in range(pending["rowspan"]):
                for dc in range(pending["colspan"]):
                    occupied.add((row + dr, col + dc))
            n_cells += 1
            if pending["rowspan"] > 1 or pending["colspan"] > 1:
                n_spanning += 1
            col += pending["colspan"]
        i += 1
    n_rows = max((r for r, _ in occupied), default=-1) + 1
    n_cols = max((c for _, c in occupied), default=-1) + 1
    return GridStats(n_rows, n_cols, n_cells, n_spanning, n_header_rows)
