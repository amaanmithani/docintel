"""Tree-Edit-Distance-based Similarity (TEDS) for HTML tables.

Follows the definition of Zhong et al. (2020), "Image-based table recognition: data, model,
and evaluation" (the PubTabNet paper):

    TEDS(Ta, Tb) = 1 - EditDist(Ta, Tb) / max(|Ta|, |Tb|)

where trees are built from the ``<table>`` element, insert/delete cost 1, and renaming a node
costs 1 if tag, colspan or rowspan differ; for two ``<td>`` nodes with the same spans the cost
is the normalised Levenshtein distance between their contents (0 in structure-only mode,
which gives TEDS-Struct).

Content is compared as a sequence of characters. Inline formatting tags (``<b>``, ``<i>``,
``<sup>`` ...) are dropped from both sides and whitespace is collapsed; the official scorer keeps
formatting tags as tokens. This matters only for full TEDS, never for TEDS-Struct.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from apted import APTED, Config
from lxml import html as lxml_html
from rapidfuzz.distance import Levenshtein as _RfLevenshtein

_KEEP_TAGS = {"table", "thead", "tbody", "tr", "td", "th"}


@dataclass
class TableNode:
    tag: str
    colspan: int = 1
    rowspan: int = 1
    content: str = ""
    children: list[TableNode] = field(default_factory=list)

    def size(self) -> int:
        return 1 + sum(c.size() for c in self.children)


def levenshtein(a: str, b: str) -> int:
    """Character edit distance with unit costs (rapidfuzz, C implementation)."""
    return int(_RfLevenshtein.distance(a, b))


@lru_cache(maxsize=200_000)
def normalized_levenshtein(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    return levenshtein(a, b) / max(len(a), len(b))


def _cell_text(el: lxml_html.HtmlElement) -> str:
    return " ".join("".join(el.itertext()).split())


def _span(el: lxml_html.HtmlElement, name: str) -> int:
    try:
        return max(1, int(el.get(name, "1")))
    except ValueError:
        return 1


def _build(el: lxml_html.HtmlElement, structure_only: bool) -> TableNode:
    tag = "td" if el.tag == "th" else str(el.tag)
    if tag == "td":
        content = "" if structure_only else _cell_text(el)
        return TableNode("td", _span(el, "colspan"), _span(el, "rowspan"), content)
    node = TableNode(tag)
    for child in el:
        if isinstance(child.tag, str) and child.tag in _KEEP_TAGS:
            node.children.append(_build(child, structure_only))
    return node


def html_to_tree(html: str, structure_only: bool = False) -> TableNode | None:
    """Parse an HTML string and return the tree rooted at its first ``<table>``."""
    if not html.strip():
        return None
    doc = lxml_html.fromstring(html)
    tables = [doc] if doc.tag == "table" else list(doc.iter("table"))
    if not tables:
        return None
    return _build(tables[0], structure_only)


class _TedsConfig(Config):  # type: ignore[misc]
    def rename(self, a: TableNode, b: TableNode) -> float:
        if a.tag != b.tag or a.colspan != b.colspan or a.rowspan != b.rowspan:
            return 1.0
        if a.tag == "td":
            return normalized_levenshtein(a.content, b.content)
        return 0.0

    def children(self, node: TableNode) -> list[TableNode]:
        return node.children


def signature(node: TableNode) -> str:
    """Canonical string of a tree; equal signatures imply TEDS = 1."""
    inner = "".join(signature(c) for c in node.children)
    return f"<{node.tag} {node.colspan} {node.rowspan} {node.content!r}>{inner}</>"


def teds_trees(pred: TableNode | None, true: TableNode | None) -> float:
    if pred is None or true is None:
        return 1.0 if pred is None and true is None else 0.0
    if signature(pred) == signature(true):
        return 1.0
    n = max(pred.size(), true.size())
    dist = float(APTED(pred, true, _TedsConfig()).compute_edit_distance())
    return 1.0 - dist / n


def teds(pred_html: str, true_html: str, structure_only: bool = False) -> float:
    """TEDS between two HTML table strings (TEDS-Struct when ``structure_only``)."""
    return teds_trees(
        html_to_tree(pred_html, structure_only), html_to_tree(true_html, structure_only)
    )
