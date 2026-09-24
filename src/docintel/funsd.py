"""Form key-value linking on FUNSD, given oracle entities.

Scope (deliberately narrow): FUNSD annotates entities (question / answer / header / other)
and links between them. We take the *gold* entities and labels as input and predict which
question each answer belongs to. Entity detection and labelling from raw OCR is NOT done
here, so these numbers are an upper-bound-style measure of the linking step only.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from docintel.geometry import Box, center, overlap_1d
from docintel.stats import bootstrap_statistic


@dataclass(frozen=True)
class Entity:
    id: int
    label: str
    text: str
    box: Box


@dataclass
class Form:
    name: str
    entities: list[Entity]
    links: set[tuple[int, int]]  # (question_id, answer_id)


@dataclass(frozen=True)
class LinkConfig:
    above_weight: float = 1.0  # penalty multiplier for "question above answer"
    max_distance: float = 400.0  # px; answers farther than this from any question stay unlinked


def parse_form(name: str, data: dict[str, Any]) -> Form:
    ents = [
        Entity(int(e["id"]), str(e["label"]), str(e.get("text", "")), tuple(e["box"]))
        for e in data["form"]
    ]
    label = {e.id: e.label for e in ents}
    links: set[tuple[int, int]] = set()
    for e in data["form"]:
        for a, b in e.get("linking", []):
            if label.get(a) == "question" and label.get(b) == "answer":
                links.add((a, b))
            elif label.get(b) == "question" and label.get(a) == "answer":
                links.add((b, a))
    return Form(name, ents, links)


def load_split(root: Path, split: str) -> list[Form]:
    folder = root / ("training_data" if split == "train" else "testing_data") / "annotations"
    return [parse_form(p.stem, json.loads(p.read_text())) for p in sorted(folder.glob("*.json"))]


def _geo_cost(q: Box, a: Box, cfg: LinkConfig) -> float | None:
    qh, ah = q[3] - q[1], a[3] - a[1]
    same_line = overlap_1d(q[1], q[3], a[1], a[3]) >= 0.5 * max(1.0, min(qh, ah))
    if same_line and q[2] <= a[0] + 0.25 * (a[2] - a[0]):
        return max(0.0, a[0] - q[2])
    if overlap_1d(q[0], q[2], a[0], a[2]) > 0 and q[3] <= a[1] + 0.5 * ah:
        return max(0.0, a[1] - q[3]) * cfg.above_weight
    return None


def link_geometric(form: Form, cfg: LinkConfig | None = None) -> set[tuple[int, int]]:
    """Each answer links to the closest question to its left (same line) or directly above."""
    cfg = cfg or LinkConfig()
    qs = [e for e in form.entities if e.label == "question"]
    out = set()
    for a in (e for e in form.entities if e.label == "answer"):
        best: tuple[float, int] | None = None
        for q in qs:
            cost = _geo_cost(q.box, a.box, cfg)
            if cost is not None and cost <= cfg.max_distance and (best is None or cost < best[0]):
                best = (cost, q.id)
        if best is not None:
            out.add((best[1], a.id))
    return out


def link_nearest(form: Form) -> set[tuple[int, int]]:
    """Baseline: each answer links to the question with the nearest centre."""
    qs = [e for e in form.entities if e.label == "question"]
    out: set[tuple[int, int]] = set()
    if not qs:
        return out
    for a in (e for e in form.entities if e.label == "answer"):
        ax, ay = center(a.box)
        q = min(qs, key=lambda q: math.dist(center(q.box), (ax, ay)))
        out.add((q.id, a.id))
    return out


def prf(tp: int, n_pred: int, n_gold: int) -> dict[str, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f}


def score_forms(
    forms: list[Form], preds: Iterable[set[tuple[int, int]]], seed: int = 0
) -> dict[str, Any]:
    counts = np.array(
        [(len(p & f.links), len(p), len(f.links)) for f, p in zip(forms, preds, strict=True)],
        dtype=float,
    ).reshape(-1, 3)

    def micro_f1(idx: np.ndarray) -> float:
        tp, npred, ngold = counts[idx].sum(axis=0)
        return prf(int(tp), int(npred), int(ngold))["f1"]

    tp, npred, ngold = (int(v) for v in counts.sum(axis=0))
    return {
        **prf(tp, npred, ngold),
        "tp": tp,
        "n_pred": npred,
        "n_gold": ngold,
        "f1_ci": bootstrap_statistic(len(forms), micro_f1, seed=seed) if len(forms) else None,
    }


def calibrate_linking(forms: list[Form]) -> tuple[LinkConfig, list[dict[str, Any]]]:
    grid = []
    for w in (0.5, 1.0, 1.5, 2.0, 3.0):
        for d in (100.0, 200.0, 400.0, 800.0):
            cfg = LinkConfig(w, d)
            res = score_forms(forms, [link_geometric(f, cfg) for f in forms])
            grid.append({"above_weight": w, "max_distance": d, "f1": res["f1"]})
    best = max(grid, key=lambda g: g["f1"])
    return LinkConfig(best["above_weight"], best["max_distance"]), grid


def evaluate_funsd(root: Path, seed: int = 0) -> dict[str, Any]:
    train, test = load_split(root, "train"), load_split(root, "test")
    cfg, grid = calibrate_linking(train)
    return {
        "task": "question->answer linking with gold entities and labels",
        "calibration": {
            "split": "train",
            "n_forms": len(train),
            "best": cfg.__dict__,
            "grid": grid,
        },
        "test": {
            "n_forms": len(test),
            "n_gold_links": sum(len(f.links) for f in test),
            "geometric": score_forms(test, [link_geometric(f, cfg) for f in test], seed=seed),
            "nearest_center_baseline": score_forms(
                test, [link_nearest(f) for f in test], seed=seed
            ),
        },
    }
