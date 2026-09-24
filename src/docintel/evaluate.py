"""PubTabNet evaluation: calibrate on a disjoint sample, then TEDS / TEDS-Struct on the eval set."""

from __future__ import annotations

import itertools
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from docintel.baselines import ink_mask, projection_table, rule_lines, textbox_cluster_table
from docintel.datasets import TableSample
from docintel.pubtabnet import grid_stats, text_boxes, to_html
from docintel.stats import bootstrap_ci
from docintel.structure import Detection, PostprocessConfig, Table, build_table
from docintel.tatr import StructureModel
from docintel.teds import html_to_tree, teds_trees

LARGE_ROWS = 20
CATEGORIES = ("has_spans", "multirow_header", "no_vertical_rules", "large")
MODEL_SYSTEM = "tatr"
BASELINES = ("textbox_cluster", "projection")


@dataclass
class Observed:
    sample: TableSample
    detections: list[Detection]
    seconds: float


def observe(samples: Iterable[TableSample], model: StructureModel) -> list[Observed]:
    """Run the model once per table; post-processing variants reuse these detections."""
    out = []
    for s in samples:
        t0 = time.perf_counter()
        dets = model.predict(s.image)
        out.append(Observed(s, dets, time.perf_counter() - t0))
    return out


def categories(sample: TableSample) -> dict[str, bool]:
    g = grid_stats(sample.annotation)
    _, v_rules = rule_lines(ink_mask(sample.image))
    return {
        "has_spans": g.n_spanning > 0,
        "multirow_header": g.n_header_rows >= 2,
        "no_vertical_rules": v_rules == 0,
        "large": g.n_rows >= LARGE_ROWS,
    }


def _table_summary(t: Table) -> dict[str, int]:
    return {
        "rows": t.n_rows,
        "cols": t.n_cols,
        "spanning": t.n_spanning,
        "header_rows": t.n_header_rows,
    }


def score_table(pred: Table, gt_full: Any, gt_struct: Any) -> dict[str, float]:
    html = pred.to_html() if pred.n_rows else ""
    return {
        "teds": round(teds_trees(html_to_tree(html), gt_full), 6),
        "teds_s": round(teds_trees(html_to_tree(html, structure_only=True), gt_struct), 6),
    }


def calibration_grid() -> list[PostprocessConfig]:
    grid = []
    for thr, span_thr, split in itertools.product(
        (0.3, 0.5, 0.7), (0.3, 0.5, 0.7, 1.01), (False, True)
    ):
        grid.append(
            PostprocessConfig(
                row_threshold=thr,
                col_threshold=thr,
                header_threshold=thr,
                span_threshold=span_thr,
                split_header_rowspans=split,
            )
        )
    return grid


def calibrate(observed: list[Observed]) -> dict[str, Any]:
    """Pick the post-processing config with the best mean TEDS-Struct on the calibration set."""
    gts = [
        html_to_tree(to_html(o.sample.annotation, with_content=False), structure_only=True)
        for o in observed
    ]
    rows: list[dict[str, Any]] = []
    cache: dict[tuple[int, str], float] = {}  # many configs yield identical HTML
    for cfg in calibration_grid():
        scores = []
        for i, (o, gt) in enumerate(zip(observed, gts, strict=True)):
            table = build_table(o.detections, text_boxes(o.sample.annotation), cfg)
            html = table.to_html() if table.n_rows else ""
            if (i, html) not in cache:
                cache[(i, html)] = teds_trees(html_to_tree(html, structure_only=True), gt)
            scores.append(cache[(i, html)])
        rows.append({"config": asdict(cfg), "teds_s": sum(scores) / max(1, len(scores))})
    best = max(rows, key=lambda r: float(r["teds_s"]))
    return {"n": len(observed), "grid": rows, "best": best}


def evaluate_tables(observed: list[Observed], config: PostprocessConfig) -> list[dict[str, Any]]:
    records = []
    for o in observed:
        s = o.sample
        words = text_boxes(s.annotation)
        gt_full = html_to_tree(to_html(s.annotation))
        gt_struct = html_to_tree(to_html(s.annotation, with_content=False), structure_only=True)
        g = grid_stats(s.annotation)
        preds = {
            MODEL_SYSTEM: build_table(o.detections, words, config),
            "textbox_cluster": textbox_cluster_table(words),
            "projection": projection_table(s.image, words),
        }
        systems = {
            name: {**score_table(t, gt_full, gt_struct), **_table_summary(t)}
            for name, t in preds.items()
        }
        records.append(
            {
                "index": s.index,
                "imgid": s.imgid,
                "filename": s.filename,
                "gt": {
                    "rows": g.n_rows,
                    "cols": g.n_cols,
                    "cells": g.n_cells,
                    "spanning": g.n_spanning,
                    "header_rows": g.n_header_rows,
                },
                "categories": categories(s),
                "model_seconds": round(o.seconds, 4),
                "systems": systems,
            }
        )
    return records


def diagnose(rec: dict[str, Any], system: str = MODEL_SYSTEM) -> list[str]:
    """Coarse structural error tags for one table (empty list = counts all match)."""
    p, g = rec["systems"][system], rec["gt"]
    tags = []
    if p["rows"] == 0:
        return ["no_table_predicted"]
    if p["rows"] != g["rows"]:
        tags.append("row_count_wrong")
    if p["cols"] != g["cols"]:
        tags.append("col_count_wrong")
    if p["spanning"] != g["spanning"]:
        tags.append("span_count_wrong")
    if p["header_rows"] != g["header_rows"]:
        tags.append("header_rows_wrong")
    return tags


def summarize(
    records: list[dict[str, Any]], seed: int, low_threshold: float = 0.9
) -> dict[str, Any]:
    systems = list(records[0]["systems"]) if records else []
    overall = {
        name: {
            metric: bootstrap_ci([r["systems"][name][metric] for r in records], seed=seed)
            for metric in ("teds", "teds_s")
        }
        for name in systems
    }
    paired = {}
    for base in BASELINES:
        if base not in systems:
            continue
        paired[f"{MODEL_SYSTEM}_minus_{base}"] = {
            metric: bootstrap_ci(
                [r["systems"][MODEL_SYSTEM][metric] - r["systems"][base][metric] for r in records],
                seed=seed,
            )
            for metric in ("teds", "teds_s")
        }
    by_category: dict[str, Any] = {}
    for cat in CATEGORIES:
        for flag in (True, False):
            subset = [r for r in records if r["categories"][cat] is flag]
            by_category[f"{cat}={str(flag).lower()}"] = {
                "n": len(subset),
                MODEL_SYSTEM: bootstrap_ci(
                    [r["systems"][MODEL_SYSTEM]["teds_s"] for r in subset], seed=seed
                ),
            }
    low = [r for r in records if r["systems"][MODEL_SYSTEM]["teds_s"] < low_threshold]
    error_tags: dict[str, int] = {}
    for r in low:
        for tag in diagnose(r) or ["counts_match_but_layout_wrong"]:
            error_tags[tag] = error_tags.get(tag, 0) + 1
    low_cats = {cat: sum(1 for r in low if r["categories"][cat]) for cat in CATEGORIES}
    all_cats = {cat: sum(1 for r in records if r["categories"][cat]) for cat in CATEGORIES}
    times = sorted(r["model_seconds"] for r in records)
    return {
        "n_tables": len(records),
        "overall": overall,
        "paired_differences": paired,
        "by_category": by_category,
        "failures": {
            "definition": f"{MODEL_SYSTEM} TEDS-Struct < {low_threshold}",
            "n": len(low),
            "error_tags": dict(sorted(error_tags.items(), key=lambda kv: -kv[1])),
            "category_counts_among_failures": low_cats,
            "category_counts_all": all_cats,
        },
        "model_seconds_per_table": {
            "median": times[len(times) // 2] if times else 0.0,
            "mean": sum(times) / len(times) if times else 0.0,
        },
    }
