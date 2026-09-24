"""Render README result tables from the committed results/*.json files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"

SYSTEM_LABELS = {
    "tatr": "TATR v1.1-all + post-processing (this repo)",
    "textbox_cluster": "Baseline: cluster oracle text boxes into rows/cols",
    "projection": "Baseline: whitespace projection on the image",
}
CATEGORY_LABELS = {
    "has_spans": "has spanning cells",
    "multirow_header": "header has >= 2 rows",
    "no_vertical_rules": "no vertical ruling lines (borderless columns)",
    "large": "large (>= 20 rows)",
}


def _ci(d: dict[str, float], digits: int = 3) -> str:
    return f"{d['mean']:.{digits}f} [{d['lo']:.{digits}f}, {d['hi']:.{digits}f}]"


def render_tables(ev: dict[str, Any]) -> str:
    s = ev["summary"]
    cfg = ev["config"]
    lines = [
        f"**PubTabNet validation, n = {s['n_tables']} tables** (seed {cfg['seed']}, sampled "
        f"from {cfg['n_total']}; post-processing chosen on a disjoint calibration sample of "
        f"{cfg['n_calibration']}). Mean with 95% percentile-bootstrap CI.",
        "",
        "| System | TEDS | TEDS-Struct |",
        "|---|---|---|",
    ]
    for name, m in s["overall"].items():
        lines.append(f"| {SYSTEM_LABELS.get(name, name)} | {_ci(m['teds'])} | {_ci(m['teds_s'])} |")
    lines += [
        "",
        "Paired differences (same tables):",
        "",
        "| Comparison | ΔTEDS | ΔTEDS-Struct |",
        "|---|---|---|",
    ]
    for name, m in s["paired_differences"].items():
        lines.append(
            f"| {name.replace('_minus_', ' - ')} | {_ci(m['teds'])} | {_ci(m['teds_s'])} |"
        )
    lines += [
        "",
        "TATR TEDS-Struct by table category:",
        "",
        "| Category | n (yes) | TEDS-Struct (yes) | n (no) | TEDS-Struct (no) |",
        "|---|---|---|---|---|",
    ]
    for cat, label in CATEGORY_LABELS.items():
        yes, no = s["by_category"][f"{cat}=true"], s["by_category"][f"{cat}=false"]
        y = _ci(yes["tatr"]) if yes["n"] else "-"
        n = _ci(no["tatr"]) if no["n"] else "-"
        lines.append(f"| {label} | {yes['n']} | {y} | {no['n']} | {n} |")
    f = s["failures"]
    lines += [
        "",
        f"Failures ({f['definition']}): **{f['n']} / {s['n_tables']}** tables.",
        "",
        "| Error tag (a table can have several) | count |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in f["error_tags"].items()]
    lines += [
        "",
        "| Category | among failures | among all tables |",
        "|---|---|---|",
    ]
    for cat, label in CATEGORY_LABELS.items():
        lines.append(
            f"| {label} | {f['category_counts_among_failures'][cat]} / {f['n']} "
            f"| {f['category_counts_all'][cat]} / {s['n_tables']} |"
        )
    t = s["model_seconds_per_table"]
    best = ev["calibration"]["best"]
    lines += [
        "",
        f"Chosen post-processing (calibration TEDS-Struct {best['teds_s']:.3f}): "
        f"`{json.dumps(best['config'])}`. Model time on CPU: median {t['median']:.2f} s/table.",
    ]
    return "\n".join(lines)


def render_funsd(fr: dict[str, Any]) -> str:
    t = fr["test"]
    lines = [
        f"**FUNSD test, {t['n_forms']} forms, {t['n_gold_links']} gold question→answer links** "
        "(gold entities and labels given; only the linking step is predicted). "
        "Micro-averaged; F1 CI is a bootstrap over forms.",
        "",
        "| Method | Precision | Recall | F1 [95% CI] |",
        "|---|---|---|---|",
    ]
    for key, label in (
        ("geometric", "Left/above geometric rule (tuned on train)"),
        ("nearest_center_baseline", "Baseline: nearest question centre"),
    ):
        m = t[key]
        ci = m["f1_ci"]
        lines.append(
            f"| {label} | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"{m['f1']:.3f} [{ci['lo']:.3f}, {ci['hi']:.3f}] |"
        )
    return "\n".join(lines)


def _matches(cfg: dict[str, Any], **want: Any) -> bool:
    return all(cfg.get(k) == v for k, v in want.items())


def render_calibration(cal: dict[str, Any]) -> str:
    """Post-processing ablation on the calibration sample (never on the eval sample)."""
    variants = [
        (
            "TATR defaults: thresholds 0.5, keep header row-spans",
            {"row_threshold": 0.5, "span_threshold": 0.5, "split_header_rowspans": False},
        ),
        (
            "thresholds 0.5, split blank header row-spans",
            {"row_threshold": 0.5, "span_threshold": 0.5, "split_header_rowspans": True},
        ),
        (
            "thresholds 0.5, spanning cells disabled",
            {"row_threshold": 0.5, "span_threshold": 1.01, "split_header_rowspans": False},
        ),
    ]
    lines = [
        f"Post-processing ablation on the **calibration** sample (n = {cal['n']}, disjoint "
        "from the eval sample; mean TEDS-Struct):",
        "",
        "| Variant | TEDS-Struct |",
        "|---|---|",
    ]
    for label, want in variants:
        hit = next((g for g in cal["grid"] if _matches(g["config"], **want)), None)
        if hit is not None:
            lines.append(f"| {label} | {hit['teds_s']:.3f} |")
    lines.append(f"| chosen (best of {len(cal['grid'])} configs) | {cal['best']['teds_s']:.3f} |")
    return "\n".join(lines)


def render(results_dir: Path) -> str:
    parts = []
    ev_path = results_dir / "pubtabnet_eval.json"
    if ev_path.exists():
        parts += [
            "### Table structure (PubTabNet)",
            "",
            render_tables(json.loads(ev_path.read_text())),
        ]
    cal_path = results_dir / "pubtabnet_calibration.json"
    if cal_path.exists():
        parts += ["", render_calibration(json.loads(cal_path.read_text()))]
    fr_path = results_dir / "funsd_linking.json"
    if fr_path.exists():
        parts += [
            "",
            "### Form key-value linking (FUNSD)",
            "",
            render_funsd(json.loads(fr_path.read_text())),
        ]
    if not parts:
        parts = ["_No results yet: run `uv run docintel eval`._"]
    return "\n".join(parts)


def update_readme(readme: Path, results_dir: Path, check: bool = False) -> bool:
    """Rewrite the block between the markers. Returns True if the README was (or is) current."""
    text = readme.read_text()
    if START not in text or END not in text:
        raise ValueError(f"{readme} lacks the {START} / {END} markers")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    new = f"{head}{START}\n{render(results_dir)}\n{END}{tail}"
    if check:
        return new == text
    readme.write_text(new)
    return True
