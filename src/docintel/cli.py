"""docintel command line: extract | eval | eval-forms | report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image

from docintel import datasets, evaluate, report
from docintel.structure import PostprocessConfig, Word, build_table
from docintel.tatr import MODEL_ID, StructureModel

RESULTS = Path("results")


def load_model(model_id: str = MODEL_ID) -> StructureModel:  # pragma: no cover - needs torch
    from docintel.tatr import TatrStructureModel

    return TatrStructureModel(model_id)


def _load_words(path: Path) -> list[Word]:
    raw = json.loads(path.read_text())
    words = []
    for w in raw:
        x1, y1, x2, y2 = (float(v) for v in w["bbox"])
        words.append(Word(str(w["text"]), (x1, y1, x2, y2)))
    return words


def _postprocess_config(results_dir: Path) -> PostprocessConfig:
    calib = results_dir / "pubtabnet_calibration.json"
    if calib.exists():
        return PostprocessConfig(**json.loads(calib.read_text())["best"]["config"])
    return PostprocessConfig()


def cmd_extract(args: argparse.Namespace) -> int:
    from docintel import ocr

    image = Image.open(args.image).convert("RGB")
    if args.words:
        words, source = _load_words(Path(args.words)), "words-file"
    elif not args.no_ocr and ocr.tesseract_available():
        words, source = ocr.ocr_words(image), "tesseract"
    else:
        words, source = [], "none (structure only)"
    model = load_model()
    table = build_table(model.predict(image), words, _postprocess_config(Path(args.results)))
    if args.csv:
        sys.stdout.write(table.to_csv())
    elif args.json:
        payload: dict[str, Any] = {
            "rows": table.n_rows,
            "cols": table.n_cols,
            "header_rows": table.n_header_rows,
            "text_source": source,
            "cells": [
                {
                    "row": c.row,
                    "col": c.col,
                    "rowspan": c.rowspan,
                    "colspan": c.colspan,
                    "header": c.header,
                    "text": c.text,
                }
                for c in table.cells
            ],
        }
        sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    else:
        sys.stdout.write(table.to_html() + "\n")
    print(f"[docintel] {table.n_rows}x{table.n_cols} grid, text source: {source}", file=sys.stderr)
    return 0


def _write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")


def cmd_eval(args: argparse.Namespace) -> int:
    out = Path(args.results)
    parquet = Path(args.parquet) if args.parquet else datasets.pubtabnet_parquet()
    n_total = datasets.count_rows(parquet)
    eval_idx, calib_idx = datasets.split_indices(n_total, args.n, args.calib_n, args.seed)
    model = load_model()

    print(f"[eval] calibration on {len(calib_idx)} tables (disjoint from eval)", file=sys.stderr)
    calib = evaluate.calibrate(evaluate.observe(datasets.load_tables(parquet, calib_idx), model))
    _write(out / "pubtabnet_calibration.json", {"seed": args.seed, "indices": calib_idx, **calib})
    cfg = PostprocessConfig(**calib["best"]["config"])

    print(f"[eval] evaluating {len(eval_idx)} tables", file=sys.stderr)
    observed = evaluate.observe(datasets.load_tables(parquet, eval_idx), model)
    records = evaluate.evaluate_tables(observed, cfg)
    summary = evaluate.summarize(records, seed=args.seed)
    config = {
        "dataset": f"hf://datasets/{datasets.PUBTABNET_REPO}/{datasets.PUBTABNET_FILE}",
        "parquet_sha256": datasets.sha256_file(parquet),
        "split": "validation",
        "n_total": n_total,
        "n_eval": len(eval_idx),
        "n_calibration": len(calib_idx),
        "seed": args.seed,
        "model": getattr(model, "name", "unknown"),
        "text_source": "PubTabNet cell text boxes (oracle OCR)",
        "postprocess": calib["best"]["config"],
    }
    _write(
        out / "pubtabnet_eval.json",
        {"config": config, "calibration": {"best": calib["best"]}, "summary": summary},
    )
    _write(out / "pubtabnet_per_table.json", records)
    print(json.dumps(summary["overall"], indent=1))
    return 0


def cmd_eval_forms(args: argparse.Namespace) -> int:
    from docintel.funsd import evaluate_funsd

    root = Path(args.funsd) if args.funsd else datasets.funsd_dir()
    res = evaluate_funsd(root, seed=args.seed)
    _write(Path(args.results) / "funsd_linking.json", res)
    print(json.dumps(res["test"], indent=1))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    ok = report.update_readme(Path(args.readme), Path(args.results), check=args.check)
    if args.check and not ok:
        print("README results are stale: run `uv run docintel report`", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="docintel", description=__doc__)
    p.add_argument("--results", default=str(RESULTS), help="results directory")
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("extract", help="table image -> HTML / CSV / JSON")
    e.add_argument("image")
    fmt = e.add_mutually_exclusive_group()
    fmt.add_argument("--html", action="store_true", help="HTML output (default)")
    fmt.add_argument("--csv", action="store_true")
    fmt.add_argument("--json", action="store_true")
    e.add_argument("--words", help="JSON list of {text, bbox:[x1,y1,x2,y2]} to use as cell text")
    e.add_argument("--no-ocr", action="store_true", help="skip the Tesseract fallback")
    e.set_defaults(func=cmd_extract)

    v = sub.add_parser("eval", help="TEDS / TEDS-Struct on a PubTabNet validation sample")
    v.add_argument("--n", type=int, default=300)
    v.add_argument("--calib-n", type=int, default=100)
    v.add_argument("--seed", type=int, default=13)
    v.add_argument("--parquet", help="local PubTabNet validation parquet (downloaded if omitted)")
    v.set_defaults(func=cmd_eval)

    f = sub.add_parser("eval-forms", help="FUNSD question->answer linking")
    f.add_argument("--funsd", help="path to the unzipped FUNSD 'dataset' folder")
    f.add_argument("--seed", type=int, default=13)
    f.set_defaults(func=cmd_eval_forms)

    r = sub.add_parser("report", help="render README tables from results/*.json")
    r.add_argument("--readme", default="README.md")
    r.add_argument("--check", action="store_true", help="exit 1 if README is stale")
    r.set_defaults(func=cmd_report)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
