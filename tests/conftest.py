"""Tiny synthetic fixtures: a drawn 3x3 table, its PubTabNet-style annotation, a stub model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image, ImageDraw

from docintel.datasets import TableSample
from docintel.pubtabnet import parse_annotation
from docintel.structure import COLUMN, HEADER, ROW, SPAN, TABLE, Detection

W, H = 300, 90
ROWS = [(0, 30), (30, 60), (60, 90)]
COLS = [(0, 100), (100, 200), (200, 300)]
TEXT = [["Name", "Age", "Dose"], ["a", "31", "5"], ["b", "42", "10"]]


def text_box(r: int, c: int, text: str) -> list[float]:
    x0 = COLS[c][0] + 10
    y0 = ROWS[r][0] + 10
    return [x0, y0, x0 + 8 * len(text), y0 + 10]


def make_image() -> Image.Image:
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.line([(0, 29), (W, 29)], fill="black", width=1)  # booktabs-style header rule
    for r, row in enumerate(TEXT):
        for c, t in enumerate(row):
            d.rectangle(text_box(r, c, t), fill="black")  # "ink" blobs stand in for glyphs
    return img


def make_annotation_dict() -> dict[str, Any]:
    structure = ["<thead>", "<tr>"]
    structure += ["<td>", "</td>"] * 3
    structure += ["</tr>", "</thead>", "<tbody>"]
    for _ in range(2):
        structure += ["<tr>", *(["<td>", "</td>"] * 3), "</tr>"]
    structure += ["</tbody>"]
    cells = []
    for r, row in enumerate(TEXT):
        for c, t in enumerate(row):
            toks = list(t)
            if r == 0:
                toks = ["<b>", *toks, "</b>"]
            cells.append({"tokens": toks, "bbox": text_box(r, c, t)})
    return {"structure": {"tokens": structure}, "cells": cells}


def perfect_detections() -> list[Detection]:
    dets = [Detection(TABLE, 0.99, (0, 0, W, H)), Detection(HEADER, 0.9, (0, 0, W, 30))]
    dets += [Detection(ROW, 0.95, (0, a, W, b)) for a, b in ROWS]
    dets += [Detection(COLUMN, 0.95, (a, 0, b, H)) for a, b in COLS]
    dets.append(Detection(SPAN, 0.1, (0, 30, 100, 90)))  # low score: filtered by threshold
    return dets


class StubModel:
    name = "stub"

    def __init__(self, detections: list[Detection] | None = None) -> None:
        self.detections = detections if detections is not None else perfect_detections()

    def predict(self, image: Image.Image) -> list[Detection]:
        return list(self.detections)


@pytest.fixture
def sample() -> TableSample:
    return TableSample(0, 42, "fake.png", make_image(), parse_annotation(make_annotation_dict()))


@pytest.fixture
def stub_model() -> StubModel:
    return StubModel()


@pytest.fixture
def tiny_parquet(tmp_path: Path) -> Path:
    import io

    import pyarrow as pa
    import pyarrow.parquet as pq

    buf = io.BytesIO()
    make_image().save(buf, format="PNG")
    n = 6
    table = pa.table(
        {
            "image": [{"bytes": buf.getvalue(), "path": f"t{i}.png"} for i in range(n)],
            "split": ["val"] * n,
            "imgid": list(range(100, 100 + n)),
            "html": [repr(make_annotation_dict())] * n,  # HF mirror stores a Python literal
            "html_table": [""] * n,
        }
    )
    path = tmp_path / "val.parquet"
    pq.write_table(table, path)
    return path


def funsd_form(
    q_box: list[int], a_box: list[int], extra: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    form = [
        {"id": 0, "label": "question", "text": "Date:", "box": q_box, "linking": [[0, 1]]},
        {"id": 1, "label": "answer", "text": "1999", "box": a_box, "linking": [[0, 1]]},
        {"id": 2, "label": "header", "text": "FORM", "box": [0, 0, 50, 10], "linking": []},
    ]
    return {"form": form + (extra or [])}


@pytest.fixture
def tiny_funsd(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    for split in ("training_data", "testing_data"):
        ann = root / split / "annotations"
        ann.mkdir(parents=True)
        (ann / "a.json").write_text(json.dumps(funsd_form([10, 50, 60, 62], [70, 50, 120, 62])))
        (ann / "b.json").write_text(json.dumps(funsd_form([10, 100, 60, 112], [12, 120, 80, 132])))
    return root
