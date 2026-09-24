import json

from docintel.pubtabnet import grid_stats, parse_annotation, text_boxes, to_html

from .conftest import make_annotation_dict

SPANNED = {
    "structure": {
        "tokens": [
            "<thead>",
            "<tr>",
            "<td",
            ' colspan="2"',
            ">",
            "</td>",
            "</tr>",
            "</thead>",
            "<tbody>",
            "<tr>",
            "<td",
            ' rowspan="2"',
            ">",
            "</td>",
            "<td>",
            "</td>",
            "</tr>",
            "<tr>",
            "<td>",
            "</td>",
            "</tr>",
            "</tbody>",
        ]
    },
    "cells": [
        {"tokens": ["<b>", "H", "&", "</b>"], "bbox": [0, 0, 10, 10]},
        {"tokens": ["x"], "bbox": [0, 20, 5, 30]},
        {"tokens": []},
        {"tokens": ["y"], "bbox": [20, 40, 25, 50]},
    ],
}


def test_parse_variants_agree():
    d = make_annotation_dict()
    a = parse_annotation(d)
    assert parse_annotation(json.dumps(d)) == a
    assert parse_annotation(repr(d)) == a
    assert a.cells[0].text == "Name"


def test_to_html_fills_cells_and_spans():
    ann = parse_annotation(SPANNED)
    html = to_html(ann)
    assert html.startswith('<table><thead><tr><td colspan="2">H&amp;</td>')
    assert '<td rowspan="2">x</td><td></td>' in html
    assert "<td>y</td>" in to_html(ann)
    assert "H" not in to_html(ann, with_content=False)


def test_grid_stats_resolves_spans():
    g = grid_stats(parse_annotation(SPANNED))
    assert (g.n_rows, g.n_cols, g.n_cells, g.n_spanning, g.n_header_rows) == (3, 2, 4, 2, 1)
    g2 = grid_stats(parse_annotation(make_annotation_dict()))
    assert (g2.n_rows, g2.n_cols, g2.n_spanning, g2.n_header_rows) == (3, 3, 0, 1)


def test_text_boxes_skip_empty():
    words = text_boxes(parse_annotation(SPANNED))
    assert [w.text for w in words] == ["H&", "x", "y"]
