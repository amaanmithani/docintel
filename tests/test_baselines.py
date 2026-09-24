import numpy as np
import pytest
from PIL import Image

from docintel.baselines import (
    _runs,
    ink_mask,
    projection_table,
    rule_lines,
    textbox_cluster_table,
)
from docintel.pubtabnet import parse_annotation, text_boxes, to_html
from docintel.teds import teds

from .conftest import make_annotation_dict, make_image


def test_runs():
    assert _runs(np.array([True, True, False, True])) == [(0, 2), (3, 4)]
    assert _runs(np.array([False])) == []


def test_rule_lines_counts_horizontal_rule():
    h, v = rule_lines(ink_mask(make_image()))
    assert (h, v) == (1, 0)
    assert rule_lines(np.zeros((0, 0), dtype=bool)) == (0, 0)


def test_projection_recovers_simple_grid():
    ann = parse_annotation(make_annotation_dict())
    t = projection_table(make_image(), text_boxes(ann))
    assert (t.n_rows, t.n_cols) == (3, 3)
    # No header in baselines, so TEDS-Struct is < 1 but the grid is right.
    assert teds(t.to_html(), to_html(ann), structure_only=True) == pytest.approx(0.8)


def test_projection_blank_image():
    assert projection_table(Image.new("RGB", (10, 10), "white"), []).n_rows == 0


def test_textbox_cluster():
    ann = parse_annotation(make_annotation_dict())
    t = textbox_cluster_table(text_boxes(ann))
    assert (t.n_rows, t.n_cols) == (3, 3)
    assert t.to_grid()[1] == ["a", "31", "5"]
    assert textbox_cluster_table([]).n_rows == 0
