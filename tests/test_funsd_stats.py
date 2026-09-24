import math

import numpy as np
import pytest

from docintel.funsd import (
    LinkConfig,
    evaluate_funsd,
    link_geometric,
    link_nearest,
    load_split,
    parse_form,
    prf,
    score_forms,
)
from docintel.stats import bootstrap_ci, bootstrap_statistic

from .conftest import funsd_form


def test_parse_form_normalises_link_direction():
    f = parse_form("x", funsd_form([0, 0, 1, 1], [2, 0, 3, 1]))
    assert f.links == {(0, 1)}
    data = funsd_form([0, 0, 1, 1], [2, 0, 3, 1])
    for e in data["form"]:
        e["linking"] = [[1, 0]] if e["id"] < 2 else []
    assert parse_form("y", data).links == {(0, 1)}


def test_geometric_prefers_same_line_left_over_nearer_above():
    data = funsd_form([10, 50, 60, 62], [70, 50, 120, 62])
    data["form"].append(
        {"id": 3, "label": "question", "text": "Other:", "box": [70, 20, 120, 32], "linking": []}
    )
    f = parse_form("x", data)
    assert link_geometric(f) == {(0, 1)}
    assert link_nearest(f) == {(3, 1)}  # centre distance is fooled by the question above


def test_geometric_above_and_max_distance():
    f = parse_form("x", funsd_form([10, 100, 60, 112], [12, 120, 80, 132]))
    assert link_geometric(f) == {(0, 1)}
    far = parse_form("x", funsd_form([10, 0, 60, 10], [12, 900, 80, 910]))
    assert link_geometric(far, LinkConfig(max_distance=100)) == set()
    no_q = parse_form("x", {"form": [{"id": 1, "label": "answer", "box": [0, 0, 1, 1]}]})
    assert link_nearest(no_q) == set()


def test_prf_and_scoring():
    assert prf(0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert prf(1, 2, 1)["f1"] == pytest.approx(2 / 3)
    f = parse_form("x", funsd_form([10, 50, 60, 62], [70, 50, 120, 62]))
    res = score_forms([f, f], [{(0, 1)}, set()])
    assert (res["tp"], res["n_pred"], res["n_gold"]) == (1, 1, 2)
    assert res["f1_ci"]["n"] == 2


def test_evaluate_funsd_end_to_end(tiny_funsd):
    assert len(load_split(tiny_funsd, "train")) == 2
    res = evaluate_funsd(tiny_funsd)
    assert res["test"]["geometric"]["f1"] == 1.0
    assert res["test"]["n_gold_links"] == 2


def test_bootstrap_ci():
    ci = bootstrap_ci([1.0, 0.0, 1.0, 0.0], seed=1)
    assert ci["mean"] == 0.5
    assert 0.0 <= ci["lo"] <= 0.5 <= ci["hi"] <= 1.0
    assert bootstrap_ci([0.7] * 5)["lo"] == pytest.approx(0.7)
    assert math.isnan(bootstrap_ci([])["mean"])


def test_bootstrap_statistic():
    vals = np.array([1.0, 2.0, 3.0])
    res = bootstrap_statistic(3, lambda idx: float(vals[idx].mean()), n_boot=200)
    assert res["value"] == 2.0
    assert res["lo"] <= 2.0 <= res["hi"]
