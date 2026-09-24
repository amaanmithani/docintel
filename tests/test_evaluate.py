from typing import Any

from docintel import evaluate
from docintel.structure import PostprocessConfig

from .conftest import StubModel


def test_observe_and_evaluate_perfect(sample, stub_model):
    observed = evaluate.observe([sample], stub_model)
    recs = evaluate.evaluate_tables(observed, PostprocessConfig())
    r = recs[0]
    assert r["systems"]["tatr"]["teds"] == 1.0
    assert r["systems"]["tatr"]["teds_s"] == 1.0
    assert r["systems"]["textbox_cluster"]["teds_s"] < 1.0  # baselines have no header
    assert r["categories"] == {
        "has_spans": False,
        "multirow_header": False,
        "no_vertical_rules": True,
        "large": False,
    }
    assert evaluate.diagnose(r) == []


def test_calibrate_picks_a_config(sample, stub_model):
    observed = evaluate.observe([sample, sample], stub_model)
    res = evaluate.calibrate(observed)
    assert res["n"] == 2
    assert len(res["grid"]) == len(evaluate.calibration_grid())
    assert res["best"]["teds_s"] == 1.0
    # A span threshold low enough to admit the bogus span must not win.
    assert res["best"]["config"]["span_threshold"] > 0.1


def test_summarize_and_failure_tags(sample, stub_model):
    good = evaluate.evaluate_tables(evaluate.observe([sample], stub_model), PostprocessConfig())
    bad = evaluate.evaluate_tables(evaluate.observe([sample], StubModel([])), PostprocessConfig())
    records = good + bad
    s = evaluate.summarize(records, seed=0)
    assert s["n_tables"] == 2
    assert s["overall"]["tatr"]["teds_s"]["mean"] == 0.5
    assert s["failures"]["n"] == 1
    assert s["failures"]["error_tags"] == {"no_table_predicted": 1}
    assert "tatr_minus_projection" in s["paired_differences"]
    assert s["by_category"]["large=true"]["n"] == 0


def test_diagnose_tags():
    rec: dict[str, Any] = {
        "gt": {"rows": 3, "cols": 3, "spanning": 1, "header_rows": 1},
        "systems": {"tatr": {"rows": 4, "cols": 2, "spanning": 0, "header_rows": 2}},
    }
    assert evaluate.diagnose(rec) == [
        "row_count_wrong",
        "col_count_wrong",
        "span_count_wrong",
        "header_rows_wrong",
    ]
    rec["systems"]["tatr"] = {"rows": 3, "cols": 3, "spanning": 1, "header_rows": 1}
    records: list[dict[str, Any]] = [
        {
            **rec,
            "categories": dict.fromkeys(evaluate.CATEGORIES, False),
            "model_seconds": 0.1,
        }
    ]
    records[0]["systems"]["tatr"].update({"teds": 0.5, "teds_s": 0.5})
    s = evaluate.summarize(records, seed=0)
    assert s["failures"]["error_tags"] == {"counts_match_but_layout_wrong": 1}
