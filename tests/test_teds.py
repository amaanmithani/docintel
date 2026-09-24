import pytest

from docintel.teds import html_to_tree, levenshtein, normalized_levenshtein, teds, teds_trees

ONE_A = "<table><tr><td>a</td></tr></table>"


def test_identical_tables_score_one():
    html = (
        '<table><thead><tr><td colspan="2">H</td></tr></thead>'
        "<tbody><tr><td>1</td><td>2</td></tr></tbody></table>"
    )
    assert teds(html, html) == 1.0
    assert teds(html, html, structure_only=True) == 1.0


def test_content_rename_costs_normalised_levenshtein():
    # 3 nodes (table, tr, td); renaming td 'a'->'b' costs 1 -> 1 - 1/3
    assert teds("<table><tr><td>b</td></tr></table>", ONE_A) == pytest.approx(2 / 3)
    # 'abcd' vs 'abce': 1 edit / 4 chars = 0.25 -> 1 - 0.25/3
    got = teds("<table><tr><td>abce</td></tr></table>", "<table><tr><td>abcd</td></tr></table>")
    assert got == pytest.approx(1 - 0.25 / 3)


def test_structure_only_ignores_content():
    assert teds("<table><tr><td>zzz</td></tr></table>", ONE_A, structure_only=True) == 1.0


def test_missing_cell_costs_one_deletion():
    two = "<table><tr><td>a</td><td>b</td></tr></table>"  # 4 nodes
    assert teds(ONE_A, two, structure_only=True) == pytest.approx(0.75)


def test_span_mismatch_is_full_rename():
    pred = '<table><tr><td colspan="2">a</td></tr></table>'
    assert teds(pred, ONE_A, structure_only=True) == pytest.approx(2 / 3)
    pred = '<table><tr><td rowspan="2">a</td></tr></table>'
    assert teds(pred, ONE_A, structure_only=True) == pytest.approx(2 / 3)


def test_missing_row():
    true = "<table><tr><td>a</td></tr><tr><td>b</td></tr></table>"  # 5 nodes
    assert teds(ONE_A, true, structure_only=True) == pytest.approx(1 - 2 / 5)


def test_th_treated_as_td_and_formatting_stripped():
    assert teds("<table><tr><th><b>a</b></th></tr></table>", ONE_A) == 1.0


def test_whole_document_and_bad_spans():
    doc = f"<html><body><p>x</p>{ONE_A}</body></html>"
    assert teds(doc, ONE_A) == 1.0
    tree = html_to_tree('<table><tr><td colspan="x">a</td></tr></table>')
    assert tree is not None
    assert tree.children[0].children[0].colspan == 1


def test_empty_inputs():
    assert teds("", ONE_A) == 0.0
    assert teds("<p>no table</p>", ONE_A) == 0.0
    assert teds_trees(None, None) == 1.0


def test_levenshtein():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert normalized_levenshtein("", "") == 0.0
    assert normalized_levenshtein("ab", "ab") == 0.0
    assert normalized_levenshtein("ab", "cd") == 1.0


def test_symmetric():
    a = '<table><tr><td>a</td><td colspan="2">b</td></tr><tr><td>c</td></tr></table>'
    b = "<table><tr><td>a</td><td>b</td><td>x</td></tr></table>"
    assert teds(a, b) == pytest.approx(teds(b, a))
