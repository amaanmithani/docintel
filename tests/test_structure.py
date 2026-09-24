from docintel.pubtabnet import parse_annotation, text_boxes, to_html
from docintel.structure import (
    COLUMN,
    HEADER,
    ROW,
    SPAN,
    Detection,
    PostprocessConfig,
    Word,
    assign_words,
    build_table,
    grid_table,
    words_inside,
)
from docintel.teds import teds

from .conftest import H, W, make_annotation_dict, perfect_detections


def test_perfect_detections_reproduce_ground_truth():
    ann = parse_annotation(make_annotation_dict())
    table = build_table(perfect_detections(), text_boxes(ann))
    assert (table.n_rows, table.n_cols, table.n_header_rows) == (3, 3, 1)
    assert teds(table.to_html(), to_html(ann)) == 1.0
    assert table.to_csv().splitlines()[0] == "Name,Age,Dose"
    assert table.to_grid()[2] == ["b", "42", "10"]


def test_low_threshold_admits_spanning_cell():
    ann = parse_annotation(make_annotation_dict())
    cfg = PostprocessConfig(span_threshold=0.05)
    table = build_table(perfect_detections(), text_boxes(ann), cfg)
    assert table.n_spanning == 1
    span = next(c for c in table.cells if c.rowspan == 2)
    assert span.text == "a b"
    assert '<td rowspan="2">a b</td>' in table.to_html()


def test_duplicate_rows_are_deduplicated():
    dets = [
        *perfect_detections(),
        Detection(ROW, 0.6, (0, 31, W, 59)),  # near-duplicate of row 2, lower score
    ]
    assert build_table(dets, []).n_rows == 3


def test_no_rows_or_cols_gives_empty_table():
    assert build_table([Detection(ROW, 0.9, (0, 0, 10, 10))], []).n_rows == 0
    assert grid_table([], [(0, 0, 1, 1)], []).n_cols == 0
    assert build_table([], []).to_html() == "<table></table>"


def test_table_extent_without_table_detection():
    dets = [d for d in perfect_detections() if d.label in (ROW, COLUMN)]
    table = build_table(dets, [])
    assert table.n_header_rows == 0
    assert "<thead>" not in table.to_html()


def test_colspan_in_header_and_straddling_span_rejected():
    rows = [(0.0, 0.0, 200.0, 10.0), (0.0, 10.0, 200.0, 20.0)]
    cols = [(0.0, 0.0, 100.0, 20.0), (100.0, 0.0, 200.0, 20.0)]
    words = [Word("Top", (60, 2, 140, 8)), Word("1", (10, 12, 20, 18))]
    t = grid_table(rows, cols, words, spans=[(0, 0, 200, 10)], header_box=(0, 0, 200, 10))
    assert '<thead><tr><td colspan="2">Top</td></tr></thead>' in t.to_html()
    # A span crossing the header/body boundary is ignored.
    t2 = grid_table(rows, cols, words, spans=[(0, 0, 100, 20)], header_box=(0, 0, 200, 10))
    assert t2.n_spanning == 0
    # Overlapping spans: the second one is dropped.
    t3 = grid_table(rows, cols, words, spans=[(0, 0, 200, 10), (0, 0, 100, 20)])
    assert t3.n_spanning == 1
    # Degenerate span covering one slot is skipped; span missing the grid is skipped.
    t4 = grid_table(rows, cols, words, spans=[(0, 0, 100, 10), (500, 500, 600, 600)])
    assert t4.n_spanning == 0


def test_split_header_rowspans():
    rows = [(0.0, 0.0, 200.0, 10.0), (0.0, 10.0, 200.0, 20.0), (0.0, 20.0, 200.0, 30.0)]
    cols = [(0.0, 0.0, 100.0, 30.0), (100.0, 0.0, 200.0, 30.0)]
    words = [Word("Subjects", (5, 12, 60, 18)), Word("Age", (105, 12, 150, 18))]
    head = (0.0, 0.0, 200.0, 20.0)
    spans = [(0.0, 0.0, 100.0, 20.0), (100.0, 0.0, 200.0, 20.0)]
    merged = grid_table(rows, cols, words, spans=spans, header_box=head)
    assert merged.n_spanning == 2
    split = grid_table(rows, cols, words, spans=spans, header_box=head, split_header_rowspans=True)
    assert split.n_spanning == 0
    assert split.to_grid()[1] == ["Subjects", "Age"]
    # A 2x2 header span with text in one row keeps its colspan on that row.
    wide = grid_table(
        rows,
        cols,
        [Word("Both", (50, 12, 150, 18))],
        spans=[(0.0, 0.0, 200.0, 20.0)],
        header_box=head,
        split_header_rowspans=True,
    )
    cell = next(c for c in wide.cells if c.colspan == 2)
    assert (cell.row, cell.rowspan, cell.text) == (1, 1, "Both")


def test_assign_words_nearest_fallback():
    rows = [(0.0, 0.0, 100.0, 10.0), (0.0, 10.0, 100.0, 20.0)]
    cols = [(0.0, 0.0, 50.0, 20.0), (50.0, 0.0, 100.0, 20.0)]
    slots = assign_words([Word("far", (120, 25, 130, 30))], rows, cols)
    assert list(slots) == [(1, 1)]
    assert assign_words([Word("x", (0, 0, 1, 1))], [], cols) == {}


def test_words_inside_and_header_detection_labels():
    words = [Word("in", (0, 0, 10, 10)), Word("out", (50, 50, 60, 60)), Word("", (0, 0, 0, 0))]
    assert [w.text for w in words_inside(words, (0, 0, 20, 20))] == ["in"]
    assert HEADER != SPAN
    assert H > 0
