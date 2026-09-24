import json
import sys
import types
from typing import Any

import pytest

from docintel import cli, datasets, ocr, report

from .conftest import StubModel, make_image


@pytest.fixture(autouse=True)
def _stub_model(monkeypatch):
    monkeypatch.setattr(cli, "load_model", lambda *a, **k: StubModel())


def test_split_indices_disjoint_and_seeded():
    e, c = datasets.split_indices(50, 10, 5, seed=3)
    assert len(e) == 10
    assert len(c) == 5
    assert not set(e) & set(c)
    assert (e, c) == datasets.split_indices(50, 10, 5, seed=3)
    with pytest.raises(ValueError, match="exceed"):
        datasets.split_indices(5, 4, 4, seed=0)


def test_load_tables(tiny_parquet):
    assert datasets.count_rows(tiny_parquet) == 6
    samples = list(datasets.load_tables(tiny_parquet, [1, 4]))
    assert [s.imgid for s in samples] == [101, 104]
    assert samples[0].annotation.cells[0].text == "Name"
    assert len(datasets.sha256_file(tiny_parquet)) == 64


def test_extract_formats(tmp_path, capsys):
    img = tmp_path / "t.png"
    make_image().save(img)
    words = tmp_path / "w.json"
    words.write_text(json.dumps([{"text": "Name", "bbox": [10, 10, 42, 20]}]))
    res = str(tmp_path / "results")
    assert cli.main(["--results", res, "extract", str(img), "--words", str(words)]) == 0
    assert "<td>Name</td>" in capsys.readouterr().out
    assert cli.main(["--results", res, "extract", str(img), "--words", str(words), "--csv"]) == 0
    assert capsys.readouterr().out.startswith("Name,,")
    assert cli.main(["--results", res, "extract", str(img), "--no-ocr", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert (out["rows"], out["cols"], out["text_source"]) == (3, 3, "none (structure only)")


def test_extract_uses_ocr_when_available(tmp_path, capsys, monkeypatch):
    img = tmp_path / "t.png"
    make_image().save(img)
    monkeypatch.setattr(ocr, "tesseract_available", lambda: True)
    monkeypatch.setattr(ocr, "ocr_words", lambda image: [])
    assert cli.main(["--results", str(tmp_path), "extract", str(img)]) == 0
    assert "tesseract" in capsys.readouterr().err


def test_eval_report_roundtrip(tiny_parquet, tmp_path, tiny_funsd, capsys):
    res = tmp_path / "results"
    argv = [
        "--results",
        str(res),
        "eval",
        "--n",
        "3",
        "--calib-n",
        "2",
        "--parquet",
        str(tiny_parquet),
    ]
    assert cli.main(argv) == 0
    ev = json.loads((res / "pubtabnet_eval.json").read_text())
    assert ev["config"]["n_eval"] == 3
    assert ev["summary"]["overall"]["tatr"]["teds"]["mean"] == 1.0
    assert len(json.loads((res / "pubtabnet_per_table.json").read_text())) == 3
    # extract now picks up the calibrated post-processing config
    assert cli._postprocess_config(res).row_threshold in (0.3, 0.5, 0.7)

    assert cli.main(["--results", str(res), "eval-forms", "--funsd", str(tiny_funsd)]) == 0
    capsys.readouterr()

    readme = tmp_path / "README.md"
    readme.write_text(f"# x\n{report.START}\nold\n{report.END}\ntail\n")
    rd = ["--results", str(res), "report", "--readme", str(readme)]
    assert cli.main([*rd, "--check"]) == 1
    assert cli.main(rd) == 0
    text = readme.read_text()
    assert "TEDS-Struct" in text
    assert "FUNSD test" in text
    assert "ablation on the **calibration** sample" in text
    assert text.endswith("tail\n")
    assert cli.main([*rd, "--check"]) == 0


def test_report_without_results_and_bad_readme(tmp_path):
    assert "No results yet" in report.render(tmp_path)
    bad = tmp_path / "R.md"
    bad.write_text("no markers")
    with pytest.raises(ValueError, match="markers"):
        report.update_readme(bad, tmp_path)


def test_ocr_parsing_and_availability(monkeypatch):
    data: dict[str, list[Any]] = {
        "text": ["", "Hello", "low", "World"],
        "conf": ["-1", "90", "10", "bad"],
        "left": [0, 1, 2, 3],
        "top": [0, 1, 2, 3],
        "width": [1, 10, 1, 1],
        "height": [1, 5, 1, 1],
    }
    words = ocr.words_from_tesseract_data(data)
    assert [w.text for w in words] == ["Hello"]
    assert words[0].bbox == (1.0, 1.0, 11.0, 6.0)
    assert ocr.scale_words(words, 2.0)[0].bbox == (0.5, 0.5, 5.5, 3.0)

    fake = types.SimpleNamespace(get_tesseract_version=lambda: "5")
    monkeypatch.setitem(sys.modules, "pytesseract", fake)
    assert ocr.tesseract_available()

    def boom():
        raise OSError("no binary")

    monkeypatch.setitem(
        sys.modules, "pytesseract", types.SimpleNamespace(get_tesseract_version=boom)
    )
    assert not ocr.tesseract_available()
