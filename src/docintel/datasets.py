"""Dataset access: PubTabNet validation (HF parquet mirror) and FUNSD (official zip)."""

from __future__ import annotations

import hashlib
import io
import os
import urllib.request
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from docintel.pubtabnet import PtnAnnotation, parse_annotation

PUBTABNET_REPO = "apoidea/pubtabnet-html"
PUBTABNET_FILE = "data/validation-00000-of-00001.parquet"
FUNSD_URL = "https://guillaumejaume.github.io/FUNSD/dataset.zip"
DATA_DIR = Path(os.environ.get("DOCINTEL_DATA", "data/raw"))


@dataclass
class TableSample:
    index: int
    imgid: int
    filename: str
    image: Image.Image
    annotation: PtnAnnotation


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pubtabnet_parquet(data_dir: Path = DATA_DIR) -> Path:  # pragma: no cover - network
    local = data_dir / PUBTABNET_FILE
    if local.exists():
        return local
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(PUBTABNET_REPO, PUBTABNET_FILE, repo_type="dataset", local_dir=data_dir)
    )


def funsd_dir(data_dir: Path = DATA_DIR) -> Path:  # pragma: no cover - network
    root = data_dir / "funsd" / "dataset"
    if root.exists():
        return root
    data_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(FUNSD_URL, timeout=120) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        zf.extractall(data_dir / "funsd")
    return root


def split_indices(
    n_total: int, n_eval: int, n_calib: int, seed: int
) -> tuple[list[int], list[int]]:
    """Disjoint eval / calibration index sets from one seeded permutation."""
    if n_eval + n_calib > n_total:
        raise ValueError("sample sizes exceed dataset size")
    perm = np.random.default_rng(seed).permutation(n_total)
    return sorted(perm[:n_eval].tolist()), sorted(perm[n_total - n_calib :].tolist())


def count_rows(parquet: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(parquet).metadata.num_rows)


def load_tables(parquet: Path, indices: list[int]) -> Iterator[TableSample]:
    """Yield the requested rows; images are decoded lazily one at a time."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet, columns=["imgid", "html", "image"]).take(indices)
    for pos, idx in enumerate(indices):
        row = table.slice(pos, 1).to_pylist()[0]
        img = row["image"]
        yield TableSample(
            index=idx,
            imgid=int(row["imgid"]),
            filename=str(img.get("path") or ""),
            image=Image.open(io.BytesIO(img["bytes"])).convert("RGB"),
            annotation=parse_annotation(row["html"]),
        )
