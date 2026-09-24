"""Thin CPU wrapper around Microsoft's Table Transformer (TATR) structure-recognition model.

Requires the ``model`` extra (torch, transformers, timm). Excluded from coverage: the unit
tests exercise everything downstream of it with a stub that returns ``Detection`` lists.
"""

from __future__ import annotations

from typing import Any, Protocol

from PIL import Image

from docintel.structure import Detection

MODEL_ID = "microsoft/table-transformer-structure-recognition-v1.1-all"
MAX_SIZE = 1000  # the TATR reference inference resizes the longest side to 1000 px
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class StructureModel(Protocol):
    name: str

    def predict(self, image: Image.Image) -> list[Detection]: ...


class TatrStructureModel:
    def __init__(self, model_id: str = MODEL_ID, min_score: float = 0.05) -> None:
        import torch
        from transformers import TableTransformerForObjectDetection

        torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        self._torch: Any = torch
        self.model: Any = TableTransformerForObjectDetection.from_pretrained(model_id).eval()
        self.id2label: dict[int, str] = dict(self.model.config.id2label)
        self.min_score = min_score
        self.name = f"tatr:{model_id.split('/')[-1]}"

    def predict(self, image: Image.Image) -> list[Detection]:
        import numpy as np

        torch = self._torch
        rgb = image.convert("RGB")
        w, h = rgb.size
        scale = MAX_SIZE / max(w, h)
        resized = rgb.resize(
            (max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.BILINEAR
        )
        x = torch.tensor(np.asarray(resized), dtype=torch.float32).permute(2, 0, 1) / 255.0
        mean = torch.tensor(_MEAN)[:, None, None]
        std = torch.tensor(_STD)[:, None, None]
        x = ((x - mean) / std)[None]
        with torch.no_grad():
            out = self.model(pixel_values=x)
        probs = out.logits.softmax(-1)[0, :, :-1]  # last class is "no object"
        scores, labels = probs.max(-1)
        dets: list[Detection] = []
        for score, label, box in zip(
            scores.tolist(), labels.tolist(), out.pred_boxes[0].tolist(), strict=True
        ):
            if score < self.min_score:
                continue
            cx, cy, bw, bh = box
            dets.append(
                Detection(
                    self.id2label[int(label)],
                    float(score),
                    ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h),
                )
            )
        return dets
