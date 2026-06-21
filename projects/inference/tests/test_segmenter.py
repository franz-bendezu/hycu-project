from __future__ import annotations

from PIL import Image

from app.schemas import SegmentationBackend
from app.services.segmenter import Segmenter


def _solid_image(width: int = 100, height: int = 80) -> Image.Image:
    return Image.new("RGB", (width, height), color=(120, 120, 120))


def test_box_rasterizer_creates_expected_mask() -> None:
    segmenter = Segmenter(model_path=None, backend=SegmentationBackend.BOX_RASTERIZER)
    image = _solid_image(100, 80)

    masks = segmenter.predict(image, [(10.0, 10.0, 30.0, 40.0)])

    assert masks.shape == (1, 80, 100)
    assert segmenter.active_backend == SegmentationBackend.BOX_RASTERIZER
    # 20x30 rectangle
    assert int(masks[0].sum()) == 600


def test_sam2_backend_falls_back_when_model_missing() -> None:
    segmenter = Segmenter(model_path=None, backend=SegmentationBackend.SAM2)
    image = _solid_image(60, 40)

    masks = segmenter.predict(image, [(5.0, 5.0, 20.0, 15.0)])

    assert masks.shape == (1, 40, 60)
    assert segmenter.active_backend == SegmentationBackend.BOX_RASTERIZER
    # 15x10 rectangle from fallback path
    assert int(masks[0].sum()) == 150
