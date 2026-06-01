from PIL import Image

from app.main import ReferenceImage, _classify, _image_features


def _solid_image(width: int, height: int, color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (width, height), color=color)


def test_classify_prefers_closest_reference(monkeypatch) -> None:
    query = _solid_image(320, 560, (230, 230, 230))
    q_ahash, q_dhash, q_edge, q_aspect, q_brightness = _image_features(query)

    refs = (
        ReferenceImage(
            category="shelf",
            average_hash=q_ahash,
            difference_hash=q_dhash,
            edge_density=q_edge,
            aspect_ratio=q_aspect,
            brightness=q_brightness,
        ),
        ReferenceImage(
            category="desk",
            average_hash=0,
            difference_hash=0,
            edge_density=1.0,
            aspect_ratio=2.4,
            brightness=0.02,
        ),
        ReferenceImage(
            category="cabinet",
            average_hash=(1 << 255),
            difference_hash=(1 << 254),
            edge_density=0.8,
            aspect_ratio=0.4,
            brightness=0.1,
        ),
    )

    monkeypatch.setattr("app.main._reference_gallery", lambda: refs)

    detected_type, confidence = _classify(query)

    assert detected_type == "shelf"
    assert confidence >= 0.8


def test_classify_returns_supported_category(monkeypatch) -> None:
    query = _solid_image(640, 380, (130, 120, 110))
    q_ahash, q_dhash, q_edge, q_aspect, q_brightness = _image_features(query)

    refs = (
        ReferenceImage(
            category="desk",
            average_hash=q_ahash,
            difference_hash=q_dhash,
            edge_density=q_edge,
            aspect_ratio=q_aspect,
            brightness=q_brightness,
        ),
        ReferenceImage(
            category="cabinet",
            average_hash=0,
            difference_hash=0,
            edge_density=0.95,
            aspect_ratio=0.35,
            brightness=0.04,
        ),
    )

    monkeypatch.setattr("app.main._reference_gallery", lambda: refs)

    detected_type, confidence = _classify(query)

    assert detected_type in {"desk", "cabinet", "shelf"}
    assert 0.0 <= confidence <= 1.0
