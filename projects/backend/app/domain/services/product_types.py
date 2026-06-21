from __future__ import annotations

# Canonical product types accepted by backend schema and mapping paths.
ALL_PRODUCT_TYPES: tuple[str, ...] = (
    "cabinet",
    "wardrobe",
    "bookcase",
    "desk",
    "table",
    "shelf",
    "nightstand",
    "dresser",
    "sideboard",
    "tv_stand",
)

# Types currently supported by generation/projection profiles.
PROFILE_PRODUCT_TYPES: tuple[str, ...] = (
    "cabinet",
    "desk",
    "shelf",
    "sideboard",
    "tv_stand",
)

CABINET_PROFILE_ALIASES: tuple[str, ...] = ("sideboard", "tv_stand")

# Runtime detected types accepted from inference payloads.
DETECTED_PRODUCT_TYPES: tuple[str, ...] = PROFILE_PRODUCT_TYPES

DEFAULT_SHELF_COUNTS: dict[str, int] = {
    "desk": 0,
    "sideboard": 0,
    "tv_stand": 1,
    "cabinet": 2,
    "shelf": 4,
}


def normalize_known_product_type(raw: object) -> str | None:
    normalized = str(raw or "").strip().lower()
    if normalized in set(ALL_PRODUCT_TYPES):
        return normalized
    return None


def normalize_detected_product_type(raw: object) -> str | None:
    normalized = str(raw or "").strip().lower()
    if normalized in set(DETECTED_PRODUCT_TYPES):
        return normalized
    return None


def generator_profile_for_inferred_type(raw: object) -> str | None:
    normalized = str(raw or "").strip().lower()
    if normalized not in set(PROFILE_PRODUCT_TYPES):
        return None
    if normalized in set(CABINET_PROFILE_ALIASES):
        return "cabinet"
    return normalized


def shelf_default_for_type(raw: object) -> int:
    normalized = str(raw or "").strip().lower()
    return int(DEFAULT_SHELF_COUNTS.get(normalized, 0))
