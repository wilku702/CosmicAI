from __future__ import annotations

PROHIBITED_SUBSTRINGS = [
    "llm_ready/",
    "ground_truth",
    "metadata.json",
    "gt_images",
]


def assert_no_gt_refs(value: str) -> None:
    lowered = value.lower()
    for token in PROHIBITED_SUBSTRINGS:
        if token in lowered:
            raise ValueError(
                f"Detected forbidden reference '{token}' in path or configuration: {value}"
            )


__all__ = ["assert_no_gt_refs"]
