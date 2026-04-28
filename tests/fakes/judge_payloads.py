"""Shared helpers for building judge-response JSON in tests.

Tests across the suite need to construct mock judge responses for
arbitrary policies. These helpers handle every scale type the rubric
supports — boolean, 0-3, and BARS labels — so individual test files
don't have to maintain parallel scale-aware payload builders.
"""
from __future__ import annotations

import json
from typing import Any

from maimonedes.core.policy import Policy, SubCondition


def most_compliant_value(s: SubCondition) -> bool | int | str:
    """Per-scale value that maps to the highest normalised score."""
    if s.scale == "boolean":
        return True
    if s.scale == "0-3":
        return 3
    # labels — pick the highest-value label.
    assert s.labels is not None
    best = max(s.labels, key=lambda label: label.value)
    return best.id


def least_compliant_value(s: SubCondition) -> bool | int | str:
    """Per-scale value that maps to the lowest normalised score."""
    if s.scale == "boolean":
        return False
    if s.scale == "0-3":
        return 0
    assert s.labels is not None
    worst = min(s.labels, key=lambda label: label.value)
    return worst.id


def value_for_target(s: SubCondition, target: float) -> bool | int | str:
    """Per-scale value whose normalised score is closest to `target`.

    Used by the calibration tests to construct judge responses that
    aggregate to a target value within the per-axis discretisation.
    """
    target = max(0.0, min(1.0, target))
    if s.scale == "boolean":
        return target >= 0.5
    if s.scale == "0-3":
        return round(target * 3)
    assert s.labels is not None
    best = min(s.labels, key=lambda label: abs(label.value - target))
    return best.id


def make_compliant_response(policy: Policy) -> dict[str, Any]:
    """Top-of-scale response — all sub-conditions report fully compliant."""
    return {
        "scores": {
            s.id: most_compliant_value(s) for s in policy.rubric.sub_conditions
        }
    }


def make_payload_for_target(policy: Policy, target: float) -> dict[str, Any]:
    """Set every sub-condition to the closest representable score for `target`."""
    return {
        "scores": {
            s.id: value_for_target(s, target) for s in policy.rubric.sub_conditions
        }
    }


def compliant_response_json(policy: Policy) -> str:
    return json.dumps(make_compliant_response(policy))


def payload_json_for_target(policy: Policy, target: float) -> str:
    return json.dumps(make_payload_for_target(policy, target))


__all__ = [
    "compliant_response_json",
    "least_compliant_value",
    "make_compliant_response",
    "make_payload_for_target",
    "most_compliant_value",
    "payload_json_for_target",
    "value_for_target",
]
