"""Tests for Policy / Rubric data model + YAML loader."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from maimonedes.core.policy import Policy, Rubric, SubCondition, load_policy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = PROJECT_ROOT / "config" / "policies" / "scope_of_practice.yaml"
RUBRIC_PATH = PROJECT_ROOT / "config" / "rubrics" / "scope_of_practice.yaml"


# ---- model invariants ------------------------------------------------------


def _make_rubric(weights: list[float]) -> Rubric:
    sub_conditions = [
        SubCondition(id=f"s{i}", description=f"d{i}", scale="boolean", weight=w)
        for i, w in enumerate(weights)
    ]
    return Rubric(sub_conditions=sub_conditions)


def test_rubric_requires_5_to_7_sub_conditions() -> None:
    with pytest.raises(ValidationError):
        _make_rubric([1.0])  # too few
    with pytest.raises(ValidationError):
        _make_rubric([0.125] * 8)  # too many


def test_rubric_weights_must_sum_to_one() -> None:
    with pytest.raises(ValidationError):
        _make_rubric([0.1, 0.1, 0.1, 0.1, 0.1])  # sum 0.5


def test_rubric_rejects_duplicate_sub_condition_ids() -> None:
    sub_conditions = [
        SubCondition(id="dup", description="a", scale="boolean", weight=0.2),
        SubCondition(id="dup", description="b", scale="boolean", weight=0.2),
        SubCondition(id="c", description="c", scale="boolean", weight=0.2),
        SubCondition(id="d", description="d", scale="boolean", weight=0.2),
        SubCondition(id="e", description="e", scale="boolean", weight=0.2),
    ]
    with pytest.raises(ValidationError):
        Rubric(sub_conditions=sub_conditions)


def test_sub_condition_rejects_unknown_scale() -> None:
    with pytest.raises(ValidationError):
        SubCondition(id="x", description="x", scale="continuous", weight=0.5)  # type: ignore[arg-type]


def test_sub_condition_rejects_negative_weight() -> None:
    with pytest.raises(ValidationError):
        SubCondition(id="x", description="x", scale="boolean", weight=-0.1)


def test_rubric_by_id_returns_correct_sub_condition() -> None:
    r = _make_rubric([0.2, 0.2, 0.2, 0.2, 0.2])
    assert r.by_id("s2").id == "s2"
    with pytest.raises(KeyError):
        r.by_id("missing")


# ---- YAML round-trip -------------------------------------------------------


def test_scope_of_practice_yaml_loads_cleanly() -> None:
    policy = load_policy(POLICY_PATH, RUBRIC_PATH)
    assert isinstance(policy, Policy)
    assert policy.id == "scope_of_practice"
    assert policy.name == "Scope of practice"
    assert "advisory scope" in policy.text.lower()

    rubric = policy.rubric
    assert 5 <= len(rubric.sub_conditions) <= 7
    assert abs(sum(s.weight for s in rubric.sub_conditions) - 1.0) < 1e-6
    # The known sub-conditions referenced by judge prompt + tests must exist.
    for required in (
        "flags_physician_review",
        "expresses_uncertainty",
        "avoids_unqualified_prescriptive_language",
        "recommendation_within_scope",
        "recommendation_appropriate_specificity",
    ):
        assert any(s.id == required for s in rubric.sub_conditions), required


def test_load_policy_rejects_non_mapping(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- not a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_policy(bad)


def test_load_policy_rejects_missing_fields(tmp_path: Path) -> None:
    bad = tmp_path / "policy.yaml"
    bad.write_text("id: x\nname: x\n", encoding="utf-8")  # missing text + rubric
    with pytest.raises(ValidationError):
        load_policy(bad)
