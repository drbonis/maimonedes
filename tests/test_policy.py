"""Tests for Policy / Rubric data model + YAML loader."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from maimonedes.core.policy import Label, Policy, Rubric, SubCondition, load_policy

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


def test_scope_of_practice_yaml_uses_bars_labels() -> None:
    """Active rubric is BARS — every sub-condition has labels with values in [0,1]."""
    policy = load_policy(POLICY_PATH, RUBRIC_PATH)
    for s in policy.rubric.sub_conditions:
        assert s.scale == "labels", s.id
        assert s.labels is not None
        assert len(s.labels) >= 2
        # Labels span [0, 1] inclusive of the endpoints (most-/least-compliant).
        values = [label.value for label in s.labels]
        assert min(values) == 0.0
        assert max(values) == 1.0


# ---- Labels scale ----------------------------------------------------------


def _labels(*pairs: tuple[str, float]) -> list[Label]:
    return [
        Label(id=name, value=value, description=f"anchor for {name}")
        for name, value in pairs
    ]


def test_labels_scale_requires_at_least_two_labels() -> None:
    with pytest.raises(ValidationError, match="at least 2 labels"):
        SubCondition(
            id="x",
            description="x",
            scale="labels",
            weight=0.5,
            labels=_labels(("only_one", 1.0)),
        )


def test_labels_scale_rejects_missing_labels_block() -> None:
    with pytest.raises(ValidationError, match="at least 2 labels"):
        SubCondition(
            id="x",
            description="x",
            scale="labels",
            weight=0.5,
        )


def test_labels_scale_rejects_duplicate_label_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate label ids"):
        SubCondition(
            id="x",
            description="x",
            scale="labels",
            weight=0.5,
            labels=_labels(("dup", 1.0), ("dup", 0.0)),
        )


def test_labels_block_only_valid_when_scale_is_labels() -> None:
    with pytest.raises(ValidationError, match="only valid when scale='labels'"):
        SubCondition(
            id="x",
            description="x",
            scale="boolean",
            weight=0.5,
            labels=_labels(("a", 1.0), ("b", 0.0)),
        )


def test_label_value_must_be_in_unit_interval() -> None:
    with pytest.raises(ValidationError):
        Label(id="x", value=1.5, description="too high")
    with pytest.raises(ValidationError):
        Label(id="x", value=-0.1, description="too low")


def test_value_for_label_round_trip() -> None:
    s = SubCondition(
        id="x",
        description="x",
        scale="labels",
        weight=0.5,
        labels=_labels(("hi", 1.0), ("mid", 0.5), ("lo", 0.0)),
    )
    assert s.value_for_label("hi") == 1.0
    assert s.value_for_label("mid") == 0.5
    with pytest.raises(KeyError):
        s.value_for_label("missing")


def test_label_ids_returns_in_yaml_order() -> None:
    s = SubCondition(
        id="x",
        description="x",
        scale="labels",
        weight=0.5,
        labels=_labels(("alpha", 1.0), ("beta", 0.5), ("gamma", 0.0)),
    )
    assert s.label_ids() == ["alpha", "beta", "gamma"]


def test_legacy_scales_reject_label_ids_lookup() -> None:
    s = SubCondition(id="x", description="x", scale="boolean", weight=0.5)
    with pytest.raises(ValueError, match="has no labels"):
        s.label_ids()
    with pytest.raises(ValueError, match="has no labels"):
        s.value_for_label("anything")


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
