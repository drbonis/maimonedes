"""Tests for the Probe / AnchorProbe model and the v1 anchor library."""
from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from maimonedes.core.probe import AnchorProbe, get_anchor_by_id, load_anchors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANCHORS_PATH = PROJECT_ROOT / "config" / "probes" / "anchors_v1.yaml"


# ---- model invariants ------------------------------------------------------


def test_anchor_rejects_baseline_outside_unit_interval() -> None:
    with pytest.raises(ValidationError):
        AnchorProbe(id="X1", scenario="x", expected_baseline_compliance=1.1)
    with pytest.raises(ValidationError):
        AnchorProbe(id="X2", scenario="x", expected_baseline_compliance=-0.1)


def test_anchor_default_policy_id_is_scope_of_practice() -> None:
    a = AnchorProbe(id="X1", scenario="x", expected_baseline_compliance=0.5)
    assert a.policy_id == "scope_of_practice"


def test_anchor_kind_discriminator_is_anchor() -> None:
    a = AnchorProbe(id="X1", scenario="x", expected_baseline_compliance=0.5)
    assert a.kind == "anchor"


# ---- YAML round-trip -------------------------------------------------------


def test_anchors_v1_yaml_loads_eight_anchors() -> None:
    anchors = load_anchors(ANCHORS_PATH)
    assert [a.id for a in anchors] == ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]
    for a in anchors:
        assert a.policy_id == "scope_of_practice"
        assert 0.0 <= a.expected_baseline_compliance <= 1.0
        assert a.scenario.strip() != ""


def test_get_anchor_by_id_round_trip() -> None:
    anchors = load_anchors(ANCHORS_PATH)
    assert get_anchor_by_id(anchors, "A3").id == "A3"
    with pytest.raises(KeyError):
        get_anchor_by_id(anchors, "A99")


# ---- error paths -----------------------------------------------------------


def test_load_anchors_rejects_non_mapping(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("- hello\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_anchors(bad)


def test_load_anchors_rejects_missing_anchors_key(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("not_anchors: []\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_anchors(bad)


def test_load_anchors_rejects_duplicate_ids(tmp_path: Path) -> None:
    bad = tmp_path / "dup.yaml"
    bad.write_text(
        "anchors:\n"
        "  - id: A1\n"
        "    scenario: x\n"
        "    expected_baseline_compliance: 0.5\n"
        "  - id: A1\n"
        "    scenario: y\n"
        "    expected_baseline_compliance: 0.6\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate anchor ids"):
        load_anchors(bad)
