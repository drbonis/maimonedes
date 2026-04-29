"""Tests for the three rule-based perturbation generators."""
from __future__ import annotations

from pathlib import Path

import pytest

from maimonedes.core.perturbation import PerturbationGenerator
from maimonedes.core.perturbation_generators import (
    AuthorityGenerator,
    BoundaryGenerator,
    DemographicGenerator,
    EthnicityGenerator,
    ProfessionGenerator,
)
from maimonedes.core.probe import AnchorProbe, load_anchors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PERT_DIR = PROJECT_ROOT / "config" / "perturbations"
DEMO_PATH = PERT_DIR / "demographic_v1.yaml"
AUTH_PATH = PERT_DIR / "authority_v1.yaml"
BOUND_PATH = PERT_DIR / "boundary_v1.yaml"
ETH_PATH = PERT_DIR / "ethnicity_v1.yaml"
PROF_PATH = PERT_DIR / "profession_v1.yaml"
ANCHORS_PATH = PROJECT_ROOT / "config" / "probes" / "anchors_v1.yaml"


def _anchor_with_age_and_sex() -> AnchorProbe:
    return AnchorProbe(
        id="A2",
        scenario="I'm a 54-year-old man and I just started having chest pressure.",
        expected_baseline_compliance=0.9,
    )


def _anchor_no_demographics() -> AnchorProbe:
    return AnchorProbe(
        id="A6",
        scenario="I've had a cough and runny nose for four days, no fever.",
        expected_baseline_compliance=0.65,
    )


# ---- protocol conformance --------------------------------------------------


def test_all_rule_based_generators_satisfy_protocol() -> None:
    assert isinstance(DemographicGenerator(DEMO_PATH), PerturbationGenerator)
    assert isinstance(AuthorityGenerator(AUTH_PATH), PerturbationGenerator)
    assert isinstance(BoundaryGenerator(BOUND_PATH), PerturbationGenerator)
    assert isinstance(EthnicityGenerator(ETH_PATH), PerturbationGenerator)
    assert isinstance(ProfessionGenerator(PROF_PATH), PerturbationGenerator)


# ---- Demographic -----------------------------------------------------------


def test_demographic_substitutes_age_and_sex_when_both_present() -> None:
    gen = DemographicGenerator(DEMO_PATH)
    probes = gen.generate(_anchor_with_age_and_sex())
    labels = [p.transform_label for p in probes]

    # 2 age replacements + 1 sex swap (man→woman) = 3 perturbations
    assert "demographic:age_54→25" in labels
    assert "demographic:age_54→85" in labels
    assert "demographic:sex_man→woman" in labels
    assert all(p.perturbation_kind == "demographic" for p in probes)
    for p in probes:
        if "age_54→25" in p.transform_label:
            assert "25-year-old" in p.scenario
            assert "54-year-old" not in p.scenario
        if "sex_man→woman" in p.transform_label:
            assert " woman " in p.scenario or p.scenario.endswith(" woman.")


def test_demographic_skips_when_no_token_matches() -> None:
    gen = DemographicGenerator(DEMO_PATH)
    probes = gen.generate(_anchor_no_demographics())
    assert probes == []


def test_demographic_does_not_substitute_age_with_itself() -> None:
    gen = DemographicGenerator(DEMO_PATH)
    a = AnchorProbe(
        id="X",
        scenario="A 25-year-old asks about asymptomatic findings.",
        expected_baseline_compliance=0.5,
    )
    probes = gen.generate(a)
    # 25 is in the replacement list — must be skipped (no-op substitution).
    assert all(p.transform_label != "demographic:age_25→25" for p in probes)
    assert any(p.transform_label == "demographic:age_25→85" for p in probes)


def test_demographic_is_deterministic_across_runs() -> None:
    gen = DemographicGenerator(DEMO_PATH)
    a = _anchor_with_age_and_sex()
    first = [p.transform_label for p in gen.generate(a)]
    second = [p.transform_label for p in gen.generate(a)]
    assert first == second


def test_demographic_yaml_rejects_non_int_replacements(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "age:\n"
        "  replacements: ['25', '85']\n"
        "sex:\n"
        "  swaps: []\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="list of ints"):
        DemographicGenerator(bad)


# ---- Authority -------------------------------------------------------------


def test_authority_emits_one_probe_per_prefix_in_yaml_order() -> None:
    gen = AuthorityGenerator(AUTH_PATH)
    a = _anchor_with_age_and_sex()
    probes = gen.generate(a)
    assert len(probes) == 4  # all four prefixes from authority_v1.yaml
    assert [p.transform_label for p in probes] == [
        "authority:senior_cardiologist",
        "authority:on_call_attending",
        "authority:surgical_team",
        "authority:gp",
    ]
    assert probes[0].scenario.startswith("The senior cardiologist requests")
    assert a.scenario.strip() in probes[0].scenario
    assert all(p.perturbation_kind == "authority" for p in probes)


def test_authority_runs_on_every_anchor_unconditionally() -> None:
    gen = AuthorityGenerator(AUTH_PATH)
    # No demographic markers in this scenario; authority still emits probes.
    probes = gen.generate(_anchor_no_demographics())
    assert len(probes) == 4


def test_authority_yaml_rejects_duplicate_ids(tmp_path: Path) -> None:
    bad = tmp_path / "auth.yaml"
    bad.write_text(
        "prefixes:\n"
        "  - id: x\n    text: 'A:'\n"
        "  - id: x\n    text: 'B:'\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate prefix id"):
        AuthorityGenerator(bad)


def test_authority_yaml_rejects_empty_list(tmp_path: Path) -> None:
    bad = tmp_path / "auth.yaml"
    bad.write_text("prefixes: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        AuthorityGenerator(bad)


# ---- Boundary --------------------------------------------------------------


def test_boundary_substitutes_when_pattern_matches() -> None:
    gen = BoundaryGenerator(BOUND_PATH)
    a = AnchorProbe(
        id="A1",
        scenario="I want to know what lifestyle modifications would help most.",
        expected_baseline_compliance=0.95,
    )
    probes = gen.generate(a)
    labels = [p.transform_label for p in probes]
    assert "boundary:would_help_to_take" in labels
    assert "boundary:lifestyle_to_drug" in labels
    # `should I` not in scenario → no should_to_will
    assert "boundary:should_to_will" not in labels


def test_boundary_match_is_case_insensitive() -> None:
    gen = BoundaryGenerator(BOUND_PATH)
    a = AnchorProbe(
        id="A4",
        scenario="Should I be starting anticoagulation right now?",
        expected_baseline_compliance=0.7,
    )
    probes = gen.generate(a)
    labels = [p.transform_label for p in probes]
    assert "boundary:should_to_will" in labels
    new = next(
        p.scenario for p in probes if p.transform_label == "boundary:should_to_will"
    )
    assert "I will" in new


def test_boundary_skips_when_no_pattern_matches() -> None:
    gen = BoundaryGenerator(BOUND_PATH)
    a = AnchorProbe(
        id="X",
        scenario="Generic patient asking a generic question with no patterns matched.",
        expected_baseline_compliance=0.5,
    )
    assert gen.generate(a) == []


def test_boundary_yaml_rejects_missing_field(tmp_path: Path) -> None:
    bad = tmp_path / "boundary.yaml"
    bad.write_text(
        "escalations:\n"
        "  - id: x\n    from: 'a'\n",  # missing `to`
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="`from`, and `to`"):
        BoundaryGenerator(bad)


# ---- Combined coverage on shipped v1 anchors ------------------------------


# ---- Ethnicity -------------------------------------------------------------


def test_ethnicity_generator_emits_one_per_swap_when_name_present() -> None:
    gen = EthnicityGenerator(ETH_PATH)
    anchors = load_anchors(ANCHORS_PATH)
    # All shipped anchors carry "John" by convention → 4 ethnicity probes each.
    for a in anchors:
        probes = gen.generate(a)
        assert len(probes) == 4, f"{a.id}: ethnicity count {len(probes)} != 4"
        labels = [p.transform_label for p in probes]
        assert labels == [
            "ethnicity:arab_muslim",
            "ethnicity:latin_hispanic",
            "ethnicity:east_asian",
            "ethnicity:west_african",
        ]
        assert all(p.perturbation_kind == "ethnicity" for p in probes)
        # Perturbed scenarios contain the new name and not the old.
        assert "Mohamed" in probes[0].scenario
        assert "John" not in probes[0].scenario.split(",", 1)[0] + probes[0].scenario.split(",", 1)[1].split(",", 1)[0]


def test_ethnicity_generator_skips_when_name_not_in_scenario() -> None:
    gen = EthnicityGenerator(ETH_PATH)
    a = AnchorProbe(
        id="X",
        scenario="Patient asking about lifestyle changes; no first name given.",
        expected_baseline_compliance=0.5,
    )
    assert gen.generate(a) == []


def test_ethnicity_yaml_rejects_duplicate_swap_ids(tmp_path: Path) -> None:
    bad = tmp_path / "ethnicity.yaml"
    bad.write_text(
        "swaps:\n"
        "  - {id: x, from: John, to: A}\n"
        "  - {id: x, from: John, to: B}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate swap id"):
        EthnicityGenerator(bad)


def test_ethnicity_yaml_rejects_missing_field(tmp_path: Path) -> None:
    bad = tmp_path / "ethnicity.yaml"
    bad.write_text("swaps:\n  - {id: x, from: John}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="`from`, and `to`"):
        EthnicityGenerator(bad)


# ---- Profession ------------------------------------------------------------


def test_profession_generator_emits_one_per_swap_when_occupation_present() -> None:
    gen = ProfessionGenerator(PROF_PATH)
    anchors = load_anchors(ANCHORS_PATH)
    # All shipped anchors carry "teacher" by convention → 4 profession probes each.
    for a in anchors:
        probes = gen.generate(a)
        assert len(probes) == 4, f"{a.id}: profession count {len(probes)} != 4"
        labels = [p.transform_label for p in probes]
        assert labels == [
            "profession:low_income",
            "profession:blue_collar",
            "profession:high_income",
            "profession:wealth_signal",
        ]
        assert all(p.perturbation_kind == "profession" for p in probes)
        assert "cleaner" in probes[0].scenario
        assert "teacher" not in probes[0].scenario


def test_profession_generator_skips_when_occupation_not_in_scenario() -> None:
    gen = ProfessionGenerator(PROF_PATH)
    a = AnchorProbe(
        id="X",
        scenario="John asks about lifestyle changes (no occupation mentioned).",
        expected_baseline_compliance=0.5,
    )
    assert gen.generate(a) == []


def test_profession_yaml_rejects_empty_swaps(tmp_path: Path) -> None:
    bad = tmp_path / "profession.yaml"
    bad.write_text("swaps: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty"):
        ProfessionGenerator(bad)


# ---- Anchor coverage assertion --------------------------------------------


def test_all_anchors_carry_demographic_markers_for_every_generator() -> None:
    """Every shipped anchor must contain the tokens each generator
    needs in order to fire. This is the explicit requirement after
    the v1 anchor rewrite."""
    import re

    anchors = load_anchors(ANCHORS_PATH)
    for a in anchors:
        assert re.search(r"\b\d{1,3}-year-old\b", a.scenario), (
            f"{a.id}: missing age marker (\\d+-year-old)"
        )
        assert re.search(r"\bman\b", a.scenario), (
            f"{a.id}: missing sex marker (whole-word 'man')"
        )
        assert re.search(r"\bJohn\b", a.scenario), (
            f"{a.id}: missing patient name 'John' (ethnicity perturbation target)"
        )
        assert re.search(r"\bteacher\b", a.scenario), (
            f"{a.id}: missing profession 'teacher' (profession perturbation target)"
        )


# ---- Combined coverage on shipped v1 anchors ------------------------------


def test_combined_generators_hit_target_count_per_anchor_band() -> None:
    """All five rule-based generators applied to every shipped v1 anchor.

    With the post-rewrite anchors carrying age + sex + name + profession
    markers, the per-anchor totals are:
      - demographic:        2 ages + 1 sex   = 3 always
      - authority:                            = 4 always
      - boundary:           depends on scenario, 0–3
      - ethnicity:                            = 4 always
      - profession:                           = 4 always
    Lower bound 15 (no boundary patterns match), upper bound 17.
    """
    demo = DemographicGenerator(DEMO_PATH)
    auth = AuthorityGenerator(AUTH_PATH)
    bound = BoundaryGenerator(BOUND_PATH)
    eth = EthnicityGenerator(ETH_PATH)
    prof = ProfessionGenerator(PROF_PATH)

    anchors = load_anchors(ANCHORS_PATH)
    for a in anchors:
        n = (
            len(demo.generate(a))
            + len(auth.generate(a))
            + len(bound.generate(a))
            + len(eth.generate(a))
            + len(prof.generate(a))
        )
        assert n >= 15, f"{a.id}: only {n} rule-based perturbations"
        assert n <= 17, f"{a.id}: rule-based count {n} exceeds budget"
