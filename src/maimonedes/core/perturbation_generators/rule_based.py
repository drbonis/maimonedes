"""Rule-based perturbation generators (Phase 2 / #18).

Three deterministic generators sharing a YAML-templated pattern:

- `DemographicGenerator` — token substitution over age and sex
- `AuthorityGenerator`   — prefix injection
- `BoundaryGenerator`    — directive-language escalation

Determinism is the contract: same anchor in → same perturbation list
out, in the same order, across runs and machines. Generators are
stateless after construction; no LLM calls.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from maimonedes.core.perturbation import PerturbationProbe
from maimonedes.core.probe import AnchorProbe


_AGE_PATTERN = re.compile(r"\b(\d{1,3})-year-old\b")


def _load_yaml(path: str | Path) -> dict[str, Any]:
    raw = Path(path).read_text(encoding="utf-8")
    parsed = yaml.safe_load(raw)
    if not isinstance(parsed, dict):
        raise ValueError(
            f"{path}: top-level YAML must be a mapping, got {type(parsed).__name__}"
        )
    return parsed


def _build_probe(
    anchor: AnchorProbe,
    *,
    perturbation_kind: str,
    transform_label: str,
    new_scenario: str,
    metadata: dict[str, Any],
) -> PerturbationProbe:
    return PerturbationProbe(
        id=f"{anchor.id}#{transform_label}",
        anchor_id=anchor.id,
        scenario=new_scenario,
        policy_id=anchor.policy_id,
        perturbation_kind=perturbation_kind,  # type: ignore[arg-type]
        transform_label=transform_label,
        generator_metadata=metadata,
    )


# ---- Demographic -----------------------------------------------------------


class DemographicGenerator:
    """Substitutes age and sex tokens in the anchor scenario."""

    def __init__(self, templates_path: str | Path) -> None:
        data = _load_yaml(templates_path)
        age_block = data.get("age", {})
        sex_block = data.get("sex", {})
        if not isinstance(age_block, dict):
            raise ValueError(f"{templates_path}: `age` must be a mapping")
        if not isinstance(sex_block, dict):
            raise ValueError(f"{templates_path}: `sex` must be a mapping")

        replacements = age_block.get("replacements", [])
        if not isinstance(replacements, list) or not all(
            isinstance(r, int) for r in replacements
        ):
            raise ValueError(
                f"{templates_path}: `age.replacements` must be a list of ints"
            )
        self._age_replacements: list[int] = replacements

        swaps = sex_block.get("swaps", [])
        if not isinstance(swaps, list):
            raise ValueError(f"{templates_path}: `sex.swaps` must be a list")
        clean_swaps: list[tuple[str, str]] = []
        for entry in swaps:
            if (
                not isinstance(entry, dict)
                or "from" not in entry
                or "to" not in entry
                or not isinstance(entry["from"], str)
                or not isinstance(entry["to"], str)
            ):
                raise ValueError(
                    f"{templates_path}: each `sex.swaps` entry must have "
                    f"string `from` and `to` fields"
                )
            clean_swaps.append((entry["from"], entry["to"]))
        self._sex_swaps = clean_swaps

    def generate(self, anchor: AnchorProbe) -> list[PerturbationProbe]:
        out: list[PerturbationProbe] = []
        out.extend(self._age_perturbations(anchor))
        out.extend(self._sex_perturbations(anchor))
        return out

    def _age_perturbations(self, anchor: AnchorProbe) -> list[PerturbationProbe]:
        match = _AGE_PATTERN.search(anchor.scenario)
        if match is None:
            return []
        original = int(match.group(1))
        out: list[PerturbationProbe] = []
        for new_age in self._age_replacements:
            if new_age == original:
                continue
            new_scenario = _AGE_PATTERN.sub(
                f"{new_age}-year-old", anchor.scenario, count=1
            )
            label = f"demographic:age_{original}→{new_age}"
            out.append(
                _build_probe(
                    anchor,
                    perturbation_kind="demographic",
                    transform_label=label,
                    new_scenario=new_scenario,
                    metadata={
                        "axis": "age",
                        "original": original,
                        "replacement": new_age,
                    },
                )
            )
        return out

    def _sex_perturbations(self, anchor: AnchorProbe) -> list[PerturbationProbe]:
        out: list[PerturbationProbe] = []
        for from_word, to_word in self._sex_swaps:
            pattern = re.compile(rf"\b{re.escape(from_word)}\b")
            if not pattern.search(anchor.scenario):
                continue
            new_scenario = pattern.sub(to_word, anchor.scenario, count=1)
            label = f"demographic:sex_{from_word}→{to_word}"
            out.append(
                _build_probe(
                    anchor,
                    perturbation_kind="demographic",
                    transform_label=label,
                    new_scenario=new_scenario,
                    metadata={
                        "axis": "sex",
                        "from": from_word,
                        "to": to_word,
                    },
                )
            )
        return out


# ---- Authority -------------------------------------------------------------


class AuthorityGenerator:
    """Prefixes the anchor scenario with one of N authority cues."""

    def __init__(self, prefixes_path: str | Path) -> None:
        data = _load_yaml(prefixes_path)
        items = data.get("prefixes", [])
        if not isinstance(items, list) or not items:
            raise ValueError(
                f"{prefixes_path}: `prefixes` must be a non-empty list"
            )
        clean: list[tuple[str, str]] = []
        ids: set[str] = set()
        for entry in items:
            if (
                not isinstance(entry, dict)
                or "id" not in entry
                or "text" not in entry
                or not isinstance(entry["id"], str)
                or not isinstance(entry["text"], str)
                or not entry["id"]
                or not entry["text"]
            ):
                raise ValueError(
                    f"{prefixes_path}: each `prefixes` entry must have "
                    f"non-empty string `id` and `text`"
                )
            if entry["id"] in ids:
                raise ValueError(
                    f"{prefixes_path}: duplicate prefix id {entry['id']!r}"
                )
            ids.add(entry["id"])
            clean.append((entry["id"], entry["text"]))
        self._prefixes = clean

    def generate(self, anchor: AnchorProbe) -> list[PerturbationProbe]:
        out: list[PerturbationProbe] = []
        for prefix_id, prefix_text in self._prefixes:
            new_scenario = f"{prefix_text}\n\n{anchor.scenario.strip()}"
            label = f"authority:{prefix_id}"
            out.append(
                _build_probe(
                    anchor,
                    perturbation_kind="authority",
                    transform_label=label,
                    new_scenario=new_scenario,
                    metadata={"prefix_id": prefix_id, "prefix_text": prefix_text},
                )
            )
        return out


# ---- Boundary --------------------------------------------------------------


class BoundaryGenerator:
    """Replaces tentative phrases with directive ones (case-insensitive)."""

    def __init__(self, escalations_path: str | Path) -> None:
        data = _load_yaml(escalations_path)
        items = data.get("escalations", [])
        if not isinstance(items, list) or not items:
            raise ValueError(
                f"{escalations_path}: `escalations` must be a non-empty list"
            )
        clean: list[tuple[str, str, str]] = []
        ids: set[str] = set()
        for entry in items:
            if (
                not isinstance(entry, dict)
                or not all(
                    isinstance(entry.get(k), str) and entry.get(k)
                    for k in ("id", "from", "to")
                )
            ):
                raise ValueError(
                    f"{escalations_path}: each `escalations` entry must have "
                    f"non-empty string `id`, `from`, and `to`"
                )
            if entry["id"] in ids:
                raise ValueError(
                    f"{escalations_path}: duplicate escalation id {entry['id']!r}"
                )
            ids.add(entry["id"])
            clean.append((entry["id"], entry["from"], entry["to"]))
        self._escalations = clean

    def generate(self, anchor: AnchorProbe) -> list[PerturbationProbe]:
        out: list[PerturbationProbe] = []
        for esc_id, from_phrase, to_phrase in self._escalations:
            pattern = re.compile(re.escape(from_phrase), re.IGNORECASE)
            if not pattern.search(anchor.scenario):
                continue
            new_scenario = pattern.sub(to_phrase, anchor.scenario, count=1)
            label = f"boundary:{esc_id}"
            out.append(
                _build_probe(
                    anchor,
                    perturbation_kind="boundary",
                    transform_label=label,
                    new_scenario=new_scenario,
                    metadata={
                        "escalation_id": esc_id,
                        "from": from_phrase,
                        "to": to_phrase,
                    },
                )
            )
        return out


__all__ = [
    "AuthorityGenerator",
    "BoundaryGenerator",
    "DemographicGenerator",
]
