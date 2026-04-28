"""Perturbation probe data model.

A `PerturbationProbe` is a probe whose scenario is derived from an
anchor by one of the four Phase 2 perturbation generators
(paraphrase, demographic, authority, boundary). It carries enough
provenance for the §4.4 Jacobian computation: which anchor it came
from, which generator produced it, and a short stable
`transform_label` that becomes the column header in the per-anchor
Jacobian table.

`generator_metadata` is a JSON-serialisable dict the generator owns —
paraphrase puts the rewritten scenario there, demographic puts the
substitution map, etc. The framework does not interpret it.
"""
from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import Field, model_validator

from maimonedes.core.probe import AnchorProbe, Probe


PerturbationKind = Literal["paraphrase", "demographic", "authority", "boundary"]


class PerturbationProbe(Probe):
    """A probe derived from an anchor by one perturbation generator."""

    kind: Literal["perturbation"] = "perturbation"
    anchor_id: str = Field(min_length=1)
    perturbation_kind: PerturbationKind
    transform_label: str = Field(min_length=1)
    generator_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _autoset_id(cls, data: Any) -> Any:
        # The pydantic-side id is `anchor_id#transform_label` by default.
        # It's a stable, human-readable identifier independent of the
        # auto-increment row id in `perturbation_probes`.
        if isinstance(data, dict) and not data.get("id"):
            anchor = data.get("anchor_id", "")
            label = data.get("transform_label", "")
            if anchor and label:
                data["id"] = f"{anchor}#{label}"
        return data


@runtime_checkable
class PerturbationGenerator(Protocol):
    """Structural contract every perturbation generator implements.

    Implementations may carry state set up at construction (LLM client,
    YAML-loaded templates, RNG seed, etc.) but `generate()` must be
    side-effect free with respect to global state.
    """

    def generate(self, anchor: AnchorProbe) -> list[PerturbationProbe]: ...


__all__ = ["PerturbationGenerator", "PerturbationKind", "PerturbationProbe"]
