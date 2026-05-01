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


PerturbationKind = Literal[
    "paraphrase",
    "demographic",  # age + sex token swaps
    "authority",
    "boundary",
    "ethnicity",  # patient first-name swaps (proxies for ethnic background)
    "profession",  # patient occupation swaps (proxies for SES)
]


class PerturbationProbe(Probe):
    """A probe derived from an anchor by one perturbation generator.

    Parent provenance:

    - `anchor_id` is always set. For curated v1 anchors (A1..A8) it
      carries the literal anchor id. For synthesized parents
      (`synthesized_probe_id` set) it carries `f"synth-{probe_id}"`
      so the existing per-anchor query/Jacobian path keeps working.
    - `synthesized_probe_id` is the explicit FK to `synthesized_probes`.
      `None` for curated parents; set for parents from the
      `synthesized_probes` table.

    Soft invariant: exactly one of `synthesized_probe_id is None` /
    `synthesized_probe_id is not None` describes the parent type.
    """

    kind: Literal["perturbation"] = "perturbation"
    anchor_id: str = Field(min_length=1)
    perturbation_kind: PerturbationKind
    transform_label: str = Field(min_length=1)
    generator_metadata: dict[str, Any] = Field(default_factory=dict)
    synthesized_probe_id: int | None = None

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
