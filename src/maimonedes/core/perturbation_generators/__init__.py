"""Concrete perturbation generators.

Each generator implements `core.perturbation.PerturbationGenerator`.
Phase 2 ships four:

- `paraphrase.ParaphraseGenerator`             — LLM-driven (#17)
- `rule_based.DemographicGenerator`             — token substitution (#18)
- `rule_based.AuthorityGenerator`               — prefix injection (#18)
- `rule_based.BoundaryGenerator`                — phrase escalation (#18)
"""
from maimonedes.core.perturbation_generators.paraphrase import ParaphraseGenerator

__all__ = ["ParaphraseGenerator"]
