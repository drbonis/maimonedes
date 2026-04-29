"""Storage subpackage.

Imports every ORM module on package init so any cross-table foreign
key (e.g. compliance_scores.llm_call_id -> llm_calls.id) can resolve
its target inside `Base.metadata` regardless of which module the
caller imports first.
"""
from maimonedes.storage import (  # noqa: F401
    compliance,
    drift,
    llm_calls,
    models,
    perturbations,
    recovery,
)
