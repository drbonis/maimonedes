"""Trained model artefacts for Phase 5.

Holds the Stage-2 classifier (#37 / #38), the GP fit (#39), and any
future learned models. Distinct from `storage/` (DB ORM) and `core/`
(domain types) because these objects are persisted as pickle blobs
on disk with rows in `storage/stage2_models.py` etc. tracking their
metadata.
"""
