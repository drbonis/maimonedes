"""Unit + DB-roundtrip tests for the Phase 3 CUSUM detector."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from alembic import command
from alembic.config import Config

from maimonedes.core.compliance import ComplianceScore
from maimonedes.monitor._baseline import (
    InsufficientBaseline,
    compute_baseline_stats,
)
from maimonedes.monitor.cusum import LowerCusum, cusum_per_anchor
from maimonedes.settings import Settings
from maimonedes.storage.compliance import record_score
from maimonedes.storage.drift import (
    create_drift_run,
    create_drift_session,
)
from maimonedes.storage.repo import init_engine, reset_engine_for_tests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ALEMBIC_INI = PROJECT_ROOT / "alembic.ini"


def _alembic_cfg(database_url: str) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", database_url)
    return cfg


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    db_path = tmp_path / "p3.sqlite"
    url = f"sqlite:///{db_path}"
    monkeypatch.setenv("DATABASE_URL", url)
    reset_engine_for_tests()
    init_engine(Settings(database_url=url))
    command.upgrade(_alembic_cfg(url), "head")
    yield url
    reset_engine_for_tests()


# ---- pure-numeric behaviour ------------------------------------------------


def test_compute_baseline_stats_raises_on_short_baseline() -> None:
    with pytest.raises(InsufficientBaseline):
        compute_baseline_stats([0.9, 0.85, 0.88, 0.92], label="short")


def test_compute_baseline_stats_warns_and_falls_back_when_sigma_zero() -> None:
    with pytest.warns(UserWarning, match="sigma is zero"):
        mu, sigma = compute_baseline_stats([0.9] * 10, label="flat")
    assert mu == pytest.approx(0.9)
    assert sigma == pytest.approx(0.01)


def test_lower_cusum_in_control_post_never_fires() -> None:
    """At exactly the baseline mean the CUSUM increment is -K < 0,
    so the running statistic stays clipped at 0 and never fires.
    """
    rng = np.random.default_rng(seed=0)
    baseline = (0.9 + rng.normal(0, 0.02, size=10)).tolist()
    post = [0.9] * 50  # deterministic, exactly at μ₀

    det = LowerCusum()
    det.fit(baseline, label="in-control")
    states = det.run(post)
    assert all(s.statistic == pytest.approx(0.0) for s in states)
    assert all(not s.fired for s in states)
    assert det.first_fire_index(post) is None


def test_lower_cusum_step_down_fires_quickly() -> None:
    rng = np.random.default_rng(seed=1)
    baseline = (0.9 + rng.normal(0, 0.02, size=10)).tolist()
    # Step the post-window down by 5σ; CUSUM must fire within a handful of samples.
    post_baseline = 0.9 + rng.normal(0, 0.02, size=10)
    post_step = 0.7 + rng.normal(0, 0.02, size=10)
    post = post_baseline.tolist() + post_step.tolist()

    det = LowerCusum()
    det.fit(baseline, label="step")
    fire_idx = det.first_fire_index(post)
    assert fire_idx is not None
    # The step starts at index 10. Detection should land within ~5 samples.
    assert 10 <= fire_idx <= 15


def test_lower_cusum_linear_ramp_fires() -> None:
    rng = np.random.default_rng(seed=2)
    baseline = (0.9 + rng.normal(0, 0.02, size=15)).tolist()
    ramp = np.linspace(0.9, 0.5, 30) + rng.normal(0, 0.02, size=30)

    det = LowerCusum()
    det.fit(baseline, label="ramp")
    assert det.first_fire_index(ramp.tolist()) is not None


def test_lower_cusum_first_fire_returns_session_index() -> None:
    rng = np.random.default_rng(seed=3)
    baseline = (0.9 + rng.normal(0, 0.01, size=10)).tolist()
    post = baseline + (0.6 + rng.normal(0, 0.01, size=10)).tolist()
    indices = list(range(100, 100 + len(post)))

    det = LowerCusum()
    det.fit(baseline, label="indices")
    fire_idx = det.first_fire_index(post, session_indices=indices)
    assert fire_idx is not None
    assert fire_idx >= 100  # honours the offset, not the implicit 0..N-1


def test_lower_cusum_step_requires_fit_first() -> None:
    det = LowerCusum()
    with pytest.raises(RuntimeError):
        det.step(0.5, session_index=0)


def test_lower_cusum_run_validates_index_length() -> None:
    det = LowerCusum()
    det.fit([0.9] * 10, label="length")
    with pytest.raises(ValueError):
        det.run([0.5, 0.5], session_indices=[1])


def test_lower_cusum_k_scales_threshold() -> None:
    det_k4 = LowerCusum(k_threshold=4.0)
    det_k4.fit([0.9, 0.92, 0.91, 0.88, 0.93], label="k4")
    det_k8 = LowerCusum(k_threshold=8.0)
    det_k8.fit([0.9, 0.92, 0.91, 0.88, 0.93], label="k8")
    assert det_k8.h == pytest.approx(2 * det_k4.h)


# ---- DB roundtrip ----------------------------------------------------------


def _seed_drift_run(
    *,
    n_baseline: int,
    n_post: int,
    baseline_mean: float = 0.9,
    post_mean: float = 0.6,
    anchors: tuple[str, ...] = ("A1",),
    rng_seed: int = 42,
) -> int:
    """Create a drift run with synthetic per-anchor scalar streams."""
    rng = np.random.default_rng(rng_seed)
    run_id = create_drift_run(
        policy_id="scope_of_practice",
        supervised_model="llama:test",
        judge_model="judge:test",
        schedule_path="config/drift/scope_of_practice_v1.yaml",
    )
    total = n_baseline + n_post
    for idx in range(total):
        stage = "baseline" if idx < n_baseline else "concise"
        sid = create_drift_session(run_id, idx, stage, "")
        for anchor in anchors:
            mean = baseline_mean if idx < n_baseline else post_mean
            agg = float(np.clip(mean + rng.normal(0, 0.02), 0.0, 1.0))
            record_score(
                ComplianceScore(
                    anchor_id=anchor,
                    policy_id="scope_of_practice",
                    per_sub_condition={"flags_physician_review": 1.0},
                    aggregate=agg,
                    judge_model="judge:test",
                    supervised_model="llama:test",
                    drift_session_id=sid,
                )
            )
    return run_id


def test_cusum_per_anchor_fires_on_step_drift(db: str) -> None:
    run_id = _seed_drift_run(n_baseline=10, n_post=20)
    traces = cusum_per_anchor(run_id)
    assert "A1" in traces
    states = traces["A1"]
    assert len(states) == 30
    fired = [s for s in states if s.fired]
    assert fired, "CUSUM should fire on a 0.9 → 0.6 step"
    # Fire must land in the post-baseline portion.
    assert fired[0].session_index >= 10


def test_cusum_per_anchor_skips_short_baseline(db: str) -> None:
    run_id = _seed_drift_run(n_baseline=4, n_post=10)
    traces = cusum_per_anchor(run_id)
    assert traces == {}  # one anchor, baseline too short → omitted


def test_cusum_per_anchor_returns_empty_on_unknown_run(db: str) -> None:
    assert cusum_per_anchor(404) == {}


def test_cusum_per_anchor_propagates_k(db: str) -> None:
    run_id = _seed_drift_run(n_baseline=10, n_post=20)
    permissive = cusum_per_anchor(run_id, k=2.0)
    strict = cusum_per_anchor(run_id, k=20.0)

    def _first_fire(states: list[object]) -> int | None:
        return next((s.session_index for s in states if s.fired), None)  # type: ignore[attr-defined]

    perm_fire = _first_fire(permissive["A1"])
    strict_fire = _first_fire(strict["A1"])
    assert perm_fire is not None
    # Larger k → larger h → can only fire later (or never), never earlier.
    assert strict_fire is None or strict_fire >= perm_fire
