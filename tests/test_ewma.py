"""Unit + DB-roundtrip tests for the Phase 3 EWMA detector."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from alembic import command
from alembic.config import Config

from maimonedes.core.compliance import ComplianceScore
from maimonedes.monitor._baseline import InsufficientBaseline
from maimonedes.monitor.ewma import LowerEwma, ewma_per_anchor
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


def test_lower_ewma_in_control_post_never_fires() -> None:
    """When post values match the underlying baseline distribution
    closely, the EWMA statistic tracks μ₀ and stays above LCL_t.
    """
    rng = np.random.default_rng(seed=0)
    baseline = (0.9 + rng.normal(0, 0.02, size=10)).tolist()
    post = [0.9] * 50  # at the true μ; sample μ_hat may differ slightly

    det = LowerEwma()
    det.fit(baseline, label="in-control")
    states = det.run(post)
    # Statistic exponentially relaxes from sample-μ₀ toward 0.9.
    final = states[-1].statistic
    assert math.isclose(final, 0.9, abs_tol=1e-3)
    assert all(not s.fired for s in states)
    assert det.first_fire_index(post) is None


def test_lower_ewma_step_down_fires() -> None:
    rng = np.random.default_rng(seed=1)
    baseline = (0.9 + rng.normal(0, 0.02, size=10)).tolist()
    post_baseline = (0.9 + rng.normal(0, 0.02, size=10)).tolist()
    post_step = (0.7 + rng.normal(0, 0.02, size=20)).tolist()

    det = LowerEwma()
    det.fit(baseline, label="step")
    fire_idx = det.first_fire_index(post_baseline + post_step)
    assert fire_idx is not None
    # Step starts at index 10; EWMA with λ=0.2 lags more than CUSUM so we
    # only assert it eventually fires within the window after the step.
    assert fire_idx >= 10


def test_lower_ewma_linear_ramp_fires() -> None:
    rng = np.random.default_rng(seed=2)
    baseline = (0.9 + rng.normal(0, 0.02, size=15)).tolist()
    ramp = np.linspace(0.9, 0.5, 30) + rng.normal(0, 0.02, size=30)

    det = LowerEwma()
    det.fit(baseline, label="ramp")
    assert det.first_fire_index(ramp.tolist()) is not None


def test_lower_ewma_first_fire_returns_session_index() -> None:
    rng = np.random.default_rng(seed=3)
    baseline = (0.9 + rng.normal(0, 0.01, size=10)).tolist()
    post = baseline + [0.6] * 20
    indices = list(range(100, 100 + len(post)))

    det = LowerEwma()
    det.fit(baseline, label="indices")
    fire_idx = det.first_fire_index(post, session_indices=indices)
    assert fire_idx is not None
    assert fire_idx >= 100  # honours the offset


def test_lower_ewma_step_requires_fit_first() -> None:
    det = LowerEwma()
    with pytest.raises(RuntimeError):
        det.step(0.5, session_index=0)


def test_lower_ewma_run_validates_index_length() -> None:
    det = LowerEwma()
    det.fit([0.9] * 10, label="length")
    with pytest.raises(ValueError):
        det.run([0.5, 0.5], session_indices=[1])


def test_insufficient_baseline_raises_on_fit() -> None:
    det = LowerEwma()
    with pytest.raises(InsufficientBaseline):
        det.fit([0.9, 0.91, 0.88], label="short")


def test_lower_ewma_lcl_is_time_varying_and_widens() -> None:
    """Early-step LCL must be wider (further below μ₀) than asymptotic."""
    det = LowerEwma()
    det.fit([0.9, 0.92, 0.91, 0.88, 0.93], label="lcl")

    # First step LCL should be MUCH closer to μ₀ than the asymptotic limit
    # (suppressing spurious early fires).
    states = det.run([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    early_lcl = states[0].lcl
    late_lcl = states[-1].lcl
    asymp = det.asymptotic_lcl
    # Larger t → tighter LCL → closer to asymptotic from above.
    assert early_lcl > late_lcl
    assert math.isclose(late_lcl, asymp, rel_tol=0.1)


# ---- DB roundtrip ----------------------------------------------------------


def _seed_drift_run(
    *,
    n_baseline: int,
    n_post: int,
    baseline_mean: float = 0.9,
    post_mean: float = 0.6,
    rng_seed: int = 42,
) -> int:
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
        mean = baseline_mean if idx < n_baseline else post_mean
        agg = float(np.clip(mean + rng.normal(0, 0.02), 0.0, 1.0))
        record_score(
            ComplianceScore(
                anchor_id="A1",
                policy_id="scope_of_practice",
                per_sub_condition={"flags_physician_review": 1.0},
                aggregate=agg,
                judge_model="judge:test",
                supervised_model="llama:test",
                drift_session_id=sid,
            )
        )
    return run_id


def test_ewma_per_anchor_fires_on_step_drift(db: str) -> None:
    run_id = _seed_drift_run(n_baseline=10, n_post=20)
    traces = ewma_per_anchor(run_id)
    assert "A1" in traces
    states = traces["A1"]
    assert len(states) == 30
    fired = [s for s in states if s.fired]
    assert fired
    assert fired[0].session_index >= 10


def test_ewma_per_anchor_skips_short_baseline(db: str) -> None:
    run_id = _seed_drift_run(n_baseline=4, n_post=10)
    assert ewma_per_anchor(run_id) == {}


def test_ewma_per_anchor_returns_empty_on_unknown_run(db: str) -> None:
    assert ewma_per_anchor(404) == {}
