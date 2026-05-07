# maimonedes

Feasibility demo of a black-box behavioral supervision framework for clinical-decision LLMs.

See `docs/roadmap.md` for the implementation plan and `docs/blackbox_supervision_architecture.md` for the canonical design.

## Installation

Requires [uv](https://github.com/astral-sh/uv) and Python 3.11.

```bash
uv sync
```

The repo lives on a vboxsf shared folder by default; `scripts/dev_env.sh` sets `UV_PROJECT_ENVIRONMENT` and `UV_LINK_MODE=copy` to avoid symlink/locking issues. See `CLAUDE.md` for environment quirks.

```bash
source scripts/dev_env.sh
uv run maimonedes --help
```

The SQLite database lives at `/home/vagrant/maimonedes.db` (ext4, off vboxsf).

---

# Operator's guide — end-to-end demo

Assumes you're at the repo root and have sourced `scripts/dev_env.sh` once.

## 0. Sanity check

```bash
uv run maimonedes ping --all
sqlite3 /home/vagrant/maimonedes.db "SELECT version_num FROM alembic_version;"   # should be 0017 (or current head)
```

`ping --all` exercises the full plumbing: settings → Ollama backend (192.168.1.30:11434) → Bio_ClinicalBERT (192.168.1.30:8000) → DB write. If it's green, the substrate works.

## 1. Phase 1 — baseline + calibrate

Establish that the judge produces stable scores and the supervised system has a measurable starting point.

```bash
uv run maimonedes calibrate                        # scores the reference corpus → calibration table
uv run maimonedes run-once A1                      # one anchor end-to-end (anchor → supervised → judge → compliance_scores row)
# ... repeat for A2..A8 if you want a fresh baseline distribution
```

**What lands**: `compliance_scores` rows for each anchor, `llm_calls` rows for each LLM round-trip.

**What to read**: dashboard page `01_compliance_scores.py`.

```bash
uv run streamlit run src/maimonedes/dashboard/app.py
```

## 2. Phase 2 — perturb to measure local fragility

Generate a perturbation cloud around each anchor; the resulting Jacobian shows which directions in input space the supervised system is fragile in.

```bash
uv run maimonedes perturb A1 --replicates 3        # one anchor, 3 replicates per transform
uv run maimonedes perturb --all-anchors            # all 8 anchors, single replicate each
uv run maimonedes fragility-report                 # writes reports/fragility_<UTC>.csv
```

**What lands**: `perturbation_probes` rows (the perturbed scenarios) + `compliance_scores` rows tagged `probe_role="perturbation"`.

**What to read**: dashboard `02_fragility.py` — the §4.4 prediction (authority/boundary redder than demographic) is the empirical test of architecture-doc §5.5.5.

## 3. Phase 3 — induce drift, watch the detectors fire

Apply staged contamination via `--schedule`, score every anchor every session, then report.

```bash
uv run maimonedes induce-drift                                       # uses config/drift/scope_of_practice_v1.yaml by default
sqlite3 /home/vagrant/maimonedes.db "SELECT id FROM drift_runs ORDER BY id DESC LIMIT 1;"  # → DRIFT_ID
uv run maimonedes drift-report <DRIFT_ID>                            # CUSUM + EWMA + lead-time table
```

**What lands**: `drift_runs` row, one `drift_sessions` row per session (with the contamination suffix), per-anchor `compliance_scores` rows for every session.

**What to read**: dashboard `03_drift.py`. The headline number is `lead = N sessions (violation - cusum)` — positive means CUSUM caught the drift before any explicit threshold violation.

## 4. Phase 4 — close the loop with feedback

Pick the worst-affected anchors, synthesize feedback per anchor, re-run with the feedback as a system message, measure Δ-toward-baseline.

```bash
uv run maimonedes apply-feedback <DRIFT_ID> --top-k 3                # clean mode (default)
uv run maimonedes recovery-report <RECOVERY_ID>                      # before/after table + verdict
```

**Harder claim** (#35 — feedback works *while contamination continues*):

```bash
uv run maimonedes apply-feedback <DRIFT_ID> --under-contamination
uv run maimonedes apply-feedback <DRIFT_ID> --under-contamination --contamination-stage concise
```

**What lands**: `recovery_runs` row (with `contamination_mode` column), `feedbacks` rows per anchor, `compliance_scores` tagged `recovery_run_id`.

**What to read**: dashboard `04_recovery.py` — the contamination-mode caption tells you which claim the verdict is testing.

## 5. Phase 5 — Stage-2 classifier, GP, probe synthesis

Once you have ≥50 (text, score) pairs in `compliance_scores`:

```bash
uv run maimonedes train-stage2 --head-kind ridge                     # or --head-kind mlp, returns a STAGE2_ID
# Now run the scoring for all anchors but using stage2 model (instead of LLM)
uv run maimonedes score-stage2 A1 --model <STAGE2_ID> --persist
uv run maimonedes score-stage2 A2 --model <STAGE2_ID> --persist
uv run maimonedes score-stage2 A3 --model <STAGE2_ID> --persist
uv run maimonedes score-stage2 A4 --model <STAGE2_ID> --persist
uv run maimonedes score-stage2 A5 --model <STAGE2_ID> --persist
uv run maimonedes score-stage2 A6 --model <STAGE2_ID> --persist
uv run maimonedes score-stage2 A7 --model <STAGE2_ID> --persist
uv run maimonedes score-stage2 A8 --model <STAGE2_ID> --persist

uv run maimonedes score-stage2 A3 --text "<some output>"             # cheap online scoring (A3 = anchor id)
uv run maimonedes audit-stage2 <STAGE2_ID>                           # gated audit: re-route recent Stage-2 scores through the LLM judge
uv run maimonedes audit-stage2 <STAGE2_ID> --force                   # bypass the n-scores/hours trigger gate

uv run maimonedes fit-gp                                             # GP over (embedding, aggregate)
uv run maimonedes fit-gp --kernel non_stationary --diagnose          # #49 — Gibbs kernel + per-PCA-1-quartile diagnostic
uv run maimonedes propose-targets <GP_FIT_ID>                        # next probe coordinates

uv run maimonedes synthesize-probes <GP_FIT_ID> --n 5                          # K-NN exemplar synthesizer (default)
uv run maimonedes synthesize-probes <GP_FIT_ID> --n 5 --strategy gradient --stage2-model <STAGE2_ID> \    # #52 — gradient-driven seeds
    
```

**What to read**: dashboard `05_gp.py` for the GP scatter + ranked targets. Synthesized probes show up as `[S] synth-N` next to library `[L] A1..A8` in the fragility selector.

**Perturbation cloud on a synthesized probe** (#53):

```bash
uv run maimonedes perturb --synthesized <SYNTH_ID>
uv run maimonedes perturb --all-synthesized
```

## 6. Phase 5 — Riemannian metric + structural signals

The metric tensor learned from Phase-2 Jacobians lets you compute geodesic distances in compliance space, which is what curvature drift compares.

```bash
uv run maimonedes fit-metric                                                   # all data → metric_fits row
uv run maimonedes metric-distance "1.0,1.0,1.0,1.0,1.0" "0.4,0.6,0.3,1.0,0.8"  # Euclidean vs Riemannian comparison
```

**Curvature drift** (#55, requires two metric fits at different time windows):

```bash
# baseline window
uv run maimonedes fit-metric --end-session <S>                                 # → METRIC_BASELINE_ID
# current window
uv run maimonedes fit-metric --start-session <S+1>                             # → METRIC_CURRENT_ID

uv run maimonedes drift-report <DRIFT_ID> \
    --metric-baseline <METRIC_BASELINE_ID> \
    --metric-current <METRIC_CURRENT_ID> \
    --decoupling-baseline-window 50 --decoupling-current-window 20
```

The drift-report grows two columns: `decoup` and `curv` (each `pass`/`fire` per anchor). Fired signals also land in `structural_signals` (idempotent — re-running with the same args doesn't duplicate rows).

**What to read**: there's no dashboard surface for #50/#51/#55 yet — query directly:

```bash
sqlite3 /home/vagrant/maimonedes.db \
  "SELECT id, anchor_id, signal_type, metric_value, threshold, fired_at FROM structural_signals ORDER BY id DESC LIMIT 10;"
```

## 7. Operational utilities

```bash
uv run maimonedes label-distribution                # #46 — per-axis rubric histogram + ⚠ near-uniform flag
uv run maimonedes backfill-llm-call-ids --dry-run   # #44 — repair legacy compliance_scores.llm_call_id NULLs
uv run maimonedes drift-report <DRIFT_ID> --metric <METRIC_ID>     # adds eucl_d/riem_d displacement columns
```

## What goes in which table (quick map)

| Table | Source command | Purpose |
|---|---|---|
| `llm_calls` | every supervised/judge/synthesizer call | full audit trail |
| `compliance_scores` | `run-once`, `perturb`, `induce-drift`, `apply-feedback`, `synthesize-probes` | every scored output |
| `perturbation_probes` | `perturb` | the perturbed scenarios |
| `drift_runs` / `drift_sessions` | `induce-drift` | drift schedule + per-session suffix |
| `recovery_runs` / `feedbacks` | `apply-feedback` | closed-loop recovery |
| `embed_calls` | embed any text | replay cache for embeddings |
| `stage2_models` | `train-stage2` | classifier registry |
| `gp_fits` | `fit-gp` | GP artefact registry |
| `synthesized_probes` | `synthesize-probes` | LLM-generated probes |
| `metric_fits` | `fit-metric` | Riemannian metric registry |
| `structural_signals` | `drift-report --metric-baseline/--metric-current` | fired decoupling/curvature alerts |
| `audit_runs` | `audit-stage2` | Stage-2 → Stage-1 re-routing log |

The natural flow is **calibrate → run-once → perturb → induce-drift → drift-report → apply-feedback → recovery-report**, with Phase 5 commands layered on top once you have enough data. Every command is idempotent in the sense that it appends — re-running won't corrupt prior data.
