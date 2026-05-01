"""Command-line entry point for maimonedes.

Subcommands:
- `ping`        — Phase 0 smoke test (settings → client → backend →
                  recording → storage)
- `run-once`    — Phase 1 deliverable: load anchor, send to supervised
                  system, score with judge, persist `ComplianceScore`
- `calibrate`   — Phase 1 calibration harness over the reference corpus
- `perturb`     — Phase 2: generate a perturbation cloud for an anchor
                  (or all eight) and persist the cloud's scores
"""
from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from pathlib import Path

import typer

from maimonedes.core.drift import DriftSchedule
from maimonedes.core.perturbation import PerturbationGenerator
from maimonedes.core.perturbation_generators import (
    AuthorityGenerator,
    BoundaryGenerator,
    DemographicGenerator,
    EthnicityGenerator,
    ParaphraseGenerator,
    ProfessionGenerator,
)
from maimonedes.core.policy import load_policy
from maimonedes.core.probe import load_anchors
from maimonedes.experiments.calibrate_judge import (
    load_references,
    run_calibration,
)
from maimonedes.experiments.apply_feedback import (
    RecoveryProgress,
    apply_feedback,
)
from maimonedes.experiments.induce_drift import (
    DriftSessionProgress,
    run_drift,
)
from maimonedes.experiments.perturbation_session import (
    PARAPHRASE_BACKEND_NAME,
    PerturbationOutcome,
    PerturbationProgress,
    run_perturbations,
)
from maimonedes.experiments.run_session import run_once as run_once_session
from maimonedes.llm.client import LLMError, Message
from maimonedes.llm.ollama_backend import OllamaBackend
from maimonedes.llm.recording_client import RecordingClient
from maimonedes.monitor.drift_report import (
    DriftReport,
    NoBaselineDataError,
    build_report,
)
from maimonedes.monitor.fragility import (
    aggregated_fragility,
    all_jacobians,
)
from maimonedes.monitor.recovery_report import (
    NoRecoveryDataError,
    OrphanRecoveryRunError,
    RecoveryReport,
    build_report as build_recovery_report,
)
from maimonedes.monitor.localizer import localize
from maimonedes.feedback.contrastive import fragility_pair, temporal_pair
from maimonedes.settings import Settings, get_settings
from maimonedes.storage.drift import get_drift_run

PING_PROMPT = "Reply with the single word PONG."

# Default config paths — relative to wherever the CLI is invoked.
# `--policy` / `--rubric` / `--probes` flags override.
DEFAULT_POLICY_PATH = Path("config/policies/scope_of_practice.yaml")
DEFAULT_RUBRIC_PATH = Path("config/rubrics/scope_of_practice.yaml")
DEFAULT_PROBES_PATH = Path("config/probes/anchors_v1.yaml")
DEFAULT_REFERENCES_PATH = Path("config/calibration/references_v1.yaml")
DEFAULT_REPORTS_DIR = Path("reports")
DEFAULT_DEMOGRAPHIC_PATH = Path("config/perturbations/demographic_v1.yaml")
DEFAULT_AUTHORITY_PATH = Path("config/perturbations/authority_v1.yaml")
DEFAULT_BOUNDARY_PATH = Path("config/perturbations/boundary_v1.yaml")
DEFAULT_ETHNICITY_PATH = Path("config/perturbations/ethnicity_v1.yaml")
DEFAULT_PROFESSION_PATH = Path("config/perturbations/profession_v1.yaml")
DEFAULT_DRIFT_SCHEDULE_PATH = Path("config/drift/scope_of_practice_v1.yaml")
ALL_KINDS = (
    "paraphrase",
    "demographic",
    "authority",
    "boundary",
    "ethnicity",
    "profession",
)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Black-box behavioral supervision framework for clinical-decision LLMs.",
)


# Forces typer to always treat `ping` as a subcommand instead of
# collapsing the single-command app at the top level.
@app.callback()
def _root() -> None:
    """maimonedes — black-box behavioral supervision framework."""


# Indirection so tests can swap in a FakeLLMClient without HTTP.
def _default_backend_factory(settings: Settings) -> object:
    return OllamaBackend(
        base_url=settings.ollama_base_url,
        api_key=settings.ollama_api_key,
        request_timeout_s=settings.ollama_request_timeout_s,
        max_retries=settings.ollama_max_retries,
    )


_backend_factory: Callable[[Settings], object] = _default_backend_factory


def set_backend_factory(factory: Callable[[Settings], object]) -> None:
    """Test hook: override the backend constructor used by `ping`."""
    global _backend_factory
    _backend_factory = factory


def reset_backend_factory() -> None:
    global _backend_factory
    _backend_factory = _default_backend_factory


def _ping_one(model: str, settings: Settings) -> None:
    backend = _backend_factory(settings)
    rc = RecordingClient(backend, backend_name="ollama", replay=False)  # type: ignore[arg-type]
    response = rc.chat_completion(
        [Message(role="user", content=PING_PROMPT)],
        model=model,
        temperature=0.0,
    )
    typer.echo(f"model={model}")
    typer.echo(f"response={response.content!r}")
    typer.echo(f"latency_ms={response.latency_ms:.1f}")
    typer.echo(
        f"prompt_tokens={response.prompt_tokens} "
        f"completion_tokens={response.completion_tokens}"
    )


def _models_to_ping(model: str | None, all_models: bool, settings: Settings) -> Iterable[str]:
    if all_models:
        return [settings.ollama_supervised_model, settings.ollama_judge_model]
    if model is None:
        raise typer.BadParameter("Provide --model <tag> or --all.")
    return [model]


@app.command()
def ping(
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Ollama model tag to ping (e.g. llama3.1:8b-instruct). Mutually "
        "exclusive with --all.",
    ),
    all_models: bool = typer.Option(
        False,
        "--all",
        help="Ping both the supervised and judge models from settings.",
    ),
) -> None:
    """Smoke-test the LLM backend and persist the round-trip to SQLite."""
    settings = get_settings()
    try:
        models = _models_to_ping(model, all_models, settings)
        for tag in models:
            _ping_one(tag, settings)
    except LLMError as exc:
        typer.echo(f"LLM backend error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except Exception as exc:  # storage / config / unexpected
        typer.echo(f"ping failed: {exc}", err=True)
        raise typer.Exit(code=3) from exc


# Indirection so tests can swap in a FakeEmbedClient without HTTP.
def _default_embed_factory(settings: Settings) -> object:
    from maimonedes.llm.clinicalbert_backend import ClinicalBertBackend

    return ClinicalBertBackend(
        base_url=settings.clinicalbert_base_url,
        model=settings.clinicalbert_model,
        request_timeout_s=settings.clinicalbert_request_timeout_s,
        max_retries=settings.clinicalbert_max_retries,
    )


_embed_factory: Callable[[Settings], object] = _default_embed_factory


def set_embed_factory(factory: Callable[[Settings], object]) -> None:
    """Test hook: override the embed-backend constructor."""
    global _embed_factory
    _embed_factory = factory


def reset_embed_factory() -> None:
    global _embed_factory
    _embed_factory = _default_embed_factory


@app.command("embed-ping")
def embed_ping(
    text: str = typer.Option(
        "Patient has a history of hypertension.",
        "--text",
        help="Sample text to embed.",
    ),
) -> None:
    """Smoke-test the Bio_ClinicalBERT embedding service."""
    from maimonedes.llm.recording_embed_client import RecordingEmbedClient

    settings = get_settings()
    backend = _embed_factory(settings)
    rc = RecordingEmbedClient(backend, backend_name="clinicalbert")  # type: ignore[arg-type]
    try:
        response = rc.embed(text, model=settings.clinicalbert_model)
    except LLMError as exc:
        typer.echo(f"embed backend error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except Exception as exc:
        typer.echo(f"embed-ping failed: {exc}", err=True)
        raise typer.Exit(code=3) from exc

    preview = ", ".join(f"{v:+.4f}" for v in response.embedding[:5])
    typer.echo(f"model={response.model}")
    typer.echo(f"dim={len(response.embedding)}")
    typer.echo(f"first_5=[{preview}]")
    typer.echo(f"latency_ms={response.latency_ms:.1f}")


DEFAULT_STAGE2_DIR = Path("models")


@app.command("train-stage2")
def train_stage2_cmd(
    policy_path: Path = typer.Option(DEFAULT_POLICY_PATH, "--policy"),
    rubric_path: Path = typer.Option(DEFAULT_RUBRIC_PATH, "--rubric"),
    output_dir: Path = typer.Option(
        DEFAULT_STAGE2_DIR,
        "--output-dir",
        help="Directory to write the Stage-2 pickle artefact under.",
    ),
    output_path: Path | None = typer.Option(
        None,
        "--output",
        help="Explicit output path. Overrides --output-dir.",
    ),
    min_samples: int = typer.Option(
        50, "--min-samples", help="Minimum training pairs required."
    ),
    eval_fraction: float = typer.Option(
        0.2, "--eval-fraction", help="Hold-out fraction (stratified by anchor)."
    ),
    ridge_alpha: float = typer.Option(
        1.0, "--ridge-alpha", help="L2 regularization for Ridge heads."
    ),
    seed: int = typer.Option(0, "--seed", help="Split + Ridge RNG seed."),
    head_kind: str = typer.Option(
        "ridge",
        "--head-kind",
        help="Per-axis head architecture: 'ridge' (default) or 'mlp'.",
    ),
    replay: bool = typer.Option(
        False,
        "--replay",
        help="Use the EmbedClient replay cache (re-runs are free if "
        "embeddings are already recorded).",
    ),
) -> None:
    """Train a Stage-2 classifier on existing supervised + judge data."""
    from datetime import datetime, timezone

    from maimonedes.llm.recording_embed_client import RecordingEmbedClient
    from maimonedes.models.stage2 import train_stage2
    from maimonedes.storage.stage2_models import record_stage2_model

    if head_kind not in {"ridge", "mlp"}:
        typer.echo(
            f"unknown --head-kind {head_kind!r}; expected 'ridge' or 'mlp'",
            err=True,
        )
        raise typer.Exit(code=5)

    try:
        policy = load_policy(policy_path, rubric_path)
    except FileNotFoundError as exc:
        typer.echo(f"config file not found: {exc}", err=True)
        raise typer.Exit(code=5) from exc
    except (ValueError, KeyError) as exc:
        typer.echo(f"invalid configuration: {exc}", err=True)
        raise typer.Exit(code=5) from exc

    settings = get_settings()
    backend = _embed_factory(settings)
    rc = RecordingEmbedClient(
        backend, backend_name="clinicalbert", replay=replay  # type: ignore[arg-type]
    )

    try:
        model = train_stage2(
            policy=policy,
            embed_client=rc,
            embedding_model=settings.clinicalbert_model,
            min_samples=min_samples,
            eval_fraction=eval_fraction,
            ridge_alpha=ridge_alpha,
            seed=seed,
            head_kind=head_kind,  # type: ignore[arg-type]
        )
    except ValueError as exc:
        typer.echo(f"training failed: {exc}", err=True)
        raise typer.Exit(code=5) from exc
    except LLMError as exc:
        typer.echo(f"embed backend error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    if output_path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = output_dir / f"stage2_{policy.id}_{timestamp}.pkl"
    saved = model.save(output_path)

    mae = {m.sub_id: m.mae for m in model.agreement_metrics}
    rho = {m.sub_id: m.spearman_rho for m in model.agreement_metrics}
    model_id = record_stage2_model(
        path=str(saved),
        policy_id=model.policy_id,
        n_samples=model.n_samples,
        embedding_model=model.embedding_model,
        mae_per_axis=mae,
        spearman_per_axis=rho,
        agreement_status=model.agreement_status,
        head_kind=model.head_kind,
    )

    typer.echo(
        f"stage2_model_id={model_id} path={saved} "
        f"head_kind={model.head_kind} "
        f"n_train={model.n_train} n_eval={model.n_eval} "
        f"agreement_status={model.agreement_status}"
    )
    for m in model.agreement_metrics:
        typer.echo(
            f"  axis={m.sub_id:<46} mae={m.mae:.3f} spearman_rho={m.spearman_rho:+.3f}"
        )

    if model.agreement_status == "red":
        # A red Stage-2 model is not deployable for online scoring.
        raise typer.Exit(code=6)


@app.command("score-stage2")
def score_stage2_cmd(
    anchor_id: str = typer.Argument(..., help="Anchor id, e.g. A3."),
    model_id: int | None = typer.Option(
        None,
        "--model",
        help="stage2_models.id to use. Defaults to the latest model for the policy.",
    ),
    text: str | None = typer.Option(
        None,
        "--text",
        help="Override the supervised output text (default: live supervised call).",
    ),
    policy_path: Path = typer.Option(DEFAULT_POLICY_PATH, "--policy"),
    rubric_path: Path = typer.Option(DEFAULT_RUBRIC_PATH, "--rubric"),
    probes_path: Path = typer.Option(DEFAULT_PROBES_PATH, "--probes"),
    persist: bool = typer.Option(
        False,
        "--persist",
        help="Persist the score row (probe_role=anchor, judge_model=stage2:<path>).",
    ),
) -> None:
    """Run Stage-2 inference on a supervised output for one anchor."""
    from maimonedes.core.probe import get_anchor_by_id
    from maimonedes.llm.recording_embed_client import RecordingEmbedClient
    from maimonedes.models.stage2 import Stage2Model
    from maimonedes.scorer.stage2 import Stage2Scorer
    from maimonedes.storage.compliance import record_score
    from maimonedes.storage.stage2_models import (
        get_stage2_model,
        latest_stage2_model_row,
    )

    try:
        policy = load_policy(policy_path, rubric_path)
        anchors = load_anchors(probes_path)
    except FileNotFoundError as exc:
        typer.echo(f"config file not found: {exc}", err=True)
        raise typer.Exit(code=5) from exc
    except (ValueError, KeyError) as exc:
        typer.echo(f"invalid configuration: {exc}", err=True)
        raise typer.Exit(code=5) from exc

    if model_id is not None:
        model_row = get_stage2_model(model_id)
        if model_row is None:
            typer.echo(f"unknown stage2 model: {model_id}", err=True)
            raise typer.Exit(code=4)
    else:
        model_row = latest_stage2_model_row(policy.id)
        if model_row is None:
            typer.echo(
                f"no Stage-2 model found for policy {policy.id!r}; "
                f"run `maimonedes train-stage2` first",
                err=True,
            )
            raise typer.Exit(code=4)

    try:
        stage2 = Stage2Model.load(model_row.path)
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(f"failed to load stage2 model: {exc}", err=True)
        raise typer.Exit(code=4) from exc

    settings = get_settings()
    embed_backend = _embed_factory(settings)
    rc = RecordingEmbedClient(embed_backend, backend_name="clinicalbert")  # type: ignore[arg-type]
    scorer = Stage2Scorer(stage2, rc, policy, model_path=model_row.path)
    anchor = get_anchor_by_id(list(anchors), anchor_id)

    if text is None:
        backend = _backend_factory(settings)
        supervised_rc = RecordingClient(
            backend,  # type: ignore[arg-type]
            backend_name="ollama-supervised",
            replay=False,
        )
        try:
            response = supervised_rc.chat_completion(
                [Message(role="user", content=anchor.scenario)],
                model=settings.ollama_supervised_model,
                temperature=0.0,
            )
        except LLMError as exc:
            typer.echo(f"supervised LLM error: {exc}", err=True)
            raise typer.Exit(code=2) from exc
        text = response.content

    score = scorer.score(
        anchor, text, supervised_model=settings.ollama_supervised_model
    )
    if persist:
        record_score(score)

    typer.echo(f"anchor={score.anchor_id}")
    typer.echo(f"judge_model={score.judge_model}")
    typer.echo(f"aggregate={score.aggregate:.3f}")
    for sub_id in sorted(score.per_sub_condition):
        typer.echo(f"  {sub_id} = {score.per_sub_condition[sub_id]:.3f}")


@app.command("audit-stage2")
def audit_stage2_cmd(
    model_id: int = typer.Argument(..., help="stage2_models.id to audit."),
    limit: int = typer.Option(
        20, "--limit", help="Max rows to re-route through Stage-1."
    ),
    policy_path: Path = typer.Option(DEFAULT_POLICY_PATH, "--policy"),
    rubric_path: Path = typer.Option(DEFAULT_RUBRIC_PATH, "--rubric"),
    probes_path: Path = typer.Option(DEFAULT_PROBES_PATH, "--probes"),
    n_scores: int = typer.Option(
        10, "--n-scores", help="N-scores trigger threshold."
    ),
    hours: float = typer.Option(
        24.0, "--hours", help="K-hours trigger threshold."
    ),
    force: bool = typer.Option(
        False,
        "--force",
        help="Skip the should_audit gate; always run.",
    ),
) -> None:
    """Force or trigger-gated Stage-1 audit of a Stage-2 model."""
    from datetime import datetime, timezone

    from maimonedes.models.stage2 import Stage2Model
    from maimonedes.monitor.stage2_audit import Stage2AuditDetector
    from maimonedes.storage.stage2_models import get_stage2_model

    try:
        policy = load_policy(policy_path, rubric_path)
        anchors = load_anchors(probes_path)
    except FileNotFoundError as exc:
        typer.echo(f"config file not found: {exc}", err=True)
        raise typer.Exit(code=5) from exc
    except (ValueError, KeyError) as exc:
        typer.echo(f"invalid configuration: {exc}", err=True)
        raise typer.Exit(code=5) from exc

    model_row = get_stage2_model(model_id)
    if model_row is None:
        typer.echo(f"unknown stage2 model: {model_id}", err=True)
        raise typer.Exit(code=4)

    try:
        stage2 = Stage2Model.load(model_row.path)
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(f"failed to load stage2 model: {exc}", err=True)
        raise typer.Exit(code=4) from exc

    settings = get_settings()
    backend = _backend_factory(settings)

    detector = Stage2AuditDetector(
        stage2_model_id=model_id, n_scores=n_scores, hours=hours
    )
    last_ended = detector.last_audit_ended_at()
    model_tag = f"stage2:{Path(model_row.path).name}"
    rows = detector.scores_since_last(model_tag, last_ended)[:limit]

    if not force:
        should, reason = detector.should_audit(
            now=datetime.now(timezone.utc),
            scores_since_last=len(rows),
            last_ended_at=last_ended,
        )
        if not should:
            typer.echo(
                f"audit skipped: trigger={reason} "
                f"(scores_since_last={len(rows)}, last_ended_at={last_ended})"
            )
            return
        trigger_reason = reason
    else:
        trigger_reason = "force"

    if not rows:
        typer.echo(
            "no scoreable Stage-2 rows since last audit; nothing to audit",
            err=True,
        )
        raise typer.Exit(code=2)

    report = detector.run_audit(
        stage2_model=stage2,
        policy=policy,
        anchors=anchors,
        judge_client=backend,  # type: ignore[arg-type]
        judge_model=settings.ollama_judge_model,
        supervised_model=settings.ollama_supervised_model,
        score_rows=rows,
        trigger_reason=trigger_reason,
    )
    typer.echo(
        f"audit_run_id={report.audit_run_id} stage2_model_id={report.stage2_model_id} "
        f"trigger={report.trigger_reason} n={report.n_samples} "
        f"agreement_status={report.agreement_status}"
    )
    for m in report.metrics:
        typer.echo(
            f"  axis={m.sub_id:<46} mae={m.mae:.3f} spearman_rho={m.spearman_rho:+.3f}"
        )


DEFAULT_GP_DIR = Path("models")


@app.command("fit-gp")
def fit_gp_cmd(
    policy_path: Path = typer.Option(DEFAULT_POLICY_PATH, "--policy"),
    rubric_path: Path = typer.Option(DEFAULT_RUBRIC_PATH, "--rubric"),
    probes_path: Path = typer.Option(DEFAULT_PROBES_PATH, "--probes"),
    output_dir: Path = typer.Option(
        DEFAULT_GP_DIR, "--output-dir", help="Directory for the GP pickle artefact."
    ),
    output_path: Path | None = typer.Option(
        None, "--output", help="Explicit output path. Overrides --output-dir."
    ),
    min_samples: int = typer.Option(
        20, "--min-samples", help="Minimum (text, score) pairs required."
    ),
    replay: bool = typer.Option(
        False,
        "--replay",
        help="Use the EmbedClient replay cache (free re-runs once embeddings are recorded).",
    ),
) -> None:
    """Fit a GP over (embedding, aggregate) pairs from compliance_scores."""
    from datetime import datetime, timezone

    from maimonedes.llm.recording_embed_client import RecordingEmbedClient
    from maimonedes.monitor.gp_layer import fit_compliance_gp
    from maimonedes.storage.gp_fits import record_gp_fit

    try:
        policy = load_policy(policy_path, rubric_path)
        anchors = load_anchors(probes_path)
    except FileNotFoundError as exc:
        typer.echo(f"config file not found: {exc}", err=True)
        raise typer.Exit(code=5) from exc
    except (ValueError, KeyError) as exc:
        typer.echo(f"invalid configuration: {exc}", err=True)
        raise typer.Exit(code=5) from exc

    settings = get_settings()
    backend = _embed_factory(settings)
    rc = RecordingEmbedClient(
        backend, backend_name="clinicalbert", replay=replay  # type: ignore[arg-type]
    )

    try:
        gp = fit_compliance_gp(
            embed_client=rc,
            policy=policy,
            embedding_model=settings.clinicalbert_model,
            min_samples=min_samples,
            library_anchor_texts={a.id: a.scenario for a in anchors},
        )
    except ValueError as exc:
        typer.echo(f"GP fit failed: {exc}", err=True)
        raise typer.Exit(code=5) from exc
    except LLMError as exc:
        typer.echo(f"embed backend error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    if output_path is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = output_dir / f"gp_{policy.id}_{timestamp}.pkl"
    saved = gp.save(output_path)
    fit_id = record_gp_fit(
        path=str(saved),
        policy_id=gp.policy_id,
        n_samples=gp.n_samples,
        kernel_name=gp.kernel_repr,
        log_marginal_likelihood=gp.log_marginal_likelihood,
        embedding_model=gp.embedding_model,
    )

    typer.echo(
        f"gp_fit_id={fit_id} path={saved} n_samples={gp.n_samples} "
        f"log_marginal_likelihood={gp.log_marginal_likelihood:.3f}"
    )
    typer.echo(f"  kernel={gp.kernel_repr}")


@app.command("propose-targets")
def propose_targets_cmd(
    fit_id: int = typer.Argument(..., help="gp_fits.id to propose from."),
    n: int = typer.Option(10, "--n", help="Number of targets to return."),
    candidate_pool_size: int = typer.Option(
        1000, "--pool", help="Candidate pool size (random perturbations)."
    ),
    seed: int = typer.Option(0, "--seed", help="RNG seed for reproducibility."),
) -> None:
    """Propose top-N embedding-space targets ranked by uncertainty × boundary risk."""
    from maimonedes.monitor.gp_layer import ComplianceGP, propose_targets
    from maimonedes.storage.gp_fits import get_gp_fit

    fit_row = get_gp_fit(fit_id)
    if fit_row is None:
        typer.echo(f"unknown gp_fit_id: {fit_id}", err=True)
        raise typer.Exit(code=4)
    try:
        gp = ComplianceGP.load(fit_row.path)
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(f"failed to load gp fit: {exc}", err=True)
        raise typer.Exit(code=4) from exc

    targets = propose_targets(
        gp,
        n_targets=n,
        candidate_pool_size=candidate_pool_size,
        seed=seed,
    )
    typer.echo(f"gp_fit_id={fit_id} n_targets={len(targets)}")
    for i, t in enumerate(targets, start=1):
        typer.echo(
            f"  T{i:02d}  expected_score={t.expected_score:+.3f} "
            f"uncertainty={t.uncertainty:.3f} score={t.score:.3f}"
        )


def _stream_synthesis_progress(progress) -> None:  # type: ignore[no-untyped-def]
    o = progress.outcome
    s = o.synthesis
    if o.error:
        typer.echo(
            f"  T{progress.target_index:02d}  FAIL  error={o.error}"
        )
        return
    score_str = (
        f"score={o.score.aggregate:.3f}" if o.score is not None else "score=skipped"
    )
    typer.echo(
        f"  T{progress.target_index:02d}  status={s.status} "
        f"cos={s.tau_distance:+.3f} retries={s.retries_used} {score_str}"
    )


@app.command("synthesize-probes")
def synthesize_probes_cmd(
    gp_fit_id: int = typer.Argument(..., help="gp_fits.id to synthesize against."),
    n: int = typer.Option(
        10, "--n", help="Number of GP-proposed targets to attempt."
    ),
    tau: float = typer.Option(
        0.7, "--tau", help="Cosine similarity threshold (achieved vs target)."
    ),
    max_retries: int = typer.Option(
        3, "--max-retries", help="Per-target retry budget on tau failures."
    ),
    generator_model: str | None = typer.Option(
        None,
        "--generator-model",
        help="LLM model used for scenario generation (default: judge model).",
    ),
    validator_model: str | None = typer.Option(
        None,
        "--validator-model",
        help="LLM model used for the quality gate (default: judge model).",
    ),
    scorer: str = typer.Option(
        "judge",
        "--scorer",
        help="Scorer for approved synthesized probes: judge | classifier.",
    ),
    stage2_model_id: int | None = typer.Option(
        None,
        "--stage2-model",
        help="stage2_models.id to use when --scorer=classifier.",
    ),
    policy_path: Path = typer.Option(DEFAULT_POLICY_PATH, "--policy"),
    rubric_path: Path = typer.Option(DEFAULT_RUBRIC_PATH, "--rubric"),
    probes_path: Path = typer.Option(DEFAULT_PROBES_PATH, "--probes"),
    replay: bool = typer.Option(
        False,
        "--replay",
        help="Use the EmbedClient + RecordingClient replay caches.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Propose targets and print the K nearest library anchors per target without LLM calls.",
    ),
) -> None:
    """Generate synthesized probes from GP targets, optionally scoring each."""
    if scorer not in ("judge", "classifier"):
        raise typer.BadParameter(
            f"--scorer must be 'judge' or 'classifier', got {scorer!r}"
        )
    if scorer == "classifier" and stage2_model_id is None:
        typer.echo(
            "--scorer=classifier requires --stage2-model <id>", err=True
        )
        raise typer.Exit(code=5)
    if n < 1:
        raise typer.BadParameter("--n must be >= 1")

    from maimonedes.experiments.synthesize_probes import synthesize_probes
    from maimonedes.feedback.probe_synthesis import _cosine_similarity
    from maimonedes.models.stage2 import Stage2Model
    from maimonedes.monitor.gp_layer import ComplianceGP, propose_targets
    from maimonedes.storage.gp_fits import get_gp_fit
    from maimonedes.storage.stage2_models import get_stage2_model

    try:
        policy = load_policy(policy_path, rubric_path)
        anchors = load_anchors(probes_path)
    except FileNotFoundError as exc:
        typer.echo(f"config file not found: {exc}", err=True)
        raise typer.Exit(code=5) from exc
    except (ValueError, KeyError) as exc:
        typer.echo(f"invalid configuration: {exc}", err=True)
        raise typer.Exit(code=5) from exc

    fit_row = get_gp_fit(gp_fit_id)
    if fit_row is None:
        typer.echo(f"unknown gp_fit_id: {gp_fit_id}", err=True)
        raise typer.Exit(code=4)
    try:
        gp = ComplianceGP.load(fit_row.path)
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(f"failed to load gp fit: {exc}", err=True)
        raise typer.Exit(code=4) from exc

    settings = get_settings()
    generator_model = generator_model or settings.ollama_judge_model
    validator_model = validator_model or settings.ollama_judge_model

    if dry_run:
        targets = propose_targets(gp, n_targets=n, candidate_pool_size=500, seed=0)
        typer.echo(
            f"gp_fit_id={gp_fit_id} n_targets={len(targets)} (dry-run, no LLM calls)"
        )
        for i, t in enumerate(targets, start=1):
            nearest: list[tuple[str, float]] = []
            for anchor_id, vec in gp.library_anchor_embeddings.items():
                cos = _cosine_similarity(t.embedding, vec)
                nearest.append((anchor_id, cos))
            nearest.sort(key=lambda kv: -kv[1])
            top = ", ".join(f"{aid}({cos:+.2f})" for aid, cos in nearest[:5])
            typer.echo(
                f"  T{i:02d}  uncertainty={t.uncertainty:.3f} "
                f"expected_score={t.expected_score:+.3f} "
                f"top_5_nearest=[{top}]"
            )
        return

    stage2_model = None
    if scorer == "classifier":
        stage2_row = get_stage2_model(stage2_model_id)  # type: ignore[arg-type]
        if stage2_row is None:
            typer.echo(
                f"unknown stage2_model id: {stage2_model_id}", err=True
            )
            raise typer.Exit(code=4)
        stage2_model = Stage2Model.load(stage2_row.path)

    embed_backend = _embed_factory(settings)
    chat_backend = _backend_factory(settings)

    try:
        summary = synthesize_probes(
            gp_fit=gp,
            n_targets=n,
            embed_client=embed_backend,  # type: ignore[arg-type]
            generator_client=chat_backend,  # type: ignore[arg-type]
            validator_client=chat_backend,  # type: ignore[arg-type]
            supervised_client=chat_backend,  # type: ignore[arg-type]
            judge_client=chat_backend,  # type: ignore[arg-type]
            embedding_model=settings.clinicalbert_model,
            generator_model=generator_model,
            validator_model=validator_model,
            supervised_model=settings.ollama_supervised_model,
            judge_model=settings.ollama_judge_model,
            policy=policy,
            anchors=anchors,
            scorer=scorer,  # type: ignore[arg-type]
            stage2_model=stage2_model,
            gp_fit_id=gp_fit_id,
            tau=tau,
            max_retries=max_retries,
            replay=replay,
            on_progress=_stream_synthesis_progress,
        )
    except ValueError as exc:
        typer.echo(f"synthesis failed: {exc}", err=True)
        raise typer.Exit(code=5) from exc
    except LLMError as exc:
        typer.echo(f"LLM backend error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    mean_str = (
        f"{summary.mean_aggregate:.3f}"
        if summary.mean_aggregate is not None
        else "n/a"
    )
    typer.echo(
        f"n_targets={summary.n_targets} approved={summary.n_approved} "
        f"rejected_tau={summary.n_rejected_tau} "
        f"rejected_validator={summary.n_rejected_validator} "
        f"scored={summary.n_scored} mean_aggregate={mean_str}"
    )


@app.command("run-once")
def run_once_cmd(
    anchor_id: str = typer.Argument(..., help="Anchor id, e.g. A3."),
    policy_path: Path = typer.Option(
        DEFAULT_POLICY_PATH, "--policy", help="Path to the policy YAML."
    ),
    rubric_path: Path = typer.Option(
        DEFAULT_RUBRIC_PATH, "--rubric", help="Path to the rubric YAML."
    ),
    probes_path: Path = typer.Option(
        DEFAULT_PROBES_PATH, "--probes", help="Path to the anchor library YAML."
    ),
    replay: bool = typer.Option(
        False,
        "--replay",
        help="Use the RecordingClient replay cache for both supervised "
        "and judge calls. Misses still go live and seed the cache.",
    ),
) -> None:
    """Score one anchor end-to-end and persist the result."""
    settings = get_settings()
    try:
        policy = load_policy(policy_path, rubric_path)
        anchors = load_anchors(probes_path)
        backend = _backend_factory(settings)
        score = run_once_session(
            anchor_id,
            policy=policy,
            anchors=anchors,
            supervised_client=backend,  # type: ignore[arg-type]
            judge_client=backend,  # type: ignore[arg-type]
            supervised_model=settings.ollama_supervised_model,
            judge_model=settings.ollama_judge_model,
            replay=replay,
        )
    except KeyError as exc:
        typer.echo(f"unknown anchor id: {exc}", err=True)
        raise typer.Exit(code=4) from exc
    except ValueError as exc:
        typer.echo(f"invalid configuration: {exc}", err=True)
        raise typer.Exit(code=5) from exc
    except LLMError as exc:
        typer.echo(f"LLM backend error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    except FileNotFoundError as exc:
        typer.echo(f"config file not found: {exc}", err=True)
        raise typer.Exit(code=5) from exc

    typer.echo(f"anchor={score.anchor_id}")
    typer.echo(f"policy={score.policy_id}")
    typer.echo(f"aggregate={score.aggregate:.3f}")
    for sub_id in sorted(score.per_sub_condition):
        typer.echo(f"  {sub_id} = {score.per_sub_condition[sub_id]:.3f}")


@app.command("calibrate")
def calibrate_cmd(
    references_path: Path = typer.Option(
        DEFAULT_REFERENCES_PATH, "--references", help="Path to the calibration corpus YAML."
    ),
    policy_path: Path = typer.Option(
        DEFAULT_POLICY_PATH, "--policy", help="Path to the policy YAML."
    ),
    rubric_path: Path = typer.Option(
        DEFAULT_RUBRIC_PATH, "--rubric", help="Path to the rubric YAML."
    ),
    probes_path: Path = typer.Option(
        DEFAULT_PROBES_PATH, "--probes", help="Path to the anchor library YAML."
    ),
    output_dir: Path = typer.Option(
        DEFAULT_REPORTS_DIR,
        "--output-dir",
        help="Where to write the calibration CSV report.",
    ),
) -> None:
    """Score the calibration corpus and emit a Spearman / MAE / status report."""
    settings = get_settings()
    try:
        references = load_references(references_path)
        policy = load_policy(policy_path, rubric_path)
        anchors = load_anchors(probes_path)
        backend = _backend_factory(settings)
        report = run_calibration(
            references,
            policy=policy,
            anchors=anchors,
            judge_client=backend,  # type: ignore[arg-type]
            judge_model=settings.ollama_judge_model,
            supervised_model=settings.ollama_supervised_model,
            output_dir=output_dir,
        )
    except FileNotFoundError as exc:
        typer.echo(f"config file not found: {exc}", err=True)
        raise typer.Exit(code=5) from exc
    except (ValueError, KeyError) as exc:
        typer.echo(f"invalid configuration: {exc}", err=True)
        raise typer.Exit(code=5) from exc
    except LLMError as exc:
        typer.echo(f"LLM backend error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(report.summary_line())
    typer.echo(f"report={report.report_path}")
    for b in report.bins:
        if b.n == 0:
            typer.echo(f"  bin [{b.lower:.2f}, {b.upper:.2f}]   empty")
        else:
            typer.echo(
                f"  bin [{b.lower:.2f}, {b.upper:.2f}] n={b.n} "
                f"pred={b.mean_predicted:.3f} hand={b.mean_hand:.3f}"
            )
    if report.status == "red":
        # Roadmap risks section: red-band judge blocks Phase 2.
        raise typer.Exit(code=6)


def _build_generators(
    kinds: list[str],
    *,
    paraphrase_client: object,
    paraphrase_model: str,
    paraphrase_n: int,
    replay: bool,
    demographic_path: Path,
    authority_path: Path,
    boundary_path: Path,
    ethnicity_path: Path,
    profession_path: Path,
) -> list[PerturbationGenerator]:
    out: list[PerturbationGenerator] = []
    for kind in kinds:
        if kind == "paraphrase":
            # Wrap the paraphrase client in a RecordingClient so its
            # LLM calls land in `llm_calls` with backend_name
            # `ollama-paraphrase`, alongside the supervised + judge
            # rows the orchestrator already records. `replay` is
            # honoured so re-runs can serve cached rewrites.
            paraphrase_rc = RecordingClient(
                paraphrase_client,  # type: ignore[arg-type]
                backend_name=PARAPHRASE_BACKEND_NAME,
                replay=replay,
            )
            out.append(
                ParaphraseGenerator(
                    paraphrase_rc,
                    model=paraphrase_model,
                    n=paraphrase_n,
                )
            )
        elif kind == "demographic":
            out.append(DemographicGenerator(demographic_path))
        elif kind == "authority":
            out.append(AuthorityGenerator(authority_path))
        elif kind == "boundary":
            out.append(BoundaryGenerator(boundary_path))
        elif kind == "ethnicity":
            out.append(EthnicityGenerator(ethnicity_path))
        elif kind == "profession":
            out.append(ProfessionGenerator(profession_path))
        else:
            raise typer.BadParameter(
                f"unknown perturbation kind: {kind!r}; "
                f"expected one of {ALL_KINDS}"
            )
    return out


def _format_outcome_line(progress: PerturbationProgress) -> str:
    """`[02/11] authority:senior_cardiologist  agg=0.420  Δ=-0.473`

    With replicates > 1, prefixes a `[rep i/N]` counter:
    `[rep 03/05][02/11] authority:senior_cardiologist  agg=0.420  Δ=-0.473`
    """
    o = progress.outcome
    digits = max(2, len(str(progress.total)))
    counter = f"[{progress.index:0{digits}d}/{progress.total:0{digits}d}]"
    if progress.replicates_total > 1:
        rep_digits = max(2, len(str(progress.replicates_total)))
        rep_counter = (
            f"[rep {progress.replicate_index + 1:0{rep_digits}d}"
            f"/{progress.replicates_total:0{rep_digits}d}]"
        )
        counter = f"{rep_counter}{counter}"
    if o.score is None:
        return f"  {counter} {o.probe.transform_label}  FAIL  ({o.error})"
    delta_str = (
        f"Δ={o.delta_aggregate:+.3f}"
        if o.delta_aggregate is not None
        else "Δ=n/a"
    )
    return (
        f"  {counter} {o.probe.transform_label}  "
        f"agg={o.score.aggregate:.3f}  {delta_str}"
    )


def _stream_outcome(progress: PerturbationProgress) -> None:
    typer.echo(_format_outcome_line(progress))


def _stream_generation(generator_name: str, n_probes: int) -> None:
    typer.echo(f"  generating {generator_name} ... {n_probes} probe(s)")


def _print_perturbation_summary(
    anchor_id: str, outcomes: list[PerturbationOutcome]
) -> None:
    """Final per-anchor summary line — per-probe lines streamed live above."""
    if not outcomes:
        typer.echo(f"  no perturbations generated for anchor={anchor_id}")
        return
    successes = [o for o in outcomes if o.score is not None]
    failures = [o for o in outcomes if o.score is None]
    if successes and any(o.delta_aggregate is not None for o in successes):
        deltas = [
            abs(o.delta_aggregate)
            for o in successes
            if o.delta_aggregate is not None
        ]
        typer.echo(
            f"  summary: n={len(successes)}/{len(outcomes)} "
            f"mean|Δ|={sum(deltas) / len(deltas):.3f} failures={len(failures)}"
        )
    else:
        typer.echo(
            f"  summary: n={len(successes)}/{len(outcomes)} "
            f"failures={len(failures)} (no baseline → Δ unavailable)"
        )


@app.command("perturb")
def perturb_cmd(
    anchor_id: str = typer.Argument(
        None,
        help="Anchor id (e.g. A3). Omit when --all-anchors / --synthesized is passed.",
    ),
    all_anchors: bool = typer.Option(
        False, "--all-anchors", help="Sweep A1..A8 sequentially."
    ),
    synthesized: int = typer.Option(
        None,
        "--synthesized",
        help="Run perturbations against synthesized_probes row #N "
        "(mutually exclusive with anchor-id and --all-anchors).",
    ),
    all_synthesized: bool = typer.Option(
        False,
        "--all-synthesized",
        help="Sweep every approved row in synthesized_probes (skip rejected).",
    ),
    kinds: str = typer.Option(
        ",".join(ALL_KINDS),
        "--kinds",
        help="Comma-separated subset of {paraphrase,demographic,authority,boundary}.",
    ),
    paraphrase_n: int = typer.Option(
        3, "--paraphrase-n", help="Number of paraphrases per anchor."
    ),
    replicates: int = typer.Option(
        1,
        "--replicates",
        "-r",
        help="Number of independent replicate runs per anchor. N>1 enables "
        "noise-floor estimation (mean Δ ± std Δ in the fragility report). "
        "Each replicate re-invokes every generator and re-scores fresh.",
    ),
    replay: bool = typer.Option(
        False,
        "--replay",
        help="Use the RecordingClient replay cache for supervised + judge calls. "
        "Note: incompatible with replicates > 1 in spirit — the cache returns "
        "identical responses, which collapses replicate variance to zero.",
    ),
    policy_path: Path = typer.Option(DEFAULT_POLICY_PATH, "--policy"),
    rubric_path: Path = typer.Option(DEFAULT_RUBRIC_PATH, "--rubric"),
    probes_path: Path = typer.Option(DEFAULT_PROBES_PATH, "--probes"),
    demographic_path: Path = typer.Option(
        DEFAULT_DEMOGRAPHIC_PATH, "--demographic-path"
    ),
    authority_path: Path = typer.Option(
        DEFAULT_AUTHORITY_PATH, "--authority-path"
    ),
    boundary_path: Path = typer.Option(
        DEFAULT_BOUNDARY_PATH, "--boundary-path"
    ),
    ethnicity_path: Path = typer.Option(
        DEFAULT_ETHNICITY_PATH, "--ethnicity-path"
    ),
    profession_path: Path = typer.Option(
        DEFAULT_PROFESSION_PATH, "--profession-path"
    ),
) -> None:
    """Generate, run, and score a perturbation cloud for one parent.

    Parent kinds:
    - curated anchor (positional `anchor-id` or `--all-anchors`)
    - synthesized probe (`--synthesized <id>` or `--all-synthesized`)
    """
    selectors = [
        anchor_id is not None,
        all_anchors,
        synthesized is not None,
        all_synthesized,
    ]
    if sum(1 for s in selectors if s) != 1:
        raise typer.BadParameter(
            "Pass exactly one of: <anchor-id>, --all-anchors, "
            "--synthesized <id>, --all-synthesized."
        )
    if replicates < 1:
        raise typer.BadParameter("--replicates must be >= 1.")
    if replicates > 1 and replay:
        typer.echo(
            "warning: --replicates > 1 with --replay returns identical cached "
            "responses for each replicate, collapsing variance to zero. "
            "Drop --replay to measure real replicate variance.",
            err=True,
        )

    kinds_list = [k.strip() for k in kinds.split(",") if k.strip()]
    for k in kinds_list:
        if k not in ALL_KINDS:
            raise typer.BadParameter(
                f"unknown kind {k!r}; expected one of {ALL_KINDS}"
            )

    settings = get_settings()
    try:
        policy = load_policy(policy_path, rubric_path)
        anchors = load_anchors(probes_path)
        backend = _backend_factory(settings)
        generators = _build_generators(
            kinds_list,
            paraphrase_client=backend,
            paraphrase_model=settings.ollama_supervised_model,
            paraphrase_n=paraphrase_n,
            replay=replay,
            demographic_path=demographic_path,
            authority_path=authority_path,
            boundary_path=boundary_path,
            ethnicity_path=ethnicity_path,
            profession_path=profession_path,
        )
    except FileNotFoundError as exc:
        typer.echo(f"config file not found: {exc}", err=True)
        raise typer.Exit(code=5) from exc
    except (ValueError, KeyError) as exc:
        typer.echo(f"invalid configuration: {exc}", err=True)
        raise typer.Exit(code=5) from exc

    # Build target list: either curated anchor ids OR synthesized
    # probe ids (using namespaced anchor labels for the print path).
    synth_targets: list[int] = []
    if synthesized is not None:
        synth_targets = [synthesized]
    elif all_synthesized:
        from maimonedes.storage.synthesized_probes import list_synthesized_probes

        approved = list_synthesized_probes(
            policy_id=policy.id, quality_status="approved"
        )
        synth_targets = [p.id for p in approved if p.id is not None]
        if not synth_targets:
            typer.echo(
                "no approved synthesized probes for this policy; "
                "run `maimonedes synthesize-probes` first.",
                err=True,
            )
            raise typer.Exit(code=4)

    target_ids: list[str]
    if synth_targets:
        target_ids = [f"synth-{i}" for i in synth_targets]
    elif all_anchors:
        target_ids = [a.id for a in anchors]
    else:
        target_ids = [anchor_id or ""]

    rep_suffix = f"; replicates={replicates}" if replicates > 1 else ""
    typer.echo(
        f"running perturbations on {len(target_ids)} parent(s); "
        f"kinds={','.join(kinds_list)}{rep_suffix}"
    )

    failures = 0
    if synth_targets:
        from maimonedes.storage.synthesized_probes import get_synthesized_probe

        DEMOGRAPHIC_HINT = "i'm "  # we expect "Hi, I'm John, ..." patterns
        for i, sid in enumerate(synth_targets, start=1):
            label = f"synth-{sid}"
            typer.echo(f"[{i}/{len(synth_targets)}] {label}")
            probe = get_synthesized_probe(sid)
            if probe is not None and DEMOGRAPHIC_HINT not in probe.scenario.lower():
                typer.echo(
                    f"warning: synthesized probe {sid} has no demographic "
                    "preamble; rule-based generators may produce zero rewrites.",
                    err=True,
                )
            try:
                outcomes = run_perturbations(
                    None,
                    policy=policy,
                    anchors=anchors,
                    supervised_client=backend,  # type: ignore[arg-type]
                    judge_client=backend,  # type: ignore[arg-type]
                    generators=generators,
                    supervised_model=settings.ollama_supervised_model,
                    judge_model=settings.ollama_judge_model,
                    replay=replay,
                    on_outcome=_stream_outcome,
                    on_generation=_stream_generation,
                    replicates=replicates,
                    synthesized_probe_id=sid,
                )
            except KeyError as exc:
                typer.echo(f"unknown synthesized id: {exc}", err=True)
                raise typer.Exit(code=4) from exc
            except ValueError as exc:
                typer.echo(f"invalid configuration: {exc}", err=True)
                raise typer.Exit(code=5) from exc
            except LLMError as exc:
                typer.echo(f"LLM backend error on {label}: {exc}", err=True)
                failures += 1
                continue
            _print_perturbation_summary(label, outcomes)
    else:
        for i, aid in enumerate(target_ids, start=1):
            if all_anchors:
                typer.echo(f"[{i}/{len(target_ids)}] anchor={aid}")
            else:
                typer.echo(f"anchor={aid}")
            try:
                outcomes = run_perturbations(
                    aid,
                    policy=policy,
                    anchors=anchors,
                    supervised_client=backend,  # type: ignore[arg-type]
                    judge_client=backend,  # type: ignore[arg-type]
                    generators=generators,
                    supervised_model=settings.ollama_supervised_model,
                    judge_model=settings.ollama_judge_model,
                    replay=replay,
                    on_outcome=_stream_outcome,
                    on_generation=_stream_generation,
                    replicates=replicates,
                )
            except KeyError as exc:
                typer.echo(f"unknown anchor id: {exc}", err=True)
                raise typer.Exit(code=4) from exc
            except ValueError as exc:
                typer.echo(f"invalid configuration: {exc}", err=True)
                raise typer.Exit(code=5) from exc
            except LLMError as exc:
                typer.echo(f"LLM backend error on {aid}: {exc}", err=True)
                failures += 1
                continue
            _print_perturbation_summary(aid, outcomes)

    if failures and failures == len(target_ids):
        raise typer.Exit(code=2)


def _stream_drift_progress(progress: DriftSessionProgress) -> None:
    """`S03 baseline A1 aggregate=0.912` — one line per (session, anchor)."""
    o = progress.outcome
    if o.score is None:
        typer.echo(
            f"  S{progress.session_index:02d} {progress.stage_label} "
            f"{progress.anchor_id}  FAIL  ({o.error})"
        )
    else:
        typer.echo(
            f"  S{progress.session_index:02d} {progress.stage_label} "
            f"{progress.anchor_id}  aggregate={o.score.aggregate:.3f}"
        )


@app.command("induce-drift")
def induce_drift_cmd(
    schedule_path: Path = typer.Option(
        DEFAULT_DRIFT_SCHEDULE_PATH,
        "--schedule",
        help="Path to the drift schedule YAML.",
    ),
    anchors_arg: str | None = typer.Option(
        None,
        "--anchors",
        help="Comma-separated anchor ids (default: all from --probes).",
    ),
    policy_path: Path = typer.Option(DEFAULT_POLICY_PATH, "--policy"),
    rubric_path: Path = typer.Option(DEFAULT_RUBRIC_PATH, "--rubric"),
    probes_path: Path = typer.Option(DEFAULT_PROBES_PATH, "--probes"),
    replay: bool = typer.Option(
        False,
        "--replay",
        help="Use the RecordingClient replay cache for supervised + judge calls. "
        "Useful for re-tuning detectors on the same data without burning tokens.",
    ),
    notes: str | None = typer.Option(
        None, "--notes", help="Free-form note stored on the drift_run row."
    ),
    k_threshold: float = typer.Option(
        4.0,
        "--k",
        help="CUSUM threshold tuner constant (h = k·σ). Persisted on the run.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print schedule and anchor count without making any LLM calls.",
    ),
) -> None:
    """Execute the synthetic drift schedule and persist per-session scores."""
    if k_threshold <= 0:
        raise typer.BadParameter("--k must be > 0")

    try:
        schedule = DriftSchedule.from_yaml(schedule_path)
        policy = load_policy(policy_path, rubric_path)
        anchors = load_anchors(probes_path)
    except FileNotFoundError as exc:
        typer.echo(f"config file not found: {exc}", err=True)
        raise typer.Exit(code=5) from exc
    except (ValueError, KeyError) as exc:
        typer.echo(f"invalid configuration: {exc}", err=True)
        raise typer.Exit(code=5) from exc

    if anchors_arg:
        wanted = {a.strip() for a in anchors_arg.split(",") if a.strip()}
        anchors = [a for a in anchors if a.id in wanted]
        missing = wanted - {a.id for a in anchors}
        if missing:
            typer.echo(
                f"unknown anchor ids: {sorted(missing)}", err=True
            )
            raise typer.Exit(code=4)

    n_sessions = schedule.total_sessions
    n_anchors = len(anchors)
    n_calls = n_sessions * n_anchors * 2  # supervised + judge per (session, anchor)
    typer.echo(
        f"schedule={schedule_path} sessions={n_sessions} "
        f"anchors={n_anchors} llm_calls~{n_calls}"
    )
    for stage in schedule.stages:
        typer.echo(
            f"  stage={stage.label:<10} sessions={stage.sessions:>3} "
            f"suffix={stage.suffix!r}"
        )

    if dry_run:
        typer.echo("dry-run: no LLM calls issued.")
        return

    settings = get_settings()
    backend = _backend_factory(settings)

    try:
        drift_run_id, summary = run_drift(
            schedule,
            policy=policy,
            anchors=anchors,
            supervised_client=backend,  # type: ignore[arg-type]
            judge_client=backend,  # type: ignore[arg-type]
            supervised_model=settings.ollama_supervised_model,
            judge_model=settings.ollama_judge_model,
            schedule_path=str(schedule_path),
            replay=replay,
            run_notes=notes,
            k_threshold=k_threshold,
            on_progress=_stream_drift_progress,
        )
    except ValueError as exc:
        typer.echo(f"invalid configuration: {exc}", err=True)
        raise typer.Exit(code=5) from exc
    except LLMError as exc:
        typer.echo(f"LLM backend error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    mean_str = (
        f"{summary.mean_baseline_aggregate:.3f}"
        if summary.mean_baseline_aggregate is not None
        else "n/a"
    )
    typer.echo(
        f"drift_run_id={drift_run_id} scored={summary.total_scored} "
        f"failures={summary.failure_count} mean_baseline_aggregate={mean_str}"
    )


def _format_report_table(report: DriftReport) -> list[str]:
    """Plain-text table — grep-friendly, no rich. Returns lines."""
    headers = (
        "anchor",
        "n_sess",
        "mean_base",
        "min_agg",
        "first_viol",
        "cusum",
        "ewma",
        "lead",
    )
    rows: list[tuple[str, ...]] = []
    for r in report.rows:
        if r.detector_skipped:
            mean_base = "n/a"
            min_agg = "n/a"
            first_viol = "n/a"
            cusum = "n/a"
            ewma = "n/a"
            lead = "n/a"
        else:
            mean_base = (
                f"{r.mean_baseline:.3f}" if r.mean_baseline is not None else "-"
            )
            min_agg = (
                f"{r.min_aggregate:.3f}" if r.min_aggregate is not None else "-"
            )
            first_viol = (
                str(r.first_violation_session)
                if r.first_violation_session is not None
                else "-"
            )
            cusum = (
                str(r.cusum_first_fire) if r.cusum_first_fire is not None else "-"
            )
            ewma = (
                str(r.ewma_first_fire) if r.ewma_first_fire is not None else "-"
            )
            lead_v = r.cusum_lead_sessions
            if lead_v is None:
                lead = "-"
            elif lead_v == float("inf"):
                lead = "+inf"
            else:
                lead = f"{lead_v:+.0f}"
        rows.append(
            (
                r.anchor_id,
                str(r.n_sessions),
                mean_base,
                min_agg,
                first_viol,
                cusum,
                ewma,
                lead,
            )
        )
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    lines: list[str] = []
    lines.append(
        "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    )
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append("  ".join(c.ljust(w) for c, w in zip(row, widths)))
    return lines


@app.command("drift-report")
def drift_report_cmd(
    run_id: int = typer.Argument(..., help="drift_run_id from `induce-drift`."),
    violation_threshold: float = typer.Option(
        0.5,
        "--violation-threshold",
        help="An aggregate score below this counts as an explicit violation.",
    ),
    k: float = typer.Option(
        4.0, "--k", help="CUSUM threshold tuner constant (h = k·σ)."
    ),
    lambda_: float = typer.Option(
        0.2, "--lambda", help="EWMA smoothing constant."
    ),
    L: float = typer.Option(
        3.0, "--L", help="EWMA control-limit width in baseline σ."
    ),
) -> None:
    """Print detection-latency summary for a drift run."""
    run = get_drift_run(run_id)
    if run is None:
        typer.echo(f"unknown drift_run_id: {run_id}", err=True)
        raise typer.Exit(code=4)

    persisted_k = run.get("k_threshold")
    if persisted_k is not None and abs(float(persisted_k) - k) > 1e-9:
        typer.echo(
            f"warning: --k={k} differs from run's persisted k_threshold="
            f"{persisted_k}; computing CUSUM with the override.",
            err=True,
        )

    try:
        report = build_report(
            run_id,
            violation_threshold=violation_threshold,
            k=k,
            lambda_=lambda_,
            L=L,
        )
    except NoBaselineDataError as exc:
        typer.echo(f"insufficient baseline data: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(
        f"drift_run_id={run_id} policy={run['policy_id']} "
        f"supervised={run['supervised_model']} judge={run['judge_model']}"
    )
    for line in _format_report_table(report):
        typer.echo(line)

    cusum_str = (
        f"{report.earliest_cusum_fire[0]} @ S{report.earliest_cusum_fire[1]}"
        if report.earliest_cusum_fire is not None
        else "no fires"
    )
    viol_str = (
        f"{report.earliest_violation[0]} @ S{report.earliest_violation[1]}"
        if report.earliest_violation is not None
        else "no violations"
    )
    typer.echo(f"earliest_cusum_fire: {cusum_str}")
    typer.echo(f"earliest_violation:  {viol_str}")
    headline = report.headline_lead_sessions
    if headline is None:
        verdict = "no detector fired"
    elif headline == float("inf"):
        verdict = "CUSUM fired with no explicit violation in this run"
    else:
        sign = "+" if headline > 0 else ""
        verdict = f"lead = {sign}{headline:.0f} sessions (violation - cusum)"
    typer.echo(f"headline:            {verdict}")


def _stream_recovery_progress(progress: RecoveryProgress) -> None:
    """`A3 evaluate  pre=0.42 post=0.81` — one line per (anchor, step)."""
    typer.echo(f"  {progress.anchor_id} {progress.step}  {progress.detail}")


@app.command("apply-feedback")
def apply_feedback_cmd(
    drift_run_id: int = typer.Argument(..., help="Parent drift_run_id."),
    contrastive_kind: str = typer.Option(
        "temporal",
        "--kind",
        help="Contrastive scenario: temporal | fragility.",
    ),
    anchors_arg: str | None = typer.Option(
        None,
        "--anchors",
        help="Comma-separated anchor ids — overrides the localizer's "
        "automatic top-k pick.",
    ),
    top_k: int = typer.Option(
        3,
        "--top-k",
        help="Number of worst-affected anchors to recover (ignored when "
        "--anchors is set).",
    ),
    policy_path: Path = typer.Option(DEFAULT_POLICY_PATH, "--policy"),
    rubric_path: Path = typer.Option(DEFAULT_RUBRIC_PATH, "--rubric"),
    probes_path: Path = typer.Option(DEFAULT_PROBES_PATH, "--probes"),
    replay: bool = typer.Option(
        False,
        "--replay",
        help="Use the RecordingClient replay cache for supervised, judge, "
        "and synthesizer calls.",
    ),
    notes: str | None = typer.Option(
        None, "--notes", help="Free-form note stored on the recovery_run row."
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the affected anchors and the contrastive pair shapes "
        "without making any LLM calls.",
    ),
) -> None:
    """Localize, synthesize feedback, and re-run anchors to close the loop."""
    if contrastive_kind not in ("temporal", "fragility"):
        raise typer.BadParameter(
            f"--kind must be 'temporal' or 'fragility', got {contrastive_kind!r}"
        )
    if top_k < 1:
        raise typer.BadParameter("--top-k must be >= 1")

    try:
        policy = load_policy(policy_path, rubric_path)
        anchors = load_anchors(probes_path)
    except FileNotFoundError as exc:
        typer.echo(f"config file not found: {exc}", err=True)
        raise typer.Exit(code=5) from exc
    except (ValueError, KeyError) as exc:
        typer.echo(f"invalid configuration: {exc}", err=True)
        raise typer.Exit(code=5) from exc

    parent = get_drift_run(drift_run_id)
    if parent is None:
        typer.echo(f"unknown drift_run_id: {drift_run_id}", err=True)
        raise typer.Exit(code=4)

    selected_ids: list[str] | None = None
    if anchors_arg:
        selected_ids = [a.strip() for a in anchors_arg.split(",") if a.strip()]

    localizations = localize(
        drift_run_id,
        policy=policy,
        top_k=top_k if selected_ids is None else None,
    )
    if selected_ids is not None:
        wanted = set(selected_ids)
        localizations = [r for r in localizations if r.anchor_id in wanted]
        missing = wanted - {r.anchor_id for r in localizations}
        if missing:
            typer.echo(f"unknown anchor ids: {sorted(missing)}", err=True)
            raise typer.Exit(code=4)

    if not localizations:
        typer.echo(
            f"no affected anchors found in drift_run {drift_run_id}; "
            f"is the run empty?",
            err=True,
        )
        raise typer.Exit(code=2)

    typer.echo(
        f"parent_drift_run={drift_run_id} kind={contrastive_kind} "
        f"anchors={len(localizations)}"
    )
    for r in localizations:
        typer.echo(
            f"  {r.anchor_id} distance={r.distance:.3f} "
            f"worst_aggregate={r.worst_score.aggregate:.3f}"
        )

    if dry_run:
        for r in localizations:
            if contrastive_kind == "temporal":
                pair = temporal_pair(r.anchor_id, drift_run_id=drift_run_id)
            else:
                pair = fragility_pair(r.anchor_id, policy=policy)
            if pair is None:
                typer.echo(f"  {r.anchor_id} pair=NONE")
            else:
                typer.echo(
                    f"  {r.anchor_id} pair=ok safe_agg="
                    f"{pair.safe_score.aggregate:.3f} "
                    f"near_agg={pair.near_boundary_score.aggregate:.3f}"
                )
        typer.echo("dry-run: no LLM calls issued.")
        return

    settings = get_settings()
    backend = _backend_factory(settings)

    try:
        recovery_run_id, summary = apply_feedback(
            drift_run_id,
            policy=policy,
            anchors=anchors,
            supervised_client=backend,  # type: ignore[arg-type]
            judge_client=backend,  # type: ignore[arg-type]
            supervised_model=settings.ollama_supervised_model,
            judge_model=settings.ollama_judge_model,
            contrastive_kind=contrastive_kind,  # type: ignore[arg-type]
            top_k=top_k,
            selected_anchor_ids=selected_ids,
            replay=replay,
            run_notes=notes,
            on_progress=_stream_recovery_progress,
        )
    except ValueError as exc:
        typer.echo(f"invalid configuration: {exc}", err=True)
        raise typer.Exit(code=5) from exc
    except LLMError as exc:
        typer.echo(f"LLM backend error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    delta_str = (
        f"{summary.mean_delta_toward_baseline:+.3f}"
        if summary.mean_delta_toward_baseline is not None
        else "n/a"
    )
    typer.echo(
        f"recovery_run_id={recovery_run_id} anchors={summary.anchor_count} "
        f"failures={summary.failure_count} "
        f"mean_delta_toward_baseline={delta_str}"
    )


def _format_recovery_table(report: RecoveryReport) -> list[str]:
    headers = (
        "anchor",
        "pre_worst",
        "pre_sess",
        "post",
        "delta",
        "recovered",
    )
    rows: list[tuple[str, ...]] = []
    for r in report.rows:
        pre = (
            f"{r.pre_worst_aggregate:.3f}"
            if r.pre_worst_aggregate is not None
            else "-"
        )
        sess = str(r.pre_worst_session) if r.pre_worst_session is not None else "-"
        post = f"{r.post_aggregate:.3f}" if r.post_aggregate is not None else "-"
        delta = (
            f"{r.delta_toward_baseline:+.3f}"
            if r.delta_toward_baseline is not None
            else "-"
        )
        recovered = "yes" if r.recovered else "no"
        rows.append((r.anchor_id, pre, sess, post, delta, recovered))

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]
    lines: list[str] = []
    lines.append("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append("  ".join(c.ljust(w) for c, w in zip(row, widths)))
    return lines


@app.command("recovery-report")
def recovery_report_cmd(
    recovery_run_id: int = typer.Argument(
        ..., help="recovery_run_id from `apply-feedback`."
    ),
    show_feedback: bool = typer.Option(
        False,
        "--show-feedback",
        help="Print each anchor's synthesized feedback text below the table.",
    ),
) -> None:
    """Print before/after summary for a recovery run."""
    try:
        report = build_recovery_report(recovery_run_id)
    except OrphanRecoveryRunError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=4) from exc
    except NoRecoveryDataError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    typer.echo(
        f"recovery_run_id={recovery_run_id} "
        f"parent_drift_run_id={report.parent_drift_run_id} "
        f"contrastive_kind={report.contrastive_kind}"
    )
    for line in _format_recovery_table(report):
        typer.echo(line)

    delta_str = (
        f"{report.mean_delta_toward_baseline:+.3f}"
        if report.mean_delta_toward_baseline is not None
        else "n/a"
    )
    typer.echo(
        f"mean_delta_toward_baseline={delta_str} "
        f"recovered={report.recovered_count}/{report.total_anchors}"
    )
    typer.echo(f"verdict: {report.verdict}")

    if show_feedback:
        typer.echo("")
        for r in report.rows:
            if r.feedback is None:
                typer.echo(f"[{r.anchor_id} contrastive=n/a]")
                typer.echo("  (no feedback recorded)")
                continue
            typer.echo(
                f"[{r.anchor_id} contrastive={r.feedback.contrastive_kind}]"
            )
            typer.echo(f"  > {r.feedback.feedback_text}")


@app.command("fragility-report")
def fragility_report_cmd(
    output_dir: Path = typer.Option(
        DEFAULT_REPORTS_DIR,
        "--output-dir",
        help="Directory under which the fragility CSV is written.",
    ),
) -> None:
    """Compute and persist the §4.4 Jacobian + aggregated fragility table.

    Reads from the DB only — no LLM calls. Writes
    `reports/fragility_<UTC>.csv` with the aggregated table at the top
    followed by a per-anchor Jacobian section.
    """
    import csv
    from datetime import datetime, timezone

    table = aggregated_fragility()
    jacobians = all_jacobians()

    if not jacobians and not table.cells:
        typer.echo(
            "no perturbation data found; run `maimonedes perturb` first",
            err=True,
        )
        raise typer.Exit(code=4)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"fragility_{timestamp}.csv"

    with report_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        # ---- Aggregated section ----------------------------------------
        # Per axis: mean Δ + between-anchor std. `n_anchors` is the
        # max number of anchors that contributed to any cell in the row
        # (some axes may have fewer if generators didn't fire on every
        # anchor; we report the maximum so the row's coverage is the
        # most-favourable interpretation).
        writer.writerow(["# aggregated fragility (mean Δ across anchors ± between-anchor std)"])
        agg_header = ["perturbation_kind"]
        for col in table.columns:
            agg_header.append(col)
            agg_header.append(f"{col}_std")
        agg_header.append("n_anchors")
        writer.writerow(agg_header)

        grid = table.as_grid()
        for kind in table.perturbation_kinds:
            row = [kind]
            counts: list[int] = []
            for col in table.columns:
                cell = grid.get((kind, col))
                if cell is None:
                    row.append("")
                    row.append("")
                else:
                    row.append(f"{cell.mean_delta:+.4f}")
                    row.append(f"{cell.std_delta:.4f}")
                    counts.append(cell.count)
            row.append(str(max(counts) if counts else 0))
            writer.writerow(row)

        # ---- Per-anchor jacobian section --------------------------------
        # Per cell: mean Δ + within-replicate std. `n_rep` is the number
        # of replicates that contributed to the row's deltas; equals 1
        # for non-replicate runs.
        writer.writerow([])
        writer.writerow(["# per-anchor jacobian (mean Δ ± within-replicate std)"])
        for anchor_id in sorted(jacobians):
            jac = jacobians[anchor_id]
            writer.writerow([])
            writer.writerow(
                [
                    f"## {anchor_id}",
                    f"baseline_aggregate={jac.baseline_aggregate:.4f}",
                ]
            )
            jac_header = ["transform_label", "perturbation_kind", "n_rep"]
            for col in jac.columns:
                jac_header.append(col)
                jac_header.append(f"{col}_std")
            writer.writerow(jac_header)

            for jrow in jac.rows:
                row = [jrow.transform_label, jrow.perturbation_kind, str(jrow.n_replicates)]
                for col in jac.columns:
                    row.append(f"{jrow.deltas.get(col, 0.0):+.4f}")
                    row.append(f"{jrow.std_deltas.get(col, 0.0):.4f}")
                writer.writerow(row)

    typer.echo(f"wrote {report_path}")
    typer.echo(
        f"summary: anchors={len(jacobians)} kinds={len(table.perturbation_kinds)} "
        f"cells={len(table.cells)}"
    )


@app.command("label-distribution")
def label_distribution_cmd(
    policy_path: Path = typer.Option(DEFAULT_POLICY_PATH, "--policy"),
    rubric_path: Path = typer.Option(DEFAULT_RUBRIC_PATH, "--rubric"),
    filter_sql: str = typer.Option(
        "",
        "--filter",
        help=(
            "Optional SQL fragment appended to WHERE; e.g. "
            "\"judge_model LIKE 'judge%'\". Operator-only; do not pass "
            "untrusted input."
        ),
    ),
) -> None:
    """Print the per-axis rubric-label distribution diagnostic.

    For each sub-condition, shows the histogram of normalised values
    seen in `compliance_scores`, sorted by `dominant_share` descending.
    Axes whose `dominant_share` exceeds 0.85 are flagged with
    `⚠ near-uniform — rubric refinement recommended` — they're not
    fixable by upgrading the Stage-2 head architecture.
    """
    from maimonedes.monitor.label_distribution import (
        compute_distribution,
        format_report,
    )
    from maimonedes.storage.repo import init_engine

    settings = get_settings()
    init_engine(settings)
    policy = load_policy(policy_path, rubric_path)
    report = compute_distribution(
        policy,
        filter_sql=filter_sql or None,
    )
    for line in format_report(report):
        typer.echo(line)


@app.command("backfill-llm-call-ids")
def backfill_llm_call_ids_cmd(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print how many rows would be updated; don't write.",
    ),
    supervised_backend_prefix: str = typer.Option(
        "ollama-supervised",
        "--supervised-prefix",
        help="LLMCall.backend_name prefix used to identify supervised rows.",
    ),
) -> None:
    """One-time backfill of `compliance_scores.llm_call_id` for legacy rows.

    Walks every `compliance_scores` row whose `llm_call_id` is NULL,
    pairs it chronologically with a supervised `llm_calls` row that
    isn't already claimed, and writes the FK. Idempotent — re-running
    against an already-backfilled DB updates zero rows.
    """
    from maimonedes.storage.llm_calls import backfill_llm_call_ids
    from maimonedes.storage.repo import init_engine

    settings = get_settings()
    init_engine(settings)
    n = backfill_llm_call_ids(
        supervised_backend_prefix=supervised_backend_prefix,
        dry_run=dry_run,
    )
    if dry_run:
        typer.echo(f"would update {n} compliance_scores row(s)")
    else:
        typer.echo(f"updated {n} compliance_scores row(s)")


def main(argv: list[str] | None = None) -> None:
    """Console-script entry point.

    Typer normally reads from `sys.argv` directly; pass `argv` for
    test harnesses that want a string-list interface.
    """
    if argv is None:
        app()
        return
    # Typer/Click read argv via sys.argv; swap it in for the call.
    saved = sys.argv
    try:
        sys.argv = ["maimonedes", *argv]
        app()
    finally:
        sys.argv = saved


__all__ = ["app", "main", "PING_PROMPT", "set_backend_factory", "reset_backend_factory"]
