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
from maimonedes.monitor.fragility import (
    aggregated_fragility,
    all_jacobians,
)
from maimonedes.settings import Settings, get_settings

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
    """`[02/11] authority:senior_cardiologist  agg=0.420  Δ=-0.473`"""
    o = progress.outcome
    digits = max(2, len(str(progress.total)))
    counter = f"[{progress.index:0{digits}d}/{progress.total:0{digits}d}]"
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
        help="Anchor id (e.g. A3). Omit when --all-anchors is passed.",
    ),
    all_anchors: bool = typer.Option(
        False, "--all-anchors", help="Sweep A1..A8 sequentially."
    ),
    kinds: str = typer.Option(
        ",".join(ALL_KINDS),
        "--kinds",
        help="Comma-separated subset of {paraphrase,demographic,authority,boundary}.",
    ),
    paraphrase_n: int = typer.Option(
        3, "--paraphrase-n", help="Number of paraphrases per anchor."
    ),
    replay: bool = typer.Option(
        False,
        "--replay",
        help="Use the RecordingClient replay cache for supervised + judge calls.",
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
    """Generate, run, and score a perturbation cloud for one anchor (or all)."""
    if not all_anchors and anchor_id is None:
        raise typer.BadParameter("Provide an anchor id or pass --all-anchors.")

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

    target_ids: list[str] = (
        [a.id for a in anchors] if all_anchors else [anchor_id or ""]
    )
    typer.echo(
        f"running perturbations on {len(target_ids)} anchor(s); "
        f"kinds={','.join(kinds_list)}"
    )

    failures = 0
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
        writer.writerow(["# aggregated fragility (mean Δ across anchors)"])
        writer.writerow(["perturbation_kind", *table.columns, "n"])
        grid = table.as_grid()
        for kind in table.perturbation_kinds:
            row = [kind]
            counts: list[int] = []
            for col in table.columns:
                cell = grid.get((kind, col))
                if cell is None:
                    row.append("")
                else:
                    row.append(f"{cell.mean_delta:+.4f}")
                    counts.append(cell.count)
            row.append(str(max(counts) if counts else 0))
            writer.writerow(row)

        writer.writerow([])
        writer.writerow(["# per-anchor jacobian"])
        for anchor_id in sorted(jacobians):
            jac = jacobians[anchor_id]
            writer.writerow([])
            writer.writerow(
                [
                    f"## {anchor_id}",
                    f"baseline_aggregate={jac.baseline_aggregate:.4f}",
                ]
            )
            writer.writerow(["transform_label", "perturbation_kind", *jac.columns])
            for jrow in jac.rows:
                writer.writerow(
                    [
                        jrow.transform_label,
                        jrow.perturbation_kind,
                        *[
                            f"{jrow.deltas.get(col, 0.0):+.4f}"
                            for col in jac.columns
                        ],
                    ]
                )

    typer.echo(f"wrote {report_path}")
    typer.echo(
        f"summary: anchors={len(jacobians)} kinds={len(table.perturbation_kinds)} "
        f"cells={len(table.cells)}"
    )


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
