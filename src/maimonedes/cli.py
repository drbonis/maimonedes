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
    ParaphraseGenerator,
)
from maimonedes.core.policy import load_policy
from maimonedes.core.probe import load_anchors
from maimonedes.experiments.calibrate_judge import (
    load_references,
    run_calibration,
)
from maimonedes.experiments.perturbation_session import (
    PerturbationOutcome,
    run_perturbations,
)
from maimonedes.experiments.run_session import run_once as run_once_session
from maimonedes.llm.client import LLMError, Message
from maimonedes.llm.ollama_backend import OllamaBackend
from maimonedes.llm.recording_client import RecordingClient
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
ALL_KINDS = ("paraphrase", "demographic", "authority", "boundary")

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
    demographic_path: Path,
    authority_path: Path,
    boundary_path: Path,
) -> list[PerturbationGenerator]:
    out: list[PerturbationGenerator] = []
    for kind in kinds:
        if kind == "paraphrase":
            out.append(
                ParaphraseGenerator(
                    paraphrase_client,  # type: ignore[arg-type]
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
        else:
            raise typer.BadParameter(
                f"unknown perturbation kind: {kind!r}; "
                f"expected one of {ALL_KINDS}"
            )
    return out


def _print_perturbation_summary(
    anchor_id: str, outcomes: list[PerturbationOutcome]
) -> None:
    if not outcomes:
        typer.echo(f"anchor={anchor_id}: no perturbations generated")
        return
    successes = [o for o in outcomes if o.score is not None]
    failures = [o for o in outcomes if o.score is None]
    typer.echo(f"anchor={anchor_id}")
    for o in outcomes:
        if o.score is None:
            typer.echo(f"  {o.probe.transform_label}  FAIL  ({o.error})")
            continue
        delta_str = (
            f"Δ={o.delta_aggregate:+.3f}"
            if o.delta_aggregate is not None
            else "Δ=n/a"
        )
        typer.echo(
            f"  {o.probe.transform_label}  agg={o.score.aggregate:.3f}  {delta_str}"
        )
    if successes and any(o.delta_aggregate is not None for o in successes):
        deltas = [
            abs(o.delta_aggregate)
            for o in successes
            if o.delta_aggregate is not None
        ]
        typer.echo(
            f"summary: n={len(successes)}/{len(outcomes)} "
            f"mean|Δ|={sum(deltas) / len(deltas):.3f} failures={len(failures)}"
        )
    else:
        typer.echo(
            f"summary: n={len(successes)}/{len(outcomes)} "
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
            demographic_path=demographic_path,
            authority_path=authority_path,
            boundary_path=boundary_path,
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
    failures = 0
    for aid in target_ids:
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
