"""Command-line entry point for maimonedes.

Phase 0 ships a single subcommand, `ping`, that exercises the entire
stack (settings -> client -> backend -> recording -> storage) so we
can prove all the wiring works before any policy / probe / scoring
code lands in Phase 1.
"""
from __future__ import annotations

import sys
from collections.abc import Callable, Iterable

import typer

from maimonedes.llm.client import LLMError, Message
from maimonedes.llm.ollama_backend import OllamaBackend
from maimonedes.llm.recording_client import RecordingClient
from maimonedes.settings import Settings, get_settings

PING_PROMPT = "Reply with the single word PONG."

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
