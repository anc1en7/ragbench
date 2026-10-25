"""RAGBench command-line interface."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import typer

from ragbench.hardware import collect_hardware

app = typer.Typer(
    name="ragbench",
    help="RAGBench: benchmark Retrieval-Augmented Generation pipelines.",
    no_args_is_help=True,
)

HARDWARE_PATH = Path("results/hardware.json")


@app.callback()
def root() -> None:
    """RAGBench command-line interface."""


@app.command("hardware")
def hardware_cmd(
    out: Path = typer.Option(HARDWARE_PATH, "--out", help="Where to write hardware.json"),
) -> None:
    """Detect hardware and package versions, writing results/hardware.json."""
    data = collect_hardware(out)
    typer.echo(json.dumps(data, indent=2))


@app.command("llm-probe")
def llm_probe_cmd(
    out: Path = typer.Option(HARDWARE_PATH, "--out", help="hardware.json to merge results into"),
) -> None:
    """Probe the configured endpoint, verify a chat completion, pin it in hardware.json."""
    from ragbench.llm_client import LLMConfigError, probe_endpoint, settings_from_env

    try:
        settings = settings_from_env()
    except LLMConfigError as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(code=1) from exc

    result = probe_endpoint(settings.base_url, settings.api_key)
    if not result.get("chat_ok"):
        typer.secho(str(result.get("error", "probe failed")), fg="red", err=True)
        raise typer.Exit(code=1)

    models = list(result.get("models", []))
    pinned = settings.model
    if pinned and pinned not in models:
        typer.secho(
            f"warning: LLM_MODEL={pinned!r} not in endpoint model list "
            f"(first 10: {models[:10]}). Proceeding anyway.",
            fg="yellow",
        )

    if out.exists():
        hardware = json.loads(out.read_text(encoding="utf-8"))
    else:
        hardware = collect_hardware(out)
    hardware["llm"] = {
        "base_url_host": settings.host,
        "model": pinned,
        "resolved_at": datetime.now(UTC).isoformat(),
        "chat_ok": True,
        "model_count": len(models),
    }
    out.write_text(json.dumps(hardware, indent=2) + "\n", encoding="utf-8")
    typer.echo(
        f"OK: {settings.host} reachable, {len(models)} models, "
        f"pinned model={pinned!r} recorded in {out}"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
