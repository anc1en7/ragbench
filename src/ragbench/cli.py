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
corpus_app = typer.Typer(name="corpus", help="Corpus fetch and build.", no_args_is_help=True)
app.add_typer(corpus_app, name="corpus")

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


@corpus_app.command("ingest")
def corpus_ingest_cmd(
    raw: Path = typer.Option(Path("data/raw"), "--raw", help="Raw corpus directory"),
    out: Path = typer.Option(Path("data/documents.jsonl"), "--out", help="Documents JSONL"),
    lock: Path = typer.Option(Path("corpus.lock"), "--lock", help="Corpus lock file"),
    force: bool = typer.Option(False, "--force", help="Re-resolve pin and re-download"),
) -> None:
    """Fetch the pinned corpus (no-op when fresh) and build documents.jsonl."""
    from ragbench.corpus.build import build_documents
    from ragbench.corpus.fetch import fetch

    lock_obj = fetch(lock_path=lock, raw_dir=raw, force=force)
    if lock_obj is None:
        typer.echo(f"corpus up to date (lock: {lock})")
    else:
        typer.echo(
            f"fetched tag {lock_obj.tag} @ {lock_obj.commit_sha[:12]} ({lock_obj.file_count} files)"
        )
    documents = build_documents(raw_dir=raw, out=out)
    sections = sum(len(doc.sections) for doc in documents)
    approx_tokens = sum(len(doc.text.split()) for doc in documents)
    typer.echo(
        f"documents={len(documents)} sections={sections} approx_tokens={approx_tokens} -> {out}"
    )


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
