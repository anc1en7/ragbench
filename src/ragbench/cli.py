"""RAGBench command-line interface."""

from __future__ import annotations

import json
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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
