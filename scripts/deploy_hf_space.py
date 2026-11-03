"""Stage and deploy the replay dashboard to a Hugging Face Docker Space (docs/DEPLOY.md).

Usage:
    python scripts/deploy_hf_space.py --stage-only          # build .tmp/hf_space only
    python scripts/deploy_hf_space.py --space <user>/ragbench

Needs a Hugging Face login with write access (`hf auth login`). The Space gets only what the
dashboard reads: no corpus text, no secrets, no model files.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / ".tmp" / "hf_space"

FRONT_MATTER = """---
title: RAGBench
emoji: 📊
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: Replay dashboard for a RAG pipeline benchmark
---

# RAGBench dashboard

Replay dashboard for [RAGBench](https://github.com/anc1en7/ragbench): retrieval and RAGAS
results for 33 RAG pipeline configs on a human-verified gold set. Every number is read from
the committed `results/` files.
"""

DOCKERFILE = """FROM python:3.13-slim

RUN useradd -m -u 1000 user
WORKDIR /app
COPY dashboard/requirements.txt dashboard/requirements.txt
RUN pip install --no-cache-dir -r dashboard/requirements.txt
COPY --chown=user . .
USER user
ENV HOME=/home/user
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.address=0.0.0.0", \\
     "--server.port=8501", "--server.headless=true"]
"""

INCLUDE = [
    "dashboard",
    "src/ragbench",
    "results/summary.csv",
    "results/runs",
    "data/goldset/verified.jsonl",
    "docs/METHODOLOGY.md",
    "docs/LIMITATIONS.md",
    ".streamlit/config.toml",
    "LICENSE",
]


def stage() -> Path:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for rel in INCLUDE:
        src, dst = ROOT / rel, STAGE / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, ignore=ignore)
        else:
            shutil.copy2(src, dst)
    (STAGE / "README.md").write_text(FRONT_MATTER, encoding="utf-8")
    (STAGE / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8")
    return STAGE


def deploy(space: str) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(
        folder_path=str(STAGE),
        repo_id=space,
        repo_type="space",
        commit_message="Deploy RAGBench replay dashboard",
        delete_patterns=["*"],  # the Space mirrors the staged folder exactly
    )
    return f"https://huggingface.co/spaces/{space}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--space", help="<user>/<space-name>")
    parser.add_argument("--stage-only", action="store_true")
    args = parser.parse_args()
    path = stage()
    files = sum(1 for p in path.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 2**20
    print(f"staged {files} files ({size:.1f} MiB) in {path}")
    if args.stage_only:
        return
    if not args.space:
        parser.error("--space is required unless --stage-only")
    print(deploy(args.space))


if __name__ == "__main__":
    main()
