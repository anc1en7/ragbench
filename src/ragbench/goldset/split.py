"""Seeded, question-type-stratified dev/test split (spec 5.6).

Dev gets 30%, test 70%; tuning happens on dev only, headline numbers on test
only. Output shape: ``{"seed": int, "dev": [qid], "test": [qid]}``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from ragbench.goldset.review import load_candidates

DEV_FRAC = 0.3
SEED = 42


def split_verified(
    verified_path: Path = Path("data/goldset/verified.jsonl"),
    out_path: Path = Path("data/goldset/splits.json"),
    *,
    dev_frac: float = DEV_FRAC,
    seed: int = SEED,
) -> dict[str, Any]:
    """Split verified questions into dev/test, stratified by question type."""
    candidates = load_candidates(verified_path)
    if not candidates:
        raise ValueError(f"no verified questions to split in {verified_path}")

    groups: dict[str, list[str]] = {}
    for candidate in candidates:
        groups.setdefault(candidate.question_type, []).append(candidate.qid)

    dev: list[str] = []
    test: list[str] = []
    for qtype in sorted(groups):
        qids = sorted(groups[qtype])
        rng = random.Random(f"{seed}:{qtype}")
        rng.shuffle(qids)
        n_dev = int(len(qids) * dev_frac + 0.5)
        dev.extend(qids[:n_dev])
        test.extend(qids[n_dev:])

    result: dict[str, Any] = {"seed": seed, "dev": sorted(dev), "test": sorted(test)}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
