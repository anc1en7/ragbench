"""Staged sweep grid: parsing, config expansion, run estimate and selection (Task 4.2).

Grid semantics (see ``configs/grid.yaml`` header): each stage's configs are
``base`` (first stage) or a parent config (stages with ``from:``), plus the stage's ``set``
overrides, plus one override dict from each ``vary`` axis (cartesian product). Dotted
keys set one leaf; a non-dotted key with a mapping value replaces that sub-config.
``only_backends`` carries parents whose backend is not listed through unchanged.
Duplicate configs (same ``config_id``) collapse, keeping first occurrence order.

Selection ranks run results by ``selection.metric`` descending, then each tiebreak
metric descending, then ``config_id`` ascending (deterministic).
"""

from __future__ import annotations

import copy
import itertools
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from ragbench.config import PipelineConfig
from ragbench.eval.results import RunResult, Split
from ragbench.eval.retrieval_metrics import METRIC_NAMES


class Selection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str = "ndcg@10"
    tiebreak: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _known_metrics(self) -> Selection:
        for name in (self.metric, *self.tiebreak):
            if name not in METRIC_NAMES:
                raise ValueError(f"unknown selection metric {name!r}; choose from {METRIC_NAMES}")
        return self


class Stage(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    name: str
    from_stage: str | None = Field(default=None, alias="from")
    top_n: int = 3
    overrides: dict[str, Any] = Field(default_factory=dict, alias="set")
    vary: list[list[dict[str, Any]]] = Field(default_factory=list)
    only_backends: list[str] | None = None


class Grid(BaseModel):
    model_config = ConfigDict(extra="forbid")

    corpus: str
    split: Split = "dev"
    seed: int = 42
    selection: Selection = Field(default_factory=Selection)
    finalists: int = 5
    base: dict[str, Any] = Field(default_factory=dict)
    stages: list[Stage]

    @model_validator(mode="after")
    def _stage_graph(self) -> Grid:
        seen: list[str] = []
        for idx, stage in enumerate(self.stages):
            if stage.name in seen:
                raise ValueError(f"duplicate stage name {stage.name!r}")
            if idx == 0 and stage.from_stage is not None:
                raise ValueError("the first stage cannot have 'from'")
            if idx > 0 and stage.from_stage not in seen:
                raise ValueError(
                    f"stage {stage.name!r} has from={stage.from_stage!r}, "
                    f"which is not an earlier stage ({seen})"
                )
            if stage.top_n < 1:
                raise ValueError(f"stage {stage.name!r}: top_n must be >= 1")
            seen.append(stage.name)
        return self

    def stage(self, name: str) -> Stage:
        return next(s for s in self.stages if s.name == name)


def load_grid(path: str | Path) -> Grid:
    """Load and validate a grid YAML file."""
    grid_path = Path(path)
    if not grid_path.exists():
        raise FileNotFoundError(f"grid file not found: {grid_path}")
    try:
        return Grid.model_validate(yaml.safe_load(grid_path.read_text(encoding="utf-8")))
    except (yaml.YAMLError, ValidationError) as exc:
        raise ValueError(f"invalid grid {grid_path}: {exc}") from exc


def apply_overrides(config: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``config`` with ``overrides`` applied (dotted leaf / mapping replace)."""
    out = copy.deepcopy(dict(config))
    for key, value in overrides.items():
        parts = key.split(".")
        node = out
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = copy.deepcopy(value)
    return out


def _validate(data: dict[str, Any], stage: Stage) -> PipelineConfig:
    try:
        return PipelineConfig.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"stage {stage.name!r} produced an invalid config: {exc}") from exc


def expand_stage(
    grid: Grid, stage: Stage, parents: Sequence[PipelineConfig] | None
) -> list[PipelineConfig]:
    """All configs of ``stage`` (deduplicated by ``config_id``, first-occurrence order)."""
    if stage.from_stage is None:
        starts: list[dict[str, Any]] = [copy.deepcopy(grid.base)]
        carried: list[PipelineConfig] = []
    else:
        if parents is None:
            raise ValueError(f"stage {stage.name!r} needs parent configs from {stage.from_stage!r}")
        starts = []
        carried = []
        for parent in parents:
            if stage.only_backends is not None and parent.backend not in stage.only_backends:
                carried.append(parent)
            else:
                starts.append(parent.model_dump(mode="json"))

    configs: list[PipelineConfig] = list(carried)
    for start in starts:
        staged = apply_overrides(start, stage.overrides)
        for combo in itertools.product(*stage.vary) if stage.vary else [()]:
            data = staged
            for variant in combo:
                data = apply_overrides(data, variant)
            data = apply_overrides(data, {"corpus": grid.corpus, "seed": grid.seed})
            configs.append(_validate(data, stage))

    unique: dict[str, PipelineConfig] = {}
    for cfg in configs:
        unique.setdefault(cfg.config_id, cfg)
    return list(unique.values())


def estimate(grid: Grid) -> list[tuple[str, int]]:
    """Planned configs per stage before reuse (parents assumed to fill ``top_n``)."""
    counts: dict[str, int] = {}
    out: list[tuple[str, int]] = []
    for stage in grid.stages:
        if stage.from_stage is None:
            n = len(expand_stage(grid, stage, None))
        else:
            parent = grid.stage(stage.from_stage)
            n_parents = min(parent.top_n, counts[stage.from_stage])
            n = n_parents * math.prod(len(axis) for axis in stage.vary)
        counts[stage.name] = n
        out.append((stage.name, n))
    return out


def rank(results: Sequence[RunResult], selection: Selection) -> list[RunResult]:
    """Results ordered best-first by the selection metric, tiebreaks, then config_id."""
    keys = (selection.metric, *selection.tiebreak)
    return sorted(
        results,
        key=lambda r: (*(-r.metrics[k].mean for k in keys), r.config_id),
    )
