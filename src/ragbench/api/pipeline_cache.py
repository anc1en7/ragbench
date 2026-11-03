"""Config discovery and a per-``config_id`` LRU cache of built pipelines (Task 5.1).

Configs are discovered from two places: pipeline YAML files in ``configs/`` (files that
are not pipeline configs, such as ``grid.yaml``, are skipped) and the ``config`` stored in
every ``results/runs/*.json``. Building a pipeline loads models, so built pipelines are
kept in a small LRU cache. A namespace that holds no data yet (``size_bytes == 0``, e.g.
the in-process BM25 control or a never-indexed config) is indexed on first use.
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from ragbench.config import PipelineConfig, load_pipeline_config
from ragbench.pipeline import Pipeline, build_pipeline, index_documents

PipelineBuilder = Callable[[PipelineConfig], Pipeline]


def discover_configs(configs_dir: Path, runs_dir: Path) -> dict[str, tuple[PipelineConfig, str]]:
    """Map ``config_id`` to ``(config, source path)``; YAML files win over run files."""
    found: dict[str, tuple[PipelineConfig, str]] = {}
    if runs_dir.is_dir():
        for path in sorted(runs_dir.glob("*.json")):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                cfg = PipelineConfig.model_validate(raw["config"])
            except (OSError, ValueError, KeyError, ValidationError):
                continue
            found.setdefault(cfg.config_id, (cfg, path.as_posix()))
    if configs_dir.is_dir():
        for path in sorted(configs_dir.glob("*.y*ml")):
            try:
                cfg = load_pipeline_config(path)
            except (FileNotFoundError, ValueError):
                continue
            found[cfg.config_id] = (cfg, path.as_posix())
    return found


def build_and_index(cfg: PipelineConfig) -> Pipeline:
    """Default builder: build the pipeline and index its corpus if the namespace is empty."""
    from ragbench.eval.runner import resolve_dataset
    from ragbench.goldset.generate import load_documents

    pipeline = build_pipeline(cfg)
    if pipeline.store.size_bytes(cfg.config_id) == 0:
        documents = load_documents(resolve_dataset(cfg.corpus).documents)
        index_documents(documents, pipeline)
    return pipeline


class PipelineCache:
    """Thread-safe LRU cache of pipelines keyed by ``config_id``."""

    def __init__(self, builder: PipelineBuilder = build_and_index, max_size: int = 4) -> None:
        if max_size < 1:
            raise ValueError(f"max_size must be >= 1, got {max_size}")
        self._builder = builder
        self._max_size = max_size
        self._items: OrderedDict[str, Pipeline] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, cfg: PipelineConfig) -> Pipeline:
        key = cfg.config_id
        with self._lock:
            if key in self._items:
                self._items.move_to_end(key)
                return self._items[key]
            pipeline = self._builder(cfg)
            self._items[key] = pipeline
            while len(self._items) > self._max_size:
                self._items.popitem(last=False)
            return pipeline

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, config_id: object) -> bool:
        return config_id in self._items
