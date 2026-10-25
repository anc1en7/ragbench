"""Hardware and environment detection.

Writes results/hardware.json so every run can record the machine it ran on
(spec 5.5). Never records secrets (LLM base URL is reduced to its host).
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import psutil

NVIDIA_SMI_QUERY = [
    "nvidia-smi",
    "--query-gpu=name,memory.total",
    "--format=csv,noheader",
]


def _gpu_info() -> dict[str, object]:
    if shutil.which(NVIDIA_SMI_QUERY[0]) is None:
        return {"gpu": "none", "gpu_vram_mb": None}
    try:
        proc = subprocess.run(
            NVIDIA_SMI_QUERY,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return {"gpu": "unknown", "gpu_vram_mb": None}
    line = proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else ""
    if not line:
        return {"gpu": "none", "gpu_vram_mb": None}
    name, _, mem = line.partition(",")
    mem = mem.strip().upper().replace(" MIB", "").replace("MB", "").strip()
    try:
        vram_mb: int | None = int(mem)
    except ValueError:
        vram_mb = None
    return {"gpu": name.strip(), "gpu_vram_mb": vram_mb}


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def _package_versions() -> dict[str, str]:
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover
        return {}
    versions: dict[str, str] = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"]
        if name:
            versions[name] = dist.version
    return dict(sorted(versions.items()))


def collect_hardware(out_path: Path) -> dict[str, object]:
    """Collect hardware facts and write them to *out_path* as JSON."""
    data: dict[str, object] = {
        "collected_at": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cpu": platform.processor() or platform.machine(),
        "cpu_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "cuda_available": _cuda_available(),
        "llm": {"base_url_host": "", "model": "", "resolved_at": ""},
        "package_versions": _package_versions(),
    }
    data.update(_gpu_info())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data
