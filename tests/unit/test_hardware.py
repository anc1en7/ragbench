import json

from ragbench.hardware import collect_hardware


def test_collect_hardware_shape(tmp_path):
    out = tmp_path / "hardware.json"
    data = collect_hardware(out)
    assert out.exists()
    on_disk = json.loads(out.read_text(encoding="utf-8"))
    for key in (
        "platform",
        "python",
        "cpu_cores",
        "ram_gb",
        "gpu",
        "gpu_vram_mb",
        "llm",
        "package_versions",
    ):
        assert key in data, key
        assert key in on_disk, key
    assert data["ram_gb"] > 0
    assert data["cpu_cores"] >= 1
    assert isinstance(data["package_versions"], dict)
    assert "llm" in data
    assert "model" in data["llm"]


def test_collect_hardware_no_secrets(tmp_path):
    out = tmp_path / "hardware.json"
    data = collect_hardware(out)
    text = json.dumps(data)
    assert "changeme" not in text
    assert "api_key" not in text.lower()
