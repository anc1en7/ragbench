"""Embedding disk-cache contract tests (Task 2.4, rulings 2, 3, 4, 6, 7).

Everything here is offline: the model layer under the cache is a counting fake, and
``SentenceTransformerEmbedder`` is only constructed (its model loads lazily on the
first embed call, which unit tests never make).
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

import ragbench.embedding.cache as cache_module
from ragbench.embedding.base import document_prefix, query_prefix
from ragbench.embedding.cache import (
    EmbeddingCache,
    cache_key_hex,
    encode_with_cache,
    hf_cache_root,
)
from ragbench.embedding.st_embedder import SentenceTransformerEmbedder

MODEL = "intfloat/e5-small-v2"
REV_A = "rev-a"
REV_B = "rev-b"
DIM = 4


class CountingCompute:
    """Fake model layer under the cache: deterministic vectors, counts every call."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        return np.stack([_fake_vec(t) for t in texts])


def _fake_vec(text: str) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(DIM).astype(np.float32)
    return vec / np.linalg.norm(vec)


def _encode(
    texts: list[str],
    cache: EmbeddingCache,
    compute: CountingCompute,
    *,
    prefix: str = "query: ",
    revision: str = REV_A,
) -> np.ndarray:
    return encode_with_cache(
        texts,
        cache=cache,
        model_name=MODEL,
        model_revision=revision,
        prefix=prefix,
        compute=compute,
    )


# --------------------------------------------------------------------------- cache key


def test_cache_key_is_deterministic() -> None:
    assert cache_key_hex(MODEL, REV_A, "hello") == cache_key_hex(MODEL, REV_A, "hello")


def test_cache_key_changes_with_each_key_component() -> None:
    base = cache_key_hex(MODEL, REV_A, "hello")
    assert cache_key_hex(MODEL, REV_A, "hello2") != base
    assert cache_key_hex(MODEL, REV_B, "hello") != base
    assert cache_key_hex("other/model", REV_A, "hello") != base


# --------------------------------------------------------------------------- store


def test_set_then_get_roundtrip_preserves_float32(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    vector = np.arange(DIM, dtype=np.float32)
    cache.set(MODEL, REV_A, "hello", vector)
    hit = cache.get(MODEL, REV_A, "hello")
    assert hit is not None
    assert hit.dtype == np.float32
    np.testing.assert_array_equal(hit, vector)


def test_get_miss_returns_none(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    assert cache.get(MODEL, REV_A, "never-stored") is None


def test_revision_change_misses_the_cache(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    cache.set(MODEL, REV_A, "hello", np.ones(DIM, dtype=np.float32))
    assert cache.get(MODEL, REV_B, "hello") is None
    assert cache.path_for(MODEL, REV_A, "hello") != cache.path_for(MODEL, REV_B, "hello")


def test_file_layout_is_keyed_sha_npy(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    cache.set(MODEL, REV_A, "hello", np.ones(DIM, dtype=np.float32))
    expected = tmp_path / f"{cache_key_hex(MODEL, REV_A, 'hello')}.npy"
    assert expected.is_file()
    assert sorted(p.name for p in tmp_path.glob("*.npy")) == [expected.name]


# --------------------------------------------------------------------------- encode seam


def test_two_identical_calls_hit_the_model_layer_once(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    compute = CountingCompute()
    first = _encode(["hello"], cache, compute)
    second = _encode(["hello"], cache, compute)
    assert len(compute.calls) == 1
    np.testing.assert_array_equal(first, second)
    assert len(list(tmp_path.glob("*.npy"))) == 1


def test_partial_hits_compute_only_missing_texts_in_one_batch(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    compute = CountingCompute()
    _encode(["t1", "t2"], cache, compute)
    compute.calls.clear()
    result = _encode(["t1", "t3", "t2"], cache, compute)
    assert compute.calls == [["query: t3"]]
    assert result.shape == (3, DIM)
    np.testing.assert_array_equal(result[0], _encode(["t1"], cache, CountingCompute())[0])
    np.testing.assert_array_equal(result[2], _encode(["t2"], cache, CountingCompute())[0])


def test_encode_returns_float32_with_document_shape(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    result = _encode(["a", "b"], cache, CountingCompute(), prefix=document_prefix(MODEL))
    assert result.dtype == np.float32
    assert result.shape == (2, DIM)


def test_prefix_reaches_model_but_is_not_part_of_the_key(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path)
    compute = CountingCompute()
    _encode(["hello"], cache, compute, prefix=query_prefix(MODEL))
    assert compute.calls == [["query: hello"]]
    assert cache.path_for(MODEL, REV_A, "hello").is_file()
    _encode(["hello"], cache, compute, prefix="")
    assert len(compute.calls) == 1


def test_encode_with_cache_rejects_empty_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        _encode([], EmbeddingCache(tmp_path), CountingCompute())


# --------------------------------------------------------------------------- root + lazy embedder


def test_hf_cache_root_defaults_to_repo_dot_hf_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_HOME", raising=False)
    expected = Path(cache_module.__file__).resolve().parents[3] / ".hf_cache"
    assert hf_cache_root() == expected


def test_hf_cache_root_respects_hf_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    custom = tmp_path / "custom-hf-home"
    monkeypatch.setenv("HF_HOME", str(custom))
    assert hf_cache_root() == custom


def test_embedder_construction_is_offline_and_lazy() -> None:
    embedder = SentenceTransformerEmbedder(MODEL)
    assert embedder.name == MODEL
    assert embedder.model_revision == "unresolved"
    with pytest.raises(RuntimeError, match="first embed"):
        _ = embedder.dim


def test_embedder_keeps_requested_revision_before_resolution() -> None:
    embedder = SentenceTransformerEmbedder(MODEL, revision="requested-sha")
    assert embedder.model_revision == "requested-sha"
