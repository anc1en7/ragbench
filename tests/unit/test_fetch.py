import io
import json
import tarfile

import httpx

from ragbench.corpus.fetch import CorpusFetchError, fetch, fingerprint

PREFIX = "fastapi-0.162.0-abc123"


def _make_tarball(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


TARBALL = _make_tarball(
    {
        f"{PREFIX}/docs/en/docs/index.md": b"# FastAPI documentation\n",
        f"{PREFIX}/docs/en/docs/tutorial/first-steps.md": b"# First steps\n",
        f"{PREFIX}/docs/en/docs/tutorial/__init__.py.txt": b"not markdown\n",
        f"{PREFIX}/README.md": b"outside the corpus\n",
        f"{PREFIX}/LICENSE": b"MIT License\n\nCopyright (c) 2018 Sebastiian Ramirez\n",
    }
)

TAG = "0.162.0"
SHA = "abc123def456"


def _handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path == "/repos/fastapi/fastapi/releases/latest":
        return httpx.Response(200, json={"tag_name": TAG})
    if path == f"/repos/fastapi/fastapi/git/ref/tags/{TAG}":
        return httpx.Response(200, json={"object": {"sha": SHA, "type": "commit"}})
    if path == f"/repos/fastapi/fastapi/tarball/{TAG}":
        return httpx.Response(200, content=TARBALL, headers={"content-type": "application/gzip"})
    return httpx.Response(404, json={"message": "not found"})


def _client() -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(_handler),
        headers={"User-Agent": "ragbench-test"},
        follow_redirects=True,
    )


def test_fetch_writes_docs_and_lock(tmp_path):
    raw_dir = tmp_path / "raw"
    lock_path = tmp_path / "corpus.lock"
    lock = fetch(
        "https://github.com/fastapi/fastapi",
        lock_path=lock_path,
        raw_dir=raw_dir,
        client=_client(),
    )
    assert lock is not None
    assert lock.tag == TAG
    assert lock.commit_sha == SHA
    assert lock.license == "MIT"
    assert lock.docs_path == "docs/en/docs"
    assert lock.file_count == 2
    assert (raw_dir / "index.md").read_bytes() == b"# FastAPI documentation\n"
    assert (raw_dir / "tutorial" / "first-steps.md").exists()
    assert not (raw_dir / "README.md").exists()
    assert not (raw_dir / "tutorial" / "__init__.py.txt").exists()
    on_disk = json.loads(lock_path.read_text(encoding="utf-8"))
    assert on_disk["tag"] == TAG
    assert on_disk["sha256_of_files"] == lock.sha256_of_files


def test_fetch_is_idempotent_second_run_noop(tmp_path):
    raw_dir = tmp_path / "raw"
    lock_path = tmp_path / "corpus.lock"
    first = fetch(
        "https://github.com/fastapi/fastapi",
        lock_path=lock_path,
        raw_dir=raw_dir,
        client=_client(),
    )
    assert first is not None
    requests: list[str] = []

    def counting_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        return _handler(request)

    client = httpx.Client(
        transport=httpx.MockTransport(counting_handler),
        headers={"User-Agent": "ragbench-test"},
        follow_redirects=True,
    )
    second = fetch(
        "https://github.com/fastapi/fastapi",
        lock_path=lock_path,
        raw_dir=raw_dir,
        client=client,
    )
    assert second is None
    assert requests == []


def test_fetch_restores_missing_raw_dir(tmp_path):
    raw_dir = tmp_path / "raw"
    lock_path = tmp_path / "corpus.lock"
    fetch(
        "https://github.com/fastapi/fastapi",
        lock_path=lock_path,
        raw_dir=raw_dir,
        client=_client(),
    )
    for path in sorted(raw_dir.rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    requests: list[str] = []

    def counting_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        return _handler(request)

    client = httpx.Client(
        transport=httpx.MockTransport(counting_handler),
        headers={"User-Agent": "ragbench-test"},
        follow_redirects=True,
    )
    restored = fetch(
        "https://github.com/fastapi/fastapi",
        lock_path=lock_path,
        raw_dir=raw_dir,
        client=client,
    )
    assert restored is None or restored.tag == TAG
    assert (raw_dir / "index.md").exists()
    assert requests  # re-downloaded the pinned tarball, no API re-resolve
    assert all("releases/latest" not in p for p in requests)


def test_fingerprint_is_order_independent():
    a = {"b.md": b"two", "a.md": b"one"}
    b = {"a.md": b"one", "b.md": b"two"}
    assert fingerprint(a) == fingerprint(b)
    assert fingerprint(a) != fingerprint({"a.md": b"one", "b.md": b"three"})


def test_fetch_rejects_missing_license(tmp_path):
    bad_tarball = _make_tarball({f"{PREFIX}/docs/en/docs/index.md": b"# x\n"})

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/repos/fastapi/fastapi/releases/latest":
            return httpx.Response(200, json={"tag_name": TAG})
        if path == f"/repos/fastapi/fastapi/git/ref/tags/{TAG}":
            return httpx.Response(200, json={"object": {"sha": SHA, "type": "commit"}})
        if path == f"/repos/fastapi/fastapi/tarball/{TAG}":
            return httpx.Response(200, content=bad_tarball)
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler), headers={"User-Agent": "t"})
    try:
        fetch(
            "https://github.com/fastapi/fastapi",
            lock_path=tmp_path / "corpus.lock",
            raw_dir=tmp_path / "raw",
            client=client,
        )
    except CorpusFetchError as exc:
        assert "license" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("expected CorpusFetchError for missing license")
