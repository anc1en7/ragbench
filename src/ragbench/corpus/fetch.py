"""Fetch the pinned FastAPI documentation corpus.

The corpus is locked to a specific release tag and commit SHA in ``corpus.lock``
(spec 2). Only English markdown under ``docs/en/docs`` is extracted into
``data/raw/`` (gitignored); ``corpus.lock`` itself is committed.
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import httpx

DEFAULT_REPO = "https://github.com/fastapi/fastapi"
GITHUB_API = "https://api.github.com"
DOCS_PATH = "docs/en/docs"
LICENSE_NAME = "LICENSE"
REQUIRED_LICENSE = "MIT License"


class CorpusFetchError(RuntimeError):
    """Raised when the corpus cannot be fetched or verified."""


@dataclass(frozen=True)
class CorpusLock:
    repo: str
    tag: str
    commit_sha: str
    docs_path: str
    license: str
    fetched_at: str
    file_count: int
    sha256_of_files: str

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> CorpusLock:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            repo=data["repo"],
            tag=data["tag"],
            commit_sha=data["commit_sha"],
            docs_path=data["docs_path"],
            license=data["license"],
            fetched_at=data["fetched_at"],
            file_count=int(data["file_count"]),
            sha256_of_files=data["sha256_of_files"],
        )


def fingerprint(files: dict[str, bytes]) -> str:
    """Deterministic sha256 over sorted relative paths and contents."""
    digest = hashlib.sha256()
    for rel in sorted(files):
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(files[rel])
        digest.update(b"\n")
    return digest.hexdigest()


def _split_repo(repo: str) -> tuple[str, str]:
    parts = PurePosixPath(httpx.URL(repo).path).parts
    if len(parts) < 3:  # noqa: PLR2004
        raise CorpusFetchError(f"cannot parse owner/name from repo URL: {repo}")
    return parts[1], parts[2].removesuffix(".git")


def _default_client() -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": "ragbench-corpus-fetch"},
        follow_redirects=True,
        timeout=120.0,
    )


def resolve_tag(client: httpx.Client, repo: str) -> tuple[str, str]:
    """Return the latest stable release ``(tag, commit_sha)`` for the repo."""
    owner, name = _split_repo(repo)
    api = f"{GITHUB_API}/repos/{owner}/{name}"
    release = client.get(f"{api}/releases/latest")
    release.raise_for_status()
    tag = str(release.json()["tag_name"])
    ref = client.get(f"{api}/git/ref/tags/{tag}")
    ref.raise_for_status()
    obj: dict[str, Any] = ref.json()["object"]
    if obj["type"] == "tag":  # annotated tag: dereference once
        tag_obj = client.get(f"{GITHUB_API}/repos/{owner}/{name}/git/tags/{obj['sha']}")
        tag_obj.raise_for_status()
        obj = dict(tag_obj.json()["object"])
    return tag, str(obj["sha"])


def _strip_prefix(name: str) -> tuple[str, ...]:
    """GitHub tarballs nest everything under one ``owner-repo-sha/`` directory."""
    parts = PurePosixPath(name).parts
    return parts[1:] if len(parts) > 1 else parts


def extract_docs(tarball: bytes, docs_path: str) -> tuple[dict[str, bytes], str]:
    """Return ``(markdown files keyed by path relative to docs_path, license text)``."""
    docs_parts = tuple(docs_path.split("/"))
    files: dict[str, bytes] = {}
    license_text = ""
    with tarfile.open(fileobj=io.BytesIO(tarball), mode="r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            rel_parts = _strip_prefix(member.name)
            if rel_parts == (LICENSE_NAME,):
                handle = tar.extractfile(member)
                if handle is not None:
                    license_text = handle.read().decode("utf-8", errors="replace")
            if len(rel_parts) <= len(docs_parts) or rel_parts[: len(docs_parts)] != docs_parts:
                continue
            rel = rel_parts[len(docs_parts) :]
            if ".." in rel or not rel[-1].endswith(".md"):
                continue
            handle = tar.extractfile(member)
            if handle is None:  # pragma: no cover
                continue
            files["/".join(rel)] = handle.read()
    if not license_text:
        raise CorpusFetchError("corpus license file (LICENSE) missing from tarball")
    if REQUIRED_LICENSE not in license_text:
        raise CorpusFetchError(
            f"corpus license is not {REQUIRED_LICENSE!r}; human review required before reuse"
        )
    return files, license_text


def _read_raw(raw_dir: Path) -> dict[str, bytes]:
    if not raw_dir.is_dir():
        return {}
    return {
        path.relative_to(raw_dir).as_posix(): path.read_bytes()
        for path in sorted(raw_dir.rglob("*.md"))
        if path.is_file()
    }


def _write_raw(raw_dir: Path, files: dict[str, bytes]) -> None:
    if raw_dir.exists():
        for path in sorted(raw_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    raw_dir.mkdir(parents=True, exist_ok=True)
    for rel, content in files.items():
        target = raw_dir.joinpath(*rel.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


def fetch(
    repo: str = DEFAULT_REPO,
    *,
    lock_path: Path = Path("corpus.lock"),
    raw_dir: Path = Path("data/raw"),
    force: bool = False,
    client: httpx.Client | None = None,
) -> CorpusLock | None:
    """Download the pinned corpus into ``raw_dir`` and maintain ``lock_path``.

    Returns ``None`` when the corpus on disk already matches the lock (no-op).
    """
    owns_client = client is None
    client = client or _default_client()
    try:
        existing = CorpusLock.read(lock_path) if lock_path.exists() else None
        on_disk = _read_raw(raw_dir)
        if existing and not force and on_disk and fingerprint(on_disk) == existing.sha256_of_files:
            return None
        if existing and not force:
            tag, sha = existing.tag, existing.commit_sha
        else:
            tag, sha = resolve_tag(client, repo)
        owner, name = _split_repo(repo)
        tarball = client.get(f"{GITHUB_API}/repos/{owner}/{name}/tarball/{tag}")
        tarball.raise_for_status()
        files, _license_text = extract_docs(tarball.content, DOCS_PATH)
        if not files:
            raise CorpusFetchError(f"no markdown found under {DOCS_PATH} at tag {tag}")
        digest = fingerprint(files)
        if existing and not force and digest != existing.sha256_of_files:
            raise CorpusFetchError(
                f"corpus drift: re-fetched tag {tag} does not match corpus.lock sha256"
            )
        _write_raw(raw_dir, files)
        if existing and not force:
            return existing
        lock = CorpusLock(
            repo=repo,
            tag=tag,
            commit_sha=sha,
            docs_path=DOCS_PATH,
            license="MIT",
            fetched_at=datetime.now(UTC).isoformat(),
            file_count=len(files),
            sha256_of_files=digest,
        )
        lock.write(lock_path)
        return lock
    finally:
        if owns_client:
            client.close()
