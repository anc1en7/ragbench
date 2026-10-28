"""OpenAI-compatible LLM client.

All generation and judging goes through this client so any OpenAI-compatible
endpoint (OmniRoute on a private VPS, Ollama, a hosted API) can be swapped in
via LLM_BASE_URL / LLM_API_KEY / LLM_MODEL without code changes (spec 2).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, TypedDict

import httpx

ENV_BASE_URL = "LLM_BASE_URL"
ENV_API_KEY = "LLM_API_KEY"
ENV_MODEL = "LLM_MODEL"

REQUIRED_ENV = (ENV_BASE_URL, ENV_API_KEY, ENV_MODEL)


class LLMConfigError(RuntimeError):
    """Raised when the LLM endpoint is not configured."""


class RateLimitError(RuntimeError):
    """Raised when the endpoint signals a cooldown/rate limit.

    ``retry_after`` is the number of seconds to wait before retrying (best effort).
    """

    def __init__(self, message: str, retry_after: float) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _parse_retry_after(response: httpx.Response) -> float:
    """Best-effort seconds-to-wait from a 429 response (header or JSON body)."""
    header = response.headers.get("retry-after", "")
    try:
        return max(0.0, float(header))
    except ValueError:
        pass
    try:
        data = response.json()
    except ValueError:
        return 60.0
    if isinstance(data, dict):
        for key in ("reset_seconds", "retry_after_seconds"):
            value = data.get(key)
            if isinstance(value, (int, float)):
                return max(0.0, float(value))
        retry_after = data.get("retry_after")
        if isinstance(retry_after, str):
            try:
                from datetime import UTC, datetime

                target = datetime.fromisoformat(retry_after.replace("Z", "+00:00"))
                return max(0.0, (target - datetime.now(UTC)).total_seconds())
            except ValueError:
                pass
    return 60.0


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    api_key: str = field(repr=False)
    model: str

    @property
    def host(self) -> str:
        """Host only - never log or persist the full URL with credentials."""
        return httpx.URL(self.base_url).host or ""


def settings_from_env() -> LLMSettings:
    """Read endpoint configuration from the environment (.env supported)."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # pragma: no cover
        pass
    missing = [k for k in REQUIRED_ENV if not os.getenv(k)]
    if missing:
        raise LLMConfigError(
            f"Missing environment variables: {', '.join(missing)}. "
            "Copy .env.example to .env and fill in your endpoint details."
        )
    values = {k: os.environ[k] for k in REQUIRED_ENV}
    if values[ENV_MODEL] == "changeme" or values[ENV_BASE_URL].endswith("example.com/v1"):
        raise LLMConfigError(
            f"{ENV_BASE_URL} / {ENV_MODEL} still contain placeholder values; "
            "fill in your real endpoint in .env."
        )
    return LLMSettings(
        base_url=values[ENV_BASE_URL].rstrip("/"),
        api_key=values[ENV_API_KEY],
        model=values[ENV_MODEL],
    )


class OpenAICompatClient:
    """Minimal chat-completions client for an OpenAI-compatible endpoint."""

    def __init__(
        self,
        settings: LLMSettings,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.settings = settings
        self._client = httpx.Client(
            base_url=settings.base_url,
            headers={"Authorization": f"Bearer {settings.api_key}"},
            timeout=timeout,
            transport=transport,
        )

    def models(self) -> list[str]:
        response = self._client.get("/models")
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        data = payload.get("data", [])
        return [str(item["id"]) for item in data]

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        body = {
            "model": model or self.settings.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = self._client.post("/chat/completions", json=body)
        if response.status_code == 429:
            raise RateLimitError(
                f"rate limited: {response.text[:120]}",
                retry_after=_parse_retry_after(response),
            )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        content = payload["choices"][0]["message"]["content"]
        return str(content)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> OpenAICompatClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class ProbeResult(TypedDict, total=False):
    models: list[str]
    chat_ok: bool
    error: str
    test_model: str
    test_reply: str


def probe_endpoint(
    base_url: str,
    api_key: str,
    *,
    transport: httpx.BaseTransport | None = None,
) -> ProbeResult:
    """List models and verify a chat completion works. Returns a JSON-safe dict."""
    client = OpenAICompatClient(
        LLMSettings(base_url=base_url.rstrip("/"), api_key=api_key, model=""),
        transport=transport,
    )
    try:
        models = client.models()
    except httpx.HTTPError as exc:
        return {"models": [], "chat_ok": False, "error": f"GET /models failed: {exc}"}
    if not models:
        return {"models": [], "chat_ok": False, "error": "endpoint returned no models"}
    first = models[0]
    try:
        reply = client.chat(
            [{"role": "user", "content": "Reply with the single word: ok"}],
            model=first,
            max_tokens=8,
        )
    except httpx.HTTPError as exc:
        return {
            "models": models,
            "chat_ok": False,
            "error": f"POST /chat/completions failed: {exc}",
        }
    return {
        "models": models,
        "chat_ok": True,
        "test_model": first,
        "test_reply": reply.strip()[:80],
    }
