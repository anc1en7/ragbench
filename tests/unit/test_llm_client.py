import httpx
import pytest

from ragbench.llm_client import (
    LLMConfigError,
    LLMSettings,
    OpenAICompatClient,
    probe_endpoint,
    settings_from_env,
)


def _mock_transport(handler):
    return httpx.MockTransport(handler)


def test_settings_from_env_rejects_missing(monkeypatch):
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: None)
    for var in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(LLMConfigError, match="LLM_BASE_URL"):
        settings_from_env()


def test_settings_from_env_rejects_placeholders(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_BASE_URL", "https://your-vps.example.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "abc")
    monkeypatch.setenv("LLM_MODEL", "changeme")
    with pytest.raises(LLMConfigError, match="placeholder"):
        settings_from_env()


def test_settings_host_never_contains_credentials(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://llm.internal.example/v1/")
    monkeypatch.setenv("LLM_API_KEY", "secret-key")
    monkeypatch.setenv("LLM_MODEL", "prov/model-x")
    s = settings_from_env()
    assert s.host == "llm.internal.example"
    assert s.base_url == "https://llm.internal.example/v1"
    assert "secret" not in repr(s)


def test_models_and_chat_with_mock_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(
                200, json={"data": [{"id": "prov/model-a"}, {"id": "prov/model-b"}]}
            )
        if request.url.path.endswith("/chat/completions"):
            body = request.read()
            assert b"prov/model-a" in body
            return httpx.Response(
                200,
                json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]},
            )
        return httpx.Response(404)

    client = OpenAICompatClient(
        LLMSettings("https://x.test/v1", "key", "prov/model-a"),
        transport=_mock_transport(handler),
    )
    assert client.models() == ["prov/model-a", "prov/model-b"]
    reply = client.chat([{"role": "user", "content": "hi"}])
    assert reply == "ok"


def test_chat_surfaces_http_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad key"})

    client = OpenAICompatClient(
        LLMSettings("https://x.test/v1", "bad", "m"),
        transport=_mock_transport(handler),
    )
    with pytest.raises(httpx.HTTPStatusError):
        client.chat([{"role": "user", "content": "hi"}])


def test_probe_endpoint_success():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "prov/good"}]})
        return httpx.Response(
            200, json={"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
        )

    result = probe_endpoint("https://x.test/v1", "k", transport=_mock_transport(handler))
    assert result["chat_ok"] is True
    assert result["models"] == ["prov/good"]


def test_probe_endpoint_chat_failure():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": "prov/good"}]})
        return httpx.Response(500)

    result = probe_endpoint("https://x.test/v1", "k", transport=_mock_transport(handler))
    assert result["chat_ok"] is False
    assert "error" in result


def test_probe_endpoint_unreachable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    result = probe_endpoint("https://x.test/v1", "k", transport=_mock_transport(handler))
    assert result["chat_ok"] is False
    assert result["models"] == []
