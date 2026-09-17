"""Defensive fetch policy checks: mocked DNS and transport, no network access."""

import asyncio
import socket

import httpx
import pytest

from src.core import research_fetch as fetch


class Body(httpx.AsyncByteStream):
    def __init__(self, chunks=(), stall=False):
        self.chunks = chunks
        self.stall = stall
        self.consumed = 0
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            self.consumed += 1
            yield chunk
        if self.stall:
            await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


@pytest.fixture
def transport(monkeypatch):
    requests = []
    responses = []
    original = httpx.AsyncClient

    def handle(request):
        requests.append(request)
        return responses.pop(0)

    def client(**kwargs):
        assert kwargs["trust_env"] is False
        assert kwargs["follow_redirects"] is False
        assert kwargs["limits"].max_keepalive_connections == 0
        return original(**kwargs, transport=httpx.MockTransport(handle))

    async def resolve(host, port, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port))]

    monkeypatch.setattr(httpx, "AsyncClient", client)
    monkeypatch.setattr(
        asyncio.BaseEventLoop, "getaddrinfo", lambda self, *a, **k: resolve(*a, **k)
    )
    return requests, responses


def run(url="https://example.test/page", timeout=1, size=10):
    return asyncio.run(fetch.fetch_public_url(url, timeout, size))


def test_pins_destination_and_preserves_host_and_tls(transport):
    requests, responses = transport
    responses.append(httpx.Response(200, stream=Body([b"hello"])))
    assert run() == "hello"
    request = requests[0]
    assert request.url.host == "93.184.216.34"
    assert request.headers["Host"] == "example.test"
    assert request.extensions["sni_hostname"] == "example.test"
    assert request.headers["Accept-Encoding"] == "identity"


@pytest.mark.parametrize("address", [
    "127.0.0.1", "10.0.0.1", "169.254.1.1", "192.168.1.1", "0.0.0.0",
    "224.0.0.1", "::1", "fc00::1", "fe80::1", "::ffff:127.0.0.1", "100.64.0.1",
])
def test_denies_nonpublic_addresses(address):
    with pytest.raises(ValueError, match="public"):
        fetch._public_address(address)


@pytest.mark.parametrize("url", ["file:///fixture", "https://user:pass@example.test", "http://"])
def test_invalid_url_never_reaches_transport(transport, url):
    with pytest.raises((ValueError, httpx.InvalidURL)):
        run(url)
    assert not transport[0]


def test_mixed_dns_answer_is_rejected(monkeypatch, transport):
    async def resolve(self, *args, **kwargs):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))
                for ip in ("93.184.216.34", "127.0.0.1")]
    monkeypatch.setattr(asyncio.BaseEventLoop, "getaddrinfo", resolve)
    with pytest.raises(ValueError):
        run()
    assert not transport[0]


def test_redirect_policy_reapplies_before_connection(transport):
    requests, responses = transport
    responses.append(httpx.Response(302, headers={"location": "http://127.0.0.1/fixture"}))
    with pytest.raises(ValueError):
        run()
    assert len(requests) == 1


def test_public_cross_host_and_relative_redirects(transport):
    requests, responses = transport
    responses.extend([
        httpx.Response(302, headers={"location": "https://other.test/a"}),
        httpx.Response(307, headers={"location": "/b"}),
        httpx.Response(200, stream=Body([b"ok"])),
    ])
    assert run() == "ok"
    assert requests[-1].headers["host"] == "other.test"
    assert requests[-1].url.path == "/b"


def test_redirect_loop_is_bounded(transport):
    requests, responses = transport
    responses.extend(httpx.Response(302, headers={"location": "/again"})
                     for _ in range(fetch.MAX_REDIRECTS + 1))
    with pytest.raises(ValueError, match="redirects"):
        run()
    assert len(requests) == fetch.MAX_REDIRECTS + 1


def test_limit_stops_consumption_and_closes(transport):
    body = Body([b"123456", b"789012", b"never read"])
    transport[1].append(httpx.Response(200, stream=body))
    result = run()
    assert result.startswith("1234567890\n[truncated")
    assert body.consumed == 2
    assert body.closed


def test_encoded_body_is_not_consumed(transport):
    body = Body([b"not decompressed"])
    transport[1].append(httpx.Response(200, headers={"content-encoding": "gzip"}, stream=body))
    with pytest.raises(ValueError, match="Encoded"):
        run()
    assert body.consumed == 0 and body.closed


def test_deadline_interrupts_stalled_body(transport):
    body = Body(stall=True)
    transport[1].append(httpx.Response(200, stream=body))
    with pytest.raises(TimeoutError):
        run(timeout=0.02)
    assert body.closed


def test_deadline_interrupts_stalled_connect(monkeypatch):
    original = httpx.AsyncClient

    async def stalled(request):
        await asyncio.Event().wait()

    monkeypatch.setattr(
        httpx, "AsyncClient",
        lambda **kwargs: original(**kwargs, transport=httpx.MockTransport(stalled)),
    )
    with pytest.raises(TimeoutError):
        run("https://93.184.216.34/fixture", timeout=0.02)


def test_deadline_includes_dns(monkeypatch, transport):
    async def stalled(self, *args, **kwargs):
        await asyncio.Event().wait()
    monkeypatch.setattr(asyncio.BaseEventLoop, "getaddrinfo", stalled)
    with pytest.raises(TimeoutError):
        run(timeout=0.02)
    assert not transport[0]
