"""Public-only research fetches; trusted MCP/A2A connections use separate policies.

Validated IPs are pinned into the request URL, with the original Host and TLS
server name retained. Proxies are disabled. Deployment egress filtering remains
necessary defense in depth. The total timeout cancels local waiting, including
DNS; an OS resolver operation already running in a thread may finish later.
"""

import asyncio
import ipaddress
import socket

import httpx

MAX_REDIRECTS = 5


def _public_address(value: str) -> str:
    address = ipaddress.ip_address(value)
    effective = getattr(address, "ipv4_mapped", None) or address
    if (not effective.is_global or effective.is_multicast or effective.is_reserved
            or getattr(address, "scope_id", None) or getattr(address, "sixtofour", None)
            or getattr(address, "teredo", None)):
        raise ValueError("Destination must be a public unicast address")
    return str(address)


async def _resolve_public(host: str, port: int) -> str:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        records = await asyncio.get_running_loop().getaddrinfo(
            host, port, type=socket.SOCK_STREAM
        )
        addresses = [_public_address(str(record[4][0])) for record in records]
        if not addresses:
            raise ValueError("Destination has no addresses") from None
        # Reject the entire answer if any address is disallowed. Only this
        # numeric address is handed to the transport (no second hostname lookup).
        return addresses[0]
    return _public_address(host)


async def fetch_public_url(url: str, timeout: float, max_bytes: int) -> str:
    """Read at most max_bytes raw body bytes under one cancellable deadline."""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    async with asyncio.timeout_at(deadline):
        current = httpx.URL(url)
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=False, trust_env=False,
            # Distinct hostnames can resolve to the same pinned IP. Never reuse
            # a TLS connection verified for a previous redirect's hostname.
            limits=httpx.Limits(max_keepalive_connections=0),
        ) as client:
            for hop in range(MAX_REDIRECTS + 1):
                if (current.scheme not in ("http", "https") or not current.host
                        or current.userinfo or "%" in current.host):
                    raise ValueError("Only public HTTP(S) URLs without credentials are allowed")
                port = current.port or (443 if current.scheme == "https" else 80)
                address = await _resolve_public(current.host, port)
                target = current.copy_with(host=address)
                headers = {
                    "Host": current.netloc.decode("ascii"),
                    "Accept-Encoding": "identity",
                }
                async with client.stream(
                    "GET", target, headers=headers,
                    extensions={"sni_hostname": current.raw_host.decode("ascii")},
                ) as response:
                    if response.status_code in (301, 302, 303, 307, 308):
                        if hop == MAX_REDIRECTS:
                            raise ValueError("Too many redirects")
                        location = response.headers.get("location")
                        if not location:
                            raise ValueError("Redirect is missing Location")
                        current = current.join(location)
                        continue
                    response.raise_for_status()
                    if response.headers.get("content-encoding", "identity").lower() != "identity":
                        raise ValueError("Encoded responses are not supported")
                    body = bytearray()
                    async for chunk in response.aiter_raw():
                        if loop.time() >= deadline:
                            raise TimeoutError("total fetch deadline exceeded")
                        remaining = max_bytes - len(body)
                        body.extend(chunk[:remaining])
                        if len(body) >= max_bytes:
                            text = body.decode("utf-8", errors="replace")
                            return text + f"\n[truncated: response reached {max_bytes} byte limit]"
                    return body.decode("utf-8", errors="replace")
    raise ValueError("No response received")
