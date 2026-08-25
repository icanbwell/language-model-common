"""Shared SSRF-prevention helpers for outbound URL construction.

Vulnerability class: Server-Side Request Forgery (SSRF). Any time a URL that
is (directly or indirectly, e.g. via an environment variable set by a
deployment pipeline) influenced by request data or external configuration is
used to make an outbound HTTP/MCP connection, the destination must be
validated before the connection is opened. Without this, an attacker who can
influence the URL (or a misconfiguration) can make the server issue requests
to internal-only services, cloud metadata endpoints (e.g.
``169.254.169.254``), or loopback addresses that would otherwise be
unreachable from outside the network.

This mirrors ``baileyai/utilities/security/url_validation.py`` (which
originated this logic). It's duplicated here, rather than baileyai importing
a single copy, because dependency direction only runs one way -- baileyai
depends on this package, not the reverse. Keep the two in sync if either
changes; both are intentionally stdlib-only so there's little to drift.
"""

from __future__ import annotations

import ipaddress
import socket
import urllib.parse

DEFAULT_ALLOWED_SCHEMES: tuple[str, ...] = ("http", "https")


def host_matches_allowlist(*, host: str, allowlist: list[str]) -> bool:
    """Return True when ``host`` matches one of the allowlist entries.

    Each entry is either an exact hostname or a ``*.suffix`` wildcard.
    Comparison is case-insensitive. An empty allowlist matches nothing
    (fail-closed).
    """
    if not allowlist or not host:
        return False
    lowered = host.lower().strip("[]")
    for entry in allowlist:
        entry = entry.lower()
        if entry.startswith("*."):
            suffix = entry[1:]  # ".example.com"
            if lowered == suffix[1:] or lowered.endswith(suffix):
                return True
        elif lowered == entry:
            return True
    return False


# RFC 6598 "Shared Address Space" (a.k.a. CGNAT range), 100.64.0.0/10.
# Python's ipaddress module does NOT classify this as private/reserved (it's
# absent from CPython's _private_networks list), but it's routinely used for
# instance-metadata-style endpoints on networks that need the SSRF
# protection here just as much as RFC 1918 space does -- e.g. Alibaba
# Cloud's instance metadata service listens on 100.100.100.200, inside this
# range, as an analogue to AWS's 169.254.169.254.
_CGNAT_RANGE = ipaddress.ip_network("100.64.0.0/10")


def is_blocked_host(*, host: str) -> bool:
    """Reject hosts that point at internal/loopback/metadata addresses.

    Catches literal-IP SSRF (``http://169.254.169.254/``, ``http://10.0.0.1/``),
    obfuscated IPv4 encodings (decimal ``http://2130706433/``, hex
    ``http://0x7f000001/``, octal ``http://0177.0.0.1/``, short forms like
    ``http://127.1/``), IPv4-mapped IPv6 (``::ffff:127.0.0.1``), CGNAT/RFC
    6598 shared address space (``100.64.0.0/10``, e.g. Alibaba Cloud's
    ``100.100.100.200`` metadata endpoint), and the obvious localhost names.

    Does NOT resolve DNS -- a hostname like ``evil.example`` that resolves
    to ``127.0.0.1`` will pass through here and must be caught at the
    HTTP-client layer (DNS pinning) for a complete SSRF defense.
    """
    if not host:
        return True
    lowered = host.lower().strip("[]")
    if lowered == "localhost" or lowered.endswith(".localhost"):
        return True

    try:
        ip = ipaddress.ip_address(lowered)
    except ValueError:
        ip = None

    if ip is None:
        # The canonical parser rejected the host, but it might still be an
        # obfuscated IPv4 (decimal, hex, octal, short form). socket.inet_aton
        # accepts those legacy forms and is what HTTP clients/glibc use to
        # resolve them -- so if it parses, treat the result as the real IP.
        try:
            packed = socket.inet_aton(lowered)
        except OSError:
            return False
        ip = ipaddress.IPv4Address(packed)

    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped

    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
        or (isinstance(ip, ipaddress.IPv4Address) and ip in _CGNAT_RANGE)
    )


def parse_and_check_host(
    *,
    url: str,
    allowed_schemes: tuple[str, ...] = DEFAULT_ALLOWED_SCHEMES,
) -> str | None:
    """Parse ``url`` and return its hostname iff scheme is allowed and the
    host is not a blocked (internal/loopback/metadata) address.

    Returns ``None`` for any of: an unparseable URL (some distro-patched
    CPython builds raise ``ValueError`` on edge-case IPv6 URLs like
    ``http://[::1]/...``), a disallowed scheme, or a blocked host. Callers
    should treat ``None`` as "reject this URL" and fail closed (skip the
    connection / degrade gracefully) -- never fall back to using the raw,
    unvalidated URL.

    Does not consult any allowlist; combine with ``host_matches_allowlist``
    when the caller also wants to restrict to a specific set of hosts.
    """
    try:
        parsed = urllib.parse.urlparse(url)
        host = parsed.hostname or ""
    except ValueError:
        return None

    if parsed.scheme not in allowed_schemes:
        return None

    if is_blocked_host(host=host):
        return None

    return host
