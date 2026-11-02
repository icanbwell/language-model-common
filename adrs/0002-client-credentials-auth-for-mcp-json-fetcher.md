# Add client-credentials (M2M) auth to McpJsonFetcher

* Status: proposed
* Deciders: language-model-common maintainers, baileyai maintainers, baileyai-skills-service maintainers
* Ticket: BAI-714
* Date: 2026-09-13

## Context and Problem Statement

`baileyai` cannot resolve MCP servers for the `bailey` plugin (`skills-library`,
`plugin-marketplace`, `skills-publisher`, `fhir-server`) in **every**
environment (dev, staging, prod, client-sandbox), confirmed in Groundcover
logs from 2026-09-11 through today. `bailey.json`'s wildcard config
(`"mcp_server": "*"`, `"skills": ["*"]`) depends on `ConfigReader` resolving
those wildcards via `McpJsonFetcher.fetch_plugins_async(["bailey"])`, which
calls the `get_mcp_servers_config` MCP tool on `baileyai-skills-service`'s
`plugin-marketplace` server. Every call is rejected:

```
get_mcp_servers_config returned an error for plugin 'bailey':
Rejected: get_mcp_servers_config requires a verified access token.
```

### Root cause

`baileyai-skills-service` PR #86 (2026-08-20, "fix: close 3 IDOR/broken-access-control
findings in MCP servers") closed a CVSS 5.3 finding: `PluginMarketplaceMCP`
had no auth provider wired in at all, unlike `SkillsMCP`/`SkillsPublisherMCP`.
`get_mcp_servers_config` specifically now calls
`_require_authenticated_caller`, rejecting any call without a token that
`AuthProvider.verify_token()` accepts — correctly, since its response can
carry embedded per-server credentials (`env`/`headers`/`auth`/`oauth` blocks;
see `serializers.py::entry_to_dict`).

`McpJsonFetcher` (this repo,
`configs/config_reader/mcp_json_fetcher.py`) was never updated to send an
`Authorization` header — its `MCPConnectionConfig` carries only `url` and
`transport`. It runs during `ConfigReader`'s bootstrap/background config
resolution, before any user request or token exists (there is no per-request
context to pass through or exchange). It used to work only because the
server-side endpoint was, until 2026-08-20, unauthenticated by mistake — that
absence of auth was itself the vulnerability the PR fixed. Nobody updated the
client side when the hole was closed.

### Why existing auth mechanisms in this codebase don't cover this case

This codebase has two auth patterns today, both scoped to a real end-user
request:

1. **Pass-through / token exchange** (`auth/pass_through_token_manager.py`,
   `auth/token_exchange/token_exchange_manager.py`, `AgentConfig.auth: "jwt_token"`
   + `auth_providers`) — forwards or exchanges the *calling user's* token.
   Requires a user token to already exist.
2. **DCR (Dynamic Client Registration)** (`oidcauthlib.auth.dcr.*`) — lets a
   new OAuth client register itself, still ultimately backing a
   user-facing authorization-code flow.

`McpJsonFetcher`'s call happens at config-load time, decoupled from any
specific user request — there is no token to pass through or exchange. This
is a genuinely new case: pure service-to-service (M2M) authentication with no
human in the loop.

## Decision Drivers

- Must not reintroduce the CVSS 5.3 hole (no reverting to anonymous access,
  no shared static secret as a substitute for OAuth — the org's baseline
  requires OAuth/OIDC for all auth flows, no custom schemes).
- Must not require a real user/session to exist — this runs at background
  config-resolution time, potentially before any request has ever hit
  baileyai.
- Reuse existing building blocks over introducing new ones.
- Minimize blast radius: `language-model-common` is consumed by every
  service on this platform; a bug here is not isolated to one repo.

## Options Considered

### Option A — OAuth2 client-credentials grant (recommended)

Standard M2M OAuth pattern: `baileyai` (or `language-model-common` on its
behalf) is registered as a confidential client with its own `client_id`/
`client_secret` in the same IdP `baileyai-skills-service` already trusts.
`McpJsonFetcher` exchanges those credentials for an access token via the
provider's `token_endpoint` (`grant_type=client_credentials`), caches it
until near-expiry, and attaches it as `Authorization: Bearer <token>`.

**What already exists and can be reused, unmodified:**
- `oidcauthlib.auth.config.auth_config.AuthConfig` already has `client_id`
  and `client_secret` fields (used today for DCR's upstream client).
- `oidcauthlib.auth.well_known_configuration.auth_server_metadata_discovery.AuthServerMetadataDiscovery`
  already discovers `token_endpoint` from the provider's OIDC metadata.
- `baileyai-skills-service`'s `TokenReaderVerifier` validates any
  correctly-signed token from the configured issuer/audience — it does not
  inspect grant type. A client-credentials token from the same IdP should
  pass verification unmodified, *provided* it isn't rejected for lacking a
  `sub` claim (client-credentials tokens typically carry only `client_id`) —
  this needs to be checked against `TokenReaderVerifier`'s actual claim
  handling before shipping (see Open Questions).

**What's missing (net-new):**
- Nothing in `oidcauthlib` currently performs the `grant_type=client_credentials`
  token request itself. This needs to be added — a single, focused capability
  (e.g. a `ClientCredentialsTokenProvider`), not a new library or vendor.
- A new OAuth client, with client-credentials grant enabled, needs to be
  registered in the IdP baileyai-skills-service trusts. This is an infra/ops
  step, not code.

**Pros:** compliant with the org's OAuth/OIDC-only auth policy; reuses
`AuthConfig`/`AuthServerMetadataDiscovery` as-is; symmetrical with how
DCR already authenticates a client to the same class of IdP.
**Cons:** genuinely new pattern in this codebase (first pure M2M flow) —
crosses `oidcauthlib`, `language-model-common`, and an IdP registration step
outside code; needs security/EA visibility per the org's "is it new" test.

### Option B — Shared static secret / API key

`McpJsonFetcher` presents a pre-shared secret (e.g. a header) instead of an
OAuth token; `get_mcp_servers_config` accepts it as an alternate credential.

**Rejected.** Violates the org's baseline directly ("All auth flows use
OAuth/OIDC. No custom authentication schemes."). Also weaker in practice:
a static secret has no expiry, no standard rotation story, and — unless
built with equal care (secret manager storage, rotation, audit) — trades a
verified-token check for a bearer-of-secret check, undermining the intent of
the CVSS 5.3 fix it would sit alongside.

### Option C — Exempt background/bootstrap calls from auth (network-level trust)

Let `get_mcp_servers_config` skip the auth check for calls that arrive over
some other trust signal (e.g. an internal-only network path, mTLS, or a
service-mesh identity) instead of a bearer token.

**Rejected for now.** Re-opens exactly the class of finding PR #86 closed if
the internal network isn't as segmented as assumed (the CVSS 5.3 finding was
that *any* caller with network reachability had access). Also requires
infra guarantees (network policy / mTLS) that don't currently exist in this
stack and would need their own review. Worth revisiting only if the org
adopts a service-mesh identity story platform-wide — out of scope here.

### Option D — Do nothing; keep MCP servers unresolved

Leave `bailey.json`'s wildcard tools/skills unresolved. Not viable — Bailey
has no access to skills-library tools at all in this state, which is the
production-impacting symptom that prompted this ADR.

## Recommendation

**Option A.** Sequence:

1. `oidcauthlib` — add a client-credentials token provider: given an
   `AuthConfig` with `client_id`/`client_secret` populated, discover
   `token_endpoint` (reusing `AuthServerMetadataDiscovery`), POST
   `grant_type=client_credentials`, cache the token until shortly before
   `expires_in` elapses.
2. `language-model-common` — `McpJsonFetcher` accepts this token provider
   (or the `AuthConfig` needed to build one) and attaches
   `Authorization: Bearer <token>` to the `MCPConnectionConfig` used for
   `get_mcp_servers_config` / `get_mcp_servers_config` (all-plugins) calls.
   Config wiring: a new env var pair analogous to the existing
   `AUTH_CLIENT_ID_<PROVIDER>`/`AUTH_WELL_KNOWN_URI_<PROVIDER>` convention,
   scoped to this specific M2M credential (exact naming TBD during
   implementation — should not collide with per-provider pass-through auth
   config already defined for user-facing flows).
3. Infra — register a new confidential OAuth client (client-credentials
   grant only, no interactive flows) in the IdP `baileyai-skills-service`
   trusts, scoped narrowly (e.g. no more than whatever
   `get_mcp_servers_config` itself checks for — currently just
   "is this a verified token," not a specific scope).
4. `baileyai-skills-service` — verify (test, not assumed) that
   `TokenReaderVerifier`/`_require_authenticated_caller` accept a
   client-credentials token (no `sub` claim) from the newly-registered
   client. If `AUTH_REQUIRED_SCOPES` is set on that deployment, confirm the
   new client's token carries a satisfying scope, or that
   `get_mcp_servers_config`'s check is scope-agnostic (it is, today — see
   Open Questions for whether that should change).

## Consequences

- Positive: closes the BAI-714 outage without reopening the CVSS 5.3 finding;
  establishes a reusable M2M auth primitive in `oidcauthlib` for any future
  background/service-to-service call in this platform.
- Negative: first pure M2M auth pattern in this codebase — real
  implementation + review cost, not a config tweak. Adds one more credential
  (client_id/secret pair) to provision and rotate.
- Neutral / governance: this is a new pattern under the org's "is it NEW"
  test (new technology/vendor/pattern not previously used) — flagging for
  EA visibility per baseline instructions, even though it composes entirely
  from already-approved building blocks (OAuth/OIDC, the existing IdP).

## Open Questions

1. Does `TokenReaderVerifier`'s claim handling assume a `sub` claim is
   present? Client-credentials tokens typically only carry `client_id`. Needs
   a direct test against a real client-credentials token from the target IdP
   before this ships, not an assumption.
2. Should `get_mcp_servers_config` require a specific scope (e.g.
   `mcp-config:read`) on top of "any verified token," now that a new,
   narrowly-scoped M2M client is being introduced specifically for this
   call? Today's check (`access_token is None`) accepts any verified caller;
   introducing a purpose-built client is a natural point to also tighten
   this to a scope check, but that's a `baileyai-skills-service`-side change
   outside this ADR's direct scope — flag for that repo's own review.
3. Exact env var naming for the new M2M credential in
   `language-model-common`/`baileyai` — should follow existing
   `AUTH_*_<PROVIDER>` conventions closely enough to avoid a parallel,
   inconsistent naming scheme, without colliding with per-provider
   pass-through auth config already defined for user-facing flows.
4. Token caching strategy: in-process cache is fine for a single worker but
   this platform runs multiple workers per service (see org baseline: no
   in-memory caching for data that changes at runtime). A client-credentials
   token is not request-scoped user data, but repeated token-endpoint calls
   from every worker on every config refresh is wasteful. Decide whether an
   external shared cache (matching the org's no-in-memory-cache-across-workers
   rule) is warranted here, or whether per-worker caching is acceptable given
   this is infrastructure credentialing, not user-facing data.

## Related Work

- BAI-714 — this outage.
- `baileyai-skills-service` PR #86 — the security fix whose client-side
  counterpart this ADR addresses.
