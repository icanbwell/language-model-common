# Review

Last verified: 2026-10-16 against 6245270 by Claude

Instructions for whoever reviews a PR in this repo, human or agent, and for authors preparing one.
Inherits the org review rubrics in `icanbwell/.github` (the `PR Architecture Review` workflow runs
them: `.github/workflows/pr-architecture-review.yml`); where this file conflicts, the org baseline
wins. This file covers only what is specific to this repo.

## What makes this repo different

A change here ships to every consumer that bumps, in-process, with no staging of its own. A signature
or behavior change can break baileyai, baileyai-skills-service or mcp-fhir-agent
without any test in this repo failing. Callers and dependencies: [RELATIONSHIPS.md](RELATIONSHIPS.md).

## Rules

| # | Rule | Severity | How it is checked | Example |
|---|---|---|---|---|
| 1 | Public functions, methods and constructors are keyword-only (`*`), call sites and tests pass keywords | REVIEW | reviewer | `AGENTS.md` (Modern, Idiomatic Code) |
| 2 | New services are registered in `register_services_in_container()`, not constructed inline | REVIEW | reviewer | `languagemodelcommon/container/container_factory.py` |
| 3 | New environment variables are a `@property` on `LanguageModelCommonEnvironmentVariables`, not `os.environ` in business logic | REVIEW | reviewer | `languagemodelcommon/utilities/environment/language_model_common_environment_variables.py` |
| 4 | A public API change (signature, SSE event shape, `AgentConfig` field) states the consumers affected and how they were checked | BLOCKING | reviewer | see Blast radius |
| 5 | No PHI, tokens or full request or response bodies in logs; follow [LOGGING.md](LOGGING.md) | BLOCKING | reviewer; org pre-commit `detect-secrets` catches secrets only | `languagemodelcommon/utilities/logger/summarization.py` |
| 6 | Every external call has an explicit timeout; retries are bounded with backoff | REVIEW | reviewer | `.claude/rules/operational.md` |
| 7 | Mypy `--strict`, ruff and bandit pass | BLOCKING | `Build and Test` pre-commit step | `.pre-commit-config.yaml` |
| 8 | A behavior change ships with tests, including error paths | BLOCKING | reviewer | `.claude/rules/testing.md` |
| 9 | Event shapes follow the contract rules | REVIEW | reviewer | `.claude/rules/event-contracts.md` |
| 10 | `AGENTS.md` and `CLAUDE.md` changes need EA review | REVIEW (convention; `require_code_owner_review` is false) | `CODEOWNERS` | `/AGENTS.md @icanbwell/enterprise-architecture` |

Branch ruleset on `main` (2026-10-11): `non_fast_forward` and `pull_request` only (1 approving review from anyone, `require_code_owner_review: false`, no `required_status_checks`). No rule above is enforced by it; green CI and EA review are conventions.

## Evidence required by change type

| Change | Evidence the PR should carry |
|---|---|
| Behavior change or fix | a test that fails without the change (`make tests`; commands in [DEPLOYMENT.md](DEPLOYMENT.md)) |
| Public API or SSE event shape | the affected consumers named, and a consumer test run against the change (`unverified` that a procedure for this exists) |
| MCP client change (session, negotiation, tool invocation) | tests under `tests/mcp/`; the relevant ADR in `adrs/` updated if the decision changes |
| Dependency bump | `Build and Test` green; for a security bump the advisory ID in the commit message |
| Log statement change | the checklist in [LOGGING.md](LOGGING.md) |
| Config schema change (`config_schema.py`) | the consuming config repos still validate (`unverified` which repos) |

## Blast radius

Check [RELATIONSHIPS.md](RELATIONSHIPS.md): the `AgentConfig` and request-wrapper rows for any change
under `languagemodelcommon/structures/` or `configs/schemas/`; the Downstream MCP row for anything in
`languagemodelcommon/mcp/`; the Databases table in [MONITORING.md](MONITORING.md) for cache name or
TTL changes.

## Getting a PR reviewed

CODEOWNERS covers only `AGENTS.md` and `CLAUDE.md` (`@icanbwell/enterprise-architecture`), so ordinary code PRs have no auto-requested reviewer: request reviewers by hand. For Bailey work the usual reviewers are the bAIley team (`unverified` for this repo; the `bailey-ai` team request works only on baileyai and mcp-fhir-agent). Branch name `{initials}-{project}-{ticket}` (`AGENTS.md`). Commit messages start with a JIRA key (`check-commit-message.yml`). PR labels required: `unverified` (no label workflow in `.github/workflows/`). For `AGENTS.md` and `CLAUDE.md`, request the EA team; GitHub does not enforce it here (`require_code_owner_review: false`).

Next: once approved, [MERGING.md](MERGING.md).
