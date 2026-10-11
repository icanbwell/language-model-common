# Merging

Last verified: 2026-10-11 against 8bc128d by Claude

What has to happen to get an approved PR merged and on its way to consumers in this repo, beyond
approval and green checks. Review criteria are in [REVIEW.md](REVIEW.md); what happens after the merge
is in [DEPLOYMENT.md](DEPLOYMENT.md).

## Before you merge

1. Commit messages start with a JIRA issue key, for example `BAI-1129 ...`. Enforced by `.github/workflows/check-commit-message.yml` (exempt: `Bump`, `Merge`, `Revert`, `Reapply`, `fix(security):`, `[Aikido]`, `build(deps): bump`). The same workflow rejects AI attribution in commits and the PR description.
2. `Build and Test` is green by convention; the ruleset has no `required_status_checks`, so GitHub does not block a merge on it: pre-commit (ruff, mypy `--strict`, bandit, detect-secrets), then the pytest suite (`.github/workflows/build_and_test.yml`, `.pre-commit-config.yaml`).
3. `PR Architecture Review` has run (`.github/workflows/pr-architecture-review.yml`, reusable workflow from `icanbwell/.github`). It does not block merge by ruleset (no `required_status_checks`).
4. A change to `AGENTS.md` or `CLAUDE.md` needs `@icanbwell/enterprise-architecture` review (`CODEOWNERS`). That is a convention, not enforced: the `main` ruleset has `require_code_owner_review: false`, so another person's approval satisfies GitHub, but request the owner anyway (`gh api repos/icanbwell/language-model-common/rules/branches/main`, 2026-10-11).
5. A behavior change ships with tests in the same PR (`.claude/rules/testing.md`); public functions are keyword-only, tests included (`AGENTS.md`). Enforced by reviewer, not CI.
6. No version bump or changelog edit is needed in the PR: the version comes from the release tag (`CONTRIBUTING.md`).
7. Branch protection on `main` blocks direct commits (`AGENTS.md`). The `main` ruleset (`gh api repos/icanbwell/language-model-common/rules/branches/main`, 2026-10-11) has only `non_fast_forward` and `pull_request`: 1 approving review from anyone, `require_code_owner_review: false`, no `required_status_checks`.

## Merge mechanics

The repository allows squash, merge commit and rebase merges and has auto-merge off (GitHub repository settings via `gh api`, 2026-10-11). The default method the team uses: `unverified`; the history on `main` shows one commit per PR with the PR number appended, for example `BAI-1129 ... (#125)`, consistent with squash. The squash commit title must still begin with a JIRA key.

## What the merge triggers

`Build and Test` runs again on `main`. Nothing builds an image, tags, releases or deploys automatically (no such workflow in `.github/workflows/`). Release is a manual step: [DEPLOYMENT.md](DEPLOYMENT.md).

## After you merge

If the change must reach a consumer, create the GitHub Release and watch `Upload Python Package` to green ([DEPLOYMENT.md](DEPLOYMENT.md)), then tell the consumer owners to bump (Slack channel: `unverified`). A failed publish leaves a GitHub Release with no PyPI version.

## Gotchas

- Merging does not publish. A fix is not available to any consumer until a release is created and the consumer bumps.
- The PR title becomes the squash commit; `check-commit-message.yml` also runs on `edited`, so fix a missing key by editing the title.
- A stacked PR targeting another branch is not covered by `Build and Test` (it triggers only on PRs to `main`).

Next: [DEPLOYMENT.md](DEPLOYMENT.md).
