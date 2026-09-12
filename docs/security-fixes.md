# Security Fixes

Running log of Aikido/Gecko-scan-driven security remediation in this repo, for
anyone auditing what was changed and why. See also
[`docs/config-reader-security.md`](./config-reader-security.md) for the
path-traversal / zip-slip hardening in `ConfigReader` (a separate, earlier
fix — not duplicated here).

---

## 2026-08: Dependency CVEs, CI Actions hardening, Docker root user

### Dependency CVEs (`uv.lock`)

Bumped via `uv lock --upgrade-package <name>` (mechanical resolver bump, no
code changes required):

| Package      | Before   | After    | Severity | Notes |
|--------------|----------|----------|----------|-------|
| anyio        | 4.13.0   | 4.14.2   | Critical | |
| cryptography | 49.0.0   | 50.0.0   | High     | CVE-2026-69247 |
| aiohttp      | 3.14.1   | 3.14.3   | High     | Covers AIKIDO-2026-168095, CVE-2026-69244, CVE-2026-59881, CVE-2026-69243, AIKIDO-2026-725081, AIKIDO-2026-675677, AIKIDO-2026-133006 |
| fsspec       | 2026.4.0 | 2026.7.0 | Medium   | AIKIDO-2026-78094 (resolver picked 2026.7.0, newer than the 2026.6.0 minimum fix) |
| yarl         | 1.24.2   | 1.24.5   | Medium   | AIKIDO-2026-472816 |
| langsmith    | 0.8.18   | 0.11.0   | Medium   | Covers AIKIDO-2026-890022, AIKIDO-2026-944829 (resolver picked 0.11.0, newer than the 0.10.9 minimum fix) |
| h2           | 4.3.0    | 4.4.1    | Medium   | CVE-2026-71554 |
| pypdf        | 6.14.2   | 6.16.1   | Medium   | Covers AIKIDO-2026-505281, CVE-2026-71852 (resolver picked 6.16.1, newer than the 6.15.0 minimum fix) |

Full test suite (`uv run pytest tests`) passes against the new lock file.
`tests/mcp/test_tool_invocation_heartbeat.py::test_slow_call_emits_periodic_heartbeats`
fails intermittently under full-suite load (timing-sensitive heartbeat-count
assertion) but passes consistently in isolation and on repeat runs — this is
pre-existing flakiness unrelated to the dependency bumps here, not a
regression.

### GitHub Actions hygiene

- `.github/workflows/build_and_test.yml`: pinned `pmeier/pytest-results-action`
  from the mutable tag `v0.8.0` to its commit SHA
  (`0841ca7226ab155943837380769373a5dd14d7ed`), and added
  `persist-credentials: false` to the `actions/checkout` step (this workflow
  never pushes back to the repo, so persisting the checkout token isn't
  needed).
- `.github/workflows/python-publish.yml`: pinned `pypa/gh-action-pypi-publish`
  from the mutable branch ref `release/v1` to the commit SHA it currently
  resolves to (`dc37677b2e1c63e2034f94d8a5b11f265b73ba33`, tag `v1.14.2`), and
  added `persist-credentials: false` to `actions/checkout`. Verified this
  workflow doesn't need to push: it only writes `VERSION` locally as build
  input to `uv build` and never commits/pushes it, and publishing goes through
  PyPI Trusted Publishing (OIDC `id-token: write`), not a git push.
- `astral-sh/setup-uv` in `python-publish.yml` was already pinned to a commit
  SHA — left unchanged.

### `pre-commit.Dockerfile` non-root user

The image previously ran `pre-commit` as root. It now creates a non-root
`appuser` and switches to it before `CMD`. Because `pre-commit-hook` bind-mounts
the host source tree (`-v "$(pwd)/:/sourcecode"`) and the host's real `.git`
directory (`-v "$GIT_COMMON_DIR:$GIT_COMMON_DIR"`) straight into the
container, two things had to change together or the non-root user would hit
permission errors on every run:

1. **UID/GID build args.** `USER_UID`/`USER_GID` default to `1000` but
   `pre-commit-hook` now passes `--build-arg USER_UID=$(id -u) --build-arg
   USER_GID=$(id -g)` at build time, so the in-container `appuser` has the
   same UID/GID as whoever is running the hook. Bind mounts preserve host
   ownership by UID number, not by username, so a mismatched UID would leave
   `appuser` unable to read the mounted source tree or write pre-commit's
   auto-fixes back to it. Verified locally: built the image with a real UID
   mismatch and confirmed a permission error on writing to the bind mount;
   after passing matching `--build-arg` values, `pre-commit` could read and
   modify (dirty) a file under the bind-mounted `/sourcecode` and the changes
   showed up on the host with correct ownership.
2. **`git config --system` instead of `--global` for `safe.directory`.**
   `--global` writes to the *running user's* `$HOME/.gitconfig`. That's fine
   for root (whose `$HOME` always exists and is writable) but fragile for a
   built-with-one-UID, run-with-another-UID non-root user. `--system` writes
   to `/etc/gitconfig`, which this Dockerfile now `chown`s to `appuser` at
   build time (via the same `USER_UID`/`USER_GID` args) so it works
   regardless of `$HOME`. The Dockerfile sets `safe.directory /sourcecode` at
   build time; `pre-commit-hook` still sets `safe.directory` for the dynamic,
   host-specific `$GIT_COMMON_DIR` at container-run time, now also via
   `--system`.

The pre-commit cache volume mount moved from `/root/.cache/pre-commit` to
`/home/appuser/.cache/pre-commit` to match the new non-root user's home
directory.

### Root.io ECR mirror gap — deferred

Aikido flagged `Dockerfile` and `pre-commit.Dockerfile` for pulling their base
image from a public registry instead of the internal Root.io ECR mirror
(`856965016623.dkr.ecr.us-east-1.amazonaws.com/root-mirror/...`). This was
**not changed** in this batch:

- Both files actually pull `public.ecr.aws/docker/library/python:3.12-alpine3.20`
  (a public *ECR* mirror of the official Python image), not `node:24-alpine`
  as initially reported against this repo — the base image is Python, not
  Node.
- No AWS credentials were available in this session to run
  `aws ecr describe-images --repository-name root-mirror` and confirm
  `python:3.12-alpine3.20` is actually mirrored yet.
- A separate branch, `feature/migrate-jfrog-ecr-mirror`, already exists with a
  full migration to the internal ECR mirror + JFrog for this repo (base image
  swap, Pipenv instead of `uv`, JFrog auth secrets in the Dockerfiles, CI
  changes). That's a substantially larger, already-in-flight change; swapping
  just the `FROM` line here without the matching JFrog auth plumbing would
  leave the build broken, and duplicating that effort risks diverging from
  it.

Recommendation: land this migration through `feature/migrate-jfrog-ecr-mirror`
(or its successor) once the mirror image is confirmed available, rather than
as a standalone `FROM`-line change here.

---

## 2026-09: Additional dependency CVEs (BAI-438 follow-up)

This branch (BAI-438) sat uncommitted since the August batch above. Before
landing it, it was rebased onto a fresh `origin/main` (which had picked up
unrelated drift in `uv.lock`, notably an `oidcauthlib` bump) and the same
`uv lock --upgrade-package` set from August was re-run. Because time passed,
the resolver picked newer patched versions than the ones documented in the
August table above for several packages already covered there — the table
above is left as-is (it accurately describes what was verified in August);
this section documents the net result actually in `uv.lock` now, plus new
packages Aikido flagged since:

| Package       | Aug table said | Actually in `uv.lock` now | Notes |
|---------------|-----------------|----------------------------|-------|
| anyio         | 4.14.2          | 4.14.2 → 4.15.1 (transitive pin) | two anyio versions now resolve in the tree; both post-fix |
| cryptography  | 50.0.0          | 50.0.1                     | newer patch since August |
| aiohttp       | 3.14.3          | 3.14.3                     | unchanged |
| fsspec        | 2026.7.0        | 2026.7.0                   | unchanged |
| yarl          | 1.24.5          | 1.24.5                     | unchanged |
| langsmith     | 0.11.0          | 0.12.2                     | newer minor since August |
| h2            | 4.4.1           | 4.4.1                      | unchanged |
| pypdf         | 6.16.1          | 6.18.0                     | newer patch since August |

New packages flagged since the August scan, bumped via the same mechanical
`uv lock --upgrade-package <name>` (no code changes):

| Package       | Before | After  | Notes |
|---------------|--------|--------|-------|
| langgraph-sdk | 0.4.2  | 0.4.4  | |
| regex         | 2026.5.9 | 2026.9.3 | |
| greenlet      | 3.5.1  | 3.5.5  | |
| numpy         | 2.4.6  | 2.5.3  | |

`uv lock` also pulled in new transitive dependencies as a result of these
bumps: `httpcore2`, `httpx2`, `httpx2-jsfetch`, `truststore`.

The `.github/workflows/*`, `pre-commit.Dockerfile`, and `pre-commit-hook`
fixes from the August section above were carried through this rebase
unchanged — verified via `git diff` against `origin/main` that they still
apply cleanly and match this section's description.

---

## 2026-09-11: Rebase onto main, reconcile with already-fixed packages

`origin/main` moved again between the September rebase above and this PR
going up (BAI-678, plus dependabot bumps to `pypdf`, `h2`, and `cryptography`
that had already landed independently). Rebasing hit a conflict in `uv.lock`;
resolved by taking `origin/main`'s lock file as the base and re-running
`uv lock --upgrade-package` for the full original CVE list (including the
packages `main` had already fixed, to pick up the newest patched version
rather than leave a stale one).

Final versions in `uv.lock` on this branch:

| Package       | Before (original CVE) | Final    | Notes |
|---------------|------------------------|----------|-------|
| anyio         | 4.13.0                 | 4.14.2, 4.15.1 (two resolve in the tree) | |
| cryptography  | 49.0.0                 | 50.0.1   | already at 50.0.0 on `main` via a separate dependabot PR; picked up 50.0.1 |
| aiohttp       | 3.14.1                 | 3.14.3   | already fixed on `main`, unchanged here |
| fsspec        | 2026.4.0               | 2026.7.0 | |
| yarl          | 1.24.2                 | 1.24.5   | |
| langsmith     | 0.8.18                 | 0.12.4   | |
| h2            | 4.3.0                  | 4.4.1    | already fixed on `main`, unchanged here |
| pypdf         | 6.14.2                 | 6.18.0   | already at 6.16.1 on `main` via #81; picked up 6.18.0 |
| langgraph-sdk | 0.4.2                  | 0.4.4    | |
| regex         | 2026.5.9               | 2026.9.10 | |
| greenlet      | 3.5.1                  | 3.5.5    | |
| numpy         | 2.4.6                  | 2.5.3    | |

No code changes were needed for any of these — all mechanical resolver
bumps. The `.github/workflows/*`, `pre-commit.Dockerfile`, and
`pre-commit-hook` changes carried through this rebase unchanged (no conflicts
outside `uv.lock`).

### Non-root `pre-commit.Dockerfile` follow-up: CI failure fixed

The first push of this branch broke CI's `run-pre-commit` step:
`error: could not lock config file /etc/gitconfig: Permission denied`. Cause:
`git config` writes its target via a temp-file-then-rename *in the same
directory as the target*, so `--system` needs write access on `/etc` itself,
not just on `/etc/gitconfig` — and `/etc` stays root-owned (0755) regardless
of what `/etc/gitconfig` is chowned to. Switched both the Dockerfile and
`pre-commit-hook` to `--global` with `ENV HOME=/home/appuser` set explicitly;
`/home/appuser` is chowned to `appuser` already, so both the build-time
`RUN git config --global ...` (still root, but `$HOME` is overridden) and the
runtime one (as `appuser`) can write there.

Also verified end-to-end locally (`./pre-commit-hook`, full run, all hooks
pass) — a fresh `pre-commit-cache` named volume inherits its ownership from
the image's `/home/appuser/.cache/pre-commit` directory (already chowned to
`appuser`), so this works cleanly on GitHub Actions' ephemeral runners. A
**pre-existing** local `pre-commit-cache` volume from before this change (root
owned, from prior runs of the old root-based image) will still need a one-time
`docker volume rm pre-commit-cache` on each developer machine — flagging this
here since it isn't something the PR itself can fix.

### "Use Alpine base images" finding — not applicable

Aikido also flagged `Dockerfile`/`pre-commit.Dockerfile` for not using an
Alpine base image. Both already pull
`public.ecr.aws/docker/library/python:3.12-alpine3.20`, i.e. they're already
Alpine-based. No change made; this looks like a stale finding that should
clear on Aikido's next scan.
