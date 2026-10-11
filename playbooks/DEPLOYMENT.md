# Deployment

Last verified: 2026-10-11 against 8bc128d by Claude

This repo is a library released as a PyPI package, `language-model-common`. Nothing is deployed from
here: a "deploy" is a release here followed by each consumer bumping its version and deploying
itself. The consumer's own `playbooks/DEPLOYMENT.md` owns its rollout, where it exists.

## Pipeline

1. Merge the PR to `main` (see [MERGING.md](MERGING.md)). `Build and Test` runs on the merge (`.github/workflows/build_and_test.yml`). The merge publishes nothing.
2. A person creates a GitHub Release in this repo. Evidence: `CONTRIBUTING.md` ("a new version is published by creating a GitHub Release") and `.github/workflows/python-publish.yml` (`on: release: types: [created]`).
3. `python-publish.yml` writes the release tag into `VERSION`, runs `uv build`, and publishes to PyPI through trusted publishing (GitHub environment `pypi`). `VERSION` is not committed back; the file on `main` (`0.0.1`) is a placeholder.
4. Tag format seen on the repo: bare versions such as `3.0.37`, and release candidates such as `3.0.34rc1` (`git tag`). The tag becomes the package version, so it must be a valid PEP 440 version. The workflow triggers on any created release, including ones marked pre-release (the Releases page shows pre-release flags on some recent releases), so a pre-release flag does not stop publication.
5. Release notes live on the GitHub Releases page; there is no `CHANGELOG.md` (`CONTRIBUTING.md`).
6. Who may create releases, and whether a version-numbering rule beyond the tag format exists: `unverified`.

`make package` and `make testpackage` upload with twine from a laptop (`Makefile`); the CI release path above is the documented one (`CONTRIBUTING.md`). Use of the Makefile path: `unverified`.

## Environments

Not applicable; library. Environments belong to the consumers.

## Before and after a deploy

**Downstream bump path (what a "deploy" of this library means):**

1. After the PyPI publish succeeds, each consumer bumps its dependency in its own `pyproject.toml` and relocks. Consumers and their current floors on `origin/main` as of this verification: baileyai `language-model-common>=3.0.37`, baileyai-skills-service `>=3.0.30`, mcp-fhir-agent `>=3.0.30`, language-model-gateway `>=2.0.99` (evidence: each repo's `pyproject.toml`).
2. `baileyai` and `baileyai-skills-service` also set `[tool.uv.exclude-newer-package] language-model-common = "0 days"` so a fresh release resolves immediately (`CONTRIBUTING.md`; confirmed for baileyai in its `pyproject.toml`). Other consumers: `unverified` whether they set it; without it `uv` applies its freshness cutoff to the new release.
3. A floor bump alone does not change what is deployed: the consumer's lock file and its own deploy do. Check the consumer's lock file for the resolved version.

**Before:** note the version each target consumer has deployed (its lock file at the deployed commit) and the consumer's own baseline test report. **Test commands for this repo:** `make tests` (runs `pytest tests languagemodelcommon` in Docker; `Makefile`); CI runs the whole suite in `build_and_test.yml`. No VPN or credentials are needed for unit tests (`unverified` for every test). **After:** run the consumer's tests against the new version in the consumer's repo, then its normal deploy. **Compare:** the consumer's own before/after procedure. **Stop rule:** a regression a consumer's test suite attributes to the bump halts that consumer's bump, not the release; fix forward with a new release. **Known artifacts:** none recorded.

## Release gotchas

- The build runs `uv build` only; tests do not run in the publish workflow (`python-publish.yml` has no test step). Green `Build and Test` on the merged commit is the gate.
- A tag that is not valid PEP 440 fails the build step after the release already exists.
- `pyproject.toml` declares `requires-python = ">=3.12"` while `README.md` says Python >= 3.10; consumers need 3.12.
- Consumers that do not use `exclude-newer-package = "0 days"` can take up to the freshness cutoff (`exclude-newer = "2 days"` in this repo's own `[tool.uv]`) to see a release (`unverified` for each consumer's setting).

## Verify in production

Nothing to verify here directly. After a consumer deploys the bump, verify in that consumer's real traffic using its `playbooks/MONITORING.md`; this library's own signals are in [MONITORING.md](MONITORING.md).

## Rollback

Publish a new release with the fix, or pin the consumer back to the previous version in its `pyproject.toml` and relock (`unverified`; not exercised). A published PyPI version cannot be replaced; a released version cannot be re-uploaded under the same number (PyPI rule, not a repo fact). Decider: the consumer's owners (`unverified`).

Next: [MONITORING.md](MONITORING.md) to verify; if verification fails, [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
