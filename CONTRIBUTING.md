# Contributing

## Consuming this package

`language-model-common` is published to PyPI and consumed as a normal dependency
(e.g. `language-model-common>=3.0.25` in a consumer's `pyproject.toml`).

**Version pinning:** adopters (`baileyai`, `baileyai-skills-service`) also set
`[tool.uv.exclude-newer-package] language-model-common = "0 days"`. This
overrides `uv`'s freshness-cutoff policy (`[tool.uv] exclude-newer`) for this
package specifically: `uv` will resolve to the latest already-published
version immediately, with none of the artificial delay the org's general
supply-chain policy applies to third-party packages. It is not a version
floor - the `>=` constraint in `dependencies` still governs the minimum
version. Treat this package like `oidcauthlib`: internal and first-party, so
there is no supply-chain rationale for delaying pickup of a new release.

**Release process:** a new version is published by creating a GitHub Release
in this repo. `VERSION` is stamped from the release tag by
`.github/workflows/python-publish.yml` at publish time and is not committed
back to the repository - the `VERSION` file in the working tree is a local
placeholder, not the published version. There is no separate `CHANGELOG.md`;
release notes live on this repo's GitHub Releases page.
