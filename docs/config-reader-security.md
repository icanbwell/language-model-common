# Config Reader: Path-Traversal Hardening

This document explains the path-traversal protections in `ConfigReader` and why they must stay, for anyone reviewing or refactoring `languagemodelcommon/configs/config_reader/`.

---

## Background

Per-client model config overrides are resolved by joining a caller-supplied `client_id` onto a base config path (`<config_path>/clients/<client_id>/...`). `client_id` is untrusted — it originates from caller input (e.g. an HTTP header) rather than a fixed, trusted source. Used unsanitized in a path join, a value like `client_id="../../etc/passwd"` (or a GitHub/S3 URI equivalent) would let a caller read files outside the intended config directory — a path-traversal / local-file-inclusion vulnerability.

A related vulnerability class is "zip-slip": extracting a ZIP archive whose member names contain `../` sequences can write files outside the intended extraction directory.

Both classes were caught by a security scan against WIP code in `baileyai` (the config-reading logic's original home) before it ever reached `main`, and were fixed prior to merge. The logic has since moved to this repo.

---

## What protects against it today

### 1. `client_id` allowlist + path containment (`ConfigReader._resolve_override_config_path`)

Two independent checks, both required:

- **Allowlist** (`ConfigReader._is_valid_client_id`): `client_id` must match `^[a-zA-Z0-9_-]+$`. This rejects `..`, `/`, and any other path-traversal metacharacters before the value is used to build a path.
- **Containment check**: after joining `client_id` onto the base config folder, the resolved path is verified to still be inside the base folder via `Path.resolve()` + `Path.relative_to()`. If it isn't, resolution is aborted and logged.

The containment check is defense in depth — the allowlist alone should already prevent traversal, but the two together mean a bug or future loosening of the allowlist doesn't silently reopen the vulnerability.

**Do not remove either check as unnecessary defensive code.** Both exist specifically because `client_id` is untrusted.

### 2. ZIP extraction was removed entirely

The original GitHub config downloader extracted a repository ZIP archive to a local directory (`GitHubConfigZipDownloader`, later `GithubConfigRepoManager`). This code has been deleted; GitHub config downloads now go through `GithubDirectoryDownloader` (`github_directory_downloader.py`), which uses `fsspec`'s GitHub filesystem to fetch individual files/directories directly — there is no ZIP extraction step anywhere in the current download path. The zip-slip bug class doesn't just have a fix here; its attack surface no longer exists in this codebase.

`GithubDirectoryDownloader` still validates that its resolved cache-directory path stays within the configured cache root, following the same "resolve and check containment" pattern as the `client_id` check above.
