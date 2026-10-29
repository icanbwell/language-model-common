# Playbooks

Last verified: 2026-10-11 against 8bc128d by Claude

Template version: 2

Operational notes for `language-model-common`, written for someone already in the repo. Standard and templates:
bwell-repo-wiki `docs/repo-playbooks/`. Edit by PR only.

This repo is a Python library, not a service. It has no deployment, no namespace and no logs of its
own: it runs inside its consumers. "Deployment" here means the package release and the downstream
version bump; "monitoring" means finding this library's log lines inside a consumer's logs.

| File | Come here if you are... |
|---|---|
| [MONITORING.md](MONITORING.md) | looking for where this library's log lines and cached data show up in a consumer |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | seeing a symptom in a consumer and need to know whether this library is involved |
| [MERGING.md](MERGING.md) | merging an approved PR |
| [LOGGING.md](LOGGING.md) | adding or changing log statements |
| [DEPLOYMENT.md](DEPLOYMENT.md) | releasing a version or getting consumers onto it |
| [REVIEW.md](REVIEW.md) | reviewing a PR here, or preparing one for review |
| [RELATIONSHIPS.md](RELATIONSHIPS.md) | changing something consumers depend on, or something this repo depends on (the tables, and the diagrams in `diagrams/` if the repo has any) |

When merging a PR, read MERGING.md, then DEPLOYMENT.md for what the merge triggers. When changing
logging, read LOGGING.md; update MONITORING.md and TROUBLESHOOTING.md if the change affects what
they point at.

Existing repo docs: [../CONTRIBUTING.md](../CONTRIBUTING.md) (consuming and releasing the package),
[../docs/](../docs/) (feature notes), [../adrs/](../adrs/) (decisions).
