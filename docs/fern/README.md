# Fern documentation CI

The Fern check and hosted preview run when an approved PR is copied to
`pull-request/<number>`. The preview workflow checks whether `docs/**` or
`.github/workflows/fern-docs-ci.yml` changed, builds the docs, and posts or
updates the preview link on the PR. It requires `DOCS_FERN_TOKEN`.
