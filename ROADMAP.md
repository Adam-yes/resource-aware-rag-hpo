# Roadmap

The reproduction default must remain faithful to the manuscript. Items below are additive or
opt-in and must not change the published computational behavior unless explicitly flagged.

## Near Term

- Keep expanding test coverage around evaluation contracts, indexing idempotency, and the
  early-stopping reproduction.
- Publish PR-style change notes for major technical upgrades.
- Render key notebooks into static documentation pages.

## Longer Term

- Add a multi-objective score that combines quality, latency, and compute budget (production
  successor, separate from the reproduction default).
- Offer an opt-in, off-by-default fitness cache and candidate-failure policy for long runs.
- Add richer observability for candidate failures and local model runtime behavior.
- Publish full reproducibility artifacts through Zenodo after paper clearance.

## Out Of Scope For The Public Repo

- Redistributing third-party technical manuals.
- Shipping model weights or local vector stores.
- Claiming full artifact reproducibility before data release.

