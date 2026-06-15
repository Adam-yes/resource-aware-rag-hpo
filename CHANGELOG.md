# Changelog

All notable changes to this project will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project uses semantic versioning once releases begin.

## [Unreleased]

### Reproduction fidelity

- Pinned the computational path to the manuscript's published behavior: Markdown
  chunk splitter, verbatim prompt template, fixed `num_ctx = 5120` for generation and judging,
  average-fitness (Delta-mu) early stopping that reproduces the six published generations, no
  fitness cache (repeated genomes are re-evaluated), and NaN-skipping fitness aggregation.
- Documented the reverted production-refactor changes and the reasoning in
  `docs/improvement-audit.md`; added an integration test that locks in the early-stopping behavior.
- Filled `docs/RESULTS.md` with the paper's actual numbers (convergence table, best configuration,
  and the quality-latency trade-off).

### Added

- Public research-artifact repository layout.
- Modern README landing page and SVG architecture assets.
- Improvement audit, design document, results summary, ADRs, and roadmap.

### Changed

- Repository positioning now emphasizes resource-aware engineering trade-offs for quality, latency, and local compute constraints.
- Refreshed README and SVG assets: tightened copy, redrew the architecture diagram to remove
  arrow/label overlaps, and rebuilt the optimization-signal chart from the published convergence
  data with a subtle grid.

