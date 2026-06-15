# Improvement Audit

This audit records the confirmed gaps behind the portfolio upgrade work. It is intentionally candid: the repository should signal technical ownership, not only attractive presentation.

## Executive Summary

The repository already has a clean public package layout, curated figures, and a strong README surface. The next maturity step is to make the engineering story match the leadership story: explicit trade-offs, documented architecture decisions, stronger measurement validity, better failure handling, and CI that proves more than importability.

## Leadership And Presentation

| Severity | Finding | Evidence | Impact |
| --- | --- | --- | --- |
| Medium | The README leads with the project name and visuals before the reviewer-facing outcome narrative. | `README.md` opening section | Busy managers and staff engineers need the outcome and trade-off story within seconds. |
| High | There is no senior-level design document explaining the quality/latency/compute trade-off. | `docs/` lacks `DESIGN.md` | The repository undersells the resource-aware thesis and the author's architectural judgment. |
| Medium | Architecture decisions are implicit in code and paper narrative, not captured as ADRs. | `docs/adr/` absent | Reviewers cannot see why GA, local inference, and FactualCorrectness were chosen. |
| Medium | Results are visually summarized but not yet documented as a reproducible results narrative. | `docs/RESULTS.md` absent | The headline numbers need a durable source separate from the README. |
| Low | Repository metadata changes are not documented for repeatable portfolio setup. | no `gh repo edit` guidance | The public About/topics setup depends on manual memory. |

## Backend Correctness

| Severity | Finding | Evidence | Impact |
| --- | --- | --- | --- |
| High | Context window is hardcoded to 5120 tokens while the search space allows `1024 * 10` retrieval content before prompt overhead. | `src/evo_rag_hpo/rag_chain_pipeline.py` sets `num_ctx=5120`; `configs/default.yaml` allows `chunk_size=1024`, `top_k=10` | Large candidates may be truncated, corrupting the measured fitness signal. |
| High | Evaluation silently truncates if RAG result count differs from reference count. | `src/evo_rag_hpo/evaluate.py` builds the dataset with `zip(...)` | Missing generations can shrink the evaluation set and inflate/deflate F1 without visibility. |
| Medium | Metric-column handling is inconsistent. | `evaluate.py` writes `row_eval.get(...)` but scores with direct `eval_df[...]` | A missing Ragas metric column can fail late or silently write inconsistent values. |
| Medium | Runtime constants are hardcoded in code. | `num_predict=1024`, `timeout=720`, `max_workers=8`, `keep_alive` values in runtime modules | Reproduction and resource tuning require code edits instead of config changes. |
| Medium | Failed candidate behavior is not explicit. | `evaluate.py` has no configurable failed-candidate fitness policy | A single timeout can stop a long run or produce inconsistent handling. |
| Medium | Indexing uses a Markdown splitter after loading PDFs. | `src/evo_rag_hpo/index.py` imports `MarkdownTextSplitter` | PDF-derived text is better served by a generic recursive splitter unless Markdown semantics are intentional. |
| Medium | Index rebuilds are not idempotent. | `index.py` always creates/adds collections; no `--force` or skip policy | Re-embedding all collections wastes compute and makes resume workflows expensive. |
| Low | The misspelled compatibility alias remains visible. | `evaluate.py` defines `run_async_aeavluate` | It preserves compatibility but should warn or be deprecated clearly. |

## Tests, CI, And Tooling

| Severity | Finding | Evidence | Impact |
| --- | --- | --- | --- |
| High | CI installs the package with `--no-deps`. | `.github/workflows/ci.yml` | A green badge currently does not prove dependency compatibility. |
| Medium | Unit tests cover only a small contract surface. | `tests/` currently covers decode/hash/import/log schema/mutation basics | Correctness fixes need regression tests around evaluation, context sizing, config validation, and early stopping. |
| Medium | No lint/format/coverage enforcement exists. | no Ruff, coverage, pre-commit, or Makefile config | Portfolio reviewers expect visible quality discipline. |
| Medium | Dependency sources can drift. | `pyproject.toml`, `requirements.txt`, and `environment.yml` are maintained separately | Install paths may diverge unless `pyproject.toml` is declared canonical. |

## Reproducibility And Artifact Policy

| Severity | Finding | Evidence | Impact |
| --- | --- | --- | --- |
| Medium | Public reproduction levels are documented, but expected runtime/hardware is incomplete. | `docs/reproduction.md` | Users need a clear distinction between smoke tests, sample analysis, and full local-model runs. |
| Medium | Full results are intentionally absent, but missing values need explicit TODO markers. | `docs/RESULTS.md` absent | Integrity requires distinguishing confirmed public artifacts from pending full-artifact values. |
| Low | Notebook outputs dominate repository weight and presentation risk. | `notebooks/` contains large curated notebooks | Stripping/rendering policy should be documented and eventually automated. |

## Implementation Direction

The upgrade should proceed in thematic commits: portfolio framing, correctness fixes, tests/CI, tooling, and final presentation cleanup. Behavioral changes must include tests and documentation updates.

