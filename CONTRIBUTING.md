# Contributing

Contributions are welcome when they improve reproducibility, code clarity, documentation, or analysis quality.

Before opening a pull request:

1. Install development dependencies with `python -m pip install -e .[test]`.
2. Run `make check` or the equivalent commands: `ruff check .`, `ruff format --check .`, `python -m compileall src`, and `pytest`.
3. Do not add raw third-party documents, model weights, local vector stores, virtual environments, or full experiment logs.
4. Keep public documentation in English.
5. Update `configs/default.yaml`, docs, and tests together when changing public runtime behavior.

For larger changes, open an issue first and describe the intended behavior, reproduction impact, and expected artifacts.

The CI intentionally stays fast and does not require Ollama, licensed manuals, or full experimental artifacts. Full reproduction remains a local workflow documented in `docs/reproduction.md`.
