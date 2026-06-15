"""Precompute the Chroma vector stores for every chunking variant in the search space.

A central design decision of the study is to *precompute the full indexing space*. Chunk size
and chunk overlap are the only two hyperparameters that affect how documents are segmented and
embedded. Rather than re-indexing the corpus for every candidate the genetic algorithm
proposes, we build one persistent Chroma collection per ``(chunk_size, chunk_overlap)``
combination ahead of time. During optimization the RAG pipeline simply selects the collection
that matches the genome of the current individual, which removes repeated re-indexing from the
hot loop and keeps the ~30-hour search tractable.

The Cartesian product of ``chunk_size`` (8 values) and ``chunk_overlap`` (3 values) yields 24
collections, each named ``chroma_<chunk_size>_<overlap_pct>`` so that
:func:`evo_rag_hpo.rag_chain_pipeline.run_async_rag_chain` can address them deterministically.

Reproduction note
-----------------
The text splitter is fixed to :class:`MarkdownTextSplitter`, exactly as in the original
experiment. The splitter choice changes chunk boundaries and therefore every downstream
retrieval result, so it must not be substituted (e.g. with a recursive character splitter) if
the goal is a faithful 1:1 reproduction of the published numbers.
"""

from __future__ import annotations

import argparse
import logging
from itertools import product
from typing import Any

from .config import load_config, resolve_project_path
from .runtime import configure_logging

logger = logging.getLogger(__name__)


def create_vector_stores(config: dict[str, Any] | None = None, force: bool = False) -> None:
    """Create one persistent Chroma collection per chunk-size/chunk-overlap pair.

    The corpus is loaded once and embedded once; only the splitting step is repeated per
    configuration. Collections are written to the configured ``persist_directory`` so they
    survive between runs and can be reused by the optimization loop.

    Args:
        config: Resolved configuration dictionary. When ``None`` the default configuration is
            loaded.
        force: When ``True`` an existing collection is deleted and rebuilt. When ``False`` (the
            default) and ``indexing.skip_existing`` is set, collections that already contain
            documents are skipped, which makes re-running the indexer idempotent.
    """

    # Heavy third-party dependencies are imported lazily so that importing this module (e.g. for
    # unit tests or documentation tooling) does not require the full RAG stack.
    from langchain_chroma import Chroma
    from langchain_community.document_loaders import DirectoryLoader
    from langchain_ollama import OllamaEmbeddings
    from langchain_text_splitters import MarkdownTextSplitter

    cfg = config or load_config()
    configure_logging(cfg["logging"]["level"])
    raw_documents = resolve_project_path(cfg["paths"]["raw_documents"])
    persist_directory = resolve_project_path(cfg["paths"]["persist_directory"])
    persist_directory.mkdir(parents=True, exist_ok=True)

    # 1. Load the source corpus exactly once. Re-loading per collection would waste minutes of
    #    disk and parsing time for no benefit, since the raw documents never change.
    logger.info("Loading documents from %s", raw_documents)
    loader = DirectoryLoader(str(raw_documents), glob="**/*.pdf", show_progress=True)
    docs = loader.load()
    logger.info("Loaded %s documents", len(docs))

    # 2. Initialize the embedding model exactly once. It is fixed to ``embeddinggemma:300m`` for
    #    the whole study to keep collections comparable.
    embeddings = OllamaEmbeddings(model=cfg["models"]["embedding"])

    # 3. Iterate over the Cartesian product of all chunk-size/overlap combinations and
    #    materialize one collection for each. Overlap is stored as a percentage and converted to
    #    an absolute token count here.
    search_space = cfg["search_space"]
    for chunk_size, overlap_pct in product(search_space["chunk_size"], search_space["chunk_overlap"]):
        overlap_tokens = int((overlap_pct / 100) * chunk_size)
        collection_name = f"chroma_{chunk_size}_{overlap_pct}"

        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=str(persist_directory),
            collection_name=collection_name,
        )

        if force:
            # Drop and recreate so a rebuild never appends to stale vectors from a previous run.
            vector_store.delete_collection()
            vector_store = Chroma(
                embedding_function=embeddings,
                persist_directory=str(persist_directory),
                collection_name=collection_name,
            )
        elif cfg["indexing"]["skip_existing"] and collection_has_documents(vector_store):
            logger.info("Skipping existing collection %s", collection_name)
            continue

        # Markdown-aware splitting preserves structural boundaries (headings, lists, code fences)
        # common in technical manuals, which is why it is preferred over a fixed-width splitter.
        text_splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap_tokens)
        splits = text_splitter.split_documents(docs)

        # Chroma's ``add_documents`` has an internal maximum batch size (~5,450). We add in
        # batches below that limit to avoid the "batch too large" error on big corpora.
        batch_size = int(cfg["indexing"]["batch_size"])
        for idx in range(0, len(splits), batch_size):
            vector_store.add_documents(splits[idx : idx + batch_size])
        logger.info("Created %s: %s chunks", collection_name, len(splits))

    logger.info("All chunking collections are ready in %s", persist_directory)


def collection_has_documents(vector_store: Any) -> bool:
    """Return whether a Chroma collection already contains at least one document.

    Used to make indexing idempotent: a partially built index can be resumed without rebuilding
    collections that are already populated.
    """

    try:
        return bool(vector_store.get(limit=1).get("ids"))
    except Exception:  # noqa: BLE001 - Chroma can raise when a collection is absent or initializing.
        return False


def main() -> None:
    """Command-line entry point: ``python -m evo_rag_hpo.index [--config ...] [--force]``."""

    parser = argparse.ArgumentParser(description="Build Chroma indexes for configured RAG chunking variants.")
    parser.add_argument("--config", default=None, help="Path to a YAML configuration file.")
    parser.add_argument("--force", action="store_true", help="Rebuild collections even if they already contain data.")
    args = parser.parse_args()
    create_vector_stores(load_config(args.config), force=args.force)


if __name__ == "__main__":
    main()
