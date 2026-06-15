"""Build Chroma vector stores for all configured chunking variants."""

from __future__ import annotations

import argparse
import logging
from itertools import product
from typing import Any

from .config import load_config, resolve_project_path
from .runtime import configure_logging

logger = logging.getLogger(__name__)


def create_vector_stores(config: dict[str, Any] | None = None, force: bool = False) -> None:
    """Create one Chroma collection per chunk-size/chunk-overlap pair."""

    from langchain_chroma import Chroma
    from langchain_community.document_loaders import DirectoryLoader
    from langchain_ollama import OllamaEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    cfg = config or load_config()
    configure_logging(cfg["logging"]["level"])
    raw_documents = resolve_project_path(cfg["paths"]["raw_documents"])
    persist_directory = resolve_project_path(cfg["paths"]["persist_directory"])
    persist_directory.mkdir(parents=True, exist_ok=True)

    loader = DirectoryLoader(str(raw_documents), glob="**/*.pdf", show_progress=True)
    docs = loader.load()
    embeddings = OllamaEmbeddings(model=cfg["models"]["embedding"])

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
            vector_store.delete_collection()
            vector_store = Chroma(
                embedding_function=embeddings,
                persist_directory=str(persist_directory),
                collection_name=collection_name,
            )
        elif cfg["indexing"]["skip_existing"] and collection_has_documents(vector_store):
            logger.info("Skipping existing collection %s", collection_name)
            continue

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap_tokens)
        splits = text_splitter.split_documents(docs)
        batch_size = int(cfg["indexing"]["batch_size"])
        for idx in range(0, len(splits), batch_size):
            vector_store.add_documents(splits[idx : idx + batch_size])
        logger.info("Created %s: %s chunks", collection_name, len(splits))


def collection_has_documents(vector_store: Any) -> bool:
    """Return whether a Chroma collection already contains documents."""

    try:
        return bool(vector_store.get(limit=1).get("ids"))
    except Exception:  # noqa: BLE001 - Chroma can raise when a collection is absent or being initialized.
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Chroma indexes for configured RAG chunking variants.")
    parser.add_argument("--config", default=None, help="Path to a YAML configuration file.")
    parser.add_argument("--force", action="store_true", help="Rebuild collections even if they already contain data.")
    args = parser.parse_args()
    create_vector_stores(load_config(args.config), force=args.force)


if __name__ == "__main__":
    main()
