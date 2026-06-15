"""Build Chroma vector stores for all configured chunking variants."""

from __future__ import annotations

import argparse
from itertools import product
from typing import Any

from .config import load_config, resolve_project_path


def create_vector_stores(config: dict[str, Any] | None = None) -> None:
    """Create one Chroma collection per chunk-size/chunk-overlap pair."""

    from langchain_chroma import Chroma
    from langchain_community.document_loaders import DirectoryLoader
    from langchain_ollama import OllamaEmbeddings
    from langchain_text_splitters import MarkdownTextSplitter

    cfg = config or load_config()
    raw_documents = resolve_project_path(cfg["paths"]["raw_documents"])
    persist_directory = resolve_project_path(cfg["paths"]["persist_directory"])
    persist_directory.mkdir(parents=True, exist_ok=True)

    loader = DirectoryLoader(str(raw_documents), glob="**/*.pdf", show_progress=True)
    docs = loader.load()
    embeddings = OllamaEmbeddings(model=cfg["models"]["embedding"])

    search_space = cfg["search_space"]
    for chunk_size, overlap_pct in product(search_space["chunk_size"], search_space["chunk_overlap"]):
        overlap_tokens = int((overlap_pct / 100) * chunk_size)
        text_splitter = MarkdownTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap_tokens)
        splits = text_splitter.split_documents(docs)
        collection_name = f"chroma_{chunk_size}_{overlap_pct}"
        vector_store = Chroma(
            embedding_function=embeddings,
            persist_directory=str(persist_directory),
            collection_name=collection_name,
            client=None,
        )
        for idx in range(0, len(splits), 5000):
            vector_store.add_documents(splits[idx : idx + 5000])
        print(f"Created {collection_name}: {len(splits)} chunks")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Chroma indexes for configured RAG chunking variants.")
    parser.add_argument("--config", default=None, help="Path to a YAML configuration file.")
    args = parser.parse_args()
    create_vector_stores(load_config(args.config))


if __name__ == "__main__":
    main()

