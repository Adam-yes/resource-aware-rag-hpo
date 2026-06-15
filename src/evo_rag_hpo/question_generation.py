"""Synthesize a single-hop QA benchmark from technical documentation with RAGAS.

This is the *first* stage of the two-stage benchmark construction described in the paper: a
knowledge graph is built from the corpus and a graph-based synthesizer generates
question-answer pairs together with source metadata. A subsequent human-in-the-loop validation
step (not automated here) reviews the pairs for factual correctness and clarity before they
become the 50-pair evaluation set used as the fitness benchmark.

Performance note
----------------
Empirically, feeding more than ~80 pages to the local LLM at once causes memory crashes or
timeouts. The robust strategy is to process **one document at a time**: build the knowledge
graph and test set per document, save to CSV, and merge the per-document CSVs afterwards (see
``notebooks/01_build_evaluation_set.ipynb``). This function encapsulates a single generation
pass so it can be driven document-by-document.
"""

from __future__ import annotations

from pathlib import Path


def generate_single_hop_testset(
    source_dir: str,
    output_csv: str,
    output_kg: str,
    generator_model: str = "qwen3-coder:30b",
    embedding_model: str = "embeddinggemma:300m",
    testset_size: int = 10,
) -> None:
    """Generate a single-hop technical-document QA test set and persist it.

    Args:
        source_dir: Directory of source PDFs to derive questions from.
        output_csv: Destination CSV for the generated question-answer pairs.
        output_kg: Destination JSON for the constructed knowledge graph (saved so generation
            can be re-run from the graph without re-parsing the documents).
        generator_model: Local Ollama model that builds the graph and synthesizes questions.
        embedding_model: Local embedding model used for graph construction and synthesis.
        testset_size: Number of question-answer pairs to generate from this corpus.

    The query distribution is 100% single-hop *specific* queries
    (:class:`SingleHopSpecificQuerySynthesizer`), reflecting the fact-lookup nature of
    technical-manual questions rather than multi-hop reasoning.
    """

    from langchain_community.document_loaders import DirectoryLoader
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from langchain_text_splitters import MarkdownTextSplitter
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.testset import TestsetGenerator
    from ragas.testset.graph import KnowledgeGraph, Node, NodeType
    from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
    from ragas.testset.transforms import apply_transforms, default_transforms

    # Temperature 0 keeps question/answer synthesis as deterministic as the backend allows.
    generator_llm = LangchainLLMWrapper(ChatOllama(model=generator_model, temperature=0))
    generator_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=embedding_model))

    docs = DirectoryLoader(source_dir, glob="**/*.pdf").load()
    # Large 7,000-character chunks (with 200 overlap) give the graph synthesizer enough context
    # per node to extract meaningful entities and relations for single-hop question generation.
    splits = MarkdownTextSplitter(chunk_size=7000, chunk_overlap=200).split_documents(docs)

    # Build the knowledge graph: one document node per chunk, carrying its content and metadata.
    kg = KnowledgeGraph()
    for doc in splits:
        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={"page_content": doc.page_content, "document_metadata": doc.metadata},
            )
        )

    transforms = default_transforms(documents=docs, llm=generator_llm, embedding_model=generator_embeddings)
    apply_transforms(kg, transforms)

    Path(output_kg).parent.mkdir(parents=True, exist_ok=True)
    kg.save(output_kg)

    generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings, knowledge_graph=kg)
    distribution = [(SingleHopSpecificQuerySynthesizer(llm=generator_llm), 1.0)]
    testset = generator.generate(testset_size=testset_size, query_distribution=distribution)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    testset.to_pandas().to_csv(output_csv, index=False)
