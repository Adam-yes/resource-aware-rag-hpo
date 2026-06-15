"""Generate evaluation questions from technical documentation with Ragas."""

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
    """Generate a single-hop technical-document QA test set."""

    from langchain_community.document_loaders import DirectoryLoader
    from langchain_ollama import ChatOllama, OllamaEmbeddings
    from langchain_text_splitters import MarkdownTextSplitter
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.testset import TestsetGenerator
    from ragas.testset.graph import KnowledgeGraph, Node, NodeType
    from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
    from ragas.testset.transforms import apply_transforms, default_transforms

    generator_llm = LangchainLLMWrapper(ChatOllama(model=generator_model, temperature=0))
    generator_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=embedding_model))

    docs = DirectoryLoader(source_dir, glob="**/*.pdf").load()
    splits = MarkdownTextSplitter(chunk_size=7000, chunk_overlap=200).split_documents(docs)

    kg = KnowledgeGraph()
    for doc in splits:
        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata,
                },
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
