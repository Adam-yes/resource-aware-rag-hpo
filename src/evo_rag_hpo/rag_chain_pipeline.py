"""Execute the Naive-RAG pipeline (retrieve -> prompt -> generate) for one configuration.

This module implements the inference half of the study: given a decoded configuration (model,
chunk size, chunk overlap, Top-k, temperature), it builds a LangChain RAG chain and runs it
asynchronously over the full 50-question benchmark. The chain has three stages:

1. **Retrieval** - embed the query and fetch the ``top_k`` most relevant chunks from the
   precomputed Chroma collection that matches the chunk-size/overlap genome. A ``top_k`` of 0
   selects a parameter-free zero-shot mode in which no context is retrieved.
2. **Prompting** - inject the retrieved context and the question into a strict, context-only
   instruction template.
3. **Generation** - decode an answer with the selected Ollama model at the configured
   temperature.

Reproduction notes
------------------
Two details are load-bearing for a faithful 1:1 reproduction and are therefore pinned:

* The **prompt template is reproduced verbatim** from the original experiment. Any change to
  wording, casing, or rule ordering changes model behavior and therefore the published scores.
* The **context window is fixed** at ``inference.num_ctx`` (5120 in the original study) for
  every candidate, rather than being derived dynamically from ``chunk_size * top_k``.
"""

from __future__ import annotations

from typing import Any

from .config import resolve_project_path
from .logger import load_column_as_list


async def run_async_rag_chain(params: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Build and run the RAG chain for every benchmark question under one configuration.

    Args:
        params: Decoded hyperparameters for a single individual, i.e. the output of
            :func:`evo_rag_hpo.config.decode_individual`.
        config: The resolved global configuration (paths, models, inference settings).

    Returns:
        One result dictionary per question, each containing the retrieved ``context``, the
        original ``question``, and the generated ``answer`` message, in evaluation-set order.
    """

    # Heavy third-party imports are deferred to keep module import cheap and side-effect free.
    from langchain_chroma import Chroma
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
    from langchain_ollama import ChatOllama, OllamaEmbeddings

    paths = config["paths"]
    models = config["models"]
    inference = config["inference"]

    # Load the user questions (the "user_input" column); reference answers are consumed later in
    # the evaluation module.
    questions = load_column_as_list(str(resolve_project_path(paths["evaluation_dataset"])), "user_input")

    # The embedding model is fixed so the query embedding is comparable across all collections.
    embeddings = OllamaEmbeddings(model=models["embedding"], keep_alive=inference["embedding_keep_alive"])
    top_k = params["top_k"]

    # --- Retrieval branch selection (Top-k switch) ---
    if top_k > 0:
        # Address the precomputed collection matching this individual's chunking genome; it is
        # never rebuilt here, only read.
        collection_name = f"chroma_{params['chunk_size']}_{params['chunk_overlap']}"
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=str(resolve_project_path(paths["persist_directory"])),
            embedding_function=embeddings,
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    else:
        # Zero-shot mode: Top-k = 0 means "no retrieval"; the model answers from the prompt
        # alone. This lets the search treat "retrieval off" as a valid configuration point.
        retriever = RunnableLambda(lambda _: [])

    # Strict, context-only instruction template - reproduced verbatim from the original study.
    # The explicit "I don't know." escape hatch and the no-outside-knowledge rule tie the
    # factual-correctness fitness signal to grounded retrieval rather than parametric recall.
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""### Instruction:
You are a concise assistant. Answer the Question strictly using only the provided Context.
Rules:
1. If the answer is not explicitly in the Context, output exactly: "I don't know."
2. Do not explain why you don't know.
3. Keep the answer short.
4. Do not use outside knowledge.
### Context:
{context}
### Question:
{question}
### Answer:
""",
    )

    # Generation model. ``num_ctx`` is fixed (not derived from chunk_size*top_k) and ``seed`` is
    # pinned so that, with temperature, decoding is as deterministic as the local backend allows.
    llm = ChatOllama(
        model=params["model_name"],
        temperature=params["temperature"],
        num_predict=inference["num_predict"],
        num_ctx=inference["num_ctx"],
        keep_alive=inference["llm_keep_alive"],
        seed=config["optimization"]["random_seed"],
    )

    def format_docs(docs):
        """Render retrieved documents into a single, enumerated context string."""

        return "\n\n".join(f"Context {idx + 1}: {doc.page_content}" for idx, doc in enumerate(docs))

    # Compose the chain: run retrieval and the pass-through question in parallel, then assign the
    # generated answer. ``abatch`` executes the chain concurrently across all questions.
    entry_point = RunnableParallel(context=retriever | format_docs, question=RunnablePassthrough())
    rag_chain = entry_point.assign(answer=prompt | llm)
    return await rag_chain.abatch(questions)
