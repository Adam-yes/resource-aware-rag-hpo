"""Naive-RAG pipeline execution with Ollama and Chroma."""

from __future__ import annotations

from typing import Any

from .config import resolve_project_path
from .logger import load_column_as_list
from .runtime import calculate_num_ctx


async def run_async_rag_chain(params: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Run a configured RAG chain for every question in the evaluation set."""

    from langchain_chroma import Chroma
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough
    from langchain_ollama import ChatOllama, OllamaEmbeddings

    paths = config["paths"]
    models = config["models"]
    questions = load_column_as_list(str(resolve_project_path(paths["evaluation_dataset"])), "user_input")

    embeddings = OllamaEmbeddings(model=models["embedding"], keep_alive=config["inference"]["embedding_keep_alive"])
    top_k = params["top_k"]

    if top_k > 0:
        collection_name = f"chroma_{params['chunk_size']}_{params['chunk_overlap']}"
        vectorstore = Chroma(
            collection_name=collection_name,
            persist_directory=str(resolve_project_path(paths["persist_directory"])),
            embedding_function=embeddings,
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    else:
        retriever = RunnableLambda(lambda _: [])

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "### Instruction:\n"
            "You are a concise assistant. Answer the question strictly using only the provided context.\n"
            'If the answer is not explicitly in the context, output exactly: "I don\'t know."\n'
            "Keep the answer short and do not use outside knowledge.\n"
            "### Context:\n{context}\n"
            "### Question:\n{question}\n"
            "### Answer:\n"
        ),
    )

    llm = ChatOllama(
        model=params["model_name"],
        temperature=params["temperature"],
        num_predict=config["inference"]["num_predict"],
        num_ctx=calculate_num_ctx(params, config),
        keep_alive=config["inference"]["llm_keep_alive"],
        seed=config["optimization"]["random_seed"],
    )

    def format_docs(docs):
        return "\n\n".join(f"Context {idx + 1}: {doc.page_content}" for idx, doc in enumerate(docs))

    entry_point = RunnableParallel(context=retriever | format_docs, question=RunnablePassthrough())
    rag_chain = entry_point.assign(answer=prompt | llm)
    return await rag_chain.abatch(questions)
