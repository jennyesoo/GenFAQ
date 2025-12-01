from pathlib import Path
from typing import Optional
from langchain_community.document_loaders import TextLoader, UnstructuredExcelLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


def _load_documents(data_path: str):
    p = Path(data_path)
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    suffix = p.suffix.lower()
    if suffix in {".txt", ".md"}:
        loader = TextLoader(str(p), encoding="utf-8")
    elif suffix in {".xlsx", ".xls"}:
        # 需要安裝: unstructured, openpyxl
        loader = UnstructuredExcelLoader(str(p))
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

    return loader.load()


def build_retriever(
    data_path: str,
    embed_model: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
    top_k: int = 4,
    score_threshold: Optional[float] = None,
    persist_directory: Optional[str] = None,
    collection_name: str = "rag_collection",
    device: str = "cuda",
):
    """Build a Chroma retriever from a text/xlsx file.

    - Uses HuggingFaceInstructEmbeddings (e.g., hkunlp/instructor-large)
    - Persists to disk if persist_directory is provided
    """
    docs = _load_documents(data_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceInstructEmbeddings(
        model_name=embed_model,
        model_kwargs={"device": device},
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    if persist_directory:
        vectordb.persist()

    # standard retriever (or with score threshold)
    retriever = vectordb.as_retriever(search_kwargs={"k": top_k})
    if score_threshold is not None:
        retriever = vectordb.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": score_threshold, "k": top_k},
        )
    return retriever