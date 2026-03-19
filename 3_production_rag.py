"""
Production RAG Ecosystem:
  - Document ingestion with recursive character splitting
  - ChromaDB vector indexing
  - LangChain orchestration
  - watsonx.ai as the generative backbone
"""

import os
from pathlib import Path

from langchain.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ibm import WatsonxLLM                # pip install langchain-ibm
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# ── Config ────────────────────────────────────────────────────────────────────
WATSONX_URL     = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY", "YOUR_API_KEY")
WATSONX_PROJECT = os.getenv("WATSONX_PROJECT_ID", "YOUR_PROJECT_ID")

DOCS_DIR        = "./documents"
CHROMA_DIR      = "./chroma_db"
EMBED_MODEL     = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 200

# ── 1. Document ingestion ─────────────────────────────────────────────────────
def load_documents(docs_dir: str):
    loaders = [
        DirectoryLoader(docs_dir, glob="**/*.pdf", loader_cls=PyPDFLoader),
        DirectoryLoader(docs_dir, glob="**/*.txt", loader_cls=TextLoader),
    ]
    docs = []
    for loader in loaders:
        try:
            docs.extend(loader.load())
        except Exception as e:
            print(f"[warn] loader error: {e}")
    print(f"Loaded {len(docs)} document pages/chunks from {docs_dir}")
    return docs

# ── 2. Recursive character splitting ─────────────────────────────────────────
def split_documents(documents):
    """
    RecursiveCharacterTextSplitter tries to split on paragraph → sentence →
    word boundaries in order, preserving semantic coherence.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size         = CHUNK_SIZE,
        chunk_overlap      = CHUNK_OVERLAP,
        separators         = ["\n\n", "\n", ". ", " ", ""],
        length_function    = len,
        is_separator_regex = False,
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks (size≤{CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks

# ── 3. ChromaDB vector indexing ───────────────────────────────────────────────
def build_vectorstore(chunks, persist_dir: str):
    embeddings = HuggingFaceEmbeddings(
        model_name       = EMBED_MODEL,
        model_kwargs     = {"device": "cpu"},
        encode_kwargs    = {"normalize_embeddings": True},
    )
    vectordb = Chroma.from_documents(
        documents        = chunks,
        embedding        = embeddings,
        persist_directory= persist_dir,
        collection_metadata={"hnsw:space": "cosine"},
    )
    vectordb.persist()
    print(f"ChromaDB persisted at {persist_dir} ({vectordb._collection.count()} vectors)")
    return vectordb, embeddings

def load_vectorstore(persist_dir: str, embeddings):
    return Chroma(
        persist_directory = persist_dir,
        embedding_function= embeddings,
    )

# ── 4. watsonx.ai LLM ────────────────────────────────────────────────────────
def build_watsonx_llm():
    params = {
        GenParams.DECODING_METHOD : "greedy",
        GenParams.MAX_NEW_TOKENS  : 512,
        GenParams.MIN_NEW_TOKENS  : 1,
        GenParams.TEMPERATURE     : 0.7,
        GenParams.TOP_K           : 50,
        GenParams.TOP_P           : 0.95,
        GenParams.REPETITION_PENALTY: 1.1,
    }
    llm = WatsonxLLM(
        model_id   = ModelTypes.LLAMA_2_70B_CHAT.value,
        url        = WATSONX_URL,
        apikey     = WATSONX_API_KEY,
        project_id = WATSONX_PROJECT,
        params     = params,
    )
    return llm

# ── 5. RAG chain ──────────────────────────────────────────────────────────────
RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Use ONLY the context below to
answer the question. If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}

Answer:"""

def build_rag_chain(vectordb, llm):
    retriever = vectordb.as_retriever(
        search_type   = "mmr",          # maximal marginal relevance for diversity
        search_kwargs = {"k": 5, "fetch_k": 20, "lambda_mult": 0.7},
    )
    prompt = PromptTemplate(
        template       = RAG_PROMPT_TEMPLATE,
        input_variables= ["context", "question"],
    )
    chain = RetrievalQA.from_chain_type(
        llm              = llm,
        chain_type       = "stuff",
        retriever        = retriever,
        return_source_documents = True,
        chain_type_kwargs= {"prompt": prompt},
    )
    return chain

# ── 6. Query interface ────────────────────────────────────────────────────────
def query_rag(chain, question: str) -> dict:
    result = chain({"query": question})
    print("\n" + "="*60)
    print(f"Q: {question}")
    print(f"\nA: {result['result']}")
    print("\n--- Sources ---")
    for doc in result["source_documents"]:
        src  = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        print(f"  • {src} (page {page})")
    return result

# ── 7. Pipeline entry point ───────────────────────────────────────────────────
def build_pipeline(docs_dir: str = DOCS_DIR, chroma_dir: str = CHROMA_DIR):
    Path(docs_dir).mkdir(exist_ok=True)

    documents  = load_documents(docs_dir)
    chunks     = split_documents(documents)
    vectordb, embeddings = build_vectorstore(chunks, chroma_dir)
    llm        = build_watsonx_llm()
    chain      = build_rag_chain(vectordb, llm)
    return chain

if __name__ == "__main__":
    chain = build_pipeline()

    questions = [
        "What are the main findings of the report?",
        "Summarize the key recommendations.",
        "What methodology was used in the study?",
    ]
    for q in questions:
        query_rag(chain, q)
