from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document
from raptor import build_raptor_tree
from adaptive_retrieval import create_adaptive_retriever

def create_vector_store(chunks):
    print("⏳ Creating embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2"
    )

    # Build RAPTOR tree — adds section + document summaries
    print("🌲 Building RAPTOR tree...")
    all_docs = build_raptor_tree(list(chunks), embeddings)

    # Store everything in FAISS
    vector_store = FAISS.from_documents(all_docs, embeddings)
    print(f"✅ Vector store created with {len(all_docs)} total docs (chunks + summaries)")
    return vector_store, all_docs

def get_retriever(vector_store, chunks):
    # FAISS retriever
    faiss_retriever = vector_store.as_retriever(
        search_kwargs={"k": 10}
    )

    # BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 10

    # CrossEncoder reranker
    reranker = HuggingFaceCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    class HybridRerankerRetriever:
        def invoke(self, query):
            # Hybrid search
            faiss_docs = faiss_retriever.invoke(query)
            bm25_docs = bm25_retriever.invoke(query)

            # Merge + deduplicate
            seen = set()
            all_docs = []
            for doc in faiss_docs + bm25_docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    all_docs.append(doc)

            if not all_docs:
                return []

            # Rerank
            pairs = [(query, doc.page_content) for doc in all_docs]
            scores = reranker.score(pairs)
            scored_docs = sorted(
                zip(scores, all_docs),
                key=lambda x: x[0],
                reverse=True
            )
            top_docs = [doc for _, doc in scored_docs[:5]]
            print(f"✅ Retrieved {len(all_docs)} → reranked to top {len(top_docs)}")
            return top_docs

    base_retriever = HybridRerankerRetriever()

    # Wrap with adaptive retrieval
    adaptive = create_adaptive_retriever(base_retriever)

    class AdaptiveHybridRetriever:
        def invoke(self, query):
            return adaptive(query)

    print("✅ Advanced retriever ready (RAPTOR + Hybrid + Reranking + Adaptive)")
    return AdaptiveHybridRetriever()