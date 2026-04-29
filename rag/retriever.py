from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents import Document

def create_vector_store(chunks):
    print("⏳ Creating embeddings... this may take a minute first time")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vector_store = FAISS.from_documents(chunks, embeddings)
    print("✅ Vector store created successfully")
    return vector_store, chunks

def get_retriever(vector_store, chunks):

    # --- RETRIEVER 1: FAISS Vector Search ---
    faiss_retriever = vector_store.as_retriever(
        search_kwargs={"k": 10}
    )

    # --- RETRIEVER 2: BM25 Keyword Search ---
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 10

    # --- RERANKER: CrossEncoder model ---
    reranker = HuggingFaceCrossEncoder(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    class HybridRerankerRetriever:
        """
        Custom retriever that:
        1. Runs FAISS + BM25 in parallel
        2. Merges and deduplicates results
        3. Reranks using CrossEncoder
        4. Returns top 5
        """
        def invoke(self, query):
            # Step 1: Get results from both retrievers
            faiss_docs = faiss_retriever.invoke(query)
            bm25_docs = bm25_retriever.invoke(query)

            # Step 2: Merge and deduplicate by content
            seen = set()
            all_docs = []
            for doc in faiss_docs + bm25_docs:
                if doc.page_content not in seen:
                    seen.add(doc.page_content)
                    all_docs.append(doc)

            if not all_docs:
                return []

            # Step 3: Rerank using CrossEncoder
            pairs = [(query, doc.page_content) for doc in all_docs]
            scores = reranker.score(pairs)

            # Step 4: Sort by score and return top 5
            scored_docs = sorted(
                zip(scores, all_docs),
                key=lambda x: x[0],
                reverse=True
            )
            top_docs = [doc for _, doc in scored_docs[:5]]

            print(f"✅ Retrieved {len(all_docs)} chunks → reranked to top {len(top_docs)}")
            return top_docs

    print("✅ Advanced RAG retriever ready (Hybrid + Reranking)")
    return HybridRerankerRetriever()