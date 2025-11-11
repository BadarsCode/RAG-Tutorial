from src.load_documents import load_all_document 
from src.embedding import EmbeddingPipeline
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch


if __name__ == "__main__":
     # docs = load_all_document("data")
     # chunks = EmbeddingPipeline().chunk_documents(docs)
     # chunk_vector = EmbeddingPipeline().embed_chunks(chunks)
     # print(f"[INFO] Embedding Shape: {chunk_vector.shape}")
     store = FaissVectorStore('faiss_store')
     # store.build_from_documents(docs)
     store.load()
     print(store.query("what is the attension mechanism in transformers?", top_k = 10))


     rag_search = RAGSearch()
     query = "what is the attension mechanism in transformers?"
     summary = rag_search.search_and_summarize(query, top_k=10)
     print("Summary:", summary)

