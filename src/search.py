import os 
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq


load_dotenv()
class RAGSearch:
    def __init__(self, persist_dir: str='faiss_store', embedding_model: str='all-MiniLM-L6-v2', chunk_size: int=1000, chunk_overlap: int=10, llm_model: str='llama-3.1-8b-instant'):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        #load or build vectors
        faiss_path = os.path.join(persist_dir, 'faiss.index')
        meta_path = os.path.join(persist_dir, 'metadata.pkl')
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from load_documents import load_all_documents
            docs = load_all_documents('data')
            self.vector_store.build_from_documents(docs)
        else:
            self.vectorstore.load()
        groq_api_key = os.getenv('GROQ_API_KEY')
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM Initialized {llm_model}")
    

    def search_and_summarize(self, query: str, top_k: int = 5, ) ->str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts =  [r['metadata'].get("text", "") for r in results if r['metadata']]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant document found"
        prompt = f""" summarize the following context for the query: {query} \n\ncontext: {context} \n\n summary: """
        response = self.llm.invoke([prompt])
        return response.content
    