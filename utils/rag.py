
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from dotenv import load_dotenv
import os
from langchain_pinecone import PineconeEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

class RAG:
    def __init__(self) -> None:
        self.pdf_folder_path = os.getenv('SOURCE_DATA')
        self.document_path = os.getenv('DOCUMENT_PATH')
        self.emb_model = os.getenv('EMB_MODEL_NAME')
        self.vector_store_path = os.getenv('VECTOR_STORE')
        self.pinecone_api_key = os.getenv('PINECONE_API_KEY')

    def get_embedding_model(self,emb_model) -> HuggingFaceBgeEmbeddings :
        model_name = emb_model
        embeddings = PineconeEmbeddings(
            model=model_name,
            pinecone_api_key = self.pinecone_api_key
        )
        return embeddings

    
    def load_vector_db(self):
        namespace = os.getenv("PINECONE_NAMESPACE")
        docsearch = PineconeVectorStore(index_name=os.getenv("PINECONE_INDEX_NAME"), embedding=self.get_embedding_model(self.emb_model), namespace=namespace)
        return docsearch
    
    def get_retriever(self):
        return self.load_vector_db().as_retriever()

