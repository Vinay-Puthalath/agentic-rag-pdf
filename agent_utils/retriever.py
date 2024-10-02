""" handles retrieving context documents from vectorstore"""
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings 
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()

def retrieve(state):
    """
    Retrieve documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents - that contains retrieved context documents
    """
    print("---RETRIEVAL FROM VECTOR DB---")

    
    embeddings = PineconeEmbeddings(
                model=os.getenv("EMB_MODEL_NAME"),
                pinecone_api_key = os.getenv("PINECONE_API_KEY")
                )
    docsearch = PineconeVectorStore.from_existing_index(  
        index_name=os.getenv("PINECONE_INDEX_NAME"), embedding=embeddings, namespace="sound-ncert"
    ) 

    question = state["question"]


    # Retrieval
    retriever = docsearch.as_retriever()
    documents = retriever.invoke(question)
    print("---RETRIEVED---")
    return {"documents": documents, "question": question}