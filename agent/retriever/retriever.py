"""
Handles retrieving context documents from the vectorstore.
"""

import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

def retrieve(state):
    """
    Retrieve documents from the Pinecone vector store.

    Args:
        state (dict): The current graph state.

    Returns:
        dict: Updated state with the retrieved context documents.
    """
    print("---RETRIEVING FROM VECTOR DB---")
    
    embeddings = PineconeEmbeddings(
        model=os.getenv("EMB_MODEL_NAME"),
        pinecone_api_key=os.getenv("PINECONE_API_KEY")
    )
    
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=os.getenv("PINECONE_INDEX_NAME"), 
        embedding=embeddings, 
        namespace="sound-ncert"
    )

    question = state["question"]

    # Retrieve documents based on the question
    retriever = docsearch.as_retriever()
    documents = retriever.invoke(question)
    print("---RETRIEVED---")
    return {"documents": documents, "question": question}
