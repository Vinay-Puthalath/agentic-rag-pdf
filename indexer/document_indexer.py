import time
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


def index_document(DOC_PATH):

    load_dotenv()

    pinecone_api_key=os.getenv("PINECONE_API_KEY")
    cloud = os.getenv('PINECONE_CLOUD') 
    region = os.getenv('AWS_REGION') 
    namespace = os.getenv("PINECONE_NAMESPACE")
    index_name = os.getenv('PINECONE_INDEX_NAME')

    loader = PyPDFLoader(DOC_PATH)
    pages = loader.load()
    # split the doc into smaller chunks i.e. chunk_size=500
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(pages)
    model_name = os.getenv("EMB_MODEL_NAME")
    embeddings = PineconeEmbeddings(
        model=model_name,
        pinecone_api_key=pinecone_api_key
    )
    pc = Pinecone(api_key=pinecone_api_key)
    spec = ServerlessSpec(cloud=cloud, region=region)
    

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=embeddings.dimension,
            metric="cosine",
            spec=spec
        )
        # wait for index to be ready
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

    PineconeVectorStore.add_documents(
        documents=chunks,
        index_name=index_name,
        embedding=embeddings,
        namespace=namespace
    )

    time.sleep(1)

if __name__ == "__main__":
    import argparse

    load_dotenv()
    parser = argparse.ArgumentParser(description="Get the path to the pdf document.")
    parser.add_argument('--filepath', type=str, default='BASIC', help='please input the path to the pdf file')
    args = parser.parse_args()

    index_document(args.filepath)