
import os
import asyncio
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings 
from langchain_openai import ChatOpenAI
from utils.add_sound import play_sound



async def predict_rag(question:str, history=None)->str:

    """
    Asynchronously performs a Retrieval-Augmented Generation (RAG) query, retrieves relevant documents, 
    and generates an answer to the input question. It also plays the generated answer as speech.

    :param question: The input question to be answered.
    :type question: str
    :param history: Optional conversation history for context (currently unused).
    :type history: Any, optional
    :return: The generated answer to the question.
    :rtype: str
    """

    print("prompt :::", question)
    if not asyncio.get_event_loop().is_running():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    embeddings = PineconeEmbeddings(
                model=os.getenv("EMB_MODEL_NAME"),
                pinecone_api_key = os.getenv("PINECONE_API_KEY")
            )
    docsearch = PineconeVectorStore.from_existing_index(  
        index_name=os.getenv("PINECONE_INDEX_NAME"), embedding=embeddings, namespace=os.getenv("PINECONE_NAMESPACE")
    ) 

    retriever = docsearch.as_retriever()
    documents = retriever.invoke(question)

    prompt = """You are an assistant for question-answering tasks.
                Use the following pieces of retrieved context to answer the question.
                If no context is present or if you don't know the answer, just say that you don't know the answer.
                Do not make up the answer unless it is there in the provided context.
                Give a detailed answer in not more than 450 characters and to the point answer with regard to the question.

                Question:
                {question}

                Context:
                {context}

                Answer:
            """
    prompt_template = ChatPromptTemplate.from_template(prompt)

    # Initialize connection with GPT-4o
    chatgpt = ChatOpenAI(model_name='gpt-4o', temperature=0)
    # Used for separating context docs with new lines
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # create QA RAG chain
    qa_rag_chain = (
        {
            "context": (itemgetter('context')
                            |
                        RunnableLambda(format_docs)),
            "question": itemgetter('question')
        }
        |
        prompt_template
        |
        chatgpt
        |
        StrOutputParser()
    )

    generation = qa_rag_chain.invoke({"context": documents, "question": question})
    # print('generation::', generation)
    play_sound(generation)
    
    return generation

