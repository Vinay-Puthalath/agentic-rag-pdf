"""
Handles creating a RAG chain and generating an answer from context documents using an LLM.
"""

from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser


def format_docs(docs):
    """
    Formats the retrieved context documents for the LLM.

    Args:
        docs (list): List of context documents.

    Returns:
        str: Formatted string combining the content of all documents.
    """
    return "\n\n".join(doc.page_content for doc in docs)


def create_rag_chain():
    """
    Creates a RAG (Retrieval-Augmented Generation) chain to generate answers from context documents.

    Returns:
        qa_rag_chain (Callable): The created RAG chain for generating answers.
    """
    prompt = """
    You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If no context is present or if you don't know the answer, just say that you don't know the answer.
    But if the question is a trivial one like greetings, respond or greet the user back.
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

    # Create QA RAG chain
    qa_rag_chain = (
        {
            "context": (
                itemgetter('context') |
                RunnableLambda(format_docs)
            ),
            "question": itemgetter('question')
        } |
        prompt_template |
        chatgpt |
        StrOutputParser()
    )
    
    return qa_rag_chain


def generate_answer(state):
    """
    Generate an answer from the context document using the LLM.

    Args:
        state (dict): The current graph state containing the question and documents.

    Returns:
        dict: Updated state with a new key 'generation' that contains the LLM generation result.
    """
    print("---GOT RELEVANT CONTENT, GENERATING ANSWER---")
    
    question = state["question"]
    documents = state.get("documents", [])

    # RAG generation
    generation = qa_rag_chain.invoke({"context": documents, "question": question})
    
    return {"documents": documents, "question": question, "generation": generation}


# Initialize the RAG chain
qa_rag_chain = create_rag_chain()
