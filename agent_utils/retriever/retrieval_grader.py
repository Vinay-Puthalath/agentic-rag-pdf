"""
Handles grading the relevance of retrieved documents to the user query.
"""

from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# Data model for LLM output format
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'."
    )


# LLM for grading
llm = ChatOpenAI(model="gpt-4o", temperature=0)
structured_llm_grader = llm.with_structured_output(GradeDocuments)

# Prompt template for grading
SYS_PROMPT = """
You are an expert grader assessing relevance of a retrieved document to a user question.
Follow these instructions for grading:
  - If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant.
  - Your grade should be either 'yes' or 'no' to indicate whether the document is relevant to the question or not.
"""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYS_PROMPT),
        ("human", """Retrieved document:
                     {document}

                     User question:
                     {question}
                  """),
    ]
)

# Build grader chain
doc_grader = grade_prompt | structured_llm_grader


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    by using an LLM Grader.

    If documents are not relevant to the question or documents are empty,
    web search needs to be done. If all documents are relevant, web search is not needed.

    Args:
        state (dict): The current graph state.

    Returns:
        dict: Updated state with only filtered relevant documents.
    """
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    print(f"RETRIEVED {len(documents)} DOCUMENTS")

    # Score each document
    filtered_docs = []
    web_search_needed = "No"

    if documents:
        for doc in documents:
            score = doc_grader.invoke(
                {"question": question, "document": doc.page_content}
            )
            grade = score.binary_score

            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(doc)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search_needed = "Yes"
    else:
        print("---NO DOCUMENTS RETRIEVED---")
        web_search_needed = "Yes"

    return {
        "documents": filtered_docs,
        "question": question,
        "web_search_needed": web_search_needed
    }
