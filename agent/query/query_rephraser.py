"""
Handles rephrasing a user query to optimize it for web search.
"""

from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# LLM for question rewriting
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Prompt template for rewriting
SYS_PROMPT = """
Act as a question re-writer and perform the following task:
  - Convert the following input question to a better version that is optimized for web search.
  - When re-writing, look at the input question and try to reason about the underlying semantic intent/meaning.
"""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYS_PROMPT),
        ("human", """Here is the initial question:
                     {question}

                     Formulate an improved question.
                  """),
    ]
)

# Create rephraser chain
question_rewriter = re_write_prompt | llm | StrOutputParser()


def rewrite_query(state):
    """
    Rewrite the query to produce a better question.

    Args:
        state (dict): The current graph state.

    Returns:
        dict: Updates the 'question' key with a rephrased or re-written question.
    """
    print("---REPHRASE QUERY---")
    print("rephrasing query for web search...")
    
    question = state["question"]
    documents = state["documents"]

    # Re-write the question
    better_question = question_rewriter.invoke({"question": question})
    
    return {"documents": documents, "question": better_question}
