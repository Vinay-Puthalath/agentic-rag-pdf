"""
Handles the websearch component
"""

from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults

# Initialize Tavily search results
tv_search = TavilySearchResults(
    max_results=3, 
    search_depth='advanced',
    max_tokens=10000
)

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state.

    Returns:
        str: Binary decision for the next node to call.
    """
    web_search_needed = state["web_search_needed"]

    if web_search_needed == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: SOME or ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, REPHRASING---")
        return "query_rewriter_node"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE RESPONSE---")
        return "answer_generation_node"

def web_search(state):
    """
    Web search based on the re-written question.

    Args:
        state (dict): The current graph state.

    Returns:
        dict: Updates the documents key with appended web results.
    """
    print("---SEARCHING THE WEB---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    docs = tv_search.invoke(question)
    web_results = "\n\n".join([d["content"] for d in docs])
    web_results = Document(page_content=web_results)
    documents.append(web_results)

    return {"documents": documents, "question": question}
