""" 
Handles creatting an agentic rag model 
"""
from langgraph.graph import END, StateGraph
from agent_utils.graphstate.graphstate import GraphState
from agent_utils.retriever.retriever import retrieve
from agent_utils.retriever.retrieval_grader import grade_documents
from agent_utils.query.query_rephraser import rewrite_query
from agent_utils.query.query_classifier import is_trivial_query
from agent_utils.search.websearch import decide_to_generate, web_search, decide_trivial
from agent_utils.rag_chain.qa_rag_chain import generate_answer

def create_rag_agent():

    """
    Creates and compiles an agentic RAG model with a 
    graph-based approach for processing queries and generating responses.

    :return: The compiled agentic RAG model.
    :rtype: StateGraph
    """

    agentic_rag = StateGraph(GraphState)

    # Define the nodes
    agentic_rag.add_node("is_trivial_query", is_trivial_query)
    agentic_rag.add_node("retrieve", retrieve)  # retrieve
    agentic_rag.add_node("grade_documents", grade_documents)  # grade documents
    agentic_rag.add_node("rewrite_query", rewrite_query)  # transform_query
    agentic_rag.add_node("web_search", web_search)  # web search
    agentic_rag.add_node("generate_answer", generate_answer)  # generate answer

    # Build graph
    agentic_rag.set_entry_point("is_trivial_query")
    agentic_rag.add_conditional_edges(
        "is_trivial_query",
        decide_trivial,
        {"generate_answer": "generate_answer", "retrieve": "retrieve"},
    )
    agentic_rag.add_edge("retrieve", "grade_documents")
    agentic_rag.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"rewrite_query": "rewrite_query", "generate_answer": "generate_answer"},
    )
    agentic_rag.add_edge("rewrite_query", "web_search")
    agentic_rag.add_edge("web_search", "generate_answer")
    agentic_rag.add_edge("generate_answer", END)

    # Compile
    agentic_rag = agentic_rag.compile()

    return agentic_rag