""" 
Handles creatting an agentic rag model 
"""
from langgraph.graph import END, StateGraph
from agent.graphstate.graphstate import GraphState
from agent.retriever.retriever import retrieve
from agent.retriever.retrieval_grader import grade_documents
from agent.query.query_rephraser import rewrite_query
from agent.query.query_classifier import is_trivial_query, decide_trivial
from agent.search.websearch import decide_to_generate, web_search
from agent.generator.answer_generator import generate_answer

def create_rag_agent():

    """
    Creates and compiles an agentic RAG model with a 
    graph-based approach for processing queries and generating responses.

    :return: The compiled agentic RAG model.
    :rtype: StateGraph
    """

    agentic_rag = StateGraph(GraphState)

    # Define the nodes
    agentic_rag.add_node("query_classifier_node", is_trivial_query)
    agentic_rag.add_node("document_retriever_node", retrieve)  # retrieve
    agentic_rag.add_node("document_grader_node", grade_documents)  # grade documents
    agentic_rag.add_node("query_rewriter_node", rewrite_query)  # transform_query
    agentic_rag.add_node("web_search_node", web_search)  # web search
    agentic_rag.add_node("answer_generation_node", generate_answer)  # generate answer

    # Build graph
    agentic_rag.set_entry_point("query_classifier_node")
    agentic_rag.add_conditional_edges(
        "query_classifier_node",
        decide_trivial,
        {"answer_generation_node": "answer_generation_node", 
         "document_retriever_node": "document_retriever_node"},
    )
    agentic_rag.add_edge("document_retriever_node", "document_grader_node")
    agentic_rag.add_conditional_edges(
        "document_grader_node",
        decide_to_generate,
        {"query_rewriter_node": "query_rewriter_node", 
         "answer_generation_node": "answer_generation_node"},
    )
    agentic_rag.add_edge("query_rewriter_node", "web_search_node")
    agentic_rag.add_edge("web_search_node", "answer_generation_node")
    agentic_rag.add_edge("answer_generation_node", END)

    # Compile
    agentic_rag = agentic_rag.compile()

    return agentic_rag