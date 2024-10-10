"""
Represents the state of our StateGraph.
"""

from typing import List, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM response generation
        web_search_needed: flag of whether to add web search - yes or no
        documents: list of context documents
    """
    messages: Annotated[list[AnyMessage], add_messages]
    question: str
    generation: str
    web_search_needed: str
    documents: List[str]