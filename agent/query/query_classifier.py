"""
Handles classifying trivial and non-trivial queries.
"""

from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Define system and human prompts for classification
SYS_PROMPT = """
You are a smart assistant. Your job is to classify whether a user's query is trivial or not.

A trivial query is one that involves greetings, small talk, farewells, or short affirmations.

If the query is trivial, respond with 'trivial'. If the query requires more complex processing, respond with 'non-trivial'.
"""

# ChatPromptTemplate to format the interaction
classifier_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYS_PROMPT),
        ("human", """
        Here is the initial question:
        {question}

        If the query is trivial, respond with 'trivial'. If the query requires more complex processing, respond with 'non-trivial'.
        """),
    ]
)

# Define the function to classify a query as trivial or non-trivial
def is_trivial_query(state):
    """
    Classify whether the given query is trivial using an LLM like GPT.
    
    Args:
        state (dict): The current graph state containing the user's question.
    
    Returns:
        dict: Updated state with a new key 'is_trivial' indicating classification result.
    """
    print("---CHECKING THE QUERY COMPLEXITY---")
    
    # Extract the question from the state
    question = state["question"]

    # Use the classifier prompt to invoke the LLM and classify the query
    try:
        # Prepare the messages based on the prompt template
        messages = classifier_prompt.format_messages(question=question)
        
        # Invoke the LLM to get the classification (trivial or non-trivial)
        response = llm.invoke(messages)
        classification = response.content.strip().lower()  # Clean the response

        print("---question---", question, "---class--", classification)

        # Return updated state with classification
        return {"is_trivial": classification, "question": question, "documents": []}
    
    except Exception as e:
        print(f"Error during classification: {e}")
        # Return a fallback state or log the error
        return {"is_trivial": "unknown", "question": question, "documents": []}

def decide_trivial(state):
    """
    Determines whether the query is trivial or not.

    Args:
        state (dict): The current graph state.

    Returns:
        node name: Decision on which node to call based on the triviality of the query.
    """
    is_trivial = state["is_trivial"]

    if is_trivial == "trivial":
        print("---TRIVIAL QUERY---")

        return "answer_generation_node"
    else:
        print("---NON TRIVIAL QUERY---")
        return "document_retriever_node"