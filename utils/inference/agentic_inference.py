import asyncio
from utils.agentic_stategraph import create_rag_agent
from utils.audio.add_sound import play_sound

async def predict_agentic_rag(qns:str, history=None)->str:

    """
    Asynchronously sends a query to the agentic RAG model, retrieves the generated response, 
    and plays the generated text as speech.

    :param qns: The question to be answered by the agentic RAG model.
    :type qns: str
    :param history: Optional conversation history for context (currently unused).
    :type history: Any, optional
    :return: The generated response from the agentic RAG model.
    :rtype: str
    """
    
    if not asyncio.get_event_loop().is_running():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    agentic_rag = create_rag_agent()

    response = agentic_rag.invoke({"question": qns})
    audio_path = play_sound(response['generation'])
    
    return response['generation'], audio_path

