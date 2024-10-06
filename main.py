import gradio as gr
import logging
from pydantic import BaseModel
from fastapi import FastAPI
from utils.inference.agentic_inference import predict_agentic_rag
from utils.inference.basic_inference import predict_rag


app = FastAPI()

class Request(BaseModel):
    prompt : str

class Response(BaseModel):
    response : str

@app.post("/basic",response_model=Response)
async def predict_api_basic(prompt:Request):
    """
    Handles POST requests for basic RAG inference.
    """
    try:
        response = await predict_rag(prompt.prompt)
        return response
    except Exception as e:
        print(f"Error processing request: {e}")
        return {"response": "An error occurred during inference."}

@app.post("/agentic",response_model=Response)
async def predict_api_agentic(prompt:Request):
    """
    Handles POST requests for smart agentiic RAG inference.
    """
    try:
        response = await predict_agentic_rag(prompt.prompt)
        return response
    except Exception as e:
        print(f"Error processing request: {e}")
        return {"response": "An error occurred during inference."}

with gr.Blocks() as demo_rag:
    gr.Markdown("BASIC RAG APP")

    text_input = gr.Textbox(
        placeholder="Ask a question", lines=1, scale=8, label="Ask a question"
    )
    text_output = gr.Textbox(label="Text Response")

    audio_output = gr.Audio(type="filepath", label="Audio Response")

    async def handle_chat_and_audio(text):
        response_text, audio_path = await predict_rag(text)  
        return response_text, audio_path  

    submit_btn = gr.Button("Submit")

    submit_btn.click(fn=handle_chat_and_audio, inputs=text_input, outputs=[text_output, audio_output])

with gr.Blocks() as demo_agentic_rag:
    gr.Markdown("SMART AGENTIC RAG APP")
    
    text_input = gr.Textbox(
        placeholder="Ask a question", lines=1, scale=8, label="Ask a question"
    )

    text_output = gr.Textbox(label="Text Response")


    audio_output = gr.Audio(type="filepath", label="Audio Response")

    async def handle_chat_and_audio_agentic(text):
        response_text, audio_path = await predict_agentic_rag(text)
        return response_text, audio_path 

    submit_btn = gr.Button("Submit")
    submit_btn.click(fn=handle_chat_and_audio_agentic, inputs=text_input, outputs=[text_output, audio_output])


if __name__ == "__main__":
    # mounting at the root path
    import os
    import uvicorn
    from dotenv import load_dotenv
    import argparse
    logging.getLogger('asyncio').setLevel(logging.CRITICAL)

    load_dotenv()
    parser = argparse.ArgumentParser(description="Select BASIC RAG APP or AGENTIC RAG APP")
    parser.add_argument('--type', type=str, default='BASIC', help='type of RAG APP')
    args = parser.parse_args()

    # Mount the selected gradio app
    if args.type == "basic":
        app = gr.mount_gradio_app(app, demo_rag, path="/rag/basic")
    elif args.type == "agentic":
        app = gr.mount_gradio_app(app, demo_agentic_rag, path="/rag/agentic")

    # Start Uvicorn server
    uvicorn.run(
        app=app,
        host=os.getenv("UVICORN_HOST", "0.0.0.0"),  
        port=int(os.getenv("UVICORN_PORT", 7860))
    )

