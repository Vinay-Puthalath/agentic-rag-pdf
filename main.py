import gradio as gr
from pydantic import BaseModel
import tempfile
from fastapi import FastAPI
from utils.agentic_inference import predict_agentic_rag
from utils.inference import predict_rag

app = FastAPI()

class Request(BaseModel):
    prompt : str

class Response(BaseModel):
    response : str

@app.post("/basic",response_model=Response)
async def predict_api(prompt:Request):
    print("prompt:::",Request.prompt)
    response = await predict_rag(Request.prompt)
    return response

@app.post("/agentic",response_model=Response)
async def predict_api(prompt:Request):
    print("prompt:::",Request.prompt)
    response = await predict_agentic_rag(Request.prompt)
    return response


with gr.Blocks() as demo_rag:

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
    
    text_input = gr.Textbox(
        placeholder="Ask a question", lines=1, scale=8, label="Ask a question"
    )

    text_output = gr.Textbox(label="Text Response")


    audio_output = gr.Audio(type="filepath", label="Audio Response")


    async def handle_chat_and_audio(text):
        response_text, audio_path = await predict_agentic_rag(text)
        return response_text, audio_path 

    submit_btn = gr.Button("Submit")


    submit_btn.click(fn=handle_chat_and_audio, inputs=text_input, outputs=[text_output, audio_output])


if __name__ == "__main__":
    # mounting at the root path
    import os
    import uvicorn
    from dotenv import load_dotenv
    import argparse

    load_dotenv()
    parser = argparse.ArgumentParser(description="Select BASIC RAG APP or AGENTIC RAG APP")
    parser.add_argument('--type', type=str, default='BASIC', help='type of RAG APP')
    args = parser.parse_args()

    if args.type == "basic":
        app = gr.mount_gradio_app(app, demo_rag, path="/rag/basic")
    elif args.type == "agentic":
        app = gr.mount_gradio_app(app, demo_agentic_rag, path="/rag/agentic")

    uvicorn.run(
        app=app,
        host=os.getenv("UVICORN_HOST"),  
        port=int(os.getenv("UVICORN_PORT"))
    )

