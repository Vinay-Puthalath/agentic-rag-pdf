import gradio as gr
from pydantic import BaseModel
from utils.agentic_inference import predict_agentic_rag
from utils.inference import predict_rag
from fastapi import FastAPI


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


demo_rag = gr.ChatInterface(
    fn=predict_rag,
    textbox=gr.Textbox(
        placeholder="Ask a question", container=False,lines=1,scale=8
    ),
    title="BASIC RAG APP",
    undo_btn="Delete Previous",
    clear_btn="Clear",
)

demo_agentic_rag = gr.ChatInterface(
    fn=predict_agentic_rag,
    textbox=gr.Textbox(
        placeholder="Ask a question", container=False,lines=1,scale=8
    ),
    title="SMART RAG AGENT APP",
    undo_btn="Delete Previous",
    clear_btn="Clear",
)






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

