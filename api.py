from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import traceback

from agent import get_agent

app = FastAPI()
agent = get_agent()

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    with open("static/chat.html") as f:
        return f.read()

@app.get("/ask")
def ask(q: str):
    try:
        response = agent.invoke(q)
        print("Response generated")
        return {"response": response}
    except Exception as e:
        print("Agent error:", e)
        return {"response": "Agent failed internally"}
    