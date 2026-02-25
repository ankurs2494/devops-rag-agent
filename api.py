from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import traceback

from agent_old import get_agent

app = FastAPI()
build_chain = get_agent()  # ✅ stores the factory function, not a chain

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    with open("static/chat.html") as f:
        return f.read()

@app.get("/ask")
def ask(q: str):
    try:
        chain = build_chain(q)   # builds the right chain per query
        result = chain.invoke(q) # invoke on the actual chain object

        # LangChain often returns dict/AIMessage/etc. Normalize to plain text.
        if isinstance(result, dict):
            text = result.get("output") or result.get("text") or str(result)
        else:
            text = getattr(result, "content", None) or str(result)

        return {"response": text}

    except Exception:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Agent failed internally")
    