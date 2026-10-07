from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_tavily import TavilySearch
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from pydantic import BaseModel

# ------------------------------------------------------------------ env + models
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)  # type: ignore
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)  # type: ignore

# ------------------------------------------------------------------ PDF store (per thread)
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    if thread_id and str(thread_id) in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[str(thread_id)]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        temp_path = tmp.name

    try:
        docs = PyPDFLoader(temp_path).load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)
        store = FAISS.from_documents(chunks, embeddings)
        _THREAD_RETRIEVERS[str(thread_id)] = store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )
        summary = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
        _THREAD_METADATA[str(thread_id)] = summary
        return summary
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


# ------------------------------------------------------------------ tools
try:
    search_tool = TavilySearch(api_key=tavily_api_key, max_results=3)
except Exception as e:
    print(f"Failed to initialize Tavily search: {e}")
    search_tool = None


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage with API key.
    """
    try:
        api_key = os.getenv("ALPHAVANTAGE_API_KEY")
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
        return requests.get(url, timeout=15).json()
    except Exception as e:
        return {"error": str(e)}


@tool
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the uploaded PDF for this chat thread.
    Always include the thread_id when calling this tool.
    """
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {"error": "No document indexed for this chat. Upload a PDF first.", "query": query}

    result = retriever.invoke(query)
    return {
        "query": query,
        "context": [d.page_content for d in result],
        "metadata": [d.metadata for d in result],
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }


tools = [t for t in [search_tool, calculator, get_stock_price, rag_tool] if t is not None]
llm_with_tools = llm.bind_tools(tools)


# ------------------------------------------------------------------ graph
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState, config=None):
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_instruction = SystemMessage(
        content=(
            "You are a helpful assistant. Format answers in clear Markdown when it helps "
            "(short paragraphs, lists, fenced code blocks). "
            f"For questions about the uploaded PDF, call the `rag_tool` and include the thread_id `{thread_id}`. "
            "You can also use the web search, stock price, and calculator tools when helpful."
        )
    )
    response = llm_with_tools.invoke([system_instruction] + state["messages"], config=config)  # type: ignore
    return {"messages": [response]}


conn = sqlite3.connect(database="chatbot_rag_langgraph.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)  # type: ignore

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", ToolNode(tools))
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")
chatbot = graph.compile(checkpointer=checkpointer)


# ------------------------------------------------------------------ helpers
def extract_message_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content:
        block = content[0]
        if isinstance(block, dict) and "text" in block:
            return block["text"]
    return str(content) if content else ""


def get_all_threads() -> list[str]:
    seen: list[str] = []
    for cp in checkpointer.list(None):
        tid = cp.config["configurable"]["thread_id"]  # type: ignore
        if tid not in seen:
            seen.append(tid)
    return seen


def load_conversation(thread_id: str):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


def make_title(messages) -> str:
    for m in messages:
        if isinstance(m, HumanMessage):
            txt = extract_message_content(m.content).strip()
            return txt[:40] + "..." if len(txt) > 40 else txt
    return "New chat"


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# ------------------------------------------------------------------ FastAPI app
app = FastAPI(title="LangGraph RAG Chatbot")
HERE = os.path.dirname(os.path.abspath(__file__))


class ChatRequest(BaseModel):
    thread_id: str
    message: str


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "hlo.html"))


@app.get("/api/threads")
def api_threads():
    out = []
    for tid in get_all_threads():
        msgs = load_conversation(tid)
        if msgs:
            out.append({"id": tid, "title": make_title(msgs)})
    return out[::-1]


@app.get("/api/threads/{thread_id}")
def api_thread(thread_id: str):
    messages = []
    for m in load_conversation(thread_id):
        if isinstance(m, HumanMessage):
            messages.append({"role": "user", "content": extract_message_content(m.content)})
        elif isinstance(m, AIMessage):
            txt = extract_message_content(m.content)
            if txt:
                messages.append({"role": "assistant", "content": txt})
    return {"messages": messages, "document": _THREAD_METADATA.get(str(thread_id))}


@app.delete("/api/threads/{thread_id}")
def api_delete_thread(thread_id: str):
    try:
        checkpointer.delete_thread(thread_id)
    except Exception as e:
        raise HTTPException(500, f"Could not delete chat: {e}")
    _THREAD_RETRIEVERS.pop(thread_id, None)
    _THREAD_METADATA.pop(thread_id, None)
    return {"deleted": thread_id}


@app.post("/api/upload")
def api_upload(file: UploadFile = File(...), thread_id: str = Form(...)):
    try:
        return ingest_pdf(file.file.read(), thread_id=thread_id, filename=file.filename)
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/api/chat")
def api_chat(req: ChatRequest):
    config = {
        "configurable": {"thread_id": req.thread_id},
        "metadata": {"thread_id": req.thread_id},
        "run_name": "chat_turn",
    }

    def generate():
        try:
            for chunk, _meta in chatbot.stream(
                {"messages": [HumanMessage(content=req.message)]},
                config=config,  # type: ignore
                stream_mode="messages",
            ):
                if isinstance(chunk, ToolMessage):
                    yield sse({"type": "tool_done", "name": getattr(chunk, "name", "tool")})
                elif isinstance(chunk, AIMessage):
                    for tc in getattr(chunk, "tool_call_chunks", None) or []:
                        if tc.get("name"):
                            yield sse({"type": "tool", "name": tc["name"]})
                    txt = extract_message_content(chunk.content)
                    if txt:
                        yield sse({"type": "token", "text": txt})
        except Exception as e:
            yield sse({"type": "error", "message": str(e)})
        yield sse({"type": "done"})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    # python hlo.py  ->  open http://localhost:8000   (API docs at /docs)
    uvicorn.run(app, host="127.0.0.1", port=8000)