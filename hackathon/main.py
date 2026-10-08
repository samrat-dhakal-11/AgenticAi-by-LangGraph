# ============================================================
# IMPORTS
# ============================================================
import os
import tempfile
import requests
from dotenv import load_dotenv

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

from llama_parse import LlamaParse

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import AIMessageChunk, HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# ============================================================
# API KEYS
# ============================================================
openai_key = os.getenv("OPENAI_API_KEY")
llama_key = os.getenv("LLAMA_PARSE_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
alphavantage_key = os.getenv("ALPHAVANTAGE_API_KEY")

if not openai_key:
    print("WARNING: OPENAI_API_KEY not set")
if not llama_key:
    print("WARNING: LLAMA_PARSE_KEY not set")
if not tavily_api_key:
    print("WARNING: TAVILY_API_KEY not set")
if not alphavantage_key:
    print("WARNING: ALPHAVANTAGE_API_KEY not set")

# ============================================================
# LLMs & EMBEDDINGS
# ============================================================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# LLAMA PARSER
# ============================================================
parser = LlamaParse(result_type="markdown", api_key=llama_key)  # type: ignore

# ============================================================
# PARSED FILE HOLDER
# ============================================================
class Parsedfile:
    def __init__(self):
        self.filename: str = ""
        self.content: str = ""
        self.vectorstore: Optional[FAISS] = None

    def set_doc(self, filename: str, content: str):
        self.filename = filename
        self.content = content

    def set_vectorstore(self, vectorstore: FAISS):
        self.vectorstore = vectorstore

    def get_content(self) -> str:
        return self.content

    def get_vs(self) -> Optional[FAISS]:
        return self.vectorstore


doc = Parsedfile()

# ============================================================
# SPLITTER + VECTORSTORE BUILDER
# ============================================================
def split_document():
    """Split doc.content into chunks and build a FAISS vectorstore."""
    text = doc.get_content()
    if not text.strip():
        raise ValueError("No content to split. Upload a document first.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.create_documents([text])
    vectorstore = FAISS.from_documents(chunks, embedding_model)
    doc.set_vectorstore(vectorstore)

# ============================================================
# TOOLS
# ============================================================
@tool
def ragtool(query: str) -> str:
    """Search the uploaded document and return the most relevant passages.
    Use this when the user asks anything about the loaded document."""
    vectorstore = doc.get_vs()
    if vectorstore is None:
        return "No document loaded yet. Please upload a file first."

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )
    results = retriever.invoke(query)
    if not results:
        return "No relevant content found in the document."
    return "\n\n".join(d.page_content for d in results)


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div."""
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
        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch the latest stock price for a symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage."""
    if not alphavantage_key:
        return {"error": "ALPHAVANTAGE_API_KEY not set"}
    try:
        r = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": alphavantage_key,
            },
            timeout=10,
        )
        data = r.json()
        # Alpha Vantage returns these keys instead of a quote when rate limited or misused
        if "Note" in data or "Information" in data or "Error Message" in data:
            return {"error": data.get("Note") or data.get("Information") or data.get("Error Message")}
        return data
    except Exception as e:
        return {"error": str(e)}


# ----- Tavily (optional) -----
# TavilySearch reads TAVILY_API_KEY from the environment by default,
# and its tool name is "tavily_search".
try:
    search_tool = TavilySearch(max_results=5)
except Exception as e:
    print(f"Failed to initialize Tavily search: {e}")
    search_tool = None


# Collect non-None tools
tools = [t for t in [ragtool, search_tool, calculator, get_stock_price] if t is not None]
llm_with_tools = llm.bind_tools(tools)

# ============================================================
# GRAPH
# ============================================================
# Build the system prompt from the tools that actually loaded,
# so the model never references a tool that doesn't exist.
tool_lines = [
    "- Use `ragtool` when the user asks about the uploaded document.",
]
if search_tool is not None:
    tool_lines.append("- Use `tavily_search` for fresh web information.")
tool_lines.append("- Use `calculator` for arithmetic.")
tool_lines.append("- Use `get_stock_price` for stock quotes.")

SYSTEM_PROMPT = SystemMessage(
    content=(
        "You are a helpful assistant with access to tools.\n"
        + "\n".join(tool_lines)
        + "\nDo not use ** or ### formatting in your replies."
    )
)


def ChatNode(state: MessagesState):
    """Call the LLM with tools. Returns an AIMessage (with tool_calls if any)."""
    messages = [SYSTEM_PROMPT] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


checkpointer = InMemorySaver()

graph = StateGraph(MessagesState)
graph.add_node("chat", ChatNode)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "chat")
graph.add_conditional_edges("chat", tools_condition)   # chat -> tools OR END
graph.add_edge("tools", "chat")                        # tools -> chat loop

chatbot = graph.compile(checkpointer=checkpointer)

# ============================================================
# UPLOAD ENDPOINT
# ============================================================
@app.post("/app/upload")
async def receive_and_parse_files(file: UploadFile = File(...)):
    tmp_path = None
    try:
        contents = await file.read()
        filename = file.filename or "unknown_file"
        suffix = os.path.splitext(filename)[1] or ".pdf"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # Async parse so the event loop isn't blocked on large files
        parse_result = await parser.aload_data(tmp_path, extra_info={"filename": filename})
        parsed_text = "\n".join(d.text for d in parse_result)

        doc.set_doc(filename, parsed_text)
        split_document()

        return {"filename": filename, "status": "success"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parsing error: {e}")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

# ============================================================
# CHAT ENDPOINT (streaming)
# ============================================================
class Query(BaseModel):
    user_query: str
    thread_id: str = "default"


@app.post("/app:chat-backend")
async def chat(msg: Query):
    async def generate():
        async for chunk, _metadata in chatbot.astream(
            {"messages": [HumanMessage(content=msg.user_query)]},
            config={"configurable": {"thread_id": msg.thread_id}},
            stream_mode="messages",
        ):
            # Only stream the model's own text, not ToolMessage results or tool-call chunks
            if isinstance(chunk, AIMessageChunk) and chunk.content and not chunk.tool_call_chunks:
                yield chunk.content

    return StreamingResponse(generate(), media_type="text/plain") #type:ignore