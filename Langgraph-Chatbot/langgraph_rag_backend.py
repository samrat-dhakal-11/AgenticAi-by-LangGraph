from __future__ import annotations

import os
import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict
import requests

from dotenv import load_dotenv
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

#-----------------------------------------------------------------------------------
# Load environment variables
load_dotenv()

# Retrieve API keys
try:
    openai_api_key = os.getenv('OPENAI_API_KEY')
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    google_api_key = os.getenv("GEMINI_API_KEY2")
    gemini_api_key = os.getenv("api_key")
    groq_api_key = os.getenv("GROQ_API_KEY")
    tavily_api_key = os.getenv('TAVILY_API_KEY')
    openai_embeddings_key = os.getenv('OPENAI_API_KEY') # Used for OpenAIEmbeddings
except Exception as e:
    print("Error is ", e)

#------------------------------------------------------------------------------------
# Initialize LLMs & Embeddings
llm = ChatOpenAI(
    model='gpt-4o-mini',
    api_key=openai_api_key #type:ignore
)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key) #type:ignore

llm1 = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash", temperature=0.7, google_api_key=gemini_api_key
)
llm2 = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite", temperature=0.7, google_api_key=google_api_key
)
llm3 = ChatGroq(model="openai/gpt-oss-20b", api_key=groq_api_key) #type:ignore

llm4 = ChatOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=nvidia_api_key, #type:ignore
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    temperature=0.2,
)

#------------------------------------------------------------------------------------
# PDF retriever store (per thread)
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None

def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Build a FAISS retriever for the uploaded PDF and store it for the thread.
    """
    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 4}
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass

#-----------------------------------tools--------------------------------------------
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
        api_key = os.getenv('ALPHAVANTAGE_API_KEY')
    except:
        print("ALPHAVANTAGE_API_KEY not found")
    
    try:    
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
        r = requests.get(url)
        return r.json()
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
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }

    result = retriever.invoke(query)
    context = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]

    return {
        "query": query,
        "context": context,
        "metadata": metadata,
        "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename"),
    }

#------------------------------------------------------------------------------------
# Binding tools with LLMs
tools = [t for t in [search_tool, calculator, get_stock_price, rag_tool] if t is not None]
llm_with_tools = llm.bind_tools(tools)

#------------------------------------------------------------------------------------
# Define State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

#------------------------------------------------------------------------------------
# Define Chat Node
def chat_node(state: ChatState, config=None):
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_instruction = SystemMessage(
        content=(
            "Important: The output should not contain ** and ###. "
            f"For questions about the uploaded PDF, call the `rag_tool` and include the thread_id `{thread_id}`. "
            "You can also use the web search, stock price, and calculator tools when helpful."
        )
    )    
    messages = [system_instruction] + state["messages"]
    response = llm_with_tools.invoke(messages, config=config) #type:ignore
    return {"messages": [response]}

tool_node = ToolNode(tools)

#------------------------------------------------------------------------------------
# Database & Checkpointer
conn = sqlite3.connect(database='chatbot_rag_langgraph.db', check_same_thread=False)
checkpointer = SqliteSaver(conn=conn) #type:ignore

#------------------------------------------------------------------------------------
# Set up Graph
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)

#------------------------------------------------------------------------------------
# Helpers
def get_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id']) #type:ignore
    return list(all_threads)

def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS

def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})


