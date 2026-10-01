import os
import requests
import asyncio
import aiosqlite
import threading
from typing import Annotated, TypedDict
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.tools import tool, BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# -----------------------------------------------------------------------------------
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
except Exception as e:
    print("Error is ", e)

# ------------------------------------------------------------------------------------
# Dedicated async loop for backend tasks (bridging sync/async seamlessly)
_ASYNC_LOOP = asyncio.new_event_loop()
_ASYNC_THREAD = threading.Thread(target=_ASYNC_LOOP.run_forever, daemon=True)
_ASYNC_THREAD.start()


def _submit_async(coro):
    return asyncio.run_coroutine_threadsafe(coro, _ASYNC_LOOP)


def run_async(coro):
    return _submit_async(coro).result()


def submit_async_task(coro):
    """Schedule a coroutine on the backend event loop."""
    return _submit_async(coro)

# ------------------------------------------------------------------------------------
# Initialize LLMs
llm = ChatOpenAI(
    model='gpt-4o-mini',
    api_key=openai_api_key  # type:ignore
)
llm1 = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash", temperature=0.7, google_api_key=gemini_api_key
)
llm2 = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite", temperature=0.7, google_api_key=google_api_key
)
llm3 = ChatGroq(model="openai/gpt-oss-20b", api_key=groq_api_key)  # type:ignore

llm4 = ChatOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=nvidia_api_key,  # type:ignore
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    temperature=0.2,
)

# ----------------------------------- Tools ------------------------------------------
search_tool = TavilySearch(api_key=tavily_api_key,max_results=3)


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
    using Alpha Vantage with API key in the URL.
    """
    try:
        api_key = os.getenv('ALPHAVANTAGE_API_KEY')
    except:
        print("ALPHAVANTAGE_API_KEY not found")
        api_key = ""
    
    try:    
        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}"
        r = requests.get(url)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# Multi-Server MCP Client Configuration
server_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp-math-server.py")

client = MultiServerMCPClient(
    {
        'arith': {
            'transport': 'stdio',
            'command': 'python3',
            'args': [server_path]
        },
        'expense': {
            'transport': 'streamable_http',  # if this fails, try "sse"
            'url': 'https://splendid-gold-dingo.fastmcp.app/mcp'
        }
    }
)


def load_mcp_tools() -> list[BaseTool]:
    try:
        return run_async(client.get_tools())
    except Exception as e:
        print(f"Failed to load MCP tools: {e}")
        return []


mcp_tools = load_mcp_tools()

# Binding all tools (local + MCP)
tools = [search_tool, calculator, get_stock_price, *mcp_tools]
llm_with_tools = llm.bind_tools(tools) if tools else llm

# ------------------------------------------------------------------------------------
# Define State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ------------------------------------------------------------------------------------
# Define Chat Node (Async implementation)
async def chat_node(state: ChatState):
    messages = state["messages"]
    system_instruction = SystemMessage(content="Important: The output should not contain ** and ###")    
    response = await llm_with_tools.ainvoke([system_instruction] + messages)
    return {"messages": [response]}


tool_node = ToolNode(tools) if tools else None

# ------------------------------------------------------------------------------------
# Asynchronous Database & Checkpointer Setup
async def _init_checkpointer():
    db_path = 'mcp_based_db.db'
    conn = await aiosqlite.connect(database=db_path)
    return AsyncSqliteSaver(conn)


checkpointer = run_async(_init_checkpointer())

# ------------------------------------------------------------------------------------
# Set up Graph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")

if tool_node:
    graph.add_node("tools", tool_node)
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge('tools', 'chat_node')
else:
    graph.add_edge('chat_node', END)

chatbot = graph.compile(checkpointer=checkpointer)

# ------------------------------------------------------------------------------------
# Helpers (Async thread-safe thread listing)
async def _alist_threads():
    all_threads = set()
    async for checkpoint in checkpointer.alist(None):
        if checkpoint.config and 'configurable' in checkpoint.config:
            thread_id = checkpoint.config['configurable'].get('thread_id')
            if thread_id:
                all_threads.add(thread_id)
    return list(all_threads)


def retrieve_all_threads():
    return run_async(_alist_threads())

