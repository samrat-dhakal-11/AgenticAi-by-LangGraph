import os
import sqlite3
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# Load environment variables
load_dotenv()

# Retrieve API keys
try:
    nvidia_api_key = os.getenv("NVIDIA_API_KEY")
    google_api_key = os.getenv("GEMINI_API_KEY2")
    gemini_api_key = os.getenv("api_key")
    groq_api_key = os.getenv("GROQ_API_KEY")
except Exception as e:
    print("Error is ", e)

# Initialize LLMs
llm = ChatGoogleGenerativeAI(
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

# Define State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Define Chat Node
def chat_node(state: ChatState):
    # Take user query from state
    messages = state["messages"]

    # Send to LLM
    response = llm2.invoke(f'{messages} \n Important:The output shouldnot contain ** and ###')

    # Response store state
    return {"messages": [response]}

#database
conn=sqlite3.connect(database='chatbot.db',check_same_thread=False)

#Checkpointer
checkpointer = SqliteSaver(conn=conn) #type:ignore

# Set up Graph and 
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)

def get_all_threads():
    all_threads=set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id']) #type:ignore

    return (list(all_threads))
